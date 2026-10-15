"""So sánh DyTimeNet baseline với IntraVMoE trên ETTh1: train 3 seed, đo FLOPs/latency, xuất bảng.

Ví dụ: python scripts/bench_moe.py            (chạy tiếp các run còn thiếu rồi viết báo cáo)
       python scripts/bench_moe.py --report_only
Kết quả từng run được lưu ngay vào results/moe_etth1_runs.json nên có thể chạy lại để tiếp tục.
"""

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from dytimenet.config import config_from_dict, load_config
from dytimenet.data.dataset import TimeSeriesDataset
from dytimenet.experiment import train
from dytimenet.layers.cross_v_block import DropFusionLinear
from dytimenet.models import DyTimeNet

CONFIGS = {
    "baseline": {},
    "moe_k1_sparse": {"use_moe": True, "num_experts": 4, "top_k": 1, "moe_dispatch": "sparse"},
    "moe_k1_dense": {"use_moe": True, "num_experts": 4, "top_k": 1, "moe_dispatch": "dense"},
    "moe_k2_sparse": {"use_moe": True, "num_experts": 4, "top_k": 2, "moe_dispatch": "sparse"},
}
SEEDS = (2021, 2022, 2023)
WARMUP, REPS = 20, 200


def timed_median_ms(fn):
    """Warm-up 20 lần rồi đo 200 lần; trả median (ms)."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(WARMUP):
        fn()
    sync()
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000


def count_dropfusion(module, inputs, output):
    """thop không biết DropFusionLinear (F.linear tùy biến): MACs = số phần tử ra × in_features."""
    module.total_ops += torch.DoubleTensor([int(output.numel() * module.in_features)])


def thop_macs(module, inputs):
    from thop import profile
    macs, _ = profile(copy.deepcopy(module), inputs=inputs, verbose=False,
                      custom_ops={DropFusionLinear: count_dropfusion})
    return float(macs)


def manual_intra_flops(kw, num_vars):
    """FLOPs (= 2·MACs, chỉ phép nhân ma trận) của FFN/MoE IntraV cho 1 mẫu, mọi layer."""
    d = kw["d_model"]
    d_ff = int(d * kw["ffn_ratio"])
    tokens = num_vars * kw["num_layers"]
    if not kw["use_moe"]:
        flops = 2 * 2 * d * d_ff
        return {"active": tokens * flops, "executed": tokens * flops}
    e, k = kw["num_experts"], kw["top_k"]
    shared, expert = 2 * 2 * d * (d_ff // 2), 2 * 2 * d * (d_ff // (2 * k))
    router = 2 * (d + 2) * e
    active = shared + k * expert + router
    executed = shared + (e if kw["moe_dispatch"] == "dense" else k) * expert + router
    return {"active": tokens * active, "executed": tokens * executed}


def measure(checkpoint_path, threads):
    """Params, FLOPs, latency toàn model và riêng IntraVBlock trên cửa sổ test thật."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    kw = ckpt["model_kwargs"]
    config = config_from_dict(dict(ckpt["config"]))
    model = DyTimeNet(**kw)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dataset = TimeSeriesDataset(config.data_path, config.seq_len, config.pred_len, "test",
                                config.target_cols, config.scale, dataset_name=config.dataset,
                                scaler_state=ckpt["scaler"])
    batch64 = torch.stack([dataset[i][0] for i in range(64)])
    num_vars = batch64.shape[1]
    out = {"params": model.count_parameters()}
    unused = 0
    if kw["use_moe"]:
        expert_params = sum(p.numel() for p in model.intra_v_blocks[0].moe.experts[0].parameters())
        unused = (kw["num_experts"] - kw["top_k"]) * expert_params * kw["num_layers"]
    out["active_params"] = out["params"] - unused

    block = model.intra_v_blocks[0]
    captured = {}
    hook = block.register_forward_pre_hook(lambda m, args: captured.__setitem__("args", args))
    with torch.no_grad():
        model(batch64)
    hook.remove()
    block_args64 = captured["args"]
    block_args1 = tuple(a[:1] if a is not None else None for a in block_args64)

    with torch.no_grad():
        # FLOPs/mẫu: profile batch 64 rồi chia 64 (sparse thực thi đúng N·k hàng mỗi expert).
        manual = manual_intra_flops(kw, num_vars)
        out["thop_flops_model"] = 2 * thop_macs(model, (batch64,)) / 64
        out["thop_flops_intra"] = 2 * thop_macs(block, block_args64) / 64 * kw["num_layers"]
        out["manual_flops_intra_active"] = manual["active"]
        out["manual_flops_intra_executed"] = manual["executed"]
        out["flops_executed"] = out["thop_flops_model"]
        out["flops_active"] = out["thop_flops_model"] - (manual["executed"] - manual["active"])

        inputs = {1: batch64[:1], 64: batch64}
        block_inputs = {1: block_args1, 64: block_args64}
        for n_threads in (threads, 1):
            torch.set_num_threads(n_threads)
            for b in (1, 64):
                out[f"lat_b{b}_t{n_threads}"] = timed_median_ms(lambda: model(inputs[b]))
                out[f"intra_b{b}_t{n_threads}"] = timed_median_ms(lambda: block(*block_inputs[b]))
        torch.set_num_threads(threads)
    return out


def run_all(args):
    runs_path = Path(args.results) / "moe_etth1_runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.is_file() else {}
    hist_dir = Path(args.results) / "moe_etth1_histories"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for name in args.configs:
        for seed in SEEDS:
            key = f"{name}_seed{seed}"
            if key in runs:
                print(f"[skip] {key} đã có kết quả")
                continue
            torch.set_num_threads(args.threads)
            overrides = {"seq_len": 96, "pred_len": 96, "seed": seed, "device": "cpu",
                         "save_dir": args.save_dir, "epochs": args.epochs, **CONFIGS[name]}
            config = load_config("configs/ETTh1.yaml", overrides)
            print(f"\n===== {key} =====")
            checkpoint, summary = train(config)
            history = pd.read_csv(Path(checkpoint).parent / "history.csv")
            history.to_csv(hist_dir / f"{key}.csv", index=False)
            row = {"config": name, "seed": seed, "mse": summary["MSE"], "mae": summary["MAE"],
                   "train_seconds": summary["train_seconds"], "epochs_run": int(len(history)),
                   "best_epoch": summary["best_epoch"],
                   "epoch_seconds": float(history["epoch_seconds"].mean()),
                   "train_epoch_seconds": float(history["train_epoch_seconds"].mean()),
                   "checkpoint": str(checkpoint)}
            if config.use_moe:
                cols = [f"train_expert_{i}" for i in range(config.num_experts)]
                row["train_expert_last"] = history[cols].iloc[-1].tolist()
                row["train_expert_max_any_epoch"] = float(history[cols].max(axis=1).max())
                row["test_expert"] = summary["test_expert_fractions"]
            row.update(measure(checkpoint, args.threads))
            runs[key] = row
            runs_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return runs


def mean_std(values, fmt="{:.3f}"):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def write_report(runs, args):
    """Bảng số liệu (mean ± std qua seed); phần kết luận viết tay ở cuối file."""
    groups = {}
    for row in runs.values():
        groups.setdefault(row["config"], []).append(row)
    names = [n for n in CONFIGS if n in groups]

    def col(name, key):
        return [r[key] for r in groups[name]]

    def label(name):
        kw = CONFIGS[name]
        return ("baseline (FFN)", "—") if not kw else (f"MoE E={kw['num_experts']} k={kw['top_k']}",
                                                       kw["moe_dispatch"])

    t = args.threads
    lines = [
        "# DyTimeNet: IntraVMoE so với baseline trên ETTh1",
        "",
        f"ETTh1, seq_len=96, pred_len=96, d_model=64, batch_size=64, lr=0.0005, p1=0.1, p2=0.1, "
        f"epochs≤100, patience=10. Seed {', '.join(map(str, SEEDS))}; số liệu là mean ± std qua "
        f"{len(SEEDS)} seed.",
        f"Thiết bị: CPU ({platform.processor() or platform.machine()}), torch {torch.__version__}, "
        f"`torch.set_num_threads({t})` khi train và đo; latency đo thêm với 1 luồng. "
        "Các cấu hình chạy tuần tự trên cùng máy.",
        "Chia dữ liệu ETTh1 theo 12/4/4 tháng (8.640 / 2.880 / 2.880 dòng, như code hiện có), "
        "không phải 7:1:2. MSE/MAE trên thang StandardScaler.",
        "",
        "## 1. Kích thước và FLOPs (mỗi mẫu)",
        "",
        "FLOPs = 2 × MACs. *Thực thi* = thop đếm trên các module thực sự chạy; *active* = thực thi trừ "
        "phần expert không được chọn mà chế độ dense vẫn tính. IntraV (tay) chỉ đếm phép nhân ma trận "
        "của FFN/MoE (gồm router), để đối chiếu với thop.",
        "",
        "| Cấu hình | Dispatch | Params | Params active | FLOPs active | FLOPs thực thi | "
        "IntraV FLOPs thop | IntraV FLOPs tay (active / thực thi) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in names:
        r = groups[n][0]   # kiến trúc giống nhau giữa các seed
        lines.append(f"| {label(n)[0]} | {label(n)[1]} | {r['params']:,} | {r['active_params']:,} | "
                     f"{r['flops_active'] / 1e6:.3f} M | {r['flops_executed'] / 1e6:.3f} M | "
                     f"{r['thop_flops_intra'] / 1e3:.1f} K | {r['manual_flops_intra_active'] / 1e3:.1f} K / "
                     f"{r['manual_flops_intra_executed'] / 1e3:.1f} K |")
    lines += [
        "",
        "## 2. Huấn luyện và độ chính xác",
        "",
        "s/epoch = train + validation của một epoch. Số epoch = số epoch đã chạy đến khi early stop.",
        "",
        "| Cấu hình | Dispatch | s/epoch | Số epoch | Best epoch | Tổng train (s) | MSE | MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in names:
        lines.append(f"| {label(n)[0]} | {label(n)[1]} | {mean_std(col(n, 'epoch_seconds'), '{:.2f}')} | "
                     f"{mean_std(col(n, 'epochs_run'), '{:.1f}')} | {mean_std(col(n, 'best_epoch'), '{:.1f}')} | "
                     f"{mean_std(col(n, 'train_seconds'), '{:.1f}')} | {mean_std(col(n, 'mse'), '{:.4f}')} | "
                     f"{mean_std(col(n, 'mae'), '{:.4f}')} |")
    lines += [
        "",
        "## 3. Latency suy luận (ms, median của 200 lần sau 20 lần warm-up, `eval()` + `no_grad()`)",
        "",
        f"| Cấu hình | Dispatch | Model b1 ({t} luồng) | Model b64 ({t} luồng) | Model b1 (1 luồng) | "
        f"Model b64 (1 luồng) | IntraV b1 ({t}) | IntraV b64 ({t}) | IntraV b1 (1) | IntraV b64 (1) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in names:
        cells = [mean_std(col(n, f"lat_b{b}_t{th}"), "{:.3f}") for th in (t, 1) for b in (1, 64)]
        cells += [mean_std(col(n, f"intra_b{b}_t{th}"), "{:.3f}") for th in (t, 1) for b in (1, 64)]
        lines.append(f"| {label(n)[0]} | {label(n)[1]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 4. Tỉ lệ token vào mỗi expert",
        "",
        "Train = epoch cuối (có noise router). Test = model tốt nhất trên toàn tập test (không noise). "
        "Với k=2 tính theo lượt gán (chia k), nên tổng = 1. Cân bằng hoàn hảo là 0,25 mỗi expert. "
        "Lịch sử từng epoch nằm trong `results/moe_etth1_histories/`.",
        "",
        "| Cấu hình | Dispatch | Train epoch cuối (e0/e1/e2/e3) | Test (e0/e1/e2/e3) | "
        "Max tỉ lệ 1 expert, mọi epoch |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for n in names:
        if not CONFIGS[n]:
            continue
        def per_expert(key):
            return " / ".join(mean_std([r[key][i] for r in groups[n]], "{:.3f}")
                              for i in range(CONFIGS[n]["num_experts"]))
        lines.append(f"| {label(n)[0]} | {label(n)[1]} | {per_expert('train_expert_last')} | "
                     f"{per_expert('test_expert')} | "
                     f"{max(col(n, 'train_expert_max_any_epoch')):.3f} |")

    base = groups.get("baseline")
    if base:
        lines += [
            "",
            "## 5. Thay đổi so với baseline (theo mean; dương = lớn hơn / chậm hơn)",
            "",
            f"| Cấu hình | Dispatch | s/epoch | Tổng train | Latency b1 ({t}) | Latency b64 ({t}) | "
            "Latency b1 (1) | Latency b64 (1) | MSE | MAE | MSE ≤ +1%? |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]

        def pct(n, key):
            b = statistics.mean(col("baseline", key))
            return (statistics.mean(col(n, key)) - b) / b * 100

        for n in names:
            if n == "baseline":
                continue
            keys = ["epoch_seconds", "train_seconds", f"lat_b1_t{t}", f"lat_b64_t{t}", "lat_b1_t1",
                    "lat_b64_t1", "mse", "mae"]
            cells = [f"{pct(n, k):+.1f}%" for k in keys]
            ok = "✅" if pct(n, "mse") <= 1.0 else "❌"
            lines.append(f"| {label(n)[0]} | {label(n)[1]} | " + " | ".join(cells) + f" | {ok} |")
    report = Path(args.results) / "moe_vs_baseline_etth1.md"
    conclusion_marker = "\n## 6. Kết luận"
    old = report.read_text(encoding="utf-8") if report.is_file() else ""
    tail = old[old.index(conclusion_marker):] if conclusion_marker in old else ""
    report.write_text("\n".join(lines) + "\n" + tail, encoding="utf-8")
    print(f"Wrote {report}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save_dir", default="checkpoints/moe_bench")
    parser.add_argument("--results", default="results")
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Chỉ để chạy thử nhanh; mặc định lấy từ config")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    if args.report_only:
        runs = json.loads((Path(args.results) / "moe_etth1_runs.json").read_text(encoding="utf-8"))
    else:
        runs = run_all(args)
    write_report(runs, args)


if __name__ == "__main__":
    main()
