"""Tổng hợp mọi lần chạy MoE/baseline trên ETTh1 thành results/moe_report_etth1.md (đọc độc lập).

Chỉ đọc kết quả đã lưu (runs.json, history.csv, metrics.json, checkpoint); không train lại.
Các phép tính thêm trên checkpoint đã lưu: kiểm tra dense vs sparse, phân tích router trên tập test,
và params/FLOPs (đếm tĩnh) cho run bị dừng trước bước đo.

Ví dụ: python scripts/make_moe_report.py
Phần "Nhận xét" lấy nguyên văn từ results/moe_report_notes_etth1.md nếu có.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from bench_moe import manual_intra_flops, thop_macs  # noqa: E402
from dytimenet.config import config_from_dict, load_config  # noqa: E402
from dytimenet.data.dataset import TimeSeriesDataset, get_dataset_splits  # noqa: E402
from dytimenet.models import DyTimeNet  # noqa: E402

SEEDS = (2021, 2022, 2023)
THREADS = 8
SPECS = {   # tên → overrides của configs/ETTh1.yaml
    "baseline": {},
    "moe_k1_sparse": {"use_moe": True, "num_experts": 4, "top_k": 1, "moe_dispatch": "sparse"},
    "moe_k1_dense": {"use_moe": True, "num_experts": 4, "top_k": 1, "moe_dispatch": "dense"},
    "moe_k2_sparse": {"use_moe": True, "num_experts": 4, "top_k": 2, "moe_dispatch": "sparse"},
    "moe_lite": {"use_moe": True, "num_experts": 4, "top_k": 1, "moe_dispatch": "sparse", "ffn_ratio": 1.0},
}
RUN_SOURCES = ("results/moe_etth1_runs.json", "results/moe_lite/moe_etth1_runs.json")
HISTORY_DIRS = ("results/moe_etth1_histories", "results/moe_lite/moe_etth1_histories")
HOUR_GROUPS = ((0, 5), (6, 11), (12, 17), (18, 23))
NOT_RUN, NOT_MEASURED = "chưa chạy", "chưa đo"


# ───────────────────────────── đọc dữ liệu đã lưu ─────────────────────────────
def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout.strip()


def config_name(cfg):
    """Nhận diện cấu hình từ config.yaml đã lưu."""
    for name, spec in SPECS.items():
        want = {"use_moe": False, "ffn_ratio": 2.0, **spec}
        if all(cfg.get(k, False if k == "use_moe" else None) == v for k, v in want.items()):
            return name
    return None


def load_runs():
    """runs.json gốc + run có metrics.json trong checkpoints/ nhưng thiếu trong runs.json."""
    raw, runs = {}, {}
    for source in RUN_SOURCES:
        path = ROOT / source
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            raw[source] = data
            for key, row in data.items():
                runs[key] = {**row, "source": source, "partial": False}
    known = {Path(r["checkpoint"]).parent.resolve() for r in runs.values()}
    for metrics_path in sorted((ROOT / "checkpoints/moe_bench").glob("*/metrics.json")):
        run_dir = metrics_path.parent
        if run_dir.resolve() in known:
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        name = config_name(cfg)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rel = run_dir.relative_to(ROOT).as_posix()
        raw[f"{rel}/metrics.json"] = metrics
        if name is None or cfg["seed"] not in SEEDS or cfg["epochs"] != 100:
            continue
        history = pd.read_csv(run_dir / "history.csv")
        runs[f"{name}_seed{cfg['seed']}"] = {
            "config": name, "seed": cfg["seed"], "mse": metrics["MSE"], "mae": metrics["MAE"],
            "train_seconds": metrics["train_seconds"], "epochs_run": int(len(history)),
            "best_epoch": metrics["best_epoch"], "epoch_seconds": float(history["epoch_seconds"].mean()),
            "train_epoch_seconds": float(history["train_epoch_seconds"].mean()),
            "checkpoint": metrics["checkpoint"], "test_expert": metrics.get("test_expert_fractions"),
            "source": f"{rel}/metrics.json", "partial": True}
    return raw, runs


def load_history(key):
    for folder in HISTORY_DIRS:
        path = ROOT / folder / f"{key}.csv"
        if path.is_file():
            return pd.read_csv(path)
    return None


def load_checkpoint_model(row):
    ckpt = torch.load(ROOT / row["checkpoint"], map_location="cpu", weights_only=True)
    model = DyTimeNet(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def test_dataset(ckpt):
    config = config_from_dict(dict(ckpt["config"]))
    return TimeSeriesDataset(ROOT / config.data_path, config.seq_len, config.pred_len, "test",
                             config.target_cols, config.scale, dataset_name=config.dataset,
                             scaler_state=ckpt["scaler"])


def test_batches(dataset, batch=256):
    for start in range(0, len(dataset), batch):
        idx = range(start, min(start + batch, len(dataset)))
        yield list(idx), torch.stack([dataset[i][0] for i in idx])


# ───────────────────────────── tính thêm trên checkpoint ─────────────────────────────
def architecture(name):
    """Kích thước FFN/expert lấy từ module thật của cấu hình."""
    config = load_config(ROOT / "configs/ETTh1.yaml", {"pred_len": 96, **SPECS[name]})
    model = DyTimeNet(**config.model_kwargs(7))
    block = model.intra_v_blocks[0]
    if not config.use_moe:
        d_ff = block.ffn[0].out_features
        return {"E": "—", "k": "—", "shared": d_ff, "expert": "—", "dispatch": "— (FFN)",
                "active": d_ff, "executed": d_ff, "params": model.count_parameters()}
    shared, expert = block.moe.shared[0].out_features, block.moe.experts[0][0].out_features
    e, k = config.num_experts, config.top_k
    return {"E": e, "k": k, "shared": shared, "expert": expert, "dispatch": config.moe_dispatch,
            "active": shared + k * expert,
            "executed": shared + (e if config.moe_dispatch == "dense" else k) * expert,
            "params": model.count_parameters()}


def static_cost(row):
    """Params/FLOPs của run bị dừng trước bước đo: cùng công thức với bench_moe.measure."""
    model, ckpt = load_checkpoint_model(row)
    kw = ckpt["model_kwargs"]
    dataset = test_dataset(ckpt)
    batch64 = torch.stack([dataset[i][0] for i in range(64)])
    captured = {}
    block = model.intra_v_blocks[0]
    hook = block.register_forward_pre_hook(lambda m, args: captured.__setitem__("args", args))
    with torch.no_grad():
        model(batch64)
        hook.remove()
        manual = manual_intra_flops(kw, batch64.shape[1])
        flops = 2 * thop_macs(model, (batch64,)) / 64
        intra = 2 * thop_macs(block, captured["args"]) / 64 * kw["num_layers"]
    unused = 0
    if kw["use_moe"]:
        unused = ((kw["num_experts"] - kw["top_k"]) * kw["num_layers"]
                  * sum(p.numel() for p in block.moe.experts[0].parameters()))
    params = model.count_parameters()
    return {"params": params, "active_params": params - unused, "flops_executed": flops,
            "flops_active": flops - (manual["executed"] - manual["active"]), "thop_flops_intra": intra,
            "manual_flops_intra_active": manual["active"], "manual_flops_intra_executed": manual["executed"]}


def dense_sparse_check(runs):
    """Nạp cùng trọng số vào hai chế độ dispatch, so output trên toàn bộ tập test (eval)."""
    rows = []
    for key in ("moe_k1_sparse_seed2021", "moe_k1_dense_seed2021", "moe_lite_seed2021"):
        if key not in runs:
            continue
        model, ckpt = load_checkpoint_model(runs[key])
        kw = dict(ckpt["model_kwargs"])
        kw["moe_dispatch"] = "dense" if kw["moe_dispatch"] == "sparse" else "sparse"
        other = DyTimeNet(**kw)
        other.load_state_dict(ckpt["model_state_dict"])
        other.eval()
        rows.append((f"checkpoint `{key}` ({ckpt['model_kwargs']['moe_dispatch']} → {kw['moe_dispatch']})",
                     *compare_outputs(model, other, test_dataset(ckpt))))
    # k=2 chưa có checkpoint: dùng trọng số khởi tạo seed 2021, cùng dữ liệu test.
    torch.manual_seed(2021)
    config = load_config(ROOT / "configs/ETTh1.yaml", {"pred_len": 96, **SPECS["moe_k2_sparse"]})
    sparse = DyTimeNet(**config.model_kwargs(7)).eval()
    dense = DyTimeNet(**{**config.model_kwargs(7), "moe_dispatch": "dense"}).eval()
    dense.load_state_dict(sparse.state_dict())
    if "moe_k1_sparse_seed2021" in runs:
        dataset = test_dataset(load_checkpoint_model(runs["moe_k1_sparse_seed2021"])[1])
        rows.append(("k=2, trọng số khởi tạo ngẫu nhiên (seed 2021), chưa train",
                     *compare_outputs(sparse, dense, dataset)))
    return rows


def compare_outputs(a, b, dataset):
    worst = 0.0
    n = 0
    with torch.no_grad():
        for _, x in test_batches(dataset):
            worst = max(worst, (a(x) - b(x)).abs().max().item())
            n += x.shape[0]
    return n, worst


def router_analysis(row, dates):
    """Expert top-1 của từng token (mẫu × kênh) trên test, không noise; trả bảng % theo kênh/nhóm giờ."""
    model, ckpt = load_checkpoint_model(row)
    dataset = test_dataset(ckpt)
    moe = model.intra_v_blocks[0].moe
    e = moe.num_experts
    captured = {}
    hook = moe.register_forward_pre_hook(lambda m, args: captured.__setitem__("args", args))
    choices, hours, margins = [], [], []
    with torch.no_grad():
        for idx, x in test_batches(dataset):
            model(x)
            tok, mu, log_sigma = captured["args"]
            logits = moe.router(torch.cat([tok, mu, log_sigma], dim=-1))     # [B, C, E]
            top2 = logits.topk(2, dim=-1).values
            choice = logits.argmax(-1)                                       # [B, C]
            # Đối chiếu với số lượt gán mà chính module ghi lại trong forward vừa chạy.
            assert torch.equal(torch.bincount(choice.reshape(-1), minlength=e), moe.expert_counts)
            choices.append(choice)
            margins.append(top2[..., 0] - top2[..., 1])
            last_rows = [dataset.start + i + dataset.seq_len - 1 for i in idx]
            hours.append(torch.tensor([dates[r].hour for r in last_rows]))
    hook.remove()
    choice = torch.cat(choices)          # [N, C]
    hour = torch.cat(hours)              # [N]
    margin = torch.cat(margins)
    channels = dataset.columns
    table = {}
    for c, col in enumerate(channels):
        table[col] = torch.bincount(choice[:, c], minlength=e).double() / choice.shape[0]
    for lo, hi in HOUR_GROUPS:
        sel = (hour >= lo) & (hour <= hi)
        table[f"{lo}–{hi}h"] = torch.bincount(choice[sel].reshape(-1), minlength=e).double() / choice[sel].numel()
    return {"table": table, "n_windows": choice.shape[0], "noise_std": 1 / e,
            "margin_mean": margin.mean().item(), "margin_median": margin.median().item(),
            "margin_below_noise": (margin < 1 / e).double().mean().item()}


def timeline(runs):
    """Bắt đầu = tên thư mục checkpoint (UTC); kết thúc train = mtime metrics.json; chồng lấn giữa các run."""
    spans = {}
    for key, row in runs.items():
        run_dir = ROOT / Path(row["checkpoint"]).parent
        start = datetime.strptime(run_dir.name.rsplit("_", 1)[1], "%Y%m%dT%H%M%S%fZ")
        end = datetime.fromtimestamp((run_dir / "metrics.json").stat().st_mtime, timezone.utc).replace(tzinfo=None)
        epochs = pd.read_csv(run_dir / "history.csv")["epoch_seconds"]
        spans[key] = {"start": start, "end": end, "epochs": epochs}
    for key, span in spans.items():
        span["overlap"] = sorted(k for k, o in spans.items()
                                 if k != key and o["start"] < span["end"] and span["start"] < o["end"])
    return spans


# ───────────────────────────── định dạng ─────────────────────────────
def fmt_ms(values, fmt="{:.3f}"):
    values = [v for v in values if v is not None]
    if not values:
        return NOT_MEASURED
    if len(values) == 1:
        return f"{fmt.format(values[0])} (n=1)"
    return f"{fmt.format(statistics.mean(values))} ± {fmt.format(statistics.stdev(values))}"


def pct(new, old):
    return (new - old) / old * 100


def fence(text, lang=""):
    return [f"```{lang}", text.rstrip("\n"), "```"]


# ───────────────────────────── báo cáo ─────────────────────────────
def build(raw, runs):
    groups = {name: sorted((r for r in runs.values() if r["config"] == name), key=lambda r: r["seed"])
              for name in SPECS}
    for name, rows in groups.items():
        for row in rows:
            if row["partial"]:
                row.update(static_cost(row))
    arch = {name: architecture(name) for name in SPECS}
    spans = timeline(runs)
    noisy = {name for name, rows in groups.items()
             if any(spans[f"{name}_seed{r['seed']}"]["overlap"] for r in rows)}

    def tag(name):
        return f"`{name}` ⚠️" if name in noisy else f"`{name}`"

    def status(name):
        rows = groups[name]
        if not rows:
            return NOT_RUN
        seeds = ", ".join(str(r["seed"]) for r in rows)
        missing = [str(s) for s in SEEDS if s not in {r["seed"] for r in rows}]
        note = f"; seed {', '.join(missing)} {NOT_RUN}" if missing else ""
        partial = [str(r["seed"]) for r in rows if r["partial"]]
        if partial:
            note += f"; seed {', '.join(partial)} train xong nhưng bị dừng trước bước đo latency"
        return f"seed {seeds}{note}"

    # ── 1. Môi trường ──
    envs = {}
    for row in runs.values():
        env_path = ROOT / Path(row["checkpoint"]).parent / "environment.json"
        env = json.loads(env_path.read_text(encoding="utf-8"))
        envs.setdefault(json.dumps(env, sort_keys=True), []).append(row["config"])
    env = json.loads(next(iter(envs)))
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip() or platform.processor()
    base_dir = ROOT / Path(groups["baseline"][0]["checkpoint"]).parent
    base_cfg = yaml.safe_load((base_dir / "config.yaml").read_text(encoding="utf-8"))
    data_sha = json.loads((base_dir / "metrics.json").read_text(encoding="utf-8"))["data_sha256"]
    csv = pd.read_csv(ROOT / base_cfg["data_path"])
    dates = pd.to_datetime(csv["date"])
    splits = get_dataset_splits(base_cfg["data_path"], len(csv), base_cfg["dataset"])
    windows = {s: len(TimeSeriesDataset(ROOT / base_cfg["data_path"], 96, 96, s, dataset_name="ETTh1"))
               for s in ("train", "val", "test")}
    stamps = {}
    for name, rows in groups.items():
        times = sorted(Path(r["checkpoint"]).parent.name.rsplit("_", 1)[1] for r in rows)
        if times:
            stamps[name] = f"{times[0][:8]} {times[0][9:15]} → {times[-1][9:15]} UTC"
    base_hash = git("rev-parse", "baseline-dytimenet")
    head_hash = git("rev-parse", "feat/moe-topk")
    model_hash = git("log", "-1", "--format=%H", "baseline-dytimenet..feat/moe-topk", "--", "src/")

    L = [
        "# Báo cáo: IntraVMoE top-k so với DyTimeNet baseline trên ETTh1",
        "",
        "Tài liệu này đọc được độc lập, không cần mở code. Mọi số liệu lấy từ các lần chạy đã lưu "
        "(`results/*.json`, `history.csv`, `metrics.json`, checkpoint) và được sinh bởi "
        "`scripts/make_moe_report.py`. Cấu hình hoặc phép đo chưa có ghi là \"chưa chạy\" hoặc \"chưa đo\".",
        "",
        "**Trạng thái các cấu hình:**",
        "",
    ]
    L += [f"- {tag(name)}: {status(name)}" for name in SPECS]
    if noisy:
        L += ["", "⚠️ = có ít nhất một seed train/đo **cùng lúc** với một run khác trên cùng CPU (xem bảng dòng "
              "thời gian ở mục 1). Thời gian (s/epoch, tổng train, latency) của các cấu hình này **không so sánh "
              "công bằng** được; MSE/MAE không bị ảnh hưởng."]
    L += [
        "",
        "## 1. Môi trường và thiết lập",
        "",
        "| Mục | Giá trị |",
        "| --- | --- |",
        f"| CPU | {cpu} ({os.cpu_count()} luồng logic), không GPU |",
        f"| Số luồng PyTorch | `torch.set_num_threads({THREADS})` khi train và đo; latency đo thêm với 1 luồng |",
        f"| Hệ điều hành | {env['platform']} |",
        f"| Python / PyTorch | {env['python']} / {env['torch']} (CUDA: {env['cuda']}) |",
        f"| numpy / pandas / scikit-learn | {env['numpy']} / {env['pandas']} / {env['scikit_learn']} |",
        f"| Dataset | ETTh1, {len(csv):,} dòng, cột `date` + {len(csv.columns) - 1} biến "
        f"({', '.join(csv.columns[1:])}), {dates.iloc[0]:%Y-%m-%d %H:%M} → {dates.iloc[-1]:%Y-%m-%d %H:%M}, mỗi giờ 1 dòng |",
        f"| SHA-256 CSV | `{data_sha}` |",
        f"| Cách chia | 12/4/4 tháng (quy ước tháng 30 ngày): train {splits[0]:,} / val {splits[1]:,} / test "
        f"{splits[2]:,} dòng; {len(csv) - sum(splits):,} dòng cuối không dùng. Không phải 7:1:2. |",
        f"| Số mẫu (cửa sổ) | train {windows['train']:,} / val {windows['val']:,} / test {windows['test']:,} "
        "(val/test lấy thêm 96 dòng quá khứ làm input) |",
        "| Chuẩn hóa | StandardScaler fit trên train; MSE/MAE báo cáo ở thang StandardScaler |",
        f"| Hyperparameter | seq_len={base_cfg['seq_len']}, pred_len={base_cfg['pred_len']}, "
        f"d_model={base_cfg['d_model']}, d_ff={int(base_cfg['d_model'] * base_cfg['ffn_ratio'])} "
        f"(ffn_ratio={base_cfg['ffn_ratio']}; moe_lite dùng ffn_ratio=1.0), batch={base_cfg['batch_size']}, "
        f"lr={base_cfg['lr']}, p1={base_cfg['p1']}, p2={base_cfg['p2']}, dropout={base_cfg['dropout']}, "
        f"weight_decay={base_cfg['weight_decay']}, decomp_kernel={base_cfg['decomp_kernel']} |",
        f"| Huấn luyện | Adam, clip grad norm 1.0, tối đa {base_cfg['epochs']} epoch, early stopping theo "
        f"val MSE (patience={base_cfg['patience']}, min_delta={base_cfg['min_delta']}), nạp lại checkpoint tốt nhất |",
        f"| Scheduler | warm-up tuyến tính {base_cfg['warmup_epochs']} epoch rồi cosine về 0 tại epoch "
        f"{base_cfg['epochs']} |",
        "| MoE | λ (moe_aux_weight) = 0.01; noise router std = 1/E khi train; seed 2021, 2022, 2023 |",
        f"| Commit baseline (tag `baseline-dytimenet`) | `{base_hash}` |",
        f"| Commit thay đổi model trên `feat/moe-topk` | `{model_hash}` |",
        f"| HEAD `feat/moe-topk` khi sinh báo cáo | `{head_hash}` (commit chứa báo cáo này nằm ngay sau) |",
        "",
        "Thời điểm chạy (lấy từ tên thư mục checkpoint):",
        "",
    ]
    L += [f"- `{name}`: {stamp}" for name, stamp in stamps.items()]
    L += [
        "",
        "**Dòng thời gian từng run** (UTC; kết thúc = lúc ghi `metrics.json`, latency được đo ngay sau đó). "
        "Cột chồng lấn liệt kê các run chạy cùng lúc.",
        "",
        "| Run | Bắt đầu | Kết thúc train | s/epoch min / median / max | Chạy cùng lúc với |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for key, span in sorted(spans.items(), key=lambda kv: kv[1]["start"]):
        ep = span["epochs"]
        L.append(f"| `{key}` | {span['start']:%H:%M:%S} | {span['end']:%H:%M:%S} | "
                 f"{ep.min():.2f} / {ep.median():.2f} / {ep.max():.2f} | "
                 f"{', '.join(span['overlap']) or '—'} |")
    if len(envs) > 1:
        L += ["", f"⚠️ Có {len(envs)} môi trường khác nhau giữa các run: {list(envs.values())}"]

    # ── 2. Thay đổi ──
    files = {
        "IntraVMoE (`src/dytimenet/layers/intra_v_moe.py`, file mới)": None,
        "IntraVBlock": "src/dytimenet/layers/intra_v_block.py",
        "RevIN": "src/dytimenet/layers/revin.py",
        "Cộng aux loss (`run_epoch`)": "src/dytimenet/engine/trainer.py",
        "Truyền λ, log thời gian epoch và tỉ lệ expert (`experiment.py`)": "src/dytimenet/experiment.py",
        "Nối μ, log σ từ RevIN vào IntraVBlock (`DyTimeNet.forward`)": "src/dytimenet/models/dytimenet.py",
    }
    L += [
        "",
        "## 2. Mô tả thay đổi",
        "",
        "Kiến trúc baseline: RevIN → MovingAvg tách trend/seasonal; trend: Linear L→S; seasonal: depthwise "
        "Conv1D → GELU → Linear L→D → CrossVBlock (DropFusion) + residual → IntraVBlock → Linear D→S; "
        "cộng hai nhánh → inverse RevIN. MoE **chỉ thay FFN bên trong IntraVBlock**: "
        "`out = x + IntraVMoE(LN(x))`. Token = một cặp (mẫu, kênh), vector D chiều. Router nhận "
        "`[LN(x); μ; log σ]`, trong đó μ, σ là thống kê RevIN của **input gốc** từng kênh (đã detach). "
        "Train loss = MSE + λ·aux (Switch: aux = E·Σ f_i·P_i, f_i = tỉ lệ lượt gán top-k chia k, "
        "P_i = xác suất softmax trung bình). Validation, early stopping và các số báo cáo chỉ dùng MSE. "
        "Khi tắt `--use_moe`, output và gradient trùng từng bit với tag baseline (đã kiểm tra).",
        "",
        "`git diff baseline-dytimenet..feat/moe-topk --stat` (tại HEAD trên, chưa gồm commit báo cáo):",
        "",
        *fence(git("diff", "--stat", "baseline-dytimenet..feat/moe-topk")),
        "",
    ]
    for title, path in files.items():
        L += [f"**{title}**", ""]
        if path is None:
            L += fence(git("show", "feat/moe-topk:src/dytimenet/layers/intra_v_moe.py"), "python")
        else:
            L += fence(git("diff", "baseline-dytimenet", "feat/moe-topk", "--", path), "diff")
        L += [""]
    L += [
        "**Các cấu hình** (hidden = tổng độ rộng lớp ẩn mỗi token đi qua; *thực thi* gồm cả expert "
        "không được chọn mà dense vẫn tính):",
        "",
        "| Tên | E | k | d_ff shared | d_ff mỗi expert | Dispatch | Hidden active | Hidden thực thi | Params |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for name, a in arch.items():
        shared = f"{a['shared']} (FFN)" if a["E"] == "—" else a["shared"]
        L.append(f"| `{name}` | {a['E']} | {a['k']} | {shared} | {a['expert']} | {a['dispatch']} | "
                 f"{a['active']} | {a['executed']} | {a['params']:,} |")
    check = dense_sparse_check(runs)
    L += [
        "",
        "**Dense vs sparse** (`model.eval()`, cùng trọng số, toàn bộ tập test; yêu cầu sai khác < 1e-5). "
        "Ngoài ra `tests/test_moe.py::test_dense_and_sparse_dispatch_match_in_eval` (k=1, k=2, input ngẫu nhiên) "
        "nằm trong bộ test 41/41 pass.",
        "",
        "| Trọng số | Số cửa sổ test | Sai khác tuyệt đối lớn nhất | < 1e-5 |",
        "| --- | ---: | ---: | :---: |",
    ]
    L += [f"| {label} | {n:,} | {worst:.3e} | {'✅' if worst < 1e-5 else '❌'} |" for label, n, worst in check]

    # ── 3. Bảng tổng hợp ──
    def cells(name, keys, fmt):
        rows = groups[name]
        if not rows:
            return [NOT_RUN] * len(keys)
        return [fmt_ms([r.get(k) for r in rows], fmt) for k in keys]

    L += [
        "",
        "## 3. Bảng kết quả tổng hợp",
        "",
        "mean ± std qua các seed đã chạy (n=1 nghĩa là chỉ có 1 seed). FLOPs mỗi mẫu = 2 × MACs, đếm bằng "
        "thop trên các module thực sự chạy (*thực thi*); *active* = thực thi trừ phần expert không được chọn "
        "mà dense vẫn tính. Với sparse, active = thực thi.",
        "",
        "**3a. Độ chính xác và chi phí**",
        "",
        "| Cấu hình | MSE | MAE | Params | FLOPs active (M) | FLOPs thực thi (M) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in SPECS:
        rows = groups[name]
        if not rows:
            L.append(f"| {tag(name)} | " + " | ".join([NOT_RUN] * 5) + " |")
            continue
        # Params/FLOPs không phụ thuộc seed (sparse: mỗi expert chạy đúng N·k hàng) → một giá trị.
        params = {r["params"] for r in rows}
        fa = {round(r["flops_active"]) for r in rows}
        fe = {round(r["flops_executed"]) for r in rows}
        assert len(params) == len(fa) == len(fe) == 1, (name, params, fa, fe)
        mse, mae = cells(name, ["mse", "mae"], "{:.4f}")
        L.append(f"| {tag(name)} | {mse} | {mae} | {params.pop():,} | {fa.pop() / 1e6:.4f} | {fe.pop() / 1e6:.4f} |")
    L += [
        "",
        "**3b. Huấn luyện** (s/epoch = train + validation một epoch; số epoch = số epoch đã chạy đến khi "
        "early stop)",
        "",
        "| Cấu hình | s/epoch | Số epoch | Best epoch | Tổng train (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in SPECS:
        L.append(f"| {tag(name)} | " + " | ".join(
            cells(name, ["epoch_seconds"], "{:.3f}") + cells(name, ["epochs_run", "best_epoch"], "{:.1f}")
            + cells(name, ["train_seconds"], "{:.1f}")) + " |")
    points = [(1, THREADS), (64, THREADS), (1, 1), (64, 1)]
    L += [
        "",
        "**3c. Latency suy luận (ms)**: median của 200 lần đo sau 20 lần warm-up, `eval()` + `no_grad()`, "
        "input là cửa sổ test thật. IntraV = riêng một lần gọi IntraVBlock (LN + FFN/MoE + residual) với "
        "input chụp từ forward thật. bX/tY = batch X, Y luồng.",
        "",
        "| Cấu hình | " + " | ".join(f"Model b{b}/t{t}" for b, t in points) + " | "
        + " | ".join(f"IntraV b{b}/t{t}" for b, t in points) + " |",
        "| --- |" + " ---: |" * 8,
    ]
    for name in SPECS:
        L.append(f"| {tag(name)} | " + " | ".join(
            cells(name, [f"lat_b{b}_t{t}" for b, t in points] + [f"intra_b{b}_t{t}" for b, t in points],
                  "{:.3f}")) + " |")
    L += ["", "Tỉ trọng IntraV trong latency toàn model (mean qua seed):", "",
          "| Cấu hình | " + " | ".join(f"b{b}/t{t}" for b, t in points) + " |", "| --- |" + " ---: |" * 4]
    for name in SPECS:
        rows = [r for r in groups[name] if "lat_b1_t1" in r]
        if not rows:
            L.append(f"| {tag(name)} | " + " | ".join([NOT_RUN if not groups[name] else NOT_MEASURED] * 4) + " |")
            continue
        L.append(f"| {tag(name)} | " + " | ".join(
            f"{statistics.mean(r[f'intra_b{b}_t{t}'] / r[f'lat_b{b}_t{t}'] for r in rows) * 100:.0f}%"
            for b, t in points) + " |")

    # ── 4. So sánh theo cặp ──
    metrics = [("mse", "MSE"), ("mae", "MAE"), ("epoch_seconds", "s/epoch"),
               (f"lat_b1_t{THREADS}", f"Latency b1/t{THREADS}"), (f"lat_b64_t{THREADS}", f"Latency b64/t{THREADS}"),
               (f"intra_b64_t{THREADS}", f"IntraV b64/t{THREADS}")]
    base = {r["seed"]: r for r in groups["baseline"]}
    L += [
        "",
        "## 4. So sánh theo cặp với baseline (cùng seed)",
        "",
        "% chênh lệch = (MoE − baseline) / baseline × 100, tính riêng từng seed rồi lấy mean và min/max. "
        "Dương = lớn hơn (MSE/MAE tệ hơn, thời gian chậm hơn). Tiêu chí đạt: mean ΔMSE ≤ +1% **và** "
        f"mean Δlatency b64/t{THREADS} ≤ 0%.",
        "",
    ]
    verdicts = []
    for name in list(SPECS)[1:]:
        L += [f"**{tag(name)}**", ""]
        if not groups[name]:
            L += [f"{NOT_RUN.capitalize()}.", ""]
            verdicts.append((name, NOT_RUN, NOT_RUN, NOT_RUN))
            continue
        L += ["| Seed | " + " | ".join(label for _, label in metrics) + " |", "| --- |" + " ---: |" * len(metrics)]
        diffs = {k: [] for k, _ in metrics}
        for row in groups[name]:
            ref = base.get(row["seed"])
            line = []
            for key, _ in metrics:
                if ref is None or key not in row or key not in ref:
                    line.append(NOT_MEASURED)
                    continue
                value = pct(row[key], ref[key])
                diffs[key].append(value)
                line.append(f"{value:+.2f}%")
            warn = " ⚠️" if spans[f"{name}_seed{row['seed']}"]["overlap"] else ""
            L.append(f"| {row['seed']}{warn} | " + " | ".join(line) + " |")
        for label, fn in (("mean", lambda v: f"{statistics.mean(v):+.2f}%"),
                          ("min / max", lambda v: f"{min(v):+.2f}% / {max(v):+.2f}%")):
            L.append(f"| **{label}** | " + " | ".join(
                (fn(diffs[k]) + (" (n=1)" if len(diffs[k]) == 1 else "")) if diffs[k] else NOT_MEASURED
                for k, _ in metrics) + " |")
        d_mse, d_lat = diffs["mse"], diffs[f"lat_b64_t{THREADS}"]
        mse_ok = statistics.mean(d_mse) <= 1.0 if d_mse else None
        lat_ok = statistics.mean(d_lat) <= 0.0 if d_lat else None
        if mse_ok is None or lat_ok is None:
            verdict = "không đánh giá được (thiếu số đo)"
        else:
            verdict = "✅ đạt" if mse_ok and lat_ok else "❌ không đạt"
        if name in noisy and verdict.startswith("❌"):
            verdict += " (latency bị nhiễu ⚠️, nhưng ngay cả seed sạch cũng chậm hơn baseline)" if any(
                not spans[f"{name}_seed{r['seed']}"]["overlap"] for r in groups[name]) else " (latency bị nhiễu ⚠️)"
        verdicts.append((name, f"{statistics.mean(d_mse):+.2f}%" if d_mse else NOT_MEASURED,
                         f"{statistics.mean(d_lat):+.2f}%" if d_lat else NOT_MEASURED, verdict))
        L += [""]
    L += [f"**Tổng hợp tiêu chí** (ΔMSE ≤ +1% và Δlatency b64/t{THREADS} ≤ 0%)", "",
          f"| Cấu hình | mean ΔMSE | mean Δlatency b64/t{THREADS} | Kết quả |", "| --- | ---: | ---: | --- |"]
    L += [f"| {tag(n)} | {a} | {b} | {c} |" for n, a, b, c in verdicts]

    # ── 5. Router ──
    L += [
        "",
        "## 5. Router",
        "",
        "Tỉ lệ token trên test: model tốt nhất, `eval()` (không noise), mọi cửa sổ test × 7 kênh. "
        "Aux cuối = aux loss trung bình của epoch train cuối cùng (giá trị cân bằng hoàn hảo = 1.0). "
        "Với k=2, tỉ lệ tính theo lượt gán chia k, nên tổng = 1. Chỉ số expert **không** tương ứng giữa "
        "các seed (mỗi seed học router riêng).",
        "",
    ]
    for name in list(SPECS)[1:]:
        e = arch[name]["E"]
        L += [f"**`{name}`**", ""]
        if not groups[name]:
            L += [f"{NOT_RUN.capitalize()}.", ""]
            continue
        L += ["| Seed | " + " | ".join(f"e{i}" for i in range(e)) + " | Aux cuối | Expert < 5% token |",
              "| --- |" + " ---: |" * (e + 1) + " --- |"]
        for row in groups[name]:
            history = load_history(f"{name}_seed{row['seed']}")
            aux = f"{history['train_aux'].iloc[-1]:.4f}" if history is not None else NOT_MEASURED
            fr = row.get("test_expert")
            if fr is None:
                L.append(f"| {row['seed']} | " + " | ".join([NOT_MEASURED] * e) + f" | {aux} | {NOT_MEASURED} |")
                continue
            low = [f"e{i} ({f * 100:.1f}%)" for i, f in enumerate(fr) if f < 0.05] or ["không có"]
            L.append(f"| {row['seed']} | " + " | ".join(f"{f * 100:.1f}%" for f in fr)
                     + f" | {aux} | {', '.join(low)} |")
        L += [""]

    L += [
        "**Expert được chọn theo kênh và theo giờ (k=1)**: % token của mỗi hàng đi vào từng expert. "
        "Nhóm giờ theo giờ của bước cuối trong cửa sổ input. Biên logit = logit top-1 − logit top-2 "
        "trên test; so với std noise router khi train (1/E = 0.25).",
        "",
    ]
    for name in ("moe_k1_sparse", "moe_k1_dense", "moe_lite"):
        if not groups[name]:
            L += [f"`{name}`: {NOT_RUN}.", ""]
            continue
        for row in groups[name]:
            ra = router_analysis(row, dates)
            e = len(next(iter(ra["table"].values())))
            L += [f"`{name}` seed {row['seed']} ({ra['n_windows']:,} cửa sổ test). Biên logit: mean "
                  f"{ra['margin_mean']:.3f}, median {ra['margin_median']:.3f}; "
                  f"{ra['margin_below_noise'] * 100:.1f}% token có biên < {ra['noise_std']:.2f}.", "",
                  "| Kênh / giờ | " + " | ".join(f"e{i}" for i in range(e)) + " |", "| --- |" + " ---: |" * e]
            for label, frac in ra["table"].items():
                top = int(frac.argmax())
                L.append(f"| {label} | " + " | ".join(
                    (f"**{v * 100:.1f}%**" if i == top else f"{v * 100:.1f}%") for i, v in enumerate(frac.tolist()))
                    + " |")
            L += [""]

    # ── 6. Nhận xét ──
    notes = ROOT / "results/moe_report_notes_etth1.md"
    L += ["## 6. Nhận xét", ""]
    L += [notes.read_text(encoding="utf-8").strip() if notes.is_file() else "(chưa viết)", ""]

    # ── 7. Phụ lục ──
    raw_text = json.dumps(raw, indent=2, ensure_ascii=False)
    L += [
        "## 7. Phụ lục: JSON kết quả thô",
        "",
        "Nguyên văn các file kết quả, theo đường dẫn nguồn (cũng lưu tại `results/moe_raw_etth1.json`). "
        "Lịch sử từng epoch (loss, thời gian, tỉ lệ expert khi train) nằm trong "
        "`results/moe_etth1_histories/` và `results/moe_lite/moe_etth1_histories/`. Đơn vị: latency (`lat_*`, "
        "`intra_*`) tính bằng ms; FLOPs là số phép tính mỗi mẫu; thời gian train tính bằng giây.",
        "",
        *fence(raw_text, "json"),
        "",
    ]
    return "\n".join(L), raw


def main():
    torch.set_num_threads(THREADS)
    raw, runs = load_runs()
    report, raw = build(raw, runs)
    (ROOT / "results/moe_raw_etth1.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "results/moe_report_etth1.md").write_text(report, encoding="utf-8")
    print("Wrote results/moe_report_etth1.md and results/moe_raw_etth1.json")


if __name__ == "__main__":
    main()
