"""Train một bộ biến thể mô hình trên một dataset (4 horizon × 3 seed), cùng pipeline, chạy tuần tự.

Ví dụ: python scripts/run_suite.py --dataset Weather --variants dyt_norm dyt_rcf rlinear cyclelinear
       python scripts/run_suite.py --dataset Weather --seq_len 720 --variants dyt_rcf timebase_lr0.02
Kết quả từng run ghi ngay vào results/suite_<dataset>.json (L = 96) hoặc results/suite_<dataset>_L<L>.json
(chạy lại để tiếp tục phần còn thiếu). TimeBase train với vài lr (paper tìm lr trong [0.01, 0.5]); chọn lr theo
val loss tốt nhất trong scripts/compare_suite.py.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from dytimenet.config import load_config

# Chu kỳ dài nhất ổn định của từng dataset (như Table 1 của CycleNet): ngày theo tần số lấy mẫu; Electricity và
# Traffic dùng tuần (168 giờ) như scripts/*/electricity.sh, traffic.sh của CycleNet/TQNet.
CYCLE = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96, "Weather": 144, "Electricity": 168, "Traffic": 168}
# Độ dài đoạn P của TimeBase theo paper (Phụ lục C.1): chu kỳ tự nhiên, Weather dùng P = 4.
TB_PERIOD = {"ETTh1": 24, "ETTh2": 24, "Weather": 4}
TB_LRS = (0.005, 0.02, 0.08)
# Siêu tham số chính thức của TimeBase theo từng horizon (scripts/etth1.sh, scripts/weather.sh trong
# github.com/hqh0728/TimeBase): (orthogonal_weight, batch_size, learning_rate); chung: period_norm, R = 6,
# 30 epoch, patience 5, lradj type3, Adam không weight decay, không clip gradient. Weather dùng individual = 1.
#   giá trị: (orthogonal_weight, batch_size, learning_rate, period_len P, basis_num R).
TB_OFFICIAL = {
    "ETTh1": {96: (0.16, 64, 0.1, 24, 6), 192: (0.16, 256, 0.4, 24, 6), 336: (0.08, 256, 0.4, 24, 6),
              720: (0.12, 64, 0.05, 24, 6)},
    "ETTh2": {96: (0.2, 512, 0.2, 24, 6), 192: (0.08, 512, 0.06, 24, 4), 336: (0.12, 64, 0.4, 24, 6),
              720: (0.12, 64, 0.4, 24, 6)},
    "ETTm1": {96: (0.04, 512, 0.02, 4, 18), 192: (0.04, 256, 0.02, 4, 20), 336: (0.08, 256, 0.02, 4, 20),
              720: (0.12, 128, 0.01, 6, 20)},
    "ETTm2": {96: (0.04, 64, 0.01, 4, 20), 192: (0.04, 64, 0.01, 4, 20), 336: (0.04, 64, 0.01, 4, 20),
              720: (0.04, 64, 0.01, 6, 20)},
    "Weather": {96: (0.04, 512, 0.02, 4, 6), 192: (0.04, 512, 0.02, 4, 6), 336: (0.08, 512, 0.05, 4, 6),
                720: (0.04, 512, 0.05, 4, 6)},
    "Electricity": {96: (0.04, 128, 0.02, 24, 6), 192: (0.04, 128, 0.02, 24, 6), 336: (0.04, 128, 0.02, 24, 6),
                    720: (0.08, 128, 0.08, 24, 6)},
    "Traffic": {96: (0.04, 128, 0.05, 24, 6), 192: (0.04, 128, 0.03, 24, 8), 336: (0.08, 128, 0.03, 24, 15),
                720: (0.04, 128, 0.03, 24, 8)},
}
TB_INDIVIDUAL = {"ETTh1": False, "ETTh2": False, "ETTm1": False, "ETTm2": False, "Weather": True,
                 "Electricity": False, "Traffic": False}
# Lịch train chung của CycleNet/TQNet (run.py chính thức): 30 epoch, patience 5, lradj type3, Adam không weight decay,
# không clip gradient. CycleNet theo scripts/CycleNet/{Linear,MLP}-Input-720/*.sh (MLP d_model 512). TQNet chỉ có
# script L = 96 (scripts/TQNet/*.sh): giữ nguyên siêu tham số, chỉ đổi seq_len; d_model 512.
#   giá trị: (batch, lr) của CycleNet/Linear, CycleNet/MLP, TQNet và dropout của TQNet (không có --dropout = 0).
PERIODIC_OFFICIAL = {
    "ETTh1": ((256, 0.005), (256, 0.0005), (256, 0.001), 0.5),
    "ETTh2": ((256, 0.005), (256, 0.0005), (256, 0.001), 0.5),
    "ETTm1": ((256, 0.005), (256, 0.0005), (256, 0.001), 0.5),
    "ETTm2": ((256, 0.005), (256, 0.0005), (256, 0.001), 0.5),
    "Weather": ((256, 0.005), (256, 0.0005), (64, 0.001), 0.5),
    "Electricity": ((64, 0.005), (64, 0.001), (32, 0.003), 0.0),
    "Traffic": ((64, 0.005), (64, 0.0005), (16, 0.003), 0.0),
}
DF_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)   # p1 (DropEdge), p2 (DropNeuron) cho lưới dytbase_d64_p*
PERIODIC_TRAIN = {"epochs": 30, "patience": 5, "lr_schedule": "type3", "weight_decay": 0.0, "grad_clip": 0.0}


def periodic(dataset):
    w = CYCLE[dataset]
    (b1, lr1), (b2, lr2), (b3, lr3), drop = PERIODIC_OFFICIAL[dataset]
    return {
        "cyclenet_off": {"model": "cyclenet", "cycle_len": w, "batch_size": b1, "lr": lr1, **PERIODIC_TRAIN},
        "cyclenet_mlp_off": {"model": "cyclenet_mlp", "cycle_len": w, "batch_size": b2, "lr": lr2, "d_model": 512,
                             **PERIODIC_TRAIN},
        "tqnet_off": {"model": "tqnet", "cycle_len": w, "batch_size": b3, "lr": lr3, "d_model": 512, "dropout": drop,
                      **PERIODIC_TRAIN},
    }


def official(dataset, model, **extra):
    """Biến thể theo horizon: trả hàm horizon → overrides với siêu tham số chính thức của TimeBase."""
    def overrides(horizon):
        orth, batch, lr, period, basis = TB_OFFICIAL[dataset][horizon]
        return {"model": model, "period": period, "basis_num": basis, "period_norm": True,
                "individual": TB_INDIVIDUAL[dataset], "orth_weight": orth, "batch_size": batch, "lr": lr,
                "epochs": 30, "patience": 5, "lr_schedule": "type3", "weight_decay": 0.0, "grad_clip": 0.0, **extra}
    return overrides


def variants(dataset):
    w = CYCLE[dataset]
    return {
        "dyt_norm": {"cross_residual": "norm"},
        "dyt_rcf": {"cross_residual": "norm", "cycle_len": w},
        "dyt_rcf_noaff": {"cross_residual": "norm", "cycle_len": w, "affine_revin": False},
        "rlinear": {"model": "rlinear"},
        "cyclelinear": {"model": "rlinear", "cycle_len": w, "affine_revin": False},
        "dlinear": {"model": "dlinear"},
        **({f"timebase_lr{lr:g}": {"model": "timebase", "period": TB_PERIOD[dataset], "basis_num": 6, "lr": lr}
            for lr in TB_LRS} if dataset in TB_PERIOD else {}),
        # DyTimeBase: backbone TimeBase (lr như lưới của TimeBase) + nhánh trộn kênh (mix_lr = lr gốc của DyTimeNet).
        **({f"dytbase_lr{lr:g}": {"model": "dytimebase", "period": TB_PERIOD[dataset], "basis_num": 6, "lr": lr,
                                  "mix_lr": 5e-4}
            for lr in TB_LRS} if dataset in TB_PERIOD else {}),
        # Cùng backbone, lịch train và siêu tham số chính thức của TimeBase; DyTimeBase chỉ thêm nhánh trộn kênh.
        **({"timebase_off": official(dataset, "timebase"),
            "dytbase_off": official(dataset, "dytimebase", mix_lr=5e-4),
            # Ablation (mô hình đề xuất vẫn là dytbase_off): tắt từng khối của nhánh trộn kênh.
            "dytbase_nocross": official(dataset, "dytimebase", mix_lr=5e-4, use_cross=False),
            "dytbase_nointra": official(dataset, "dytimebase", mix_lr=5e-4, use_intra=False),
            # Cùng kiến trúc, lr nhánh trộn kênh gấp 4 (lịch type3 giảm lr rất nhanh).
            "dytbase_mixlr2e-3": official(dataset, "dytimebase", mix_lr=2e-3)} if dataset in TB_OFFICIAL else {}),
        **periodic(dataset),
        # Lưới DropFusion của nhánh trộn kênh (d_model 64 cố định, xương sống như dytbase_off); chọn theo val loss.
        **({f"dytbase_d64_p{p1:g}_{p2:g}": official(dataset, "dytimebase", mix_lr=5e-4, d_model=64, p1=p1, p2=p2)
            for p1 in DF_GRID for p2 in DF_GRID} if dataset in TB_OFFICIAL else {}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(CYCLE))
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="cuda: chạy trên GPU (ví dụ Colab)")
    parser.add_argument("--results", default=None, help="file JSON kết quả (mặc định results/suite_<tag>.json)")
    args = parser.parse_args(argv)
    table = variants(args.dataset)
    unknown = set(args.variants) - set(table)
    if unknown:
        parser.error(f"Unknown variants: {sorted(unknown)}")
    from dytimenet.experiment import train
    tag = args.dataset if args.seq_len == 96 else f"{args.dataset}_L{args.seq_len}"
    results = Path(args.results or f"results/suite_{tag}.json")
    runs = json.loads(results.read_text(encoding="utf-8")) if results.is_file() else {}
    # Xen kẽ biến thể trong cùng (horizon, seed) để độ trôi của máy ảnh hưởng đều.
    for horizon in args.horizons:
        for seed in args.seeds:
            for name in args.variants:
                key = f"{name}_S{horizon}_seed{seed}"
                if key in runs:
                    continue
                torch.set_num_threads(args.threads)
                overrides = table[name](horizon) if callable(table[name]) else table[name]
                config = load_config(f"configs/{args.dataset}.yaml", {
                    "seq_len": args.seq_len, "pred_len": horizon, "seed": seed, "device": args.device,
                    "save_dir": f"checkpoints/suite_{tag}", **overrides})
                print(f"\n===== {args.dataset} {key} =====", flush=True)
                checkpoint, summary = train(config)
                history = pd.read_csv(Path(checkpoint).parent / "history.csv")
                runs[key] = {"variant": name, "overrides": overrides, "seq_len": args.seq_len, "device": args.device,
                             "pred_len": horizon, "seed": seed, "best_val": float(history["val_loss"].min()),
                             "mse": summary["MSE"], "mae": summary["MAE"], "best_epoch": summary["best_epoch"],
                             "epochs_run": int(len(history)), "train_seconds": summary["train_seconds"],
                             "mean_epoch_seconds": float(history["epoch_seconds"].mean())
                             if "epoch_seconds" in history else None,
                             "checkpoint": str(checkpoint)}
                results.parent.mkdir(parents=True, exist_ok=True)
                # Đọc lại và gộp trước khi ghi để không xoá kết quả của tiến trình khác ghi cùng file.
                runs = {**(json.loads(results.read_text(encoding="utf-8")) if results.is_file() else {}), **runs}
                results.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
