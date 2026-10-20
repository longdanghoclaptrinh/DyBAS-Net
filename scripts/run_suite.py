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

# Chu kỳ dài nhất ổn định của từng dataset (như Table 1 của CycleNet): ngày theo tần số lấy mẫu.
CYCLE = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96, "Weather": 144}
# Độ dài đoạn P của TimeBase theo paper (Phụ lục C.1): chu kỳ tự nhiên, Weather dùng P = 4.
TB_PERIOD = {"ETTh1": 24, "ETTh2": 24, "Weather": 4}
TB_LRS = (0.005, 0.02, 0.08)


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
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(CYCLE))
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=96)
    args = parser.parse_args(argv)
    table = variants(args.dataset)
    unknown = set(args.variants) - set(table)
    if unknown:
        parser.error(f"Unknown variants: {sorted(unknown)}")
    from dytimenet.experiment import train
    tag = args.dataset if args.seq_len == 96 else f"{args.dataset}_L{args.seq_len}"
    results = Path(f"results/suite_{tag}.json")
    runs = json.loads(results.read_text(encoding="utf-8")) if results.is_file() else {}
    # Xen kẽ biến thể trong cùng (horizon, seed) để độ trôi của máy ảnh hưởng đều.
    for horizon in args.horizons:
        for seed in args.seeds:
            for name in args.variants:
                key = f"{name}_S{horizon}_seed{seed}"
                if key in runs:
                    continue
                torch.set_num_threads(args.threads)
                config = load_config(f"configs/{args.dataset}.yaml", {
                    "seq_len": args.seq_len, "pred_len": horizon, "seed": seed, "device": "cpu",
                    "save_dir": f"checkpoints/suite_{tag}", **table[name]})
                print(f"\n===== {args.dataset} {key} =====", flush=True)
                checkpoint, summary = train(config)
                history = pd.read_csv(Path(checkpoint).parent / "history.csv")
                runs[key] = {"variant": name, "overrides": table[name], "seq_len": args.seq_len,
                             "pred_len": horizon, "seed": seed, "best_val": float(history["val_loss"].min()),
                             "mse": summary["MSE"], "mae": summary["MAE"], "best_epoch": summary["best_epoch"],
                             "epochs_run": int(len(history)), "train_seconds": summary["train_seconds"],
                             "mean_epoch_seconds": float(history["epoch_seconds"].mean())
                             if "epoch_seconds" in history else None,
                             "checkpoint": str(checkpoint)}
                results.parent.mkdir(parents=True, exist_ok=True)
                results.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
