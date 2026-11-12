"""Thử các cách khắc phục DyTimeBase (L = 720) trên ETT và Weather, cùng pipeline với run_suite.py.

Biến thể (backbone và lịch train theo siêu tham số chính thức của TimeBase; nhánh trộn kênh d_model 64, p1/p2 theo
compare_all.TUNED):
  timebase_std        TimeBase + chia std cửa sổ (inst_std).
  dytbase_detach      DyTimeBase, chặn gradient từ nhánh trộn kênh về basis (mix_detach).
  dytbase_2stage      Giai đoạn 2: nạp TimeBase* (timebase_off cùng horizon/seed), đóng băng backbone, chỉ train nhánh
                      trộn kênh; epoch 0 = TimeBase* nên val không thể kém hơn backbone.
  dytbase_std         DyTimeBase + inst_std, train một giai đoạn.
  dytbase_std_2stage  Như dytbase_2stage nhưng nạp timebase_std.
  dytbase_2stage_mlr<lr>  Như dytbase_2stage, lr nhánh trộn kênh <lr> thay 5e-4 (chọn chung cho mọi dataset).
  dytbase_2stage_long Như dytbase_2stage, huấn luyện lâu hơn: lịch cosine (warmup 2), 50 epoch, patience 10 thay lịch
                      type3 (lr giảm 0.8 mỗi epoch từ epoch 4), 30 epoch, patience 5.
  dytbase_2stage_R<r> Nạp timebase_off_R<r> (run_suite.R_GRID: R lớn hơn cho ba bộ phụ tải), basis R = <r>.
  dytbase_2stage_p<p1>_<p2>  Lưới DropFusion của giai đoạn 2 (p1, p2 ∈ run_suite.DF_GRID, d_model 64), chọn theo val
                      loss với seed 2021 như scripts/summarize_grid.py; (0.1, 0.1) chính là dytbase_2stage của Load.
Ví dụ: python scripts/run_fix.py --dataset ETTm2 --variants timebase_std dytbase_2stage
Kết quả ghi vào results/fix_<dataset>_L720.json (chạy lại để tiếp tục phần còn thiếu).
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

from dytimenet.config import load_config
from run_suite import DF_GRID, R_GRID, official

SEQ_LEN = 720
# p1, p2 của nhánh trộn kênh (d_model 64) như mô hình đang báo cáo: ETTh2, ETTm2 theo compare_all.TUNED, còn lại theo
# configs/*.yaml (Đồ án I, Bảng 3.1).
TUNED = {"ETTh1": (0.1, 0.1), "ETTh2": (0.3, 0.5), "ETTm1": (0.1, 0.2), "ETTm2": (0.4, 0.6), "Weather": (0.1, 0.1),
         "Load": (0.1, 0.1),   # Load, Exchange: như ETTh1
         "Exchange": (0.1, 0.1),
         # Bộ phụ tải mới: lưới dytbase_d64_p* một giai đoạn, chọn theo val loss seed 2021 (scripts/summarize_grid.py).
         "Tetouan": (0.2, 0.5), "Panama": (0.2, 0.1), "PJM": (0.5, 0.1)}
# File chứa timebase_off (giai đoạn 1); Weather dùng bản chạy lại theo siêu tham số chính thức (compare_all.DATASETS).
SUITE = {"Weather": f"results/suite_Weather_L{SEQ_LEN}_official.json"}


def branch(dataset):
    p1, p2 = TUNED[dataset]
    return {"mix_lr": 5e-4, "d_model": 64, "p1": p1, "p2": p2}


def stage1_checkpoint(dataset, variant, horizon, seed):
    """Checkpoint backbone của giai đoạn 1: timebase_off trong suite_*, timebase_std trong fix_*."""
    source = SUITE.get(dataset, f"results/suite_{dataset}_L{SEQ_LEN}.json") if variant.startswith("timebase_off") else \
        f"results/fix_{dataset}_L{SEQ_LEN}.json"
    runs = json.loads(Path(source).read_text(encoding="utf-8"))
    key = f"{variant}_S{horizon}_seed{seed}"
    if key not in runs:
        raise KeyError(f"Run {key} first (stage 1 for the two-stage variant)")
    return runs[key]["checkpoint"]


def overrides(dataset, name, horizon, seed):
    tb, dyt = official(dataset, "timebase"), official(dataset, "dytimebase", **branch(dataset))
    if name == "timebase_std":
        return {**tb(horizon), "inst_std": True}
    if name == "dytbase_detach":
        return {**dyt(horizon), "mix_detach": True}
    if name == "dytbase_std":
        return {**dyt(horizon), "inst_std": True}
    if name.startswith("dytbase_2stage_mlr"):
        return {**overrides(dataset, "dytbase_2stage", horizon, seed),
                "mix_lr": float(name.removeprefix("dytbase_2stage_mlr"))}
    if name == "dytbase_2stage_long":
        return {**overrides(dataset, "dytbase_2stage", horizon, seed), "lr_schedule": "cosine", "epochs": 50,
                "patience": 10}
    if name.startswith("dytbase_2stage_R"):
        r = int(name.removeprefix("dytbase_2stage_R"))
        ckpt = stage1_checkpoint(dataset, f"timebase_off_R{r}", horizon, seed)
        return {**dyt(horizon), "basis_num": r, "init_from": ckpt, "freeze_backbone": True}
    if name.startswith("dytbase_2stage_p"):
        p1, p2 = (float(x) for x in name.removeprefix("dytbase_2stage_p").split("_"))
        return {**overrides(dataset, "dytbase_2stage", horizon, seed), "p1": p1, "p2": p2}
    if name in ("dytbase_2stage", "dytbase_std_2stage"):
        std = name == "dytbase_std_2stage"
        ckpt = stage1_checkpoint(dataset, "timebase_std" if std else "timebase_off", horizon, seed)
        return {**dyt(horizon), "inst_std": std, "init_from": ckpt, "freeze_backbone": True}
    raise KeyError(name)


MIX_LRS = (2.5e-4, 1e-3, 2e-3)   # lr nhánh trộn kênh thử thay 5e-4
VARIANTS = ("timebase_std", "dytbase_detach", "dytbase_2stage", "dytbase_std", "dytbase_std_2stage",
            "dytbase_2stage_long", *(f"dytbase_2stage_mlr{lr:g}" for lr in MIX_LRS),
            *sorted({f"dytbase_2stage_R{r}" for rs in R_GRID.values() for r in rs}),
            *(f"dytbase_2stage_p{p1:g}_{p2:g}" for p1 in DF_GRID for p2 in DF_GRID))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(TUNED))
    parser.add_argument("--variants", nargs="+", required=True, choices=VARIANTS)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    from dytimenet.experiment import train
    results = Path(f"results/fix_{args.dataset}_L{SEQ_LEN}.json")
    runs = json.loads(results.read_text(encoding="utf-8")) if results.is_file() else {}
    for horizon in args.horizons:
        for seed in args.seeds:
            for name in args.variants:
                key = f"{name}_S{horizon}_seed{seed}"
                if key in runs:
                    continue
                torch.set_num_threads(args.threads)
                ov = overrides(args.dataset, name, horizon, seed)
                config = load_config(f"configs/{args.dataset}.yaml", {
                    "seq_len": SEQ_LEN, "pred_len": horizon, "seed": seed, "device": "cpu",
                    "save_dir": f"checkpoints/fix_{args.dataset}_L{SEQ_LEN}", **ov})
                print(f"\n===== {args.dataset} {key} =====", flush=True)
                checkpoint, summary = train(config)
                history = pd.read_csv(Path(checkpoint).parent / "history.csv")
                runs[key] = {"variant": name, "overrides": ov, "seq_len": SEQ_LEN, "pred_len": horizon, "seed": seed,
                             "best_val": float(summary["best_val_loss"]), "mse": summary["MSE"], "mae": summary["MAE"],
                             "best_epoch": summary["best_epoch"], "epochs_run": int(len(history)),
                             "train_seconds": summary["train_seconds"], "checkpoint": str(checkpoint)}
                runs = {**(json.loads(results.read_text(encoding="utf-8")) if results.is_file() else {}), **runs}
                results.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
