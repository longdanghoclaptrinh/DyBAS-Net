"""Train RLinear và DLinear trên ETTh1 trong đúng pipeline của DyTimeNet (4 horizon × 3 seed).

Ví dụ: python scripts/run_baselines.py
Kết quả từng run ghi ngay vào results/baselines_etth1.json (chạy lại để tiếp tục).
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from dytimenet.config import load_config

MODELS = ("rlinear", "dlinear")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
RESULTS = Path("results/baselines_etth1.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save_dir", default="checkpoints/baselines")
    args = parser.parse_args(argv)
    from dytimenet.experiment import train
    runs = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.is_file() else {}
    for horizon in HORIZONS:
        for seed in SEEDS:
            for name in MODELS:
                key = f"{name}_S{horizon}_seed{seed}"
                if key in runs:
                    continue
                torch.set_num_threads(args.threads)
                config = load_config("configs/ETTh1.yaml", {
                    "seq_len": 96, "pred_len": horizon, "seed": seed, "device": "cpu",
                    "save_dir": args.save_dir, "model": name})
                print(f"\n===== {key} =====", flush=True)
                checkpoint, summary = train(config)
                history = pd.read_csv(Path(checkpoint).parent / "history.csv")
                runs[key] = {"model": name, "pred_len": horizon, "seed": seed, "mse": summary["MSE"],
                             "mae": summary["MAE"], "best_epoch": summary["best_epoch"],
                             "epochs_run": int(len(history)), "train_seconds": summary["train_seconds"],
                             "checkpoint": str(checkpoint)}
                RESULTS.parent.mkdir(parents=True, exist_ok=True)
                RESULTS.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
