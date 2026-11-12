"""Tổng hợp lưới DropFusion (p1, p2) của DyTimeBase: chọn cặp có val MSE trung bình 4 horizon thấp nhất.

Đọc results/grid_<dataset>_L720*.json (do scripts/run_suite.py ghi với các biến thể dytbase_d64_p<p1>_<p2>, seed 2021).
Chỉ dùng val loss để chọn; test MSE in kèm để tham khảo. Ví dụ: python scripts/summarize_grid.py --dataset ETTh2 ETTm2
"""

import argparse
import json
from pathlib import Path
import re
import statistics

HORIZONS = (96, 192, 336, 720)
SELECT_SEED = 2021
NAME = re.compile(r"dytbase_d64_p([\d.]+)_([\d.]+)$")


def load(dataset):
    runs = {}
    for path in sorted(Path("results").glob(f"grid_{dataset}_L720*.json")):
        runs.update(json.loads(path.read_text(encoding="utf-8")))
    table = {}
    for r in runs.values():
        m = NAME.match(r["variant"])
        if m and r["seed"] == SELECT_SEED:      # seed khác là lần chạy kiểm chứng sau khi đã chọn
            table.setdefault((float(m[1]), float(m[2])), {})[r["pred_len"]] = r
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="+", default=["ETTh2", "ETTm2"])
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)
    for ds in args.dataset:
        table = load(ds)
        done = {k: v for k, v in table.items() if all(h in v for h in HORIZONS)}
        print(f"\n## {ds}: {len(done)}/{len(table)} cặp (p1, p2) đủ 4 horizon")
        rows = sorted(((statistics.mean(v[h]["best_val"] for h in HORIZONS),
                        statistics.mean(v[h]["mse"] for h in HORIZONS), k, v) for k, v in done.items()))
        print("| hạng | p1 | p2 | val MSE TB | test MSE TB | " + " | ".join(f"test S={h}" for h in HORIZONS) + " |")
        print("| ---: | ---: | ---: | ---: | ---: |" + " ---: |" * len(HORIZONS))
        for i, (val, test, (p1, p2), v) in enumerate(rows[:args.top], 1):
            print(f"| {i} | {p1:g} | {p2:g} | {val:.4f} | {test:.4f} | "
                  + " | ".join(f"{v[h]['mse']:.4f}" for h in HORIZONS) + " |")


if __name__ == "__main__":
    main()
