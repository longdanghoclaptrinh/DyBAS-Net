"""Bảng độ chính xác trên 5 dataset (L = 720): DyTimeBase và TimeBase tự chạy + 10 baseline theo số công bố.

Tự chạy (trung bình 3 seed, cùng pipeline, siêu tham số chính thức của TimeBase): biến thể timebase_off / dytbase_off
của scripts/run_suite.py. Số công bố: results/published_timebase_table10.json (Table 10 của TimeBase, ICML 2025).
Ghi results/compare_all_L720.md. Ví dụ: python scripts/compare_all.py
"""

import json
from pathlib import Path
import statistics

DATASETS = {"ETTh1": "results/suite_ETTh1_L720.json", "ETTh2": "results/suite_ETTh2_L720.json",
            "ETTm1": "results/suite_ETTm1_L720.json", "ETTm2": "results/suite_ETTm2_L720.json",
            "Weather": "results/suite_Weather_L720_official.json"}
HORIZONS = ("96", "192", "336", "720")
SEEDS = (2021, 2022, 2023)
OURS = {"DyTimeBase": "dytbase_off", "TimeBase*": "timebase_off"}
PUBLISHED = ["TimeBase", "SparseTSF", "TimeMixer", "FITS", "iTransformer", "DLinear", "PatchTST", "TimesNet",
             "FEDformer", "Autoformer", "Informer"]


def main():
    pub = json.loads(Path("results/published_timebase_table10.json").read_text(encoding="utf-8"))["results"]
    cols = list(OURS) + PUBLISHED
    table, std = {}, {}
    for ds, path in DATASETS.items():
        runs = json.loads(Path(path).read_text(encoding="utf-8"))
        for h in HORIZONS:
            row = {}
            for label, v in OURS.items():
                rs = [runs[f"{v}_S{h}_seed{s}"] for s in SEEDS]
                row[label] = (statistics.mean(r["mse"] for r in rs), statistics.mean(r["mae"] for r in rs))
                std[(ds, h, label)] = statistics.stdev(r["mse"] for r in rs)
            for m in PUBLISHED:
                row[m] = (pub[ds][m][h]["mse"], pub[ds][m][h]["mae"])
            table[(ds, h)] = row
        table[(ds, "Avg")] = {c: tuple(statistics.mean(table[(ds, h)][c][i] for h in HORIZONS) for i in (0, 1))
                              for c in cols}

    def cell(ds, h, c, i):
        vals = sorted({round(table[(ds, h)][k][i], 3) for k in cols})
        v = round(table[(ds, h)][c][i], 3)
        s = f"{v:.3f}"
        return f"**{s}**" if v == vals[0] else f"<u>{s}</u>" if len(vals) > 1 and v == vals[1] else s

    lines = ["# Độ chính xác dài hạn, L = 720 cho mọi mô hình", "",
             "DyTimeBase và TimeBase\\* tự chạy trong cùng pipeline (siêu tham số chính thức của TimeBase, trung bình 3 seed "
             "2021–2023, CPU). Các cột còn lại là số công bố trong Table 10 của TimeBase (Huang et al., ICML 2025), cũng "
             "với L = 720 (cột TimeBase là số họ công bố, chạy 1 lần). **Đậm** = tốt nhất, <u>gạch chân</u> = thứ hai "
             "(làm tròn 3 chữ số).", "",
             "| Dataset | S | " + " | ".join(f"{c} MSE | {c} MAE" for c in cols) + " |",
             "| --- | --- |" + " ---: |" * (2 * len(cols))]
    wins = {c: 0 for c in cols}
    for ds in DATASETS:
        for h in HORIZONS + ("Avg",):
            lines.append(f"| {ds} | {h} | " + " | ".join(f"{cell(ds, h, c, 0)} | {cell(ds, h, c, 1)}" for c in cols) + " |")
            if h != "Avg":
                for i in (0, 1):
                    best = min(round(table[(ds, h)][k][i], 3) for k in cols)
                    for c in cols:
                        wins[c] += round(table[(ds, h)][c][i], 3) == best
    lines += ["", "Số lần đứng nhất (40 ô = 5 dataset × 4 horizon × MSE/MAE, tính cả hòa): "
              + ", ".join(f"{c} {n}" for c, n in sorted(wins.items(), key=lambda kv: -kv[1])) + ".", "",
              "## DyTimeBase so với TimeBase tự chạy (cùng pipeline, cùng siêu tham số)", "",
              "| Dataset | S | DyTimeBase MSE | TimeBase\\* MSE | Chênh lệch | std DyTimeBase | std TimeBase\\* |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for ds in DATASETS:
        for h in HORIZONS + ("Avg",):
            a, b = table[(ds, h)]["DyTimeBase"][0], table[(ds, h)]["TimeBase*"][0]
            sa = f"{std[(ds, h, 'DyTimeBase')]:.4f}" if h != "Avg" else ""
            sb = f"{std[(ds, h, 'TimeBase*')]:.4f}" if h != "Avg" else ""
            lines.append(f"| {ds} | {h} | {a:.4f} | {b:.4f} | {100 * (a - b) / b:+.1f}% | {sa} | {sb} |")
    Path("results/compare_all_L720.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-32:]))
    print(lines[-34])


if __name__ == "__main__":
    main()
