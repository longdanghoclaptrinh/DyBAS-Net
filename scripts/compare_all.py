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


def head_to_head(table, cols):
    """DyTimeBase thắng/thua từng mô hình theo MSE: số ô (dataset × horizon) và AVG MSE từng dataset."""
    out = ["", "## DyTimeBase đối đầu từng mô hình (MSE, 20 ô = 5 dataset × 4 horizon, làm tròn 3 chữ số)", "",
           "| Mô hình | Thắng | Hòa | Thua | DyTimeBase thắng (AVG MSE) | DyTimeBase thua (AVG MSE) |",
           "| --- | ---: | ---: | ---: | --- | --- |"]
    for c in cols[1:]:
        w = t = l = 0
        for ds in DATASETS:
            for h in HORIZONS:
                a, b = round(table[(ds, h)]["DyTimeBase"][0], 3), round(table[(ds, h)][c][0], 3)
                w, t, l = w + (a < b), t + (a == b), l + (a > b)
        win = [ds for ds in DATASETS if table[(ds, "Avg")]["DyTimeBase"][0] < table[(ds, "Avg")][c][0]]
        lose = [ds for ds in DATASETS if ds not in win]
        out.append(f"| {c} | {w} | {t} | {l} | {', '.join(win) or '—'} | {', '.join(lose) or '—'} |")
    return out


ABLATION = {"DyTimeBase (đầy đủ)": "dytbase_off", "bỏ CrossV": "dytbase_nocross", "bỏ IntraV": "dytbase_nointra",
            "bỏ cả nhánh trộn kênh (= TimeBase\\*)": "timebase_off", "đầy đủ, mix_lr 2e-3": "dytbase_mixlr2e-3"}


def ablation():
    out = ["", "## Ablation nhánh trộn kênh (AVG MSE trên 4 horizon, trung bình 3 seed, tự chạy)", "",
           "| Biến thể | " + " | ".join(DATASETS) + " | Trung bình |", "| --- |" + " ---: |" * (len(DATASETS) + 1)]
    all_runs = {ds: json.loads(Path(p).read_text(encoding="utf-8")) for ds, p in DATASETS.items()}
    for label, v in ABLATION.items():
        vals = []
        for ds, runs in all_runs.items():
            keys = [f"{v}_S{h}_seed{s}" for h in HORIZONS for s in SEEDS]
            vals.append(statistics.mean(runs[k]["mse"] for k in keys) if all(k in runs for k in keys) else None)
        cells = [f"{x:.4f}" if x is not None else "chưa chạy" for x in vals]
        mean = f"{statistics.mean(vals):.4f}" if None not in vals else "—"
        out.append(f"| {label} | " + " | ".join(cells) + f" | {mean} |")
    return out


def speed():
    path = Path("results/bench_efficiency.json")
    if not path.is_file():
        return []
    r = json.loads(path.read_text(encoding="utf-8"))
    ref = r["DyTimeBase"]["infer_ms_t8"]
    count = lambda v: f"{v / 1e9:.2f}G" if v >= 1e9 else f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.2f}K"  # noqa: E731
    out = ["", "## Tốc độ trên cùng máy (Electricity, C = 321, L = S = 720; chi tiết: results/bench_efficiency.md)", "",
           "| Mô hình | Params | MACs | Max Mem (MB) | Epoch Time (s) | Infer (ms) | Infer so với DyTimeBase |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n, x in sorted(r.items(), key=lambda kv: kv[1]["infer_ms_t8"]):
        mem = f"> {x['mem_over_gb'] * 1024:.0f}" if "mem_over_gb" in x else f"{x['max_mem_mb']:.0f}"
        out.append(f"| {n} | {count(x['params'])} | {count(x['macs'])} | {mem} | {x['epoch_s']:.0f} | "
                   f"{x['infer_ms_t8']:.2f} | {x['infer_ms_t8'] / ref:.2f}× |")
    return out


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
    lines += head_to_head(table, cols) + ablation() + speed()
    Path("results/compare_all_L720.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
