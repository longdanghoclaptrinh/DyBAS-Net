"""Bảng độ chính xác và latency cho các biến thể train bằng scripts/run_suite.py (cùng pipeline).

Đọc results/suite_<dataset>.json (và các file kết quả cũ truyền qua --extra), ghi results/compare_suite_<dataset>.md
và results/compare_suite_<dataset>_latency.json. Latency đo xen kẽ ở seed 2021; biến thể DyTimeNet đo cả bản đã gộp.
Ví dụ: python scripts/compare_suite.py --dataset Weather
       python scripts/compare_suite.py --dataset ETTh1 --extra results/rcf_etth1.json --no_latency
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dytimenet.config import config_from_dict
from dytimenet.data.dataset import TimeSeriesDataset
from dytimenet.models import build_model
from dytimenet.models.folded import fold

HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
LABELS = {"dyt_norm": "DyTimeNet (norm)", "dyt_rcf": "DyTimeNet-RCF", "dyt_rcf_noaff": "DyTimeNet-RCF (RevIN không affine)",
          "rlinear": "RLinear", "cyclelinear": "CycleNet/Linear", "dlinear": "DLinear"}
POINTS = [(1, 8), (64, 8), (1, 1), (64, 1)]


def ms(values, fmt="{:.4f}"):
    return f"{fmt.format(statistics.mean(values))} ± {fmt.format(statistics.stdev(values))}" if len(values) > 1 else "—"


def median_ms(fn, warmup=50, reps=300):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
    return statistics.median(t) * 1e3


def load_model(run):
    ck = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    cfg = config_from_dict(dict(ck["config"]))
    model = build_model(cfg.model, **ck["model_kwargs"])
    model.load_state_dict(ck["model_state_dict"])
    return model.eval(), ck, cfg


def latency(runs, names, horizon, rounds=5):
    """Latency median (ms), xen kẽ các model qua nhiều vòng; mọi model nhận cùng 64 cửa sổ test đầu."""
    models, x_all = {}, None
    for name in names:
        model, ck, cfg = load_model(runs[f"{name}_S{horizon}_seed2021"])
        if x_all is None:
            ds = TimeSeriesDataset(cfg.data_path, cfg.seq_len, cfg.pred_len, "test", dataset_name=cfg.dataset,
                                   scaler_state=ck["scaler"])
            x_all = torch.stack([ds[i][0] for i in range(64)])
            row_all = torch.arange(64) + ds.start       # chỉ số dòng toàn cục; pha = row mod cycle_len
        cycle = getattr(model, "cycle_len", 0)
        models[LABELS[name]] = (model, cycle)
        if name.startswith("dyt"):
            models[f"{LABELS[name]} — đã gộp"] = (fold(model), cycle)
    out = {}
    with torch.inference_mode():
        for b, t in POINTS:
            torch.set_num_threads(t)
            x = x_all[:b].contiguous()
            calls = {label: ((lambda m=m, i=(row_all[:b] % c): m(x, i)) if c else (lambda m=m: m(x)))
                     for label, (m, c) in models.items()}
            vals = {label: [] for label in models}
            for _ in range(rounds):
                for label, call in calls.items():
                    vals[label].append(median_ms(call))
            for label, v in vals.items():
                out.setdefault(label, {})[f"b{b}/t{t}"] = statistics.median(v)
    torch.set_num_threads(8)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--extra", nargs="*", default=[], help="file kết quả cũ (dict key → run có 'variant')")
    parser.add_argument("--no_latency", action="store_true")
    args = parser.parse_args(argv)
    tag = args.dataset.lower()
    runs = json.loads(Path(f"results/suite_{args.dataset}.json").read_text(encoding="utf-8"))
    for path in args.extra:
        for run in json.loads(Path(path).read_text(encoding="utf-8")).values():
            runs.setdefault(f"{run['variant']}_S{run['pred_len']}_seed{run['seed']}", run)
    names = [n for n in LABELS if all(f"{n}_S{h}_seed{s}" in runs for h in HORIZONS for s in SEEDS)]
    lines = [
        f"# {args.dataset}: DyTimeNet, DyTimeNet-RCF và baseline tuyến tính (cùng pipeline)",
        "",
        "L = 96, Adam lr 5e-4, cosine, tối đa 100 epoch, patience 10, batch 64, seed 2021–2023, CPU 8 luồng. "
        "DyTimeNet dùng residual `norm`; RCF thay MovingAvg bằng chu kỳ học được (CycleNet). "
        "CycleNet/Linear = RevIN không affine + RCF + Linear L→S.",
        "",
        "## 1. Độ chính xác (MSE, mean ± std qua 3 seed)",
        "",
        "| Mô hình | " + " | ".join(f"S={h}" for h in HORIZONS) + " | AVG MSE | AVG MAE | epoch TB |",
        "| --- |" + " ---: |" * (len(HORIZONS) + 3),
    ]
    for name in names:
        get = lambda h, s: runs[f"{name}_S{h}_seed{s}"]                            # noqa: E731
        cells = [ms([get(h, s)["mse"] for s in SEEDS]) for h in HORIZONS]
        avg_mse = [statistics.mean(get(h, s)["mse"] for h in HORIZONS) for s in SEEDS]
        avg_mae = [statistics.mean(get(h, s)["mae"] for h in HORIZONS) for s in SEEDS]
        epochs = statistics.mean(get(h, s)["epochs_run"] for h in HORIZONS for s in SEEDS)
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + f" | {ms(avg_mse)} | {ms(avg_mae)} | {epochs:.1f} |")
    params = {LABELS[n]: load_model(runs[f"{n}_S96_seed2021"])[0].count_parameters() for n in names}
    lines += ["", "Số tham số (S = 96): " + ", ".join(f"{k} {v:,}" for k, v in params.items()) + ".", ""]

    if not args.no_latency:
        lat = {h: latency(runs, names, h) for h in (96, 720)}
        Path(f"results/compare_suite_{tag}_latency.json").write_text(
            json.dumps(lat, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        for sec, (h, table) in enumerate(lat.items(), 2):
            lines += [f"## {sec}. Latency suy luận S = {h} (ms, median 5 vòng xen kẽ × 300 lần, eager, inference_mode)",
                      "", "| Mô hình | " + " | ".join(f"b{b}/t{t}" for b, t in POINTS) + " |",
                      "| --- |" + " ---: |" * len(POINTS)]
            for label, row in table.items():
                lines.append(f"| {label} | " + " | ".join(f"{row[f'b{b}/t{t}']:.3f}" for b, t in POINTS) + " |")
            lines.append("")
    Path(f"results/compare_suite_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
