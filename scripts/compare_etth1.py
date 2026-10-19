"""So sánh DyTimeNet (double/norm, gốc và đã gộp) với RLinear/DLinear trên ETTh1: độ chính xác + latency.

Đọc results/residual_ablation_etth1.json và results/baselines_etth1.json; đo latency xen kẽ
(seed 2021, mỗi horizon) rồi ghi results/compare_etth1.md và results/compare_etth1_latency.json.
Ví dụ: python scripts/compare_etth1.py
"""

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
ROWS = [("DyTimeNet double", "double"), ("DyTimeNet norm", "norm"), ("RLinear", "rlinear"), ("DLinear", "dlinear")]
POINTS = [(1, 8), (64, 8), (1, 1), (64, 1)]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get(runs, key, h, s):
    return runs.get(f"{key}_S{h}_seed{s}")


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


def latency(runs_all, horizon, rounds=5):
    """Latency median (ms) của các model ở seed 2021, đo xen kẽ qua nhiều vòng."""
    models, x_all = {}, None
    for label, key in ROWS:
        run = get(runs_all, key, horizon, 2021)
        model, ck, cfg = load_model(run)
        if x_all is None:
            ds = TimeSeriesDataset(cfg.data_path, cfg.seq_len, cfg.pred_len, "test", dataset_name=cfg.dataset,
                                   scaler_state=ck["scaler"])
            x_all = torch.stack([ds[i][0] for i in range(64)])
        models[label] = model
        if key == "norm":
            models["DyTimeNet norm (đã gộp)"] = fold(model)
    out = {}
    with torch.inference_mode():
        for b, t in POINTS:
            torch.set_num_threads(t)
            x = x_all[:b].contiguous()
            vals = {label: [] for label in models}
            for _ in range(rounds):
                for label, m in models.items():
                    vals[label].append(median_ms(lambda: m(x)))
            for label, v in vals.items():
                out.setdefault(label, {})[f"b{b}/t{t}"] = statistics.median(v)
    torch.set_num_threads(8)
    return out


def main():
    dyt = load("results/residual_ablation_etth1.json")
    base = load("results/baselines_etth1.json")
    runs_all = {**{k: v for k, v in dyt.items()}, **base}
    lines = [
        "# ETTh1: DyTimeNet so với RLinear / DLinear trong cùng pipeline",
        "",
        "Cùng split 12/4/4 tháng, StandardScaler, L = 96, Adam lr 5e-4, cosine, tối đa 100 epoch, patience 10, "
        "batch 64, `drop_last=False` ở test, seed 2021–2023, CPU 8 luồng. RLinear = RevIN + Linear L→S; "
        "DLinear = MovingAvg(25) + 2 Linear L→S (đều dùng chung cho mọi kênh).",
        "",
        "## 1. Độ chính xác (MSE / MAE, mean ± std qua 3 seed)",
        "",
        "| Mô hình | " + " | ".join(f"S={h} MSE" for h in HORIZONS) + " | AVG MSE | AVG MAE |",
        "| --- |" + " ---: |" * (len(HORIZONS) + 2),
    ]
    for label, key in ROWS:
        cells = [ms([r["mse"] for s in SEEDS if (r := get(runs_all, key, h, s))]) for h in HORIZONS]
        avg_mse = [statistics.mean(get(runs_all, key, h, s)["mse"] for h in HORIZONS) for s in SEEDS]
        avg_mae = [statistics.mean(get(runs_all, key, h, s)["mae"] for h in HORIZONS) for s in SEEDS]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {ms(avg_mse)} | {ms(avg_mae)} |")
    params = {}
    for label, key in ROWS:
        model, _, _ = load_model(get(runs_all, key, 96, 2021))
        params[label] = model.count_parameters()
    lines += ["", "Số tham số (S = 96): " + ", ".join(f"{k} {v:,}" for k, v in params.items()) + ".", ""]

    lat = {h: latency(runs_all, h) for h in (96, 720)}
    Path("results/compare_etth1_latency.json").write_text(json.dumps(lat, indent=1, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")
    for h, table in lat.items():
        lines += [f"## 2. Latency suy luận S = {h} (ms, median 5 vòng xen kẽ × 300 lần, eager, inference_mode)", "",
                  "| Mô hình | " + " | ".join(f"b{b}/t{t}" for b, t in POINTS) + " |",
                  "| --- |" + " ---: |" * len(POINTS)]
        for label, row in table.items():
            lines.append(f"| {label} | " + " | ".join(f"{row[f'b{b}/t{t}']:.3f}" for b, t in POINTS) + " |")
        lines.append("")
    Path("results/compare_etth1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
