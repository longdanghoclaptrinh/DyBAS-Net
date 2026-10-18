"""So sánh 3 cách đặt residual quanh CrossVBlock trên ETTh1 (4 horizon × 3 seed), rồi xuất bảng.

  double : s ← cross_v(s) + s = 2s + f(LN s)   (code hiện tại, giữ từ notebook)
  single : s ← s + f(LN s)                     (Fig. 5 của paper)
  norm   : s ← LN(s) + f(LN s)                 (eq. 2.14 đồ án, eq. 25 paper)

Ví dụ: python scripts/run_residual_ablation.py            (chạy tiếp các run còn thiếu rồi viết bảng)
       python scripts/run_residual_ablation.py --report_only
Kết quả từng run được ghi ngay vào results/residual_ablation_etth1.json.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from dytimenet.config import load_config

VARIANTS = ("double", "single", "norm")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
RESULTS = Path("results/residual_ablation_etth1.json")
REPORT = Path("results/residual_ablation_etth1.md")


def run_all(args):
    from dytimenet.experiment import train
    runs = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.is_file() else {}
    # Xen kẽ biến thể trong cùng (horizon, seed) để độ trôi của máy ảnh hưởng đều.
    for horizon in HORIZONS:
        for seed in SEEDS:
            for variant in VARIANTS:
                key = f"{variant}_S{horizon}_seed{seed}"
                if key in runs:
                    continue
                torch.set_num_threads(args.threads)
                config = load_config("configs/ETTh1.yaml", {
                    "seq_len": 96, "pred_len": horizon, "seed": seed, "device": "cpu",
                    "save_dir": args.save_dir, "cross_residual": variant})
                print(f"\n===== {key} =====", flush=True)
                checkpoint, summary = train(config)
                history = pd.read_csv(Path(checkpoint).parent / "history.csv")
                runs[key] = {"variant": variant, "pred_len": horizon, "seed": seed,
                             "mse": summary["MSE"], "mae": summary["MAE"],
                             "best_epoch": summary["best_epoch"], "epochs_run": int(len(history)),
                             "best_val_loss": summary["best_val_loss"],
                             "train_seconds": summary["train_seconds"], "checkpoint": str(checkpoint)}
                RESULTS.parent.mkdir(parents=True, exist_ok=True)
                RESULTS.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    return runs


def ms(values, fmt="{:.4f}"):
    if len(values) < 2:
        return fmt.format(values[0]) + " (n=1)" if values else "chưa chạy"
    return f"{fmt.format(statistics.mean(values))} ± {fmt.format(statistics.stdev(values))}"


def write_report(runs, threads):
    get = lambda v, h, s: runs.get(f"{v}_S{h}_seed{s}")  # noqa: E731
    lines = [
        "# Residual quanh CrossVBlock: double vs single vs norm (ETTh1)",
        "",
        "- `double`: `s ← cross_v(s) + s = 2s + f(LN s)` — code hiện tại (giữ từ notebook).",
        "- `single`: `s ← s + f(LN s)` — Fig. 5 của paper.",
        "- `norm`: `s ← LN(s) + f(LN s)` — eq. 2.14 trong báo cáo đồ án, eq. 25 của paper.",
        "",
        "Mọi thứ khác giữ nguyên `configs/ETTh1.yaml` (seq_len=96, d_model=64, batch=64, lr=0.0005, "
        f"p1=p2=0.1, tối đa 100 epoch, patience=10). CPU, `torch.set_num_threads({threads})`, chạy tuần tự, "
        f"seed {', '.join(map(str, SEEDS))}. MSE/MAE trên tập test, thang StandardScaler. "
        "Ba biến thể có cùng số tham số.",
        "",
        f"## 1. MSE / MAE (mean ± std qua {len(SEEDS)} seed)",
        "",
        "| pred_len | " + " | ".join(f"{v} MSE" for v in VARIANTS) + " | "
        + " | ".join(f"{v} MAE" for v in VARIANTS) + " |",
        "| ---: |" + " ---: |" * (2 * len(VARIANTS)),
    ]
    for h in HORIZONS:
        cells = [ms([r["mse"] for s in SEEDS if (r := get(v, h, s))]) for v in VARIANTS]
        cells += [ms([r["mae"] for s in SEEDS if (r := get(v, h, s))]) for v in VARIANTS]
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    avg = []
    for metric in ("mse", "mae"):
        for v in VARIANTS:
            per_seed = [statistics.mean(get(v, h, s)[metric] for h in HORIZONS)
                        for s in SEEDS if all(get(v, h, s) for h in HORIZONS)]
            avg.append(ms(per_seed))
    lines.append(f"| **AVG {len(HORIZONS)} horizon** | " + " | ".join(avg) + " |")

    lines += [
        "",
        "## 2. Chênh lệch theo cặp so với `double` (cùng horizon, cùng seed)",
        "",
        "Δ = (biến thể − double) / double × 100. Âm = tốt hơn `double`. "
        "Cột \"thắng\" = số cặp (horizon, seed) mà biến thể có MSE thấp hơn `double`.",
        "",
        "| Biến thể | pred_len | ΔMSE mean | ΔMSE min / max | ΔMAE mean | Thắng (MSE) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in VARIANTS[1:]:
        all_mse, wins, total = [], 0, 0
        for h in HORIZONS:
            pairs = [(get(v, h, s), get("double", h, s)) for s in SEEDS if get(v, h, s) and get("double", h, s)]
            if not pairs:
                lines.append(f"| `{v}` | {h} | chưa chạy | | | |")
                continue
            d_mse = [(a["mse"] - b["mse"]) / b["mse"] * 100 for a, b in pairs]
            d_mae = [(a["mae"] - b["mae"]) / b["mae"] * 100 for a, b in pairs]
            w = sum(d < 0 for d in d_mse)
            all_mse += d_mse
            wins += w
            total += len(d_mse)
            lines.append(f"| `{v}` | {h} | {statistics.mean(d_mse):+.2f}% | {min(d_mse):+.2f}% / "
                         f"{max(d_mse):+.2f}% | {statistics.mean(d_mae):+.2f}% | {w}/{len(d_mse)} |")
        if all_mse:
            lines.append(f"| `{v}` | **tất cả** | **{statistics.mean(all_mse):+.2f}%** | {min(all_mse):+.2f}% / "
                         f"{max(all_mse):+.2f}% | | **{wins}/{total}** |")

    lines += [
        "",
        f"## 3. Huấn luyện (mean ± std qua {len(SEEDS)} seed)",
        "",
        "| pred_len | " + " | ".join(f"{v} epoch (best)" for v in VARIANTS) + " | "
        + " | ".join(f"{v} train (s)" for v in VARIANTS) + " |",
        "| ---: |" + " ---: |" * (2 * len(VARIANTS)),
    ]
    for h in HORIZONS:
        cells = [ms([r["best_epoch"] for s in SEEDS if (r := get(v, h, s))], "{:.1f}") for v in VARIANTS]
        cells += [ms([r["train_seconds"] for s in SEEDS if (r := get(v, h, s))], "{:.1f}") for v in VARIANTS]
        lines.append(f"| {h} | " + " | ".join(cells) + " |")

    lines += ["", "## 4. Số liệu từng run", "", "| Run | MSE | MAE | best epoch | epoch đã chạy |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for key in sorted(runs, key=lambda k: (runs[k]["pred_len"], runs[k]["seed"], VARIANTS.index(runs[k]["variant"]))):
        r = runs[key]
        lines.append(f"| `{key}` | {r['mse']:.4f} | {r['mae']:.4f} | {r['best_epoch']} | {r['epochs_run']} |")

    old = REPORT.read_text(encoding="utf-8") if REPORT.is_file() else ""
    marker = "\n## 5. Kết luận"
    tail = old[old.index(marker):] if marker in old else ""
    REPORT.write_text("\n".join(lines) + "\n" + tail, encoding="utf-8")
    print(f"Wrote {REPORT}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save_dir", default="checkpoints/residual_ablation")
    parser.add_argument("--report_only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    runs = json.loads(RESULTS.read_text(encoding="utf-8")) if args.report_only else run_all(args)
    write_report(runs, args.threads)


if __name__ == "__main__":
    main()
