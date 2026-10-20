"""Quét kích thước (C, L, S) để so latency của hai cách gộp (channel / shared) với bản gốc của DyTimeNet.

Dùng để chọn ngưỡng AUTO_SHARED_BYTES. Model khởi tạo ngẫu nhiên (latency không phụ thuộc trọng số), batch 1,
8 luồng, median 5 vòng xen kẽ. Ghi results/bench_fold_modes.json và .md. Ví dụ: python scripts/bench_fold_modes.py
"""

import gc
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dytimenet.models import build_model
from dytimenet.models.folded import fold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_vs_timebase import timed  # noqa: E402

SIZES = [(7, 96, 96), (21, 96, 96), (21, 96, 720), (7, 336, 336), (21, 336, 336), (137, 96, 96),
         (321, 96, 96), (21, 720, 720), (137, 336, 336), (321, 336, 336), (321, 720, 720)]


def main():
    torch.set_num_threads(8)
    rows = []
    for C, L, S in SIZES:
        torch.manual_seed(0)
        model = build_model("dytimenet", num_vars=C, seq_len=L, pred_len=S, d_model=64, cross_residual="norm").eval()
        models = {"gốc": model, "channel": fold(model, mode="channel"), "shared": fold(model, mode="shared")}
        x = torch.randn(1, C, L)
        vals = {k: [] for k in models}
        with torch.inference_mode():
            for _ in range(5):
                for k, m in models.items():
                    vals[k].append(timed(lambda m=m: m(x)))
        row = {"C": C, "L": L, "S": S, "channel_MB": 4 * C * (L + S) * L / 2**20,
               **{k: statistics.median(v) for k, v in vals.items()}}
        rows.append(row)
        print(row, flush=True)
        del models, model; gc.collect()
    lines = ["# Latency DyTimeNet (norm): bản gốc, gộp channel, gộp shared (ms, b1/t8)", "",
             "| C | L | S | Bộ đệm channel (MB) | gốc | gộp channel | gộp shared | nhanh nhất |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        best = min(("gốc", "channel", "shared"), key=lambda k: r[k])
        lines.append(f"| {r['C']} | {r['L']} | {r['S']} | {r['channel_MB']:.1f} | {r['gốc']:.3f} | "
                     f"{r['channel']:.3f} | {r['shared']:.3f} | {best} |")
    Path("results/bench_fold_modes.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    Path("results/bench_fold_modes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
