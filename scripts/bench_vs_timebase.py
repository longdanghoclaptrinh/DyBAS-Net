"""Đo latency CPU và MACs của DyTimeNet so với TimeBase, SparseTSF và các baseline tuyến tính trên cùng một máy.

Latency không phụ thuộc giá trị trọng số nên dùng model khởi tạo ngẫu nhiên (eval). Hai cấu hình:
  paper : như Table 3 của TimeBase (Electricity: C = 321, L = 720, S = 720), latency mỗi mẫu (batch 1)
  ours  : Weather như pipeline của mình (C = 21, L = 96, S = 96 và 720)
Ghi results/bench_vs_timebase.json và .md. Ví dụ: python scripts/bench_vs_timebase.py
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.flop_counter import FlopCounterMode

from dytimenet.models import build_model
from dytimenet.models.folded import fold

SETTINGS = {
    "paper": dict(C=321, L=720, S=(720,), period=24, cycle=24, points=[(1, 8), (1, 1), (12, 8)]),
    "ours": dict(C=21, L=96, S=(96, 720), period=4, cycle=144, points=[(1, 8), (1, 1), (64, 8)]),
}


def candidates(C, L, S, period, cycle, with_fold):
    base = dict(num_vars=C, seq_len=L, pred_len=S)
    out = {
        "TimeBase": build_model("timebase", **base, period=period, basis_num=6),
        "SparseTSF": build_model("sparsetsf", **base, period=period),
        "DLinear": build_model("dlinear", **base),
        "RLinear": build_model("rlinear", **base),
        "CycleNet/Linear": build_model("rlinear", **base, cycle_len=cycle, affine_revin=False),
        "DyTimeNet (norm)": build_model("dytimenet", **base, d_model=64, cross_residual="norm"),
        "DyTimeNet-RCF": build_model("dytimenet", **base, d_model=64, cross_residual="norm", cycle_len=cycle),
    }
    for m in out.values():
        m.eval()
    for mode in ("channel", "shared"):
        if mode == "shared" or with_fold:
            out[f"DyTimeNet (norm) — gộp {mode}"] = fold(out["DyTimeNet (norm)"], mode=mode)
            out[f"DyTimeNet-RCF — gộp {mode}"] = fold(out["DyTimeNet-RCF"], mode=mode)
    return out


def timed(fn, budget=0.4):
    for _ in range(10):
        fn()
    s = time.perf_counter(); fn(); est = time.perf_counter() - s
    reps = max(30, min(300, int(budget / max(est, 1e-6))))
    t = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
    return statistics.median(t) * 1e3


def macs(model, x, idx):
    with FlopCounterMode(display=False) as fc:
        model(x, idx) if getattr(model, "cycle_len", 0) else model(x)
    return fc.get_total_flops() / 2


def bench(name, cfg, S, with_fold, rounds=5):
    torch.manual_seed(0)
    models = candidates(cfg["C"], cfg["L"], S, cfg["period"], cfg["cycle"], with_fold)
    bmax = max(b for b, _ in cfg["points"])
    x_all = torch.randn(bmax, cfg["C"], cfg["L"]) * 2 + 1
    idx_all = torch.randint(0, cfg["cycle"], (bmax,))
    rows = {}
    with torch.inference_mode():
        for label, m in models.items():
            params = sum(p.numel() for p in m.parameters()) if "gộp" not in label else None
            buffers = sum(b.numel() * b.element_size() for b in m.buffers())
            rows[label] = {"params": params, "MACs/mẫu": macs(m, x_all[:1], idx_all[:1]), "buffer_MB": buffers / 2**20}
        for b, t in cfg["points"]:
            torch.set_num_threads(t)
            x, idx = x_all[:b].contiguous(), idx_all[:b]
            calls = {lab: ((lambda m=m: m(x, idx)) if getattr(m, "cycle_len", 0) else (lambda m=m: m(x)))
                     for lab, m in models.items()}
            vals = {lab: [] for lab in calls}
            for _ in range(rounds):
                for lab, call in calls.items():
                    vals[lab].append(timed(call))
            for lab, v in vals.items():
                rows[lab][f"b{b}/t{t}"] = statistics.median(v)
    torch.set_num_threads(8)
    del models; gc.collect()
    return rows


def fmt_macs(v):
    return f"{v / 1e9:.2f}G" if v >= 1e9 else f"{v / 1e6:.2f}M"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no_fold_paper", action="store_true", help="bỏ bản gộp channel ở cấu hình paper (tốn RAM)")
    args = parser.parse_args(argv)
    results, lines = {}, ["# DyTimeNet so với TimeBase/SparseTSF/baseline tuyến tính: latency CPU và MACs (cùng máy)", "",
                          "CPU AMD Ryzen 7 7840HS, PyTorch eager, `inference_mode`, model khởi tạo ngẫu nhiên (latency không "
                          "phụ thuộc trọng số). Median 5 vòng xen kẽ. MACs đếm bằng `torch.utils.flop_counter` (FLOPs/2), "
                          "mỗi mẫu. TimeBase: P = chu kỳ, R = 6, không loss trực giao (không ảnh hưởng suy luận). "
                          "Gộp channel: ma trận riêng từng kênh (bộ nhớ O(C·L·(L+S))); gộp shared: Conv depthwise + "
                          "Linear trend dùng chung (bộ nhớ O(L·S)).", ""]
    for name, cfg in SETTINGS.items():
        for S in cfg["S"]:
            key = f"{name}_C{cfg['C']}_L{cfg['L']}_S{S}"
            print("=====", key, flush=True)
            rows = bench(name, cfg, S, with_fold=not (name == "paper" and args.no_fold_paper))
            results[key] = rows
            pts = [f"b{b}/t{t}" for b, t in cfg["points"]]
            lines += [f"## {key} (P = {cfg['period']}, W = {cfg['cycle']})", "",
                      "| Mô hình | Tham số | MACs/mẫu | Buffer (MB) | " + " | ".join(f"{p} (ms)" for p in pts) + " |",
                      "| --- | ---: | ---: | ---: |" + " ---: |" * len(pts)]
            for lab, r in rows.items():
                par = f"{r['params']:,}" if r["params"] is not None else "(như trên)"
                lines.append(f"| {lab} | {par} | {fmt_macs(r['MACs/mẫu'])} | {r['buffer_MB']:.1f} | "
                             + " | ".join(f"{r[p]:.3f}" for p in pts) + " |")
            lines.append("")
            Path("results/bench_vs_timebase.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n",
                                                              encoding="utf-8")
    Path("results/bench_vs_timebase.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
