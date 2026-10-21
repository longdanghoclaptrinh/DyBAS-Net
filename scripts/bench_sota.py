"""Đo tốc độ DyTimeBase so với TimeBase, baseline tuyến tính và các mô hình SOTA trên cùng một máy CPU.

PatchTST, TimeMixer, iTransformer, TimeXer lấy từ Time-Series-Library (github.com/thuml/Time-Series-Library), không
chép vào repo: truyền đường dẫn bản clone qua --tslib. Kiến trúc của chúng theo script chính thức của TSLib
(scripts/long_term_forecast/{Weather,ECL}_script, cấu hình S = 96), chỉ đổi seq_len/pred_len cho giống cấu hình đo.
Latency không phụ thuộc giá trị trọng số nên mọi model khởi tạo ngẫu nhiên.

Hai cấu hình:
  weather     : C = 21,  L = 720, S ∈ {96, 720} — cấu hình train thật của TimeBase/DyTimeBase (P = 4, individual).
  electricity : C = 321, L = S = 720 — như Table 3 của TimeBase (P = 24, dùng chung trọng số), chỉ đo batch 1.
Đo: latency suy luận (median, các model xen kẽ qua nhiều vòng), một bước train (forward + backward + Adam, chỉ ở
weather), số tham số và MACs mỗi mẫu (torch.utils.flop_counter, FLOPs/2).
Ghi results/bench_sota.json và .md. Ví dụ: python scripts/bench_sota.py --tslib <đường dẫn TSLib>
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from dytimenet.models import build_model
from dytimenet.models.folded import fold

SETTINGS = {
    "weather": dict(C=21, L=720, S=(96, 720), period=4, individual=True, cycle=144, data="Weather",
                    points=[(1, 8), (1, 1), (32, 8)], train_batch=32),
    "electricity": dict(C=321, L=720, S=(720,), period=24, individual=False, cycle=24, data="ECL",
                        points=[(1, 8), (1, 1)], train_batch=None),
}

# Kiến trúc theo script chính thức của TSLib (cấu hình pred_len = 96 của từng dataset).
TSLIB = {
    # PatchTST: script ECL của TSLib để d_model mặc định 512 (39.5M tham số); dùng cấu hình của repo PatchTST chính
    # thức (yuqinie98/PatchTST: e_layers 3, n_heads 16, d_model 128, d_ff 256) — nhỏ hơn, khớp 8.69M ở Table 3 TimeBase.
    "PatchTST": {d: dict(e_layers=3, n_heads=16, d_model=128, d_ff=256, label_len=48) for d in ("Weather", "ECL")},
    "TimeMixer": {d: dict(e_layers=3, d_model=16, d_ff=32, down_sampling_layers=3, down_sampling_window=2,
                          down_sampling_method="avg", label_len=0) for d in ("Weather", "ECL")},
    "iTransformer": {d: dict(e_layers=3, d_model=512, d_ff=512, label_len=48) for d in ("Weather", "ECL")},
    "TimeXer": {"Weather": dict(e_layers=1, d_model=256, d_ff=512, label_len=48),
                "ECL": dict(e_layers=4, d_model=512, d_ff=512, label_len=48)},
}


def tslib_config(C, L, S, **over):
    cfg = dict(task_name="long_term_forecast", seq_len=L, label_len=48, pred_len=S, enc_in=C, dec_in=C, c_out=C,
               d_model=512, d_ff=2048, n_heads=8, e_layers=2, d_layers=1, factor=3, dropout=0.1, embed="timeF",
               freq="h", activation="gelu", use_norm=1, features="M", moving_avg=25, channel_independence=1,
               decomp_method="moving_avg", top_k=5, num_class=1, patch_len=16, down_sampling_layers=0,
               down_sampling_window=1, down_sampling_method=None, output_attention=False)
    cfg.update(over)
    return types.SimpleNamespace(**cfg)


class TSLibWrap(nn.Module):
    """Gọi model TSLib như vòng lặp chuẩn của họ: (x_enc, x_mark_enc, x_dec, x_mark_dec), x_dec = [label | 0]."""

    def __init__(self, model, label_len, pred_len, n_marks=4):
        super().__init__()
        self.model, self.label_len, self.pred_len, self.n_marks = model, label_len, pred_len, n_marks

    def forward(self, x):                                       # x: [B, C, L] → [B, C, S]
        xe = x.transpose(1, 2)
        B, L, C = xe.shape
        mark_enc = torch.zeros(B, L, self.n_marks)
        dec = torch.cat([xe[:, L - self.label_len:], xe.new_zeros(B, self.pred_len, C)], 1)
        mark_dec = torch.zeros(B, self.label_len + self.pred_len, self.n_marks)
        return self.model(xe, mark_enc, dec, mark_dec)[:, -self.pred_len:].transpose(1, 2)


def candidates(name, cfg, S, tslib):
    C, L, P = cfg["C"], cfg["L"], cfg["period"]
    base = dict(num_vars=C, seq_len=L, pred_len=S)
    seg = dict(period=P, basis_num=6, period_norm=True, individual=cfg["individual"])
    models = {
        "DyTimeBase": build_model("dytimebase", **base, **seg, d_model=64),
        "TimeBase": build_model("timebase", **base, **seg),
        "SparseTSF": build_model("sparsetsf", **base, period=P),
        "DLinear": build_model("dlinear", **base),
        "RLinear": build_model("rlinear", **base),
        "CycleNet/Linear": build_model("rlinear", **base, cycle_len=cfg["cycle"], affine_revin=False),
        "DyTimeNet (cũ)": build_model("dytimenet", **base, d_model=64, cross_residual="norm"),
    }
    models["DyTimeNet (cũ) — đã gộp"] = fold(models["DyTimeNet (cũ)"].eval())
    if tslib:
        sys.path.append(tslib)
        import importlib
        for m, per_ds in TSLIB.items():
            over = dict(per_ds[cfg["data"]])
            label_len = over.pop("label_len")
            mod = importlib.import_module(f"models.{m}")
            models[m] = TSLibWrap(mod.Model(tslib_config(C, L, S, label_len=label_len, **over)), label_len, S)
    for m in models.values():
        m.eval()
    return models


def call(model, x, idx):
    return model(x, idx) if getattr(model, "cycle_len", 0) else model(x)


def timed(fn, budget=0.6, min_reps=8, max_reps=300):
    for _ in range(3):
        fn()
    s = time.perf_counter(); fn(); est = time.perf_counter() - s
    for _ in range(max(0, min(20, int(0.3 / max(est, 1e-6))) - 3)):
        fn()
    reps = max(min_reps, min(max_reps, int(budget / max(est, 1e-6))))
    t = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
    return statistics.median(t) * 1e3


def train_step_ms(model, x, y, idx, rounds=3):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        nn.functional.mse_loss(call(model, x, idx), y).backward()
        opt.step()
    vals = [timed(step, budget=1.0, min_reps=5, max_reps=50) for _ in range(rounds)]
    model.eval()
    return statistics.median(vals)


def bench(name, cfg, S, tslib, rounds):
    torch.manual_seed(0)
    models = candidates(name, cfg, S, tslib)
    bmax = max([b for b, _ in cfg["points"]] + [cfg["train_batch"] or 1])
    x_all = torch.randn(bmax, cfg["C"], cfg["L"])
    y_all = torch.randn(bmax, cfg["C"], S)
    idx_all = torch.randint(0, cfg["cycle"], (bmax,))
    rows = {}
    with torch.inference_mode():
        for label, m in models.items():
            with FlopCounterMode(display=False) as fc:
                call(m, x_all[:1], idx_all[:1])
            folded = "gộp" in label
            rows[label] = {"params": None if folded else sum(p.numel() for p in m.parameters()),
                           "macs": fc.get_total_flops() / 2}
        for b, t in cfg["points"]:
            torch.set_num_threads(t)
            x, idx = x_all[:b].contiguous(), idx_all[:b]
            vals = {label: [] for label in models}
            for _ in range(rounds):
                for label, m in models.items():
                    vals[label].append(timed(lambda m=m: call(m, x, idx)))
            for label, v in vals.items():
                rows[label][f"b{b}/t{t}"] = statistics.median(v)
            print(f"  {name} S={S} b{b}/t{t} xong", flush=True)
    torch.set_num_threads(8)
    if cfg["train_batch"]:
        b = cfg["train_batch"]
        for label, m in models.items():
            if "gộp" not in label:
                rows[label][f"train_b{b}"] = train_step_ms(m, x_all[:b], y_all[:b], idx_all[:b])
        print(f"  {name} S={S} train xong", flush=True)
    del models; gc.collect()
    return rows


def fmt_count(v):
    if v is None:
        return "(như trên)"
    return f"{v / 1e9:.2f}G" if v >= 1e9 else f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.1f}K"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tslib", default=None, help="đường dẫn bản clone Time-Series-Library (bỏ trống: không đo SOTA)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")      # chạy được cả với python -I (bỏ qua PYTHONIOENCODING)
    torch.set_num_threads(8)
    results = {"cpu": platform.processor(), "torch": torch.__version__, "runs": {}}
    lines = ["# Tốc độ trên cùng một máy: DyTimeBase so với TimeBase, baseline tuyến tính và SOTA", "",
             f"CPU: {platform.processor()}; PyTorch {torch.__version__}, eager, `inference_mode`. Latency = median của "
             f"{args.rounds} vòng đo xen kẽ (ms/lần gọi). bB/tT: batch B, T luồng. Train = một bước forward + backward + "
             "Adam (ms). Kiến trúc PatchTST/TimeMixer/iTransformer/TimeXer theo script chính thức của "
             "Time-Series-Library (cấu hình S = 96; PatchTST theo repo chính thức của PatchTST), seq_len/pred_len đổi theo cấu hình đo. MACs mỗi mẫu.", ""]
    for name in args.settings:
        cfg = SETTINGS[name]
        for S in cfg["S"]:
            key = f"{name}_C{cfg['C']}_L{cfg['L']}_S{S}"
            print("=====", key, flush=True)
            rows = bench(name, cfg, S, args.tslib, args.rounds)
            results["runs"][key] = rows
            cols = [f"b{b}/t{t}" for b, t in cfg["points"]] + ([f"train_b{cfg['train_batch']}"] if cfg["train_batch"] else [])
            ref = rows["DyTimeBase"]["b1/t8"]
            lines += [f"## {key} (P = {cfg['period']}, individual = {cfg['individual']})", "",
                      "| Mô hình | Tham số | MACs/mẫu | " + " | ".join(f"{c} (ms)" for c in cols) + " | b1/t8 so với DyTimeBase |",
                      "| --- | ---: | ---: |" + " ---: |" * (len(cols) + 1)]
            for label, r in sorted(rows.items(), key=lambda kv: kv[1]["b1/t8"]):
                cells = [f"{r[c]:.3f}" if c in r else "—" for c in cols]
                lines.append(f"| {label} | {fmt_count(r['params'])} | {fmt_count(r['macs'])} | " + " | ".join(cells)
                             + f" | {r['b1/t8'] / ref:.2f}× |")
            lines.append("")
            Path("results/bench_sota.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
    Path("results/bench_sota.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
