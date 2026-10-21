"""Bảng hiệu năng kiểu Table 3 của TimeBase, đo trên máy này (CPU): Params, MACs, Max Mem, Epoch Time, Infer Time.

Cấu hình như TimeBase: Electricity (C = 321), L = 720, S = 720, bộ nhớ đo khi train với batch 12. Dữ liệu ngẫu nhiên
(tốc độ và bộ nhớ không phụ thuộc giá trị). Mỗi mô hình chạy trong một tiến trình riêng để đo bộ nhớ đỉnh sạch:
  Params     : số tham số (tham số phức của FITS tính là 1, như thop).
  MACs       : mỗi mẫu, torch.utils.flop_counter (FLOPs/2). FFT không được flop_counter đếm.
  Max Mem    : đỉnh bộ nhớ tiến trình (working set) khi train batch 12, trừ mức trước khi tạo mô hình.
  Epoch Time : thời gian một bước train (forward + backward + Adam, batch 12) × số bước một epoch Electricity
               (⌊16 973 / 12⌋ = 1 414 bước; 16 973 = số cửa sổ train với L = S = 720, split 7:1:2).
  Infer Time : suy luận 1 mẫu, 8 luồng (và 1 luồng).
Tiến trình nào vượt --mem_limit_gb bị dừng; khi đó bước train được đo lại với batch nhỏ hơn và ghi chú rõ.

Mô hình SOTA lấy từ Time-Series-Library (--tslib) với cấu hình script ECL chính thức; PatchTST theo repo PatchTST;
FITS từ repo chính thức (--fits, file models/FITS.py) với cut_freq = (L/24 + 1)·8 + 10 = 258 (133.6K tham số như
bảng TimeBase). Ví dụ: python scripts/bench_efficiency.py --tslib <TSLib> --fits <FITS.py>
"""

import argparse
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

C, L, S, MEM_BATCH = 321, 720, 720, 12
TRAIN_WINDOWS = int(26304 * 0.7) - L - S + 1          # Electricity: 26 304 dòng, train 70%
MODELS = ["Informer", "Autoformer", "FEDformer", "TimesNet", "PatchTST", "DLinear", "FITS", "iTransformer",
          "TimeMixer", "SparseTSF", "TimeBase", "DyTimeNet (cũ)", "DyTimeBase"]
TSLIB = {   # script ECL chính thức của TSLib (PatchTST theo repo PatchTST)
    "Informer": dict(label_len=48, e_layers=2, d_layers=1, factor=3),
    "Autoformer": dict(label_len=48, e_layers=2, d_layers=1, factor=3),
    "FEDformer": dict(label_len=48, e_layers=2, d_layers=1, factor=3),
    "TimesNet": dict(label_len=48, e_layers=2, d_layers=1, factor=3, d_model=256, d_ff=512, top_k=5),
    "PatchTST": dict(label_len=48, e_layers=3, n_heads=16, d_model=128, d_ff=256),
    "iTransformer": dict(label_len=48, e_layers=3, d_model=512, d_ff=512),
    "TimeMixer": dict(label_len=0, e_layers=3, d_model=16, d_ff=32, down_sampling_layers=3, down_sampling_window=2,
                      down_sampling_method="avg"),
}


def build(name, tslib, fits):
    import torch
    from torch import nn
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dytimenet.models import build_model
    base = dict(num_vars=C, seq_len=L, pred_len=S)
    seg = dict(period=24, basis_num=6, period_norm=True)
    if name == "DyTimeBase":
        return build_model("dytimebase", **base, **seg, d_model=64)
    if name == "TimeBase":
        return build_model("timebase", **base, **seg)
    if name == "SparseTSF":
        return build_model("sparsetsf", **base, period=24)
    if name == "DLinear":
        return build_model("dlinear", **base)
    if name == "DyTimeNet (cũ)":
        return build_model("dytimenet", **base, d_model=64, cross_residual="norm")
    if name == "FITS":
        sys.modules.setdefault("models", types.ModuleType("models"))
        sys.modules["models.NLinear"] = types.ModuleType("models.NLinear")
        spec = importlib.util.spec_from_file_location("fits_official", fits)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        cut = int(L // 24 + 1) * 8 + 10
        model = mod.Model(types.SimpleNamespace(seq_len=L, pred_len=S, individual=False, enc_in=C, cut_freq=cut))

        class FITSWrap(nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m

            def forward(self, x):                                   # [B, C, L] → [B, C, S]
                return self.m(x.transpose(1, 2))[0][:, -S:].transpose(1, 2)
        return FITSWrap(model)
    from bench_sota import TSLibWrap, tslib_config
    sys.path.append(tslib)
    over = dict(TSLIB[name]); label_len = over.pop("label_len")
    mod = importlib.import_module(f"models.{name}")
    return TSLibWrap(mod.Model(tslib_config(C, L, S, label_len=label_len, **over)), label_len, S)


def worker(args):
    """Đo một mô hình; in một dòng JSON. Gọi bởi tiến trình cha."""
    import psutil
    import torch
    from torch import nn
    from torch.utils.flop_counter import FlopCounterMode
    sys.stdout.reconfigure(encoding="utf-8")
    proc = psutil.Process()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    batch = args.batch
    x = torch.randn(batch, C, L); y = torch.randn(batch, C, S)
    before = proc.memory_info().rss
    model = build(args.worker, args.tslib, args.fits)
    out = {"params": sum(p.numel() for p in model.parameters()), "batch": batch}
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    step()                                                       # khởi động + cấp phát trạng thái Adam
    times = []
    deadline = time.perf_counter() + args.train_budget
    while len(times) < 3 or (time.perf_counter() < deadline and len(times) < 30):
        s = time.perf_counter(); step(); times.append(time.perf_counter() - s)
    out["step_s"] = statistics.median(times)
    out["epoch_s"] = out["step_s"] * (TRAIN_WINDOWS // batch)
    out["max_mem_mb"] = (proc.memory_info().peak_wset - before) / 2**20 if hasattr(proc.memory_info(), "peak_wset") \
        else None
    model.eval()
    x1 = x[:1].contiguous()
    with torch.inference_mode():
        with FlopCounterMode(display=False) as fc:
            model(x1)
        out["macs"] = fc.get_total_flops() / 2
        for t in (8, 1):
            torch.set_num_threads(t)
            for _ in range(3):
                model(x1)
            s = time.perf_counter(); model(x1); est = time.perf_counter() - s
            reps = max(5, min(200, int(args.infer_budget / max(est, 1e-6))))
            ts = []
            for _ in range(reps):
                s = time.perf_counter(); model(x1); ts.append(time.perf_counter() - s)
            out[f"infer_ms_t{t}"] = statistics.median(ts) * 1e3
    print("RESULT " + json.dumps(out), flush=True)


def run_child(name, batch, args):
    import psutil
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", name, "--batch", str(batch),
           "--tslib", args.tslib, "--fits", args.fits, "--train_budget", str(args.train_budget),
           "--infer_budget", str(args.infer_budget)]
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    ps, peak, killed = psutil.Process(child.pid), 0, False
    while child.poll() is None:
        try:
            peak = max(peak, ps.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        if peak > args.mem_limit_gb * 2**30:
            child.kill(); killed = True
            break
        time.sleep(0.2)
    stdout, stderr = child.communicate()
    if killed:
        return {"killed": True, "peak_gb": peak / 2**30}
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": (stderr or stdout).strip().splitlines()[-1] if (stderr or stdout).strip() else "không rõ"}


def fmt(v, unit=""):
    if v is None:
        return "—"
    if unit == "count":
        return f"{v / 1e9:.2f}G" if v >= 1e9 else f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.2f}K"
    return f"{v:.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", default=None)
    parser.add_argument("--batch", type=int, default=MEM_BATCH)
    parser.add_argument("--tslib", required=True)
    parser.add_argument("--fits", required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--mem_limit_gb", type=float, default=11.0)
    parser.add_argument("--train_budget", type=float, default=20.0, help="giây tối đa để đo bước train")
    parser.add_argument("--infer_budget", type=float, default=3.0)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)
    sys.stdout.reconfigure(encoding="utf-8")
    path = Path("results/bench_efficiency.json")
    results = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for name in args.models:
        if name in results and "error" not in results[name]:
            continue
        print(f"===== {name}", flush=True)
        r = run_child(name, MEM_BATCH, args)
        if r.get("killed"):                     # vượt RAM: đo lại bước train với batch nhỏ hơn
            note = {"mem_over_gb": args.mem_limit_gb, "peak_gb_seen": r["peak_gb"]}
            for b in (4, 1):
                r = run_child(name, b, args)
                if not r.get("killed"):
                    break
            r.update(note)
        results[name] = r
        print("  ", r, flush=True)
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    import torch
    lines = ["# Hiệu năng trên máy này (kiểu Table 3 của TimeBase): Electricity, L = S = 720", "",
             f"CPU {platform.processor()}, PyTorch {torch.__version__}, eager. Max Mem: đỉnh bộ nhớ tiến trình khi train "
             f"batch {MEM_BATCH} (CPU RAM, không phải GPU). Epoch Time: bước train batch {MEM_BATCH} đo thật × "
             f"{TRAIN_WINDOWS // MEM_BATCH} bước. Infer Time: 1 mẫu. MACs mỗi mẫu (FFT không được đếm).", "",
             "| Mô hình | Params | MACs | Max Mem (MB) | Epoch Time (s) | Infer t8 (ms) | Infer t1 (ms) | Ghi chú |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    order = [n for n in MODELS if n in results]
    for n in order:
        r = results[n]
        if "error" in r:
            lines.append(f"| {n} | lỗi: {r['error']} |  |  |  |  |  |  |"); continue
        note = ""
        if "mem_over_gb" in r:
            note = (f"train batch {MEM_BATCH} vượt {r['mem_over_gb']:.0f} GB RAM; Epoch Time đo ở batch {r['batch']} "
                    f"× {TRAIN_WINDOWS // r['batch']} bước")
        mem = f"> {r['mem_over_gb'] * 1024:.0f}" if "mem_over_gb" in r else fmt(r["max_mem_mb"])
        lines.append(f"| {n} | {fmt(r['params'], 'count')} | {fmt(r['macs'], 'count')} | {mem} | {fmt(r['epoch_s'])} | "
                     f"{fmt(r['infer_ms_t8'])} | {fmt(r['infer_ms_t1'])} | {note} |")
    Path("results/bench_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
