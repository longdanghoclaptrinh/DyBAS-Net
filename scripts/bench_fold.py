"""Đo latency suy luận: DyTimeNet gốc so với các mức gộp (ablation), eager và TorchScript.

Ví dụ: python scripts/bench_fold.py --checkpoint checkpoints/residual_ablation/<run>/best.pt
Ghi JSON vào results/fold_latency_<tên>.json. Chạy tuần tự, không chạy gì khác cùng lúc.
"""

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dytimenet.config import config_from_dict
from dytimenet.data.dataset import TimeSeriesDataset
from dytimenet.models import DyTimeNet
from dytimenet.models.folded import fold

VARIANTS = {
    "(a) gốc": None,
    "(a') viết lại dạng hàm, chưa gộp": dict(fold_trend=False, fold_seasonal=False),
    "(b) + gộp trend & inverse RevIN (K)": dict(fold_seasonal=False),
    "(c) + gộp RevIN-norm, MovingAvg, Conv (M)": dict(concat=False),
    "(d) + ghép [M; K] thành 1 einsum": {},
}


def timed(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    t.sort()
    return {"median_ms": statistics.median(t) * 1e3, "p90_ms": t[int(0.9 * len(t))] * 1e3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[8, 1])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DyTimeNet(**ck["model_kwargs"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    cfg = config_from_dict(dict(ck["config"]))
    ds = TimeSeriesDataset(cfg.data_path, cfg.seq_len, cfg.pred_len, "test", cfg.target_cols, cfg.scale,
                           dataset_name=cfg.dataset, scaler_state=ck["scaler"])
    x_all = torch.stack([ds[i][0] for i in range(max(args.batches))])

    models = {name: (model if st is None else fold(model, **st)) for name, st in VARIANTS.items()}
    results = {"checkpoint": args.checkpoint, "model_kwargs": ck["model_kwargs"],
               "cpu": platform.processor(), "torch": torch.__version__, "warmup": args.warmup,
               "reps": args.reps, "rounds": args.rounds, "timings": {}}
    with torch.inference_mode():
        runs = {}
        for name, m in models.items():
            # Kiểm tra tương đương trước khi đo.
            diff = (m(x_all) - model(x_all)).abs().max().item()
            assert diff < 1e-4, (name, diff)
            runs[("eager", name)] = m
            runs[("torchscript", name)] = torch.jit.freeze(torch.jit.trace(m, x_all[:1]).eval())
        # Đo xen kẽ: mỗi vòng đo lần lượt mọi biến thể, để độ trôi của máy ảnh hưởng đều.
        for t in args.threads:
            torch.set_num_threads(t)
            for b in args.batches:
                x = x_all[:b].contiguous()
                rounds = {key: [] for key in runs}
                for _ in range(args.rounds):
                    for key, run in runs.items():
                        rounds[key].append(timed(lambda: run(x), args.warmup, args.reps))
                for (mode, name), vals in rounds.items():
                    meds = [v["median_ms"] for v in vals]
                    key = f"{mode} | b{b}/t{t} | {name}"
                    results["timings"][key] = {"median_ms": statistics.median(meds), "min_ms": min(meds),
                                               "max_ms": max(meds), "p90_ms": statistics.median(v["p90_ms"] for v in vals)}
                    print(f"{key}: {statistics.median(meds):.3f} ms (vòng: {min(meds):.3f}–{max(meds):.3f})", flush=True)
    out = Path(args.out or f"results/fold_latency_{Path(args.checkpoint).parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
