"""Chạy cấu hình moe_lite (shared d_ff=32, 4 expert × d_ff=32, top-1, sparse) với cùng quy trình đo
của scripts/bench_moe.py. moe_lite = MoE với ffn_ratio=1.0 (d_ff=64 → shared d_ff/2=32, expert d_ff/2k=32).

Ví dụ: python scripts/run_moe_lite.py
Kết quả ghi vào results/moe_lite/ (không đụng tới results/moe_etth1_runs.json).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_moe  # noqa: E402

bench_moe.CONFIGS["moe_lite"] = {"use_moe": True, "num_experts": 4, "top_k": 1,
                                 "moe_dispatch": "sparse", "ffn_ratio": 1.0}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save_dir", default="checkpoints/moe_bench")
    parser.add_argument("--results", default="results/moe_lite")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)
    args.configs = ["moe_lite"]
    bench_moe.torch.set_num_threads(args.threads)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    bench_moe.run_all(args)


if __name__ == "__main__":
    main()
