"""CLI: evaluate a saved model using its exact architecture and training scaler."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a DyTimeNet best.pt checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None, help="Relocated original CSV, with identical contents")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default=None, help="Optional JSON metrics path")
    args = vars(parser.parse_args(argv))
    args["checkpoint_path"] = args.pop("checkpoint")
    try:
        from dytimenet.experiment import evaluate_checkpoint
        evaluate_checkpoint(**args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
