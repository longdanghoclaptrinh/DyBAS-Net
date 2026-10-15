"""Exercise real train/evaluate entrypoints on a tiny generated CSV, without downloads."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from dytimenet.config import load_config, training_parser


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, check=True):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                            text=True, encoding="utf-8", capture_output=True, timeout=120)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_yaml_cli_override_and_invalid_config():
    args = vars(training_parser().parse_args(["--pred_len", "336", "--epochs", "3", "--seed", "7"]))
    config = load_config(args.pop("config"), args)
    assert (config.pred_len, config.epochs, config.seed, config.p1) == (336, 3, 7, 0.1)
    with pytest.raises(ValueError, match="odd"):
        load_config("configs/ETTm2.yaml", {"decomp_kernel": 24})
    with pytest.raises(ValueError, match="Unknown"):
        load_config("configs/ETTm2.yaml", {"learning_rate": 0.1})


def test_train_save_evaluate_round_trip(tmp_path):
    path = tmp_path / "synthetic.csv"
    steps = np.arange(320, dtype=np.float32)
    pd.DataFrame({"date": ["2020-01-01"] * 320,
                  "x": np.sin(steps / 10), "y": np.cos(steps / 7)}).to_csv(path, index=False)
    save_dir = tmp_path / "runs"
    run_cli("scripts/train.py", "--config", "configs/Weather.yaml", "--data_path", path,
            "--save_dir", save_dir, "--seq_len", "16", "--pred_len", "8", "--d_model", "4",
            "--decomp_kernel", "5", "--batch_size", "32", "--epochs", "1", "--seed", "7",
            "--device", "cpu")
    checkpoints = list(save_dir.glob("*/best.pt"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    run_dir = checkpoint.parent
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["pred_len"] == 8 and config["seed"] == 7 and config["epochs"] == 1
    assert (run_dir / "history.csv").is_file() and (run_dir / "environment.json").is_file()
    original = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    output = tmp_path / "evaluated.json"
    run_cli("scripts/evaluate.py", "--checkpoint", checkpoint, "--data_path", path,
            "--device", "cpu", "--output", output)
    restored = json.loads(output.read_text(encoding="utf-8"))
    for metric in ("MSE", "MAE", "RMSE"):
        assert restored[metric] == original[metric]
    assert restored["best_epoch"] == 1
    with path.open("a") as stream:
        stream.write("2020-01-01,0,0\n")
    failure = run_cli("scripts/evaluate.py", "--checkpoint", checkpoint, check=False)
    assert failure.returncode == 2 and "SHA-256 mismatch" in failure.stderr


def test_help_and_missing_data(tmp_path):
    assert "--pred_len" in run_cli("scripts/train.py", "--help").stdout
    assert "--checkpoint" in run_cli("scripts/evaluate.py", "--help").stdout
    result = run_cli("scripts/train.py", "--data_path", tmp_path / "missing.csv", check=False)
    assert result.returncode == 2 and "data/README.md" in result.stderr
