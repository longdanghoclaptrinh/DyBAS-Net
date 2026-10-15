"""Test metrics and readable summaries on the dataset's standardized scale."""

import math

import pandas as pd
from torch import nn

from .trainer import run_epoch


def evaluate(model, loader, device="cpu"):
    """Aggregate all test windows, including the final partial batch."""
    metrics = run_epoch(model, loader, nn.MSELoss(), device=device)
    return {"mse": metrics["mse"], "mae": metrics["mae"], "rmse": math.sqrt(metrics["mse"])}


def print_summary(config, metrics, train_seconds=None):
    """Print one experiment and return its machine-readable result row."""
    names = ("dataset", "seq_len", "pred_len", "d_model", "decomp_kernel", "batch_size",
             "lr", "p1", "p2", "seed")
    row = {name: getattr(config, name) for name in names}
    row["metric_scale"] = "standardized" if config.scale else "original"
    if train_seconds is not None:
        row["train_seconds"] = round(train_seconds, 3)
    row.update({name.upper(): value for name, value in metrics.items()})
    print("\nTest results:")
    print(pd.DataFrame([row]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return row
