"""Verify metric weighting, best-checkpoint selection, and scheduler endpoints."""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dytimenet.engine.evaluate import evaluate
from dytimenet.engine.trainer import EarlyStopping, cosine_lr, run_epoch


def test_metrics_include_partial_batch():
    x = torch.zeros(5, 1, 1)
    y = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    metrics = evaluate(nn.Identity(), loader)
    assert metrics == {"mse": 6.0, "mae": 2.0, "rmse": math.sqrt(6)}
    with pytest.raises(ValueError, match="no batches"):
        run_epoch(nn.Identity(), [], nn.MSELoss())


def test_early_stopping_reloads_minimum(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    early = EarlyStopping(patience=2, save_path=tmp_path / "best.pt")
    assert not early(0.5, model)
    model.weight.data.fill_(2)
    assert not early(0.4, model)
    model.weight.data.fill_(3)
    assert not early(0.45, model)
    assert early(0.44, model)
    early.load_best(model)
    assert model.weight.item() == 2


def test_original_cosine_schedule():
    optimizer = torch.optim.Adam(nn.Linear(1, 1).parameters(), lr=0.0005)
    assert cosine_lr(optimizer, 0, 0.0005, 100, 2) == 0.00025
    assert cosine_lr(optimizer, 1, 0.0005, 100, 2) == 0.0005
    assert cosine_lr(optimizer, 2, 0.0005, 100, 2) == 0.0005
    assert 0 < cosine_lr(optimizer, 99, 0.0005, 100, 2) < 0.000001
