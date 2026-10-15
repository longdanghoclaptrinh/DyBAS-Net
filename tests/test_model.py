"""Check model shapes, gradients, reversible normalization, and DropFusion modes."""

import pytest
import torch
from torch.nn import functional as F

from dytimenet.config import load_config
from dytimenet.layers.cross_v_block import DropFusionLinear
from dytimenet.layers.decomposition import MovingAvg
from dytimenet.layers.revin import RevIN
from dytimenet.models import DyTimeNet
from dytimenet.utils.seed import set_seed


@pytest.mark.parametrize("dataset", ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"])
@pytest.mark.parametrize("horizon", [96, 192, 336, 720])
def test_benchmark_forward_and_backward(dataset, horizon):
    config = load_config(f"configs/{dataset}.yaml", {"pred_len": horizon})
    num_vars = 21 if dataset == "Weather" else 7
    model = DyTimeNet(**config.model_kwargs(num_vars))
    inputs = torch.randn(2, num_vars, 96)
    output = model(inputs)
    assert output.shape == (2, num_vars, horizon)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_revin_and_decomposition():
    x = torch.randn(2, 7, 96) * 3 + 10
    revin = RevIN(7)
    with pytest.raises(RuntimeError):
        revin(x, "denorm")
    torch.testing.assert_close(revin(revin(x, "norm"), "denorm"), x, rtol=1e-4, atol=1e-4)
    seasonal, trend = MovingAvg(25)(x)
    assert seasonal.shape == trend.shape == x.shape
    torch.testing.assert_close(seasonal + trend, x)


def test_dropfusion_stochastic_training_and_dense_eval():
    set_seed(42)
    layer = DropFusionLinear(12, 8, p1=0.6, p2=0.6)
    x = torch.randn(4, 5, 12)
    first = layer(x)
    assert not torch.equal(first, layer(x))
    layer.eval()
    torch.testing.assert_close(layer(x), F.linear(x, layer.weight, layer.bias), rtol=0, atol=0)
    torch.testing.assert_close(layer(x), layer(x), rtol=0, atol=0)


def test_seed_repeats_model_initialization():
    set_seed(42)
    first = DyTimeNet(7, 96, 192, d_model=32)
    set_seed(42)
    second = DyTimeNet(7, 96, 192, d_model=32)
    assert first.count_parameters() == 32501
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
