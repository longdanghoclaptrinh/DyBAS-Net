"""CycleNet/TQNet: shape, gradient, chu kỳ khởi tạo 0 và cấu hình bắt buộc có cycle_len."""

import pytest
import torch

from dytimenet.config import load_config
from dytimenet.models import build_model


@pytest.mark.parametrize("name", ["cyclenet", "cyclenet_mlp", "tqnet"])
def test_shapes_and_gradient(name):
    model = build_model(name, num_vars=5, seq_len=48, pred_len=24, cycle_len=24, d_model=32)
    out = model(torch.randn(3, 5, 48), torch.tensor([0, 7, 23]))
    assert out.shape == (3, 5, 24)
    out.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_cyclenet_zero_cycle_is_normalized_linear():
    model = build_model("cyclenet", num_vars=4, seq_len=48, pred_len=12, cycle_len=24).eval()
    x = torch.randn(2, 4, 48)
    mean, std = x.mean(-1, keepdim=True), (x.var(-1, keepdim=True) + 1e-5).sqrt()
    expected = model.model((x - mean) / std) * std + mean
    torch.testing.assert_close(model(x, torch.tensor([3, 11])), expected)


@pytest.mark.parametrize("name", ["cyclenet", "tqnet"])
def test_config_requires_cycle_len(name):
    with pytest.raises(ValueError, match="cycle_len"):
        load_config("configs/ETTh1.yaml", {"model": name, "seq_len": 720, "pred_len": 96})
    config = load_config("configs/ETTh1.yaml", {"model": name, "seq_len": 720, "pred_len": 96, "cycle_len": 24})
    assert build_model(config.model, **config.model_kwargs(7))(torch.randn(2, 7, 720),
                                                               torch.tensor([0, 1])).shape == (2, 7, 96)
