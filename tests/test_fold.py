"""Exact folding must reproduce DyTimeNet outputs; linear baselines must run in the same pipeline."""

import pytest
import torch

from dytimenet.config import load_config
from dytimenet.models import DyTimeNet, build_model
from dytimenet.models.folded import fold
from dytimenet.utils.seed import set_seed

STAGES = [dict(fold_trend=False, fold_seasonal=False), dict(fold_seasonal=False), dict(concat=False), {}]


@pytest.mark.parametrize("variant", ["double", "single", "norm"])
@pytest.mark.parametrize("stages", STAGES)
def test_fold_matches_original(variant, stages):
    set_seed(0)
    model = DyTimeNet(7, 96, 192, d_model=32, cross_residual=variant).eval()
    with torch.no_grad():
        model.revin.affine_weight.uniform_(0.5, 1.5)
        model.revin.affine_bias.normal_()
        x = torch.randn(8, 7, 96) * 3 + 5
        ref = model(x)
        out = fold(model, **stages)(x)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=5e-5)


def test_fold_requires_eval_and_handles_plain_revin():
    model = DyTimeNet(7, 96, 96, d_model=32, affine_revin=False)
    with pytest.raises(ValueError, match="eval"):
        fold(model)
    model.eval()
    x = torch.randn(4, 7, 96)
    with torch.no_grad():
        torch.testing.assert_close(fold(model)(x), model(x), rtol=1e-5, atol=5e-5)


@pytest.mark.parametrize("name", ["rlinear", "dlinear"])
def test_linear_baselines_build_from_config(name):
    config = load_config("configs/ETTh1.yaml", {"pred_len": 96, "model": name})
    model = build_model(config.model, **config.model_kwargs(7))
    out = model(torch.randn(2, 7, 96))
    assert out.shape == (2, 7, 96)
    out.square().mean().backward()
