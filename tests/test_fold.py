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


@pytest.mark.parametrize("affine", [True, False])
def test_fold_matches_original_with_rcf(affine):
    set_seed(0)
    model = DyTimeNet(7, 96, 192, d_model=32, cross_residual="norm", cycle_len=24, affine_revin=affine).eval()
    with torch.no_grad():
        model.cycle.data.normal_()
        if affine:
            model.revin.affine_weight.uniform_(0.5, 1.5)
            model.revin.affine_bias.normal_()
        x = torch.randn(8, 7, 96) * 3 + 5
        idx = torch.randint(0, 24, (8,))
        torch.testing.assert_close(fold(model)(x, idx), model(x, idx), rtol=1e-5, atol=5e-5)


def test_rcf_dataset_phase_and_cyclelinear():
    config = load_config("configs/ETTh1.yaml", {"pred_len": 96, "model": "rlinear", "cycle_len": 24,
                                                 "affine_revin": False})
    model = build_model(config.model, **config.model_kwargs(7))
    out = model(torch.randn(3, 7, 96), torch.tensor([0, 5, 23]))
    assert out.shape == (3, 7, 96)


@pytest.mark.parametrize("cycle_len", [0, 24])
def test_fold_stable_on_constant_windows(cycle_len):
    """Cửa sổ hằng số (σ = 0, như lượng mưa = 0 trên Weather) không được khuếch đại sai số float32."""
    set_seed(0)
    model = DyTimeNet(7, 96, 96, d_model=32, cross_residual="norm", cycle_len=cycle_len).eval()
    x = torch.randn(8, 7, 96)
    x[:, 2] = 37.5                                    # kênh hằng số với mức lớn
    x[:4, 5] = -12.25
    idx = torch.randint(0, 24, (8,)) if cycle_len else None
    args = (idx,) if cycle_len else ()
    with torch.no_grad():
        torch.testing.assert_close(fold(model)(x, *args), model(x, *args), rtol=1e-5, atol=5e-5)


@pytest.mark.parametrize("name, params", [("timebase", 396), ("sparsetsf", 925)])
def test_segment_baselines_match_paper_size(name, params):
    """Cấu hình Table 3 của TimeBase (Electricity, L = S = 720, P = 24): TimeBase 0.39K tham số."""
    model = build_model(name, num_vars=3, seq_len=720, pred_len=720, period=24)
    assert model.count_parameters() == params
    assert model(torch.randn(2, 3, 720)).shape == (2, 3, 720)
    assert build_model(name, num_vars=3, seq_len=100, pred_len=50, period=24)(torch.randn(2, 3, 100)).shape == (2, 3, 50)


@pytest.mark.parametrize("variant", ["double", "norm"])
@pytest.mark.parametrize("cycle_len, affine", [(0, True), (0, False), (24, True), (24, False)])
def test_shared_fold_matches_original(variant, cycle_len, affine):
    """Gộp dùng chung (bộ nhớ O(L·S)) phải cho cùng kết quả với mô hình gốc, kể cả cửa sổ hằng số."""
    set_seed(0)
    model = DyTimeNet(7, 96, 192, d_model=32, cross_residual=variant, cycle_len=cycle_len,
                      affine_revin=affine).eval()
    with torch.no_grad():
        if cycle_len:
            model.cycle.data.normal_()
        if affine:
            model.revin.affine_weight.uniform_(0.5, 1.5)
            model.revin.affine_bias.normal_()
        x = torch.randn(8, 7, 96) * 3 + 5
        x[:, 2] = 37.5
        args = (torch.randint(0, 24, (8,)),) if cycle_len else ()
        folded = fold(model, mode="shared")
        assert folded.mode == "shared" and not hasattr(folded, "MK")
        torch.testing.assert_close(folded(x, *args), model(x, *args), rtol=1e-5, atol=5e-5)


def test_auto_mode_picks_shared_for_large_inputs():
    small = DyTimeNet(7, 96, 96, d_model=32, cross_residual="norm").eval()
    large = DyTimeNet(64, 336, 336, d_model=32, cross_residual="norm").eval()
    assert fold(small).mode == "channel" and fold(large).mode == "shared"
