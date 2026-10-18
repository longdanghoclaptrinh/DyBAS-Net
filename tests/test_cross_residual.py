"""Check the three residual layouts around CrossVBlock and that "double" stays the default."""

import pytest
import torch

from dytimenet.config import load_config, training_parser
from dytimenet.models import DyTimeNet
from dytimenet.utils.seed import set_seed


def built(cross_residual):
    set_seed(0)
    config = load_config("configs/ETTh1.yaml", {"pred_len": 96, "cross_residual": cross_residual})
    return DyTimeNet(**config.model_kwargs(7)).eval()


def cross_input_output(model, x):
    """Input of CrossVBlock and the tensor that reaches IntraVBlock."""
    seen = {}
    model.cross_v_blocks[0].register_forward_pre_hook(lambda m, a: seen.__setitem__("s", a[0]))
    model.intra_v_blocks[0].register_forward_pre_hook(lambda m, a: seen.__setitem__("intra", a[0]))
    with torch.no_grad():
        model(x)
    return seen["s"], seen["intra"]


@pytest.mark.parametrize("variant", ["double", "single", "norm"])
def test_residual_formulas(variant):
    model = built(variant)
    block = model.cross_v_blocks[0]
    s, intra_in = cross_input_output(model, torch.randn(4, 7, 96))
    with torch.no_grad():
        normed = block.norm(s)
        mixed = block.df_linear2(block.activation(block.df_linear1(normed.transpose(1, 2)))).transpose(1, 2)
    expected = {"double": 2 * s + mixed, "single": s + mixed, "norm": normed + mixed}[variant]
    torch.testing.assert_close(intra_in, expected, rtol=1e-5, atol=1e-5)


def test_default_is_double_and_variants_share_parameters():
    assert load_config("configs/ETTh1.yaml").cross_residual == "double"
    keys = {v: list(built(v).state_dict()) for v in ("double", "single", "norm")}
    assert keys["double"] == keys["single"] == keys["norm"]
    args = vars(training_parser().parse_args(["--config", "configs/ETTh1.yaml", "--cross_residual", "norm"]))
    assert load_config(args.pop("config"), args).cross_residual == "norm"
    with pytest.raises(ValueError, match="cross_residual"):
        load_config("configs/ETTh1.yaml", {"cross_residual": "triple"})
