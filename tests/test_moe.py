"""Check IntraVMoE: dense/sparse equivalence, expert sizes, aux loss, and baseline isolation."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dytimenet.config import load_config, training_parser
from dytimenet.engine.trainer import run_epoch
from dytimenet.layers.intra_v_moe import IntraVMoE
from dytimenet.layers.revin import RevIN
from dytimenet.models import DyTimeNet
from dytimenet.utils.seed import set_seed


def moe_model(top_k, dispatch, seed=0):
    set_seed(seed)
    config = load_config("configs/ETTh1.yaml", {"pred_len": 96, "use_moe": True,
                                                 "top_k": top_k, "moe_dispatch": dispatch})
    return DyTimeNet(**config.model_kwargs(7))


@pytest.mark.parametrize("top_k", [1, 2])
def test_dense_and_sparse_dispatch_match_in_eval(top_k):
    sparse = moe_model(top_k, "sparse").eval()
    dense = moe_model(top_k, "dense", seed=1).eval()
    dense.load_state_dict(sparse.state_dict())
    x = torch.randn(64, 7, 96) * 2 + 1
    with torch.no_grad():
        out_sparse, out_dense = sparse(x), dense(x)
    assert (out_sparse - out_dense).abs().max().item() < 1e-5
    counts = sparse.moe_expert_counts()
    assert counts.sum().item() == 64 * 7 * top_k
    torch.testing.assert_close(counts, dense.moe_expert_counts(), rtol=0, atol=0)


@pytest.mark.parametrize("top_k, expert_ff", [(1, 64), (2, 32)])
def test_expert_sizes_and_training_gradients(top_k, expert_ff):
    model = moe_model(top_k, "sparse").train()
    moe = model.intra_v_blocks[0].moe
    assert moe.shared[0].out_features == 64 and moe.router.in_features == 64 + 2
    assert [e[0].out_features for e in moe.experts] == [expert_ff] * 4
    output = model(torch.randn(8, 7, 96))
    aux = model.moe_aux_loss()
    assert output.shape == (8, 7, 96) and aux.ndim == 0 and aux.item() >= 1 - 1e-6
    (output.square().mean() + 0.01 * aux).backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    model.eval()
    model(torch.randn(2, 7, 96))
    assert model.moe_aux_loss() is None


def test_sparse_gates_sum_shared_plus_selected_expert():
    set_seed(0)
    moe = IntraVMoE(8, 16, num_experts=3, top_k=1, dropout=0.0).eval()
    x, mu, log_sigma = torch.randn(2, 5, 8), torch.randn(2, 5, 1), torch.randn(2, 5, 1)
    with torch.no_grad():
        out = moe(x, mu, log_sigma)
        choice = moe.router(torch.cat([x, mu, log_sigma], -1)).argmax(-1)
        expected = moe.shared(x) + torch.stack(
            [moe.experts[int(c)](t) for c, t in zip(choice.reshape(-1), x.reshape(-1, 8))]).reshape(x.shape)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_revin_returns_detached_input_stats():
    x = torch.randn(3, 7, 96) * 4 + 2
    revin = RevIN(7)
    normed, mu, sigma = revin(x, "norm", return_stats=True)
    torch.testing.assert_close(normed, RevIN(7)(x, "norm"))
    assert mu.shape == sigma.shape == (3, 7, 1) and not mu.requires_grad
    torch.testing.assert_close(mu, x.mean(-1, keepdim=True))


def test_baseline_unchanged_when_moe_disabled():
    config = load_config("configs/ETTh1.yaml", {"pred_len": 96})
    assert not config.use_moe
    set_seed(42)
    model = DyTimeNet(**config.model_kwargs(7))
    assert not any("moe" in key for key in model.state_dict())
    assert model.moe_aux_loss() is None and model.moe_expert_counts() is None


def test_run_epoch_adds_aux_only_when_training():
    model = moe_model(1, "sparse")
    loader = DataLoader(TensorDataset(torch.randn(16, 7, 96), torch.randn(16, 7, 96)), batch_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train = run_epoch(model, loader, nn.MSELoss(), optimizer, aux_weight=0.5)
    assert train["loss"] > train["mse"] and train["aux"] > 0
    assert sum(train[f"expert_{i}"] for i in range(4)) == pytest.approx(1.0)
    val = run_epoch(model, loader, nn.MSELoss())
    assert val["loss"] == pytest.approx(val["mse"]) and "aux" not in val


def test_moe_cli_flags_and_validation():
    args = vars(training_parser().parse_args(["--config", "configs/ETTh1.yaml", "--use_moe",
                                              "--num_experts", "4", "--top_k", "2",
                                              "--moe_aux_weight", "0.01", "--moe_dispatch", "dense"]))
    config = load_config(args.pop("config"), args)
    assert (config.use_moe, config.num_experts, config.top_k, config.moe_dispatch) == (True, 4, 2, "dense")
    with pytest.raises(ValueError, match="top_k"):
        load_config("configs/ETTh1.yaml", {"top_k": 5})
    with pytest.raises(SystemExit):
        training_parser().parse_args(["--moe_dispatch", "gather"])
