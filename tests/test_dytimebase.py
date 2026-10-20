"""DyTimeBase khởi đầu đúng bằng TimeBase; nhánh trộn kênh có lr riêng và giữ tỉ lệ qua cosine."""

import torch

from dytimenet.config import load_config
from dytimenet.engine.trainer import cosine_lr
from dytimenet.models import DyTimeBase, TimeBase, build_model


def test_init_equals_timebase_and_rcf_shapes():
    model = DyTimeBase(7, 100, 50, period=24, basis_num=6, d_model=16).eval()
    base = TimeBase(7, 100, 50, period=24, basis_num=6)
    base.load_state_dict({k: v for k, v in model.state_dict().items() if k.split(".")[0] in ("basis", "forecast")})
    x = torch.randn(3, 7, 100)
    torch.testing.assert_close(model(x), base(x))
    rcf = DyTimeBase(7, 96, 192, period=24, d_model=16, cycle_len=24)
    out = rcf(x[..., :96], torch.tensor([0, 5, 23]))
    assert out.shape == (3, 7, 192)
    out.square().mean().backward()


def test_mix_lr_param_groups_follow_cosine():
    config = load_config("configs/ETTh1.yaml", {"model": "dytimebase", "seq_len": 720, "pred_len": 96,
                                                 "lr": 0.02, "mix_lr": 5e-4})
    model = build_model(config.model, **config.model_kwargs(7))
    mix = model.mix_parameters()
    assert mix and len(mix) < len(list(model.parameters()))
    ids = {id(p) for p in mix}
    opt = torch.optim.Adam([{"params": [p for p in model.parameters() if id(p) not in ids], "base_lr": 0.02},
                            {"params": mix, "base_lr": 5e-4}], lr=0.02)
    cosine_lr(opt, 10, 0.02, 100)
    ratio = opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"]
    assert abs(ratio - 5e-4 / 0.02) < 1e-12


def test_type3_schedule_matches_timebase():
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    lrs = [cosine_lr(opt, e, 0.1, 30, kind="type3") for e in range(7)]
    assert lrs[:4] == [0.1] * 4
    assert abs(lrs[4] - 0.08) < 1e-12 and abs(lrs[6] - 0.1 * 0.8 ** 3) < 1e-12


def test_official_options_and_aux_loss():
    config = load_config("configs/Weather.yaml", {"model": "dytimebase", "seq_len": 720, "pred_len": 96, "period": 4,
                                                   "period_norm": True, "individual": True, "orth_weight": 0.04,
                                                   "lr_schedule": "type3", "grad_clip": 0.0, "mix_lr": 5e-4})
    model = build_model(config.model, **config.model_kwargs(21)).train()
    out = model(torch.randn(2, 21, 720))
    assert out.shape == (2, 21, 96) and model.aux_loss is not None and model.aux_loss.ndim == 0
    model.eval(); model(torch.randn(2, 21, 720))
    assert model.aux_loss is None
