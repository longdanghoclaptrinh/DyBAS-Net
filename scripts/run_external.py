"""Train các mô hình ngoài repo (Time-Series-Library và FITS) trong cùng cách chia dữ liệu / chuẩn hóa / đánh giá.

Dữ liệu: TimeSeriesDataset của repo (cùng split, scaler fit trên train, test trên mọi cửa sổ). Mô hình và giao thức
train theo repo chính thức của từng mô hình, chỉ đổi seq_len và số biến:
  TSLib (DLinear, PatchTST, iTransformer, TimesNet, TimeMixer, FEDformer, Autoformer, Informer): kiến trúc đọc từ
      scripts/long_term_forecast/ETT_script/<Model>_<script>.sh (khối có cùng pred_len); Adam, MSE, lr/batch/epoch/
      patience theo script (mặc định run.py: lr 1e-4, batch 32, 10 epoch, patience 3, lradj type1). Dấu thời gian
      timeF (freq h: giờ, thứ, ngày trong tháng, ngày trong năm, mỗi đặc trưng trong [-0.5, 0.5]) như TSLib.
  FITS: models/FITS.py chính thức, scripts/FITS/etth1_best.sh (H_order 6, base_T 24, batch 64, lr 5e-4,
      patience 20, train_mode 1 = loss trên phần dự báo), run_longExp_F.py: 50 epoch, lradj type3.
  DUET (KDD 2025, decisionintelligence/DUET): ts_benchmark/baselines/duet/models/duet_model.py; siêu tham số theo
      scripts/multivariate_forecast/<bộ>_script/DUET.sh (khối cùng horizon, seq_len đổi thành L), loss của script (MAE)
      cộng loss cân bằng chuyên gia khi train (như _process của DUET), Adam, lịch type1 như TFB.
  MS-FLOW (ICML 2026, Cola-Fsm/MS-FLOW): models/MS-FLOW.py; repo chưa công bố script nên dùng mặc định của run.py
      (patch 16, stride 8, d_model 256, 3 lớp, top_k 10, dropout 0.1, batch 16, lr 1e-4, 100 epoch, patience 3, type1)
      và loss MAE nhân trọng số arctan theo bước dự báo cho cả train lẫn val như exp/exp_main.py.
  TimeEmb (NeurIPS 2025, showmeon/TimeEmb): models/TimeEmb.py; siêu tham số theo scripts/TimeEmb/MLP-Input-720/<bộ>.sh
      (khối cùng horizon), mặc định run.py cho phần còn lại (d_model 512, RevIN, lradj type3); train bằng
      rec_lambda · MSE + auxi_lambda · MAE trên FFT theo trục thời gian, val bằng MSE như exp/exp_main.py; chỉ số giờ
      lấy từ giờ của bước dự báo đầu tiên như data_loader.py.
  GTR (ICLR 2026, macovaseas/GTR): models/GTR.py (GTR + MLP); siêu tham số theo scripts/Input-720/<bộ>.sh, mặc định
      run.py cho phần còn lại (d_model 512, dropout 0, RevIN, MSE, lradj type3); chỉ số chu kỳ = (chỉ số dòng của bước
      dự báo đầu tiên) mod cycle như data_loader.py.
  LightGTS (ICML 2025, decisionintelligence/LightGTS): src/models/LightGTS_resample.py, nạp checkpoints/LightGTS_4M.pth
      rồi tinh chỉnh toàn bộ như script/Decoder_forcasting_finetune/exp_fullshot_4M.sh + finetune.py (3+3 lớp, d_model
      256, AdamW, one-cycle lr_max = lr/2 = 1e-4 trong 20 epoch, batch 64, loss L1, chọn epoch theo val MSE, RevIN).
      Patch theo chu kỳ: script dùng 48 cho dữ liệu theo giờ (giữ nguyên, 15 token với L = 720) và 192/288 cho 15/10
      phút với L dài hơn; với L = 720 dùng 96/144 (một ngày) để còn 7/5 token thay vì 3/2.
Ví dụ: python scripts/run_external.py --dataset Load --tslib D:/Time-Series-Library-main --fits <FITS.py>
           --models DLinear PatchTST --seeds 2021
Kết quả từng run ghi ngay vào results/external_<dataset>_L<L>.json (chạy lại để tiếp tục phần còn thiếu).
"""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bench_sota import tslib_config
from dytimenet.data.dataset import TimeSeriesDataset
from dytimenet.engine.evaluate import evaluate
from dytimenet.engine.trainer import EarlyStopping, cosine_lr, run_epoch
from dytimenet.utils.seed import set_seed

SEQ_LEN = 720
DATA = {"Load": ("load_new.csv", "timestamp"), "Exchange": ("data/exchange_rate.csv", "date"),
        "Tetouan": ("data/tetouan.csv", "date"), "Panama": ("data/panama.csv", "date"), "PJM": ("data/pjm.csv", "date"),
        **{ds: (f"data/{ds}.csv", "date") for ds in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")},
        "Weather": ("data/weather.csv", "date")}
# base_T của FITS = chu kỳ chính của dữ liệu (như scripts chính thức của FITS); Tetouan 10 phút: một ngày = 144 bước.
FITS_BASE_T = {"Tetouan": 144}
# Script ETT chính thức của TSLib dùng cho từng mô hình (iTransformer không có script ETTh1, dùng ETTh2).
TSLIB_SCRIPT = {"DLinear": "ETTh1", "PatchTST": "ETTh1", "iTransformer": "ETTh2", "TimesNet": "ETTh1",
                "TimeMixer": "ETTh1", "FEDformer": "ETTh1", "Autoformer": "ETTh1", "Informer": "ETTh1"}
TSLIB_RUN = {"learning_rate": 1e-4, "batch_size": 32, "train_epochs": 10, "patience": 3, "lradj": "type1"}
ARCH_KEYS = {"label_len", "e_layers", "d_layers", "factor", "d_model", "d_ff", "n_heads", "top_k", "dropout",
             "down_sampling_layers", "down_sampling_window", "down_sampling_method", "channel_independence",
             "moving_avg", "num_kernels", "use_norm"}
FITS_RUN = {"learning_rate": 5e-4, "batch_size": 64, "train_epochs": 50, "patience": 20, "lradj": "type3",
            "H_order": 6, "base_T": 24}
# Script DUET theo bộ dữ liệu; bộ không có script dùng bộ cùng tần số (như TimeBase*): theo giờ → ETTh1, 10 phút →
# Weather.
DUET_SCRIPT = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2", "Weather": "Weather",
               "Tetouan": "Weather"}
DUET_DEFAULTS = {"activation": "gelu", "output_attention": 0, "moving_avg": 25, "hidden_size": 256,
                 "noisy_gating": True, "factor": 1, "n_heads": 8, "dropout": 0.2, "fc_dropout": 0.2}
DUET_RUN_KEYS = {"lr": "learning_rate", "batch_size": "batch_size", "num_epochs": "train_epochs",
                 "patience": "patience", "lradj": "lradj", "loss": "loss"}
MSFLOW_ARCH = {"patch_len": 16, "stride": 8, "d_model": 256, "e_layers": 3, "top_k": 10, "dropout": 0.1}
MSFLOW_RUN = {"learning_rate": 1e-4, "batch_size": 16, "train_epochs": 100, "patience": 3, "lradj": "type1",
              "loss": "arctan_mae"}
# Script TimeEmb theo bộ dữ liệu; bộ không có script dùng bộ cùng tần số như DUET: theo giờ → etth1, 10 phút → weather.
TIMEEMB_SCRIPT = {"ETTh1": "etth1", "ETTh2": "etth2", "ETTm1": "ettm1", "ETTm2": "ettm2", "Weather": "weather",
                  "Tetouan": "weather"}
TIMEEMB_DEFAULTS = {"d_model": 512, "use_revin": 1, "use_hour_index": 1, "use_day_index": 0, "hour_length": 24,
                    "day_length": 7, "rec_lambda": 0.0, "auxi_lambda": 1.0, "learning_rate": 1e-4, "batch_size": 128,
                    "train_epochs": 30, "patience": 5, "lradj": "type3"}
# Script GTR theo bộ dữ liệu, cùng cách chọn như TimeEmb (cycle: Tetouan 144 như Weather, Panama 24 như ETTh1).
GTR_DEFAULTS = {"d_model": 512, "dropout": 0.0, "use_revin": 1, "individual": 0, "cycle": 24, "learning_rate": 1e-4,
                "batch_size": 128, "train_epochs": 30, "patience": 5, "lradj": "type3"}
LIGHTGTS_PATCH = {"ETTm1": 96, "ETTm2": 96, "Weather": 144, "Tetouan": 144}   # còn lại (theo giờ): 48
LIGHTGTS_ARCH = {"e_layers": 3, "d_layers": 3, "n_heads": 8, "d_model": 256, "d_ff": 512, "dropout": 0.2,
                 "attn_dropout": 0.2, "head_dropout": 0.2}
LIGHTGTS_RUN = {"learning_rate": 2e-4, "batch_size": 64, "train_epochs": 20, "patience": 50, "lradj": "onecycle",
                "pct_start": 0.3, "loss": "MAE", "checkpoint": "checkpoints/LightGTS_4M.pth"}
MODELS = tuple(TSLIB_SCRIPT) + ("FITS", "DUET", "MSFLOW", "TimeEmb", "GTR", "LightGTS")


def script_args(tslib, model, pred_len):
    """Tham số của khối python trong script TSLib có --pred_len bằng pred_len (thay biến shell name=value)."""
    path = Path(tslib) / "scripts/long_term_forecast/ETT_script" / f"{model}_{TSLIB_SCRIPT[model]}.sh"
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    env = dict(re.findall(r"^\s*(\w+)=(\S+)\s*$", text, flags=re.M))
    for line in text.splitlines():
        if "run.py" not in line:
            continue
        tokens = shlex.split(re.sub(r"\$(\w+)", lambda m: env.get(m.group(1), m.group(0)), line.split("|")[0]))
        args = {tokens[i][2:]: tokens[i + 1] for i in range(len(tokens) - 1)
                if tokens[i].startswith("--") and not tokens[i + 1].startswith("--")}
        if int(args.get("pred_len", -1)) == pred_len:
            return args
    raise KeyError(f"{path.name}: no block with --pred_len {pred_len}")


def time_features(stamps):
    """timeF của TSLib với freq h: HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, mỗi cột trong [-0.5, 0.5]."""
    t = pd.DatetimeIndex(stamps)
    return np.stack([t.hour / 23.0 - 0.5, t.dayofweek / 6.0 - 0.5, (t.day - 1) / 30.0 - 0.5,
                     (t.dayofyear - 1) / 365.0 - 0.5], 1).astype(np.float32)


class Marked(Dataset):
    """Cửa sổ của TimeSeriesDataset kèm dấu thời gian [L + S, 4] (TSLibWrap tách phần encoder/decoder).

    with_row: thêm chỉ số dòng (trong cả file) của bước dự báo đầu tiên, cho chỉ số chu kỳ của GTR.
    """

    def __init__(self, base, marks, with_row=False):
        self.base, self.marks, self.with_row = base, torch.tensor(marks[base.start:base.end]), with_row

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        out = (x, y, self.marks[idx:idx + self.base.seq_len + self.base.pred_len])
        return out + (self.base.start + idx + self.base.seq_len,) if self.with_row else out


class TSLibWrap(nn.Module):
    """Vòng lặp chuẩn của TSLib: (x_enc, x_mark_enc, x_dec, x_mark_dec), x_dec = [label_len cuối | 0]."""

    def __init__(self, model, seq_len, label_len, pred_len):
        super().__init__()
        self.model, self.seq_len, self.label_len, self.pred_len = model, seq_len, label_len, pred_len

    def forward(self, x, marks):                                 # x: [B, C, L], marks: [B, L + S, 4] → [B, C, S]
        xe = x.transpose(1, 2)
        dec = torch.cat([xe[:, self.seq_len - self.label_len:], xe.new_zeros(xe.shape[0], self.pred_len, xe.shape[2])], 1)
        out = self.model(xe, marks[:, :self.seq_len], dec, marks[:, self.seq_len - self.label_len:])
        return out[:, -self.pred_len:].transpose(1, 2)


class FITSWrap(nn.Module):
    def __init__(self, model, pred_len):
        super().__init__()
        self.model, self.pred_len = model, pred_len

    def forward(self, x, marks=None):                            # FITS trả [B, L + S, C]; train_mode 1: lấy S cuối
        return self.model(x.transpose(1, 2))[0][:, -self.pred_len:].transpose(1, 2)


class DUETWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.aux_loss = model, None

    def forward(self, x, marks=None):                            # DUET: [B, L, C] → ([B, S, C], loss cân bằng)
        out, self.aux_loss = self.model(x.transpose(1, 2))
        return out.transpose(1, 2)


class MSFLOWWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, marks=None):
        return self.model(x.transpose(1, 2)).transpose(1, 2)


class TimeEmbWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, marks):                                 # chỉ số giờ của bước dự báo đầu tiên (cột giờ timeF)
        hour = ((marks[:, SEQ_LEN, 0] + 0.5) * 23).round().long()
        return self.model(x.transpose(1, 2), hour).transpose(1, 2)


class GTRWrap(nn.Module):
    def __init__(self, model, cycle):
        super().__init__()
        self.model, self.cycle = model, cycle

    def forward(self, x, marks, row):
        return self.model(x.transpose(1, 2), row % self.cycle).transpose(1, 2)


class LightGTSWrap(nn.Module):
    """RevIN (không affine) + chia patch lấy phần cuối cửa sổ như RevInCB/PatchCB của LightGTS."""

    def __init__(self, model, patch_len):
        super().__init__()
        self.model, self.patch_len = model, patch_len

    def forward(self, x, marks=None):                            # x: [B, C, L] → [B, C, S]
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)
        z = ((x - mean) / std).transpose(1, 2)                   # [B, L, C]
        n = z.shape[1] // self.patch_len
        z = z[:, z.shape[1] - n * self.patch_len:].unfold(1, self.patch_len, self.patch_len)   # [B, n, C, patch]
        return self.model(z).transpose(1, 2) * std + mean


class TrainEvalLoss(nn.Module):
    """Loss train khác loss val: train_loss khi tính gradient, MSE khi không (val, chọn epoch theo val MSE)."""

    def __init__(self, train_loss):
        super().__init__()
        self.train_loss = train_loss

    def forward(self, out, y):
        return self.train_loss(out, y) if torch.is_grad_enabled() else ((out - y) ** 2).mean()


class OneCycleOptimizer:
    """AdamW + OneCycleLR bước theo từng lô (như callback OneCycleLR của LightGTS); run_epoch gọi zero_grad/step."""

    def __init__(self, optimizer, scheduler):
        self.optimizer, self.scheduler = optimizer, scheduler
        self.param_groups = optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.scheduler.step()


class TimeEmbLoss(nn.Module):
    """Loss train của TimeEmb: rec · MSE + auxi · MAE của hiệu FFT theo trục thời gian; khi không tính gradient (val) là MSE."""

    def __init__(self, rec, auxi):
        super().__init__()
        self.rec, self.auxi = rec, auxi

    def forward(self, out, y):
        mse = ((out - y) ** 2).mean()
        if not torch.is_grad_enabled():
            return mse
        return self.rec * mse + self.auxi * (torch.fft.fft(out, dim=-1) - torch.fft.fft(y, dim=-1)).abs().mean()


def loop_script_args(path, pred_len):
    """Tham số trong script shell của vòng `for pred_len in ...` chứa pred_len (thay biến shell name=value)."""
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    env = dict(re.findall(r"^\s*(\w+)=(\S+)\s*$", text, flags=re.M))
    for block in text.split("for pred_len in")[1:]:
        horizons, body = block.split("do", 1)
        if pred_len in map(int, horizons.split()):
            tokens = shlex.split(re.sub(r"\$(\w+)", lambda m: env.get(m.group(1), m.group(0)), body))
            return {tokens[i][2:]: tokens[i + 1] for i in range(len(tokens) - 1)
                    if tokens[i].startswith("--") and not tokens[i + 1].startswith("--")}
    raise KeyError(f"{path}: no block with pred_len {pred_len}")


def timeemb_args(timeemb, dataset, pred_len):
    return loop_script_args(Path(timeemb) / "scripts/TimeEmb/MLP-Input-720" /
                            f"{TIMEEMB_SCRIPT.get(dataset, 'etth1')}.sh", pred_len)


class ArctanMAE(nn.Module):
    """Loss của MS-FLOW: MAE với trọng số 1 + π/4 − arctan(s + 1) ở bước dự báo s (out, y: [B, C, S])."""

    def forward(self, out, y):
        s = torch.arange(1, out.shape[-1] + 1, dtype=out.dtype, device=out.device)
        w = 1 + torch.pi / 4 - torch.atan(s)
        return (w * (out - y)).abs().mean()


def duet_args(duet, dataset, pred_len):
    """Siêu tham số trong scripts/multivariate_forecast/<bộ>_script/DUET.sh của khối có horizon bằng pred_len."""
    path = Path(duet) / "scripts/multivariate_forecast" / f"{DUET_SCRIPT.get(dataset, 'ETTh1')}_script/DUET.sh"
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"--model-hyper-params '(\{.*?\})'", line)
        if m and json.loads(m.group(1))["horizon"] == pred_len:
            return json.loads(m.group(1))
    raise KeyError(f"{path}: no block with horizon {pred_len}")


def build(name, num_vars, pred_len, tslib, fits, base_t=FITS_RUN["base_T"], duet=None, msflow=None, dataset=None,
          timeemb=None, gtr=None, lightgts=None):
    if name == "LightGTS":
        if str(Path(lightgts)) not in sys.path:
            sys.path.append(str(Path(lightgts)))
        mod = importlib.import_module("src.models.LightGTS_resample")
        patch = LIGHTGTS_PATCH.get(dataset, 48)
        model = mod.LightGTS(c_in=num_vars, target_dim=pred_len, patch_len=patch, stride=patch,
                             num_patch=SEQ_LEN // patch, shared_embedding=True, act="relu", head_type="prediction",
                             res_attention=False, learn_pe=False, **LIGHTGTS_ARCH)
        # Như transfer_weights(exclude_head=False): chép mọi tham số trùng tên và cùng kích thước (weights_only).
        state = torch.load(Path(lightgts) / LIGHTGTS_RUN["checkpoint"], map_location="cpu", weights_only=True)["model"]
        own = model.state_dict()
        loaded = [k for k in own if k in state and state[k].shape == own[k].shape]
        model.load_state_dict({**own, **{k: state[k] for k in loaded}})
        print(f"LightGTS: nạp {len(loaded)}/{len(own)} tham số từ checkpoint; khác: "
              f"{[k for k in own if k not in loaded]}", flush=True)
        return LightGTSWrap(model, patch), {**LIGHTGTS_RUN, **LIGHTGTS_ARCH, "patch_len": patch}
    if name == "GTR":
        spec = importlib.util.spec_from_file_location("gtr_official", Path(gtr) / "models/GTR.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args = loop_script_args(Path(gtr) / "scripts/Input-720" / f"{TIMEEMB_SCRIPT.get(dataset, 'etth1')}.sh", pred_len)
        cfg = {k: type(v)(args[k]) if k in args else v for k, v in GTR_DEFAULTS.items()}
        model = mod.Model(types.SimpleNamespace(seq_len=SEQ_LEN, pred_len=pred_len, enc_in=num_vars, **cfg))
        return GTRWrap(model, cfg["cycle"]), cfg
    if name == "TimeEmb":
        spec = importlib.util.spec_from_file_location("timeemb_official", Path(timeemb) / "models/TimeEmb.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args = timeemb_args(timeemb, dataset, pred_len)
        cfg = {k: type(v)(args[k]) if k in args else v for k, v in TIMEEMB_DEFAULTS.items()}
        model = mod.Model(types.SimpleNamespace(seq_len=SEQ_LEN, pred_len=pred_len, enc_in=num_vars, **cfg))
        return TimeEmbWrap(model), cfg
    if name == "DUET":
        root = Path(duet) / "ts_benchmark/baselines/duet"
        if str(Path(duet)) not in sys.path:
            sys.path.append(str(Path(duet)))
        # Bỏ qua duet/__init__.py (adapter TFB kéo theo cả framework), chỉ nạp mô hình.
        pkg = types.ModuleType("ts_benchmark.baselines.duet")
        pkg.__path__ = [str(root)]
        sys.modules.setdefault("ts_benchmark.baselines.duet", pkg)
        mod = importlib.import_module("ts_benchmark.baselines.duet.models.duet_model")
        hp = duet_args(duet, dataset, pred_len)
        cfg = {**DUET_DEFAULTS, **hp, "seq_len": SEQ_LEN, "pred_len": pred_len, "enc_in": num_vars,
               "dec_in": num_vars, "c_out": num_vars}
        run = {DUET_RUN_KEYS[k]: cfg[k] for k in DUET_RUN_KEYS}
        arch = {k: v for k, v in cfg.items() if k not in DUET_RUN_KEYS and k not in ("horizon", "norm")}
        return DUETWrap(mod.DUETModel(types.SimpleNamespace(**cfg))), {**run, **arch}
    if name == "MSFLOW":
        spec = importlib.util.spec_from_file_location("msflow_official", msflow)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.Model(types.SimpleNamespace(enc_in=num_vars, seq_len=SEQ_LEN, pred_len=pred_len, **MSFLOW_ARCH))
        return MSFLOWWrap(model), {**MSFLOW_RUN, **MSFLOW_ARCH}
    if name == "FITS":
        sys.modules.setdefault("models", types.ModuleType("models"))
        sys.modules.setdefault("models.NLinear", types.ModuleType("models.NLinear"))
        spec = importlib.util.spec_from_file_location("fits_official", fits)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cut = int(SEQ_LEN // base_t + 1) * FITS_RUN["H_order"] + 10
        model = mod.Model(types.SimpleNamespace(seq_len=SEQ_LEN, pred_len=pred_len, individual=False,
                                                enc_in=num_vars, cut_freq=cut))
        return FITSWrap(model, pred_len), {**FITS_RUN, "base_T": base_t}
    args = script_args(tslib, name, pred_len)
    arch = {k: type_of(k, v) for k, v in args.items() if k in ARCH_KEYS}
    run = {**TSLIB_RUN, **{k: type_of(k, args[k]) for k in TSLIB_RUN if k in args}}
    if tslib not in sys.path:
        sys.path.append(tslib)
    mod = importlib.import_module(f"models.{name}")
    label_len = arch.pop("label_len", 48)
    cfg = tslib_config(num_vars, SEQ_LEN, pred_len, label_len=label_len, **arch)
    return TSLibWrap(mod.Model(cfg), SEQ_LEN, label_len, pred_len), {**run, **arch, "label_len": label_len}


# Lệch khỏi script chính thức (ghi trong slide): TimeMixer trên Panama với lr 0.01 của script ETTh1 (viết cho cửa sổ 96)
# phân kỳ ngay epoch 1 ở L=720 (train loss tới ~2e5); các lượt đó lưu ở results/external_Panama_L720_TimeMixer_lr1e-2.json.
RUN_OVERRIDE = {("TimeMixer", "Panama"): {"learning_rate": 1e-3}}


def type_of(key, value):
    if key in ("learning_rate", "dropout"):
        return float(value)
    if key in ("lradj", "down_sampling_method"):
        return value
    return int(value)


def set_lr(optimizer, epoch, run):
    if run["lradj"] == "type3":                                  # FITS/SparseTSF: như lịch type3 của repo
        return cosine_lr(optimizer, epoch, run["learning_rate"], run["train_epochs"], kind="type3")
    lr = run["learning_rate"] * 0.5 ** max(0, epoch - 1)         # TSLib type1: chỉnh sau mỗi epoch, bắt đầu từ epoch 2
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def train_one(name, dataset, horizon, seed, tslib, fits, save_dir, threads, duet=None, msflow=None, device="cpu",
              timeemb=None, gtr=None, lightgts=None):
    torch.set_num_threads(threads)
    set_seed(seed)
    path, stamp_col = DATA[dataset]
    kwargs = dict(data_path=path, seq_len=SEQ_LEN, pred_len=horizon, dataset_name=dataset)
    train_ds = TimeSeriesDataset(**kwargs, split="train")
    sets = [train_ds] + [TimeSeriesDataset(**kwargs, split=s, scaler_state=train_ds.scaler_state())
                         for s in ("val", "test")]
    marks = time_features(pd.read_csv(path, usecols=[stamp_col])[stamp_col])
    model, run = build(name, train_ds.num_vars, horizon, tslib, fits, FITS_BASE_T.get(dataset, FITS_RUN["base_T"]),
                       duet, msflow, dataset, timeemb, gtr, lightgts)
    run = {**run, **RUN_OVERRIDE.get((name, dataset), {})}
    model = model.to(device)
    loaders = [DataLoader(Marked(ds, marks, with_row=(name == "GTR")), batch_size=run["batch_size"], shuffle=(i == 0), drop_last=False)
               for i, ds in enumerate(sets)]
    run_dir = Path(save_dir) / f"{name}_S{horizon}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    early = EarlyStopping(run["patience"], 0.0, run_dir / "best.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=run["learning_rate"])
    if name == "LightGTS":                                       # fine_tune: fit_one_cycle(lr_max = lr / 2)
        adamw = torch.optim.AdamW(model.parameters(), lr=run["learning_rate"])
        optimizer = OneCycleOptimizer(adamw, torch.optim.lr_scheduler.OneCycleLR(
            adamw, max_lr=run["learning_rate"] / 2, epochs=run["train_epochs"], steps_per_epoch=len(loaders[0]),
            pct_start=run["pct_start"]))
    criterion = {"MAE": nn.L1Loss(), "arctan_mae": ArctanMAE()}.get(run.get("loss"), nn.MSELoss())
    if name == "TimeEmb":
        criterion = TimeEmbLoss(run["rec_lambda"], run["auxi_lambda"])
    if name == "LightGTS":
        criterion = TrainEvalLoss(nn.L1Loss())
    aux = 1.0 if name == "DUET" else 0.0                         # DUET: loss + loss cân bằng chuyên gia
    print(f"{name} | {dataset} | L={SEQ_LEN} S={horizon} seed={seed} | params="
          f"{sum(p.numel() for p in model.parameters()):,} | {run}", flush=True)
    start, history = time.perf_counter(), []
    for epoch in range(run["train_epochs"]):
        lr = optimizer.param_groups[0]["lr"] if run["lradj"] == "onecycle" else set_lr(optimizer, epoch, run)
        t0 = time.perf_counter()
        tr = run_epoch(model, loaders[0], criterion, optimizer, device, aux_weight=aux, grad_clip=0.0)
        va = run_epoch(model, loaders[1], criterion, device=device)
        history.append({"epoch": epoch + 1, "lr": lr, "train_loss": tr["loss"], "val_loss": va["loss"],
                        "epoch_seconds": time.perf_counter() - t0})
        print(f"Epoch {epoch + 1:3}/{run['train_epochs']} | lr={lr:.2e} | train_loss={tr['loss']:.6f} | "
              f"val_loss={va['loss']:.6f} | {history[-1]['epoch_seconds']:.0f}s", flush=True)
        if early(va["loss"], model, epoch + 1):
            break
    seconds = time.perf_counter() - start
    early.load_best(model)
    metrics = evaluate(model, loaders[2], device)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print(f"Test MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f}", flush=True)
    return {"variant": name, "run": run, "device": device, "seq_len": SEQ_LEN, "pred_len": horizon, "seed": seed,
            "best_val": early.best_loss, "mse": metrics["mse"], "mae": metrics["mae"], "best_epoch": early.best_epoch,
            "epochs_run": len(history), "train_seconds": seconds,
            "mean_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])),
            "checkpoint": str(run_dir / "best.pt")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATA))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--tslib", default="D:/Time-Series-Library-main")
    parser.add_argument("--fits", default="D:/FITS-official/models/FITS.py")
    parser.add_argument("--duet", default="D:/DUET-official")
    parser.add_argument("--msflow", default="D:/MSFLOW-official/models/MS-FLOW.py")
    parser.add_argument("--timeemb", default="D:/TimeEmb-official/TimeEmb-main")
    parser.add_argument("--gtr", default="D:/GTR-official")
    parser.add_argument("--lightgts", default="D:/LightGTS-official")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    results = Path(f"results/external_{args.dataset}_L{SEQ_LEN}.json")
    runs = json.loads(results.read_text(encoding="utf-8")) if results.is_file() else {}
    for horizon in args.horizons:
        for seed in args.seeds:
            for name in args.models:
                key = f"{name}_S{horizon}_seed{seed}"
                if key in runs:
                    continue
                print(f"\n===== {args.dataset} {key} =====", flush=True)
                runs[key] = train_one(name, args.dataset, horizon, seed, args.tslib, args.fits,
                                      f"checkpoints/external_{args.dataset}_L{SEQ_LEN}", args.threads,
                                      args.duet, args.msflow, args.device, args.timeemb, args.gtr, args.lightgts)
                runs = {**(json.loads(results.read_text(encoding="utf-8")) if results.is_file() else {}), **runs}
                results.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
