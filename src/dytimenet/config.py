"""Validated experiment configuration; CLI values override YAML/JSON values."""

import argparse
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    """Notebook defaults; dataset YAML files supply the tuned hyperparameters."""

    dataset: str = "ETTm2"
    data_path: str = "data/ETTm2.csv"
    save_dir: str = "checkpoints"
    seq_len: int = 96
    pred_len: int = 192
    d_model: int = 32
    num_layers: int = 1
    batch_size: int = 64
    lr: float = 0.0005
    weight_decay: float = 0.0001
    epochs: int = 100
    warmup_epochs: int = 2
    patience: int = 10
    min_delta: float = 0.00001
    p1: float = 0.1
    p2: float = 0.1
    decomp_kernel: int = 25
    seed: int = 42
    ffn_ratio: float = 2.0
    hidden_ratio: float = 0.75
    dropout: float = 0.1
    affine_revin: bool = True
    scale: bool = True
    target_cols: list[str] | None = None
    num_workers: int = 0
    device: str = "auto"
    cross_residual: str = "double"
    model: str = "dytimenet"
    cycle_len: int = 0   # > 0: thay MovingAvg bằng RCF (CycleNet) với chu kỳ dài cycle_len bước
    period: int = 24     # độ dài đoạn P của TimeBase/SparseTSF
    basis_num: int = 6   # số basis R của TimeBase
    mix_lr: float = 0.0  # > 0: lr riêng cho nhánh trộn kênh của DyTimeBase (lr dùng cho backbone)
    period_norm: bool = False  # TimeBase/DyTimeBase: trừ mean theo pha (use_period_norm của code chính thức)
    individual: bool = False   # TimeBase/DyTimeBase: Linear basis/forecast riêng từng kênh
    orth_weight: float = 0.0   # trọng số loss trực giao trên basis (TimeBase)
    lr_schedule: str = "cosine"   # cosine (mặc định) hoặc type3 (lịch của TimeBase/SparseTSF)
    grad_clip: float = 1.0     # chuẩn tối đa của gradient; 0 = không clip

    def validate(self):
        """Reject invalid settings before reading data or creating a checkpoint."""
        if self.dataset not in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"):
            raise ValueError("dataset must be ETTh1, ETTh2, ETTm1, ETTm2, or Weather")
        for name in ("seq_len", "pred_len", "d_model", "num_layers", "batch_size",
                     "epochs", "patience", "decomp_kernel", "period", "basis_num"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_epochs", "num_workers", "seed", "cycle_len"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.seed >= 2**32:
            raise ValueError("seed must be smaller than 2**32")
        if self.d_model >= self.seq_len:
            raise ValueError("d_model must be smaller than seq_len")
        if self.decomp_kernel % 2 != 1:
            raise ValueError("decomp_kernel must be odd to preserve the time dimension")
        for name in ("lr", "weight_decay", "min_delta", "p1", "p2", "dropout",
                     "ffn_ratio", "hidden_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("p1", "p2", "dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.lr <= 0 or self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("lr must be positive; weight_decay and min_delta cannot be negative")
        if self.ffn_ratio <= 0 or int(self.d_model * self.ffn_ratio) < 1 or self.hidden_ratio <= 0:
            raise ValueError("ffn_ratio and hidden_ratio must produce positive hidden dimensions")
        for name in ("scale", "affine_revin", "period_norm", "individual"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("data_path", "save_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty path string")
        if self.model not in ("dytimenet", "rlinear", "dlinear", "timebase", "sparsetsf", "dytimebase"):
            raise ValueError("model must be dytimenet, rlinear, dlinear, timebase, sparsetsf, or dytimebase")
        for name in ("mix_lr", "orth_weight", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        if self.lr_schedule not in ("cosine", "type3"):
            raise ValueError("lr_schedule must be cosine or type3")
        if self.cross_residual not in ("double", "single", "norm"):
            raise ValueError("cross_residual must be double, single, or norm")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if self.target_cols is not None:
            if (not isinstance(self.target_cols, list) or not self.target_cols
                    or any(not isinstance(c, str) or not c for c in self.target_cols)
                    or len(set(self.target_cols)) != len(self.target_cols)):
                raise ValueError("target_cols must be null or a non-empty list of unique column names")
        return self

    def model_kwargs(self, num_vars):
        """All constructor arguments needed to rebuild the exact architecture."""
        names = ("seq_len", "pred_len", "d_model", "num_layers", "p1", "p2",
                 "ffn_ratio", "hidden_ratio", "dropout", "affine_revin", "decomp_kernel",
                 "cross_residual", "cycle_len")
        if self.model in ("timebase", "sparsetsf", "dytimebase"):
            names += ("period", "basis_num", "period_norm", "individual")
        return {"num_vars": num_vars, **{name: getattr(self, name) for name in names}}

    def to_dict(self):
        return asdict(self)


def config_from_dict(values):
    """Load a mapping strictly, catching misspelled YAML options."""
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML/JSON object")
    unknown = set(values) - {f.name for f in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(map(str, unknown))}")
    return ExperimentConfig(**values).validate()


def load_config(path, overrides=None):
    """Paths inside config are relative to the working directory (normally repo root)."""
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML/JSON object")
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return config_from_dict(values)


def training_parser():
    """Expose notebook settings with None defaults so YAML is never masked."""
    parser = argparse.ArgumentParser(description="Train DyTimeNet v2 and evaluate the best checkpoint.")
    parser.add_argument("--config", default="configs/ETTm2.yaml")
    defaults = ExperimentConfig()
    for field in fields(ExperimentConfig):
        name = field.name
        value = getattr(defaults, name)
        if name == "target_cols":
            parser.add_argument("--target_cols", nargs="+", default=None)
        elif name == "model":
            parser.add_argument("--model", choices=("dytimenet", "rlinear", "dlinear", "timebase", "sparsetsf", "dytimebase"), default=None)
        elif name == "lr_schedule":
            parser.add_argument("--lr_schedule", choices=("cosine", "type3"), default=None)
        elif name == "cross_residual":
            parser.add_argument("--cross_residual", choices=("double", "single", "norm"), default=None)
        elif isinstance(value, bool):
            parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=None)
    return parser
