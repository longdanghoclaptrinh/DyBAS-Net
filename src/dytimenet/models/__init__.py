"""Public forecasting model."""

from .dytimenet import DyTimeNet
from .linear_baselines import DLinear, RLinear, SparseTSF, TimeBase

MODELS = {"dytimenet": DyTimeNet, "rlinear": RLinear, "dlinear": DLinear,
          "sparsetsf": SparseTSF, "timebase": TimeBase}


def build_model(name="dytimenet", **kwargs):
    """Tạo model theo tên trong config; mặc định DyTimeNet."""
    return MODELS[name](**kwargs)

__all__ = ["DyTimeNet", "RLinear", "DLinear", "SparseTSF", "TimeBase", "MODELS", "build_model"]
