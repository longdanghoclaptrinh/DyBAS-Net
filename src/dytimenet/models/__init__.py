"""Public forecasting model."""

from .dytimebase import DyTimeBase
from .dytimenet import DyTimeNet
from .linear_baselines import DLinear, RLinear, SparseTSF, TimeBase

MODELS = {"dytimenet": DyTimeNet, "rlinear": RLinear, "dlinear": DLinear,
          "sparsetsf": SparseTSF, "timebase": TimeBase,
          "dytimebase": DyTimeBase}


def build_model(name="dytimenet", **kwargs):
    """Tạo model theo tên trong config; mặc định DyTimeNet."""
    return MODELS[name](**kwargs)

__all__ = ["DyTimeBase", "DyTimeNet", "RLinear", "DLinear", "SparseTSF", "TimeBase", "MODELS", "build_model"]
