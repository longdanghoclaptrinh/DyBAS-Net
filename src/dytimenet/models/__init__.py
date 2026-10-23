"""Public forecasting model."""

from functools import partial

from .dytimebase import DyTimeBase
from .dytimenet import DyTimeNet
from .linear_baselines import DLinear, RLinear, SparseTSF, TimeBase
from .periodic_baselines import CycleNet, TQNet

MODELS = {"dytimenet": DyTimeNet, "rlinear": RLinear, "dlinear": DLinear,
          "sparsetsf": SparseTSF, "timebase": TimeBase,
          "dytimebase": DyTimeBase, "cyclenet": CycleNet,
          "cyclenet_mlp": partial(CycleNet, mlp=True), "tqnet": TQNet}


def build_model(name="dytimenet", **kwargs):
    """Tạo model theo tên trong config; mặc định DyTimeNet."""
    return MODELS[name](**kwargs)

__all__ = ["DyTimeBase", "DyTimeNet", "RLinear", "DLinear", "SparseTSF", "TimeBase", "CycleNet", "TQNet",
           "MODELS", "build_model"]
