"""DeviceWindowLoader cho đúng các lô như DataLoader (cùng cửa sổ, pha chu kỳ, drop_last ở train)."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dytimenet.data.dataset import DeviceWindowLoader, TimeSeriesDataset


def test_same_batches_as_dataloader(tmp_path):
    path = tmp_path / "weather.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"date": ["2020-01-01"] * 1000, **{f"v{i}": rng.normal(size=1000) for i in range(3)}}).to_csv(
        path, index=False)
    for split, shuffle in (("val", False), ("test", False), ("train", True)):
        ds = TimeSeriesDataset(path, 24, 12, split, dataset_name="Weather", cycle_len=7)
        fast = DeviceWindowLoader(ds, 16, shuffle, "cpu")
        torch.manual_seed(0)
        got = list(fast)
        torch.manual_seed(0)
        order = torch.randperm(len(ds)) if shuffle else torch.arange(len(ds))
        sampler = order.tolist()[:len(fast) * 16] if shuffle else None
        ref = list(DataLoader(ds, batch_size=16, sampler=sampler, drop_last=shuffle))
        assert len(got) == len(ref) == len(fast)
        assert sum(b[0].shape[0] for b in got) == (len(ds) // 16 * 16 if shuffle else len(ds))
        for a, b in zip(got, ref):
            for u, v in zip(a, b):
                assert torch.equal(u, v)
