"""DeviceWindowLoader cho đúng các lô như DataLoader: cùng cửa sổ, pha chu kỳ, drop_last và cùng thứ tự xáo trộn."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dytimenet.data.dataset import DeviceWindowLoader, TimeSeriesDataset


def test_same_batches_and_shuffle_order_as_dataloader(tmp_path):
    path = tmp_path / "weather.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"date": ["2020-01-01"] * 1000, **{f"v{i}": rng.normal(size=1000) for i in range(3)}}).to_csv(
        path, index=False)
    train, val = (TimeSeriesDataset(path, 24, 12, s, dataset_name="Weather", cycle_len=7) for s in ("train", "val"))

    def epochs(make):
        # Như vòng train: 2 epoch, mỗi epoch duyệt train (xáo trộn) rồi val.
        torch.manual_seed(2021)
        tr, va = make(train, True), make(val, False)
        return [list(loader) for _ in range(2) for loader in (tr, va)]

    ref = epochs(lambda ds, shuffle: DataLoader(ds, batch_size=16, shuffle=shuffle, drop_last=shuffle))
    got = epochs(lambda ds, shuffle: DeviceWindowLoader(ds, 16, shuffle, "cpu"))
    assert len(got[0]) == len(train) // 16 and sum(b[0].shape[0] for b in got[1]) == len(val)
    for a, b in zip(got, ref):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            for u, v in zip(x, y):
                assert torch.equal(u, v)
    assert not torch.equal(got[0][0][0], got[2][0][0])   # hai epoch xáo trộn khác nhau
