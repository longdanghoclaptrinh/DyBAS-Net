"""Guard chronological split boundaries, training-only scaling, and invalid CSVs."""

import numpy as np
import pandas as pd
import pytest
import torch

from dytimenet.data.dataset import TimeSeriesDataset, get_dataloaders, get_dataset_splits


def test_windows_and_train_only_scaler(tmp_path):
    path = tmp_path / "weather.csv"
    values = np.arange(1000, dtype=np.float32)
    pd.DataFrame({"date": ["2020-01-01"] * 1000, "x": values, "y": values * 2}).to_csv(path, index=False)
    train, val, test, channels = get_dataloaders(path, 16, 8, 32, verbose=False)
    assert channels == 2
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (677, 93, 193)
    assert train.drop_last and not val.drop_last and not test.drop_last
    np.testing.assert_allclose(train.dataset.scaler.mean_, [349.5, 699])
    np.testing.assert_equal(val.dataset.scaler.mean_, train.dataset.scaler.mean_)
    np.testing.assert_equal(test.dataset.scaler.mean_, train.dataset.scaler.mean_)
    for split, start, end in (("train", 16, 699), ("val", 700, 799), ("test", 800, 999)):
        ds = TimeSeriesDataset(path, 16, 8, split, scale=False)
        x, y = ds[0]
        assert x.shape == (2, 16) and y.shape == (2, 8)
        assert x.is_contiguous() and y.is_contiguous()
        assert y[0, 0].item() == start
        assert ds[len(ds) - 1][1][0, -1].item() == end
    # Reusing a checkpoint scaler must not fit on modified train observations.
    state = train.dataset.scaler_state()
    frame = pd.read_csv(path)
    frame.loc[:699, "x"] += 10000
    frame.to_csv(path, index=False)
    restored = TimeSeriesDataset(path, 16, 8, "test", scaler_state=state,
                                 expected_columns=["x", "y"])
    torch.testing.assert_close(restored.data, test.dataset.data, rtol=0, atol=0)


def test_ett_splits_and_insufficient_rows():
    assert get_dataset_splits("renamed.csv", 70000, "ETTm2") == (34560, 11520, 11520)
    assert get_dataset_splits("ETTh1.csv", 17420) == (8640, 2880, 2880)
    with pytest.raises(ValueError, match="57,600"):
        get_dataset_splits("ETTm2.csv", 500)


def test_invalid_data_and_empty_training_batch(tmp_path):
    path = tmp_path / "weather.csv"
    pd.DataFrame({"x": np.arange(200)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="batch_size"):
        get_dataloaders(path, 16, 8, batch_size=1000)
    with pytest.raises(ValueError, match="names/order"):
        TimeSeriesDataset(path, 16, 8, expected_columns=["other"])
    with pytest.raises(ValueError, match="Not enough"):
        TimeSeriesDataset(path, 96, 96)
    pd.DataFrame({"x": [np.nan] + list(range(199))}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="NaN"):
        TimeSeriesDataset(path, 16, 8)
