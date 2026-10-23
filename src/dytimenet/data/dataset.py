"""CSV windows and the notebook's ETT 12/4/4-month or Weather 70/10/20 splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset


def get_dataset_splits(data_path, n_rows, dataset_name=None):
    """Return train/val/test row counts; explicit dataset avoids filename ambiguity."""
    name = (dataset_name or Path(data_path).stem).lower()
    if "ettm" in name:
        counts = (34560, 11520, 11520)
    elif "etth" in name or "ett" in name:
        counts = (8640, 2880, 2880)
    else:
        n_train, n_val = int(n_rows * 0.7), int(n_rows * 0.1)
        counts = (n_train, n_val, n_rows - n_train - n_val)
    if sum(counts) > n_rows:
        raise ValueError(f"{name} needs at least {sum(counts):,} rows; CSV has {n_rows:,}")
    return counts


class TimeSeriesDataset(Dataset):
    """Forecast every selected numeric variable as contiguous [C, L] / [C, S] tensors."""

    def __init__(self, data_path, seq_len, pred_len, split="train", target_cols=None,
                 scale=True, dataset_name=None, scaler_state=None, expected_columns=None, cycle_len=0):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if seq_len <= 0 or pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len, self.pred_len, self.scale = seq_len, pred_len, scale
        self.cycle_len = cycle_len   # > 0: trả thêm pha chu kỳ (chỉ số dòng toàn cục mod cycle_len) cho RCF
        if not Path(data_path).is_file():
            raise FileNotFoundError(f"CSV not found: {data_path}. See data/README.md or use --data_path.")
        df = pd.read_csv(data_path)
        date_cols = [c for c in df.columns
                     if any(kw in c.lower() for kw in ("date", "time", "timestamp"))]
        df = df.drop(columns=date_cols)
        if target_cols:
            missing = set(target_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing target columns: {sorted(missing)}")
            df = df[target_cols]
        numeric = df.select_dtypes(include=[np.number])
        if target_cols and len(numeric.columns) != len(target_cols):
            raise ValueError("All target_cols must contain numeric data")
        self.columns = list(numeric.columns)
        if not self.columns:
            raise ValueError("CSV has no numeric feature columns")
        if expected_columns is not None and self.columns != list(expected_columns):
            raise ValueError("CSV feature names/order differ from the training checkpoint")
        data = numeric.values.astype(np.float32)
        if not np.isfinite(data).all():
            raise ValueError("CSV contains NaN or infinite values; clean it before training")
        self.split_counts = get_dataset_splits(data_path, len(data), dataset_name)
        n_train, n_val, n_test = self.split_counts
        if n_train < seq_len + pred_len or min(n_val, n_test) < pred_len:
            raise ValueError("Not enough rows in each split for seq_len/pred_len")
        borders = {
            "train": (0, n_train),
            "val": (n_train - seq_len, n_train + n_val),
            "test": (n_train + n_val - seq_len, n_train + n_val + n_test),
        }
        self.scaler = StandardScaler()
        if scale:
            if scaler_state is None:
                self.scaler.fit(data[:n_train])
            else:
                for name in ("mean_", "scale_", "var_"):
                    values = np.asarray(scaler_state[name], dtype=np.float64)
                    if values.shape != (len(self.columns),) or not np.isfinite(values).all():
                        raise ValueError("Checkpoint scaler does not match the CSV features")
                    setattr(self.scaler, name, values)
                self.scaler.n_features_in_ = len(self.columns)
                self.scaler.n_samples_seen_ = scaler_state["n_samples_seen_"]
            data = self.scaler.transform(data)
        self.start, self.end = borders[split]
        self.data = torch.tensor(data[self.start:self.end], dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        if self.cycle_len:
            return x.T.contiguous(), y.T.contiguous(), (self.start + idx) % self.cycle_len
        return x.T.contiguous(), y.T.contiguous()

    @property
    def num_vars(self):
        return self.data.shape[1]

    def scaler_state(self):
        """Primitive values suitable for torch.load(weights_only=True)."""
        if not self.scale:
            return None
        return {
            **{name: getattr(self.scaler, name).tolist() for name in ("mean_", "scale_", "var_")},
            "n_samples_seen_": int(self.scaler.n_samples_seen_),
        }


class DeviceWindowLoader:
    """Thay DataLoader khi chạy GPU: giữ cả split trên thiết bị và cắt một lô cửa sổ bằng một phép chỉ mục.

    Cùng cửa sổ, cùng thứ tự cột và pha chu kỳ như TimeSeriesDataset; train xáo trộn bằng torch.randperm (RNG đã cố
    định seed) và bỏ lô cuối, val/test giữ thứ tự và dùng mọi cửa sổ. Tránh nút thắt CPU của việc cắt từng mẫu.
    """

    def __init__(self, dataset, batch_size, shuffle, device):
        self.dataset, self.batch_size, self.shuffle, self.drop_last = dataset, batch_size, shuffle, shuffle
        self.data = dataset.data.to(device)                                       # [T, C]
        self.offsets = torch.arange(dataset.seq_len + dataset.pred_len, device=device)
        self.device = device

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n, L = len(self.dataset), self.dataset.seq_len
        order = torch.randperm(n) if self.shuffle else torch.arange(n)
        for k in range(len(self)):
            idx = order[k * self.batch_size:(k + 1) * self.batch_size].to(self.device)
            win = self.data[idx[:, None] + self.offsets]                          # [B, L + S, C]
            x, y = win[:, :L].transpose(1, 2).contiguous(), win[:, L:].transpose(1, 2).contiguous()
            if self.dataset.cycle_len:
                yield x, y, (self.dataset.start + idx) % self.dataset.cycle_len
            else:
                yield x, y


def get_dataloaders(data_path, seq_len, pred_len, batch_size=32, num_workers=0,
                    target_cols=None, scale=True, verbose=True, dataset_name=None, cycle_len=0, device=None):
    """Preserve shuffled/drop-last training and complete ordered validation/test.

    device là GPU: dùng DeviceWindowLoader (cùng cửa sổ và cách chia lô, nhanh hơn nhiều); CPU giữ DataLoader.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    kwargs = dict(data_path=data_path, seq_len=seq_len, pred_len=pred_len,
                  target_cols=target_cols, scale=scale, dataset_name=dataset_name, cycle_len=cycle_len)
    train_ds = TimeSeriesDataset(**kwargs, split="train")
    val_ds = TimeSeriesDataset(**kwargs, split="val", scaler_state=train_ds.scaler_state())
    test_ds = TimeSeriesDataset(**kwargs, split="test", scaler_state=train_ds.scaler_state())
    if len(train_ds) < batch_size:
        raise ValueError("Training split has fewer windows than batch_size; reduce --batch_size")

    def loader(ds, shuffle):
        if device is not None and torch.device(device).type == "cuda":
            return DeviceWindowLoader(ds, batch_size, shuffle, device)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=shuffle)

    if verbose:
        print(f"[Dataset] Variables: {train_ds.num_vars}; windows: "
              f"train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")
    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False), train_ds.num_vars
