"""End-to-end training and standalone evaluation with saved configuration/scaler."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml

from .config import config_from_dict
from .data.dataset import TimeSeriesDataset, get_dataloaders
from .engine.evaluate import evaluate, expert_usage, print_summary
from .engine.trainer import EarlyStopping, cosine_lr, run_epoch
from .models import DyTimeNet
from .utils.seed import set_seed


def resolve_device(name):
    """Use CUDA when available; fail clearly for an unavailable explicit request."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch")
    return torch.device(name)


def file_sha256(path):
    """Identify the exact CSV used for a reproducible experiment."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def train(config):
    """Train with Adam/MSE, reload best validation weights, and save test metrics."""
    # 1. Kiểm tra cấu hình, cố định seed và chọn CPU/GPU.
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    # 2. Đọc CSV, tạo các tập dữ liệu và khởi tạo model.
    train_loader, val_loader, test_loader, num_vars = get_dataloaders(
        config.data_path, config.seq_len, config.pred_len, config.batch_size,
        config.num_workers, config.target_cols, config.scale, dataset_name=config.dataset)
    model = DyTimeNet(**config.model_kwargs(num_vars)).to(device)
    moe_info = (f" | MoE E={config.num_experts} k={config.top_k} {config.moe_dispatch} "
                f"aux_weight={config.moe_aux_weight}" if config.use_moe else "")
    print(f"DyTimeNet v2 | {config.dataset} | L={config.seq_len} S={config.pred_len} | "
          f"device={device} | parameters={model.count_parameters():,}{moe_info}")
    # 3. Tạo thư mục riêng và lưu thông tin để tái lập lần chạy.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(config.save_dir) / f"{config.dataset}_L{config.seq_len}_S{config.pred_len}_seed{config.seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "pandas": str(pd.__version__), "scikit_learn": str(sklearn.__version__),
        "pyyaml": str(yaml.__version__), "device": str(device), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    write_json(run_dir / "environment.json", environment)
    metadata = {
        "config": config.to_dict(), "model_kwargs": config.model_kwargs(num_vars),
        "feature_columns": train_loader.dataset.columns,
        "scaler": train_loader.dataset.scaler_state(),
        "split_counts": list(train_loader.dataset.split_counts),
        "data_sha256": file_sha256(config.data_path), "environment": environment,
    }
    checkpoint_path = run_dir / "best.pt"
    # 4. Chuẩn bị loss, optimizer và bộ chọn checkpoint theo validation loss.
    early = EarlyStopping(config.patience, config.min_delta, checkpoint_path, metadata)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    aux_weight = config.moe_aux_weight if config.use_moe else 0.0
    history = []
    start = time.perf_counter()
    # 5. Mỗi epoch: cập nhật LR → train → validation → lưu checkpoint nếu cải thiện.
    for epoch in range(config.epochs):
        lr = cosine_lr(optimizer, epoch, config.lr, config.epochs, config.warmup_epochs)
        epoch_start = time.perf_counter()
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, aux_weight)
        train_epoch_seconds = time.perf_counter() - epoch_start
        val_m = run_epoch(model, val_loader, criterion, device=device)
        history.append({"epoch": epoch + 1, "lr": lr,
                        "train_epoch_seconds": train_epoch_seconds,
                        "epoch_seconds": time.perf_counter() - epoch_start,
                        **{f"train_{k}": v for k, v in train_m.items()},
                        **{f"val_{k}": v for k, v in val_m.items()}})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        usage = ""
        if config.use_moe:
            usage = " | experts(train)=" + "/".join(
                f"{train_m[f'expert_{i}']:.2f}" for i in range(config.num_experts))
            usage += f" | aux={train_m['aux']:.4f}"
        print(f"Epoch {epoch + 1:3}/{config.epochs} | lr={lr:.2e} | "
              f"train_loss={train_m['loss']:.6f} | val_loss={val_m['loss']:.6f}{usage}")
        if early(val_m["loss"], model, epoch + 1):
            print(f"Early stopping at epoch {epoch + 1}.")
            break
    train_seconds = time.perf_counter() - start
    # 6. Nạp checkpoint tốt nhất, đánh giá trên test và lưu kết quả.
    early.load_best(model)
    metrics = evaluate(model, test_loader, device)
    summary = print_summary(config, metrics, train_seconds)
    summary.update(best_epoch=early.best_epoch, best_val_loss=early.best_loss, epochs_run=len(history),
                   mean_epoch_seconds=float(np.mean([h["epoch_seconds"] for h in history])),
                   checkpoint=str(checkpoint_path), data_sha256=metadata["data_sha256"])
    if config.use_moe:
        summary["test_expert_fractions"] = expert_usage(model, test_loader, device)
        print("Test expert usage: " + "/".join(f"{f:.3f}" for f in summary["test_expert_fractions"]))
    write_json(run_dir / "metrics.json", summary)
    pd.DataFrame([{k: v for k, v in summary.items() if not isinstance(v, list)}]).to_csv(
        run_dir / "metrics.csv", index=False)
    print(f"Checkpoint: {checkpoint_path}")
    return checkpoint_path, summary


def evaluate_checkpoint(checkpoint_path, data_path=None, batch_size=None, device="auto",
                        num_workers=None, output=None):
    """Rebuild the model and preprocessing from a training checkpoint, without fitting."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Use a best.pt checkpoint produced by scripts/train.py (format_version=1)")
    values = dict(checkpoint["config"])
    for key, value in dict(data_path=data_path, batch_size=batch_size,
                           device=device, num_workers=num_workers).items():
        if value is not None:
            values[key] = value
    config = config_from_dict(values)
    set_seed(config.seed)
    resolved = resolve_device(config.device)
    if not Path(config.data_path).is_file():
        raise FileNotFoundError(f"CSV not found: {config.data_path}. Use --data_path to relocate it.")
    if file_sha256(config.data_path) != checkpoint["data_sha256"]:
        raise ValueError("CSV differs from the training data (SHA-256 mismatch); use the original CSV")
    if config.scale and checkpoint.get("scaler") is None:
        raise ValueError("Checkpoint is missing the fitted training scaler")
    dataset = TimeSeriesDataset(
        config.data_path, config.seq_len, config.pred_len, "test", config.target_cols, config.scale,
        dataset_name=config.dataset, scaler_state=checkpoint["scaler"],
        expected_columns=checkpoint["feature_columns"])
    if list(dataset.split_counts) != checkpoint["split_counts"]:
        raise ValueError("Dataset split counts differ from the checkpoint")
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                        shuffle=False, drop_last=False)
    model = DyTimeNet(**checkpoint["model_kwargs"]).to(resolved)
    model.load_state_dict(checkpoint["model_state_dict"])
    summary = print_summary(config, evaluate(model, loader, resolved))
    summary.update(checkpoint=str(checkpoint_path), best_epoch=checkpoint["best_epoch"],
                   data_sha256=checkpoint["data_sha256"])
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write_json(output, summary)
    return summary
