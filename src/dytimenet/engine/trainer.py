"""Notebook training equations, early stopping, and portable best checkpoints."""

import math
from pathlib import Path

import torch
from torch import nn


class EarlyStopping:
    """Select the minimum validation loss with the notebook's min_delta rule."""

    def __init__(self, patience=10, min_delta=1e-5, save_path="best_model.pt", metadata=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = Path(save_path)
        self.metadata = metadata
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = None
        self.stopped = False

    def __call__(self, val_loss, model, epoch=None):
        if not math.isfinite(val_loss):
            raise ValueError("Validation loss is not finite")
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.counter = val_loss, epoch, 0
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            state = model.state_dict()
            if self.metadata is not None:
                state = {**self.metadata, "format_version": 1, "model_state_dict": state,
                         "best_val_loss": val_loss, "best_epoch": epoch}
            torch.save(state, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
        return False

    def load_best(self, model):
        checkpoint = torch.load(self.save_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return model


def cosine_lr(optimizer, epoch, lr_init, total_epochs, warmup_epochs=2):
    """Zero-based warmup followed by the original cosine schedule."""
    if epoch < warmup_epochs:
        lr = lr_init * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = lr_init * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def run_epoch(model, loader, criterion, optimizer=None, device="cpu", aux_weight=0.0):
    """Train/evaluate with sample-weighted MSE/MAE and gradient norm clipping at 1.

    Model MoE: khi train, loss = criterion + aux_weight·aux (validation/test chỉ là criterion);
    trả thêm "aux" và "expert_i" = tỉ lệ lượt gán token vào expert i trong epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_mse = total_mae = total_aux = 0.0
    n = 0
    expert_counts = None
    with torch.set_grad_enabled(is_train):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            aux = model.moe_aux_loss() if hasattr(model, "moe_aux_loss") else None
            if is_train and aux is not None:
                total_aux += aux.item() * bx.size(0)
                loss = loss + aux_weight * aux
            counts = model.moe_expert_counts() if hasattr(model, "moe_expert_counts") else None
            if counts is not None:
                expert_counts = counts.cpu() if expert_counts is None else expert_counts + counts.cpu()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss encountered; check data and hyperparameters")
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = bx.size(0)
            total_loss += loss.item() * bs
            total_mse += ((out - by) ** 2).mean().item() * bs
            total_mae += (out - by).abs().mean().item() * bs
            n += bs
    if n == 0:
        raise ValueError("DataLoader produced no batches; check split sizes and batch_size")
    metrics = {"loss": total_loss / n, "mse": total_mse / n, "mae": total_mae / n}
    if expert_counts is not None:
        if is_train:
            metrics["aux"] = total_aux / n
        fractions = expert_counts.double() / expert_counts.sum()
        metrics.update({f"expert_{i}": f.item() for i, f in enumerate(fractions)})
    return metrics
