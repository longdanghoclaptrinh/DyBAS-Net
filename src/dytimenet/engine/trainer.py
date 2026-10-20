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


def cosine_lr(optimizer, epoch, lr_init, total_epochs, warmup_epochs=2, kind="cosine"):
    """Zero-based warmup followed by the original cosine schedule.

    kind="type3": lịch của TimeBase/SparseTSF (lradj type3) — giữ lr 4 epoch đầu (epoch 0–3), sau đó nhân 0.8 mỗi epoch.
    """
    if kind == "type3":
        lr = lr_init if epoch < 4 else lr_init * 0.8 ** (epoch - 3)
    elif epoch < warmup_epochs:
        lr = lr_init * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = lr_init * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:   # nhóm có base_lr riêng giữ nguyên tỉ lệ so với lr_init
        pg["lr"] = lr * pg.get("base_lr", lr_init) / lr_init
    return lr


def run_epoch(model, loader, criterion, optimizer=None, device="cpu", aux_weight=0.0, grad_clip=1.0):
    """Train/evaluate with sample-weighted MSE/MAE and gradient norm clipping (grad_clip; 0 = tắt).

    aux_weight > 0: cộng aux_weight · model.aux_loss (vd. loss trực giao của TimeBase) vào loss khi train;
    loss báo cáo vẫn là MSE.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_mse = total_mae = 0.0
    n = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            bx, by, *extra = batch      # extra = [pha chu kỳ] khi dùng RCF
            bx, by = bx.to(device), by.to(device)
            out = model(bx, *[e.to(device) for e in extra])
            loss = criterion(out, by)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss encountered; check data and hyperparameters")
            if is_train:
                optimizer.zero_grad()
                aux = getattr(model, "aux_loss", None)
                (loss + aux_weight * aux if aux_weight and aux is not None else loss).backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = bx.size(0)
            total_loss += loss.item() * bs
            total_mse += ((out - by) ** 2).mean().item() * bs
            total_mae += (out - by).abs().mean().item() * bs
            n += bs
    if n == 0:
        raise ValueError("DataLoader produced no batches; check split sizes and batch_size")
    return {"loss": total_loss / n, "mse": total_mse / n, "mae": total_mae / n}
