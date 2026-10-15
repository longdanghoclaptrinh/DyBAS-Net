# Báo cáo: IntraVMoE top-k so với DyTimeNet baseline trên ETTh1

Tài liệu này đọc được độc lập, không cần mở code. Mọi số liệu lấy từ các lần chạy đã lưu (`results/*.json`, `history.csv`, `metrics.json`, checkpoint) và được sinh bởi `scripts/make_moe_report.py`. Cấu hình hoặc phép đo chưa có ghi là "chưa chạy" hoặc "chưa đo".

**Trạng thái các cấu hình:**

- `baseline`: seed 2021, 2022, 2023
- `moe_k1_sparse`: seed 2021, 2022, 2023
- `moe_k1_dense` ⚠️: seed 2021, 2022, 2023
- `moe_k2_sparse` ⚠️: seed 2021, 2022, 2023
- `moe_lite` ⚠️: seed 2021, 2022, 2023

⚠️ = có ít nhất một seed train/đo **cùng lúc** với một run khác trên cùng CPU (xem bảng dòng thời gian ở mục 1). Thời gian (s/epoch, tổng train, latency) của các cấu hình này **không so sánh công bằng** được; MSE/MAE không bị ảnh hưởng.

## 1. Môi trường và thiết lập

| Mục | Giá trị |
| --- | --- |
| CPU | AMD Ryzen 7 7840HS w/ Radeon 780M Graphics (16 luồng logic), không GPU |
| Số luồng PyTorch | `torch.set_num_threads(8)` khi train và đo; latency đo thêm với 1 luồng |
| Hệ điều hành | Windows-11-10.0.26200-SP0 |
| Python / PyTorch | 3.13.5 / 2.6.0+cpu (CUDA: None) |
| numpy / pandas / scikit-learn | 2.1.2 / 2.3.3 / 1.7.2 |
| Dataset | ETTh1, 17,420 dòng, cột `date` + 7 biến (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT), 2016-07-01 00:00 → 2018-06-26 19:00, mỗi giờ 1 dòng |
| SHA-256 CSV | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |
| Cách chia | 12/4/4 tháng (quy ước tháng 30 ngày): train 8,640 / val 2,880 / test 2,880 dòng; 3,020 dòng cuối không dùng. Không phải 7:1:2. |
| Số mẫu (cửa sổ) | train 8,449 / val 2,785 / test 2,785 (val/test lấy thêm 96 dòng quá khứ làm input) |
| Chuẩn hóa | StandardScaler fit trên train; MSE/MAE báo cáo ở thang StandardScaler |
| Hyperparameter | seq_len=96, pred_len=96, d_model=64, d_ff=128 (ffn_ratio=2.0; moe_lite dùng ffn_ratio=1.0), batch=64, lr=0.0005, p1=0.1, p2=0.1, dropout=0.1, weight_decay=0.0001, decomp_kernel=25 |
| Huấn luyện | Adam, clip grad norm 1.0, tối đa 100 epoch, early stopping theo val MSE (patience=10, min_delta=1e-05), nạp lại checkpoint tốt nhất |
| Scheduler | warm-up tuyến tính 2 epoch rồi cosine về 0 tại epoch 100 |
| MoE | λ (moe_aux_weight) = 0.01; noise router std = 1/E khi train; seed 2021, 2022, 2023 |
| Commit baseline (tag `baseline-dytimenet`) | `7e1f91f5649574ce6c24688f9500b5116d5a0555` |
| Commit thay đổi model trên `feat/moe-topk` | `b33ef2a1e4c461c1df7934ea497dcf2d4a6bca2f` |
| HEAD `feat/moe-topk` khi sinh báo cáo | `4767bb5697d966c464c0e983edf5a817a4467173` (commit chứa báo cáo này nằm ngay sau) |

Thời điểm chạy (lấy từ tên thư mục checkpoint):

- `baseline`: 20261015 090804 → 090954 UTC
- `moe_k1_sparse`: 20261015 091105 → 091614 UTC
- `moe_k1_dense`: 20261015 091820 → 092232 UTC
- `moe_k2_sparse`: 20261015 092442 → 092818 UTC
- `moe_lite`: 20261015 092210 → 092859 UTC

**Dòng thời gian từng run** (UTC; kết thúc = lúc ghi `metrics.json`, latency được đo ngay sau đó). Cột chồng lấn liệt kê các run chạy cùng lúc.

| Run | Bắt đầu | Kết thúc train | s/epoch min / median / max | Chạy cùng lúc với |
| --- | --- | --- | ---: | --- |
| `baseline_seed2021` | 09:08:04 | 09:08:48 | 1.15 / 1.19 / 1.37 | — |
| `baseline_seed2022` | 09:08:50 | 09:09:53 | 1.14 / 1.18 / 1.24 | — |
| `baseline_seed2023` | 09:09:54 | 09:11:03 | 1.15 / 1.19 / 1.27 | — |
| `moe_k1_sparse_seed2021` | 09:11:05 | 09:12:58 | 1.66 / 1.79 / 3.09 | — |
| `moe_k1_sparse_seed2022` | 09:13:03 | 09:16:09 | 2.81 / 2.87 / 3.21 | — |
| `moe_k1_sparse_seed2023` | 09:16:14 | 09:18:16 | 2.82 / 2.87 / 3.07 | — |
| `moe_k1_dense_seed2021` | 09:18:20 | 09:20:09 | 1.66 / 1.78 / 3.26 | — |
| `moe_k1_dense_seed2022` | 09:20:12 | 09:22:26 | 1.73 / 1.89 / 5.34 | moe_lite_seed2021 |
| `moe_lite_seed2021` | 09:22:10 | 09:24:40 | 2.07 / 3.38 / 3.80 | moe_k1_dense_seed2022, moe_k1_dense_seed2023 |
| `moe_k1_dense_seed2023` | 09:22:32 | 09:24:37 | 2.08 / 3.39 / 3.78 | moe_lite_seed2021 |
| `moe_k2_sparse_seed2021` | 09:24:42 | 09:26:14 | 3.18 / 3.30 / 3.88 | moe_lite_seed2022 |
| `moe_lite_seed2022` | 09:24:46 | 09:28:54 | 2.99 / 3.23 / 3.65 | moe_k2_sparse_seed2021, moe_k2_sparse_seed2022, moe_k2_sparse_seed2023 |
| `moe_k2_sparse_seed2022` | 09:26:19 | 09:28:13 | 3.22 / 3.31 / 3.52 | moe_lite_seed2022 |
| `moe_k2_sparse_seed2023` | 09:28:18 | 09:31:03 | 3.09 / 3.31 / 3.72 | moe_lite_seed2022, moe_lite_seed2023 |
| `moe_lite_seed2023` | 09:28:59 | 09:31:37 | 2.78 / 3.13 / 3.62 | moe_k2_sparse_seed2023 |

## 2. Mô tả thay đổi

Kiến trúc baseline: RevIN → MovingAvg tách trend/seasonal; trend: Linear L→S; seasonal: depthwise Conv1D → GELU → Linear L→D → CrossVBlock (DropFusion) + residual → IntraVBlock → Linear D→S; cộng hai nhánh → inverse RevIN. MoE **chỉ thay FFN bên trong IntraVBlock**: `out = x + IntraVMoE(LN(x))`. Token = một cặp (mẫu, kênh), vector D chiều. Router nhận `[LN(x); μ; log σ]`, trong đó μ, σ là thống kê RevIN của **input gốc** từng kênh (đã detach). Train loss = MSE + λ·aux (Switch: aux = E·Σ f_i·P_i, f_i = tỉ lệ lượt gán top-k chia k, P_i = xác suất softmax trung bình). Validation, early stopping và các số báo cáo chỉ dùng MSE. Khi tắt `--use_moe`, output và gradient trùng từng bit với tag baseline (đã kiểm tra).

`git diff baseline-dytimenet..feat/moe-topk --stat` (tại HEAD trên, chưa gồm commit báo cáo):

```
scripts/bench_moe.py                  | 315 ++++++++++++++++++++++++++++++++++
 src/dytimenet/config.py               |  24 ++-
 src/dytimenet/engine/evaluate.py      |   9 +
 src/dytimenet/engine/trainer.py       |  26 ++-
 src/dytimenet/experiment.py           |  29 +++-
 src/dytimenet/layers/intra_v_block.py |  36 ++--
 src/dytimenet/layers/intra_v_moe.py   |  85 +++++++++
 src/dytimenet/layers/revin.py         |   5 +-
 src/dytimenet/models/dytimenet.py     |  27 ++-
 tests/test_moe.py                     | 103 +++++++++++
 10 files changed, 629 insertions(+), 30 deletions(-)
```

**IntraVMoE (`src/dytimenet/layers/intra_v_moe.py`, file mới)**

```python
"""Top-k Mixture-of-Experts thay cho FFN của IntraVBlock (token = một cặp mẫu × kênh)."""

import torch
from torch import nn
from torch.nn import functional as F


def _ffn(d_model: int, d_ff: int, dropout: float) -> nn.Sequential:
    """Đúng cấu trúc FFN cũ của IntraVBlock, chỉ khác độ rộng ẩn."""
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout),
    )


class IntraVMoE(nn.Module):
    """
    Shared expert (d_ff/2, luôn bật) + E expert được route (mỗi cái d_ff/(2k)).

    Router: Linear(D + 2 → E) trên [token ; μ ; log σ], μ/σ lấy từ RevIN đầu vào.
    Khi train cộng noise Gaussian std = 1/E vào logits; chọn top-k, softmax trên k logit.
    dispatch="dense": chạy mọi expert rồi nhân mặt nạ gate (đối chứng).
    dispatch="sparse": mỗi expert chỉ chạy trên các token được route tới nó, rồi index_add_.
    Sau mỗi forward: self.aux_loss (Switch load-balance, chỉ khi train) và
    self.expert_counts (số lượt gán token → expert, [E]).
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 1,
                 dropout: float = 0.1, dispatch: str = "sparse"):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if dispatch not in ("dense", "sparse"):
            raise ValueError("dispatch must be dense or sparse")
        self.num_experts = num_experts
        self.top_k = top_k
        self.dispatch = dispatch
        self.shared = _ffn(d_model, max(1, d_ff // 2), dropout)
        self.experts = nn.ModuleList([
            _ffn(d_model, max(1, d_ff // (2 * top_k)), dropout) for _ in range(num_experts)])
        self.router = nn.Linear(d_model + 2, num_experts)
        self.aux_loss = None
        self.expert_counts = None

    def forward(self, x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """x : [B, C, D], mu/log_sigma : [B, C, 1] → [B, C, D]"""
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])                                   # [N, D]
        logits = self.router(torch.cat([x, mu, log_sigma], dim=-1)).reshape(-1, self.num_experts)
        if self.training:
            logits = logits + torch.randn_like(logits) / self.num_experts
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)              # [N, k]
        gates = F.softmax(top_logits, dim=-1)                               # [N, k]

        out = self.shared(tokens)
        if self.dispatch == "dense":
            # [N, k, E] → [N, E]: gate tại expert được chọn, 0 ở chỗ khác.
            weights = (F.one_hot(top_idx, self.num_experts).to(gates.dtype) * gates.unsqueeze(-1)).sum(1)
            expert_out = torch.stack([expert(tokens) for expert in self.experts], dim=1)   # [N, E, D]
            out = out + (weights.unsqueeze(-1) * expert_out).sum(1)
        else:
            flat_idx = top_idx.reshape(-1)                                  # [N·k]
            flat_gate = gates.reshape(-1, 1)
            token_ids = torch.arange(tokens.shape[0], device=x.device).repeat_interleave(self.top_k)
            routed = torch.zeros_like(tokens)
            for e, expert in enumerate(self.experts):
                sel = (flat_idx == e).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                ids = token_ids[sel]
                routed.index_add_(0, ids, expert(tokens[ids]) * flat_gate[sel])
            out = out + routed

        counts = torch.bincount(top_idx.reshape(-1), minlength=self.num_experts)
        self.expert_counts = counts.detach()
        if self.training:
            # Switch: aux = E · Σ f_i · P_i; f_i = tỉ lệ lượt gán top-k (chia k), P_i = xác suất router.
            frac = counts.to(logits.dtype) / (tokens.shape[0] * self.top_k)
            prob = F.softmax(logits, dim=-1).mean(0)
            self.aux_loss = self.num_experts * (frac * prob).sum()
        else:
            self.aux_loss = None
        return out.reshape(shape)
```

**IntraVBlock**

```diff
diff --git a/src/dytimenet/layers/intra_v_block.py b/src/dytimenet/layers/intra_v_block.py
index 34fb836..5aacd42 100644
--- a/src/dytimenet/layers/intra_v_block.py
+++ b/src/dytimenet/layers/intra_v_block.py
@@ -3,24 +3,36 @@
 import torch
 from torch import nn
 
+from dytimenet.layers.intra_v_moe import IntraVMoE
+
 
 class IntraVBlock(nn.Module):
-    """Intra-Variable Block — Dense FFN trên chiều thời gian D (shared across C)."""
-    def __init__(self, d_model: int, ffn_ratio: float = 2.0, dropout: float = 0.1):
+    """Intra-Variable Block — Dense FFN trên chiều thời gian D (shared across C).
+
+    use_moe=True thay FFN bằng IntraVMoE; LayerNorm và residual giữ nguyên.
+    """
+    def __init__(self, d_model: int, ffn_ratio: float = 2.0, dropout: float = 0.1,
+                 use_moe: bool = False, num_experts: int = 4, top_k: int = 1,
+                 moe_dispatch: str = "sparse"):
         super().__init__()
         d_ff = int(d_model * ffn_ratio)
+        self.use_moe = use_moe
         self.norm = nn.LayerNorm(d_model)
-        self.ffn  = nn.Sequential(
-            nn.Linear(d_model, d_ff),
-            nn.GELU(),
-            nn.Dropout(dropout),
-            nn.Linear(d_ff, d_model),
-            nn.Dropout(dropout),
-        )
+        if use_moe:
+            self.moe = IntraVMoE(d_model, d_ff, num_experts, top_k, dropout, moe_dispatch)
+        else:
+            self.ffn  = nn.Sequential(
+                nn.Linear(d_model, d_ff),
+                nn.GELU(),
+                nn.Dropout(dropout),
+                nn.Linear(d_ff, d_model),
+                nn.Dropout(dropout),
+            )
 
-    def forward(self, x: torch.Tensor) -> torch.Tensor:
-        """x : [B, C, D] → [B, C, D]"""
+    def forward(self, x: torch.Tensor, mu: torch.Tensor = None,
+                log_sigma: torch.Tensor = None) -> torch.Tensor:
+        """x : [B, C, D] → [B, C, D]; mu/log_sigma [B, C, 1] chỉ dùng khi use_moe."""
         residual = x
         x = self.norm(x)
-        x = self.ffn(x)
+        x = self.moe(x, mu, log_sigma) if self.use_moe else self.ffn(x)
         return x + residual
```

**RevIN**

```diff
diff --git a/src/dytimenet/layers/revin.py b/src/dytimenet/layers/revin.py
index 139fbbc..f95c54a 100644
--- a/src/dytimenet/layers/revin.py
+++ b/src/dytimenet/layers/revin.py
@@ -18,13 +18,16 @@ class RevIN(nn.Module):
         self._mean: Optional[torch.Tensor] = None
         self._std:  Optional[torch.Tensor] = None
 
-    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
+    def forward(self, x: torch.Tensor, mode: str = "norm", return_stats: bool = False):
+        """return_stats=True (chỉ mode "norm") trả thêm μ, σ đã detach, shape [B, C, 1]."""
         if mode == "norm":
             self._mean = x.mean(dim=-1, keepdim=True).detach()
             self._std  = x.std(dim=-1, keepdim=True, unbiased=False).detach() + self.eps
             x = (x - self._mean) / self._std
             if self.affine:
                 x = x * self.affine_weight[None, :, None] + self.affine_bias[None, :, None]
+            if return_stats:
+                return x, self._mean, self._std
             return x
         elif mode == "denorm":
             if self._mean is None:
```

**Cộng aux loss (`run_epoch`)**

```diff
diff --git a/src/dytimenet/engine/trainer.py b/src/dytimenet/engine/trainer.py
index 8edaefb..b5c7c79 100644
--- a/src/dytimenet/engine/trainer.py
+++ b/src/dytimenet/engine/trainer.py
@@ -56,17 +56,29 @@ def cosine_lr(optimizer, epoch, lr_init, total_epochs, warmup_epochs=2):
     return lr
 
 
-def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
-    """Train/evaluate with sample-weighted MSE/MAE and gradient norm clipping at 1."""
+def run_epoch(model, loader, criterion, optimizer=None, device="cpu", aux_weight=0.0):
+    """Train/evaluate with sample-weighted MSE/MAE and gradient norm clipping at 1.
+
+    Model MoE: khi train, loss = criterion + aux_weight·aux (validation/test chỉ là criterion);
+    trả thêm "aux" và "expert_i" = tỉ lệ lượt gán token vào expert i trong epoch.
+    """
     is_train = optimizer is not None
     model.train(is_train)
-    total_loss = total_mse = total_mae = 0.0
+    total_loss = total_mse = total_mae = total_aux = 0.0
     n = 0
+    expert_counts = None
     with torch.set_grad_enabled(is_train):
         for bx, by in loader:
             bx, by = bx.to(device), by.to(device)
             out = model(bx)
             loss = criterion(out, by)
+            aux = model.moe_aux_loss() if hasattr(model, "moe_aux_loss") else None
+            if is_train and aux is not None:
+                total_aux += aux.item() * bx.size(0)
+                loss = loss + aux_weight * aux
+            counts = model.moe_expert_counts() if hasattr(model, "moe_expert_counts") else None
+            if counts is not None:
+                expert_counts = counts.cpu() if expert_counts is None else expert_counts + counts.cpu()
             if not torch.isfinite(loss):
                 raise ValueError("Non-finite loss encountered; check data and hyperparameters")
             if is_train:
@@ -81,4 +93,10 @@ def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
             n += bs
     if n == 0:
         raise ValueError("DataLoader produced no batches; check split sizes and batch_size")
-    return {"loss": total_loss / n, "mse": total_mse / n, "mae": total_mae / n}
+    metrics = {"loss": total_loss / n, "mse": total_mse / n, "mae": total_mae / n}
+    if expert_counts is not None:
+        if is_train:
+            metrics["aux"] = total_aux / n
+        fractions = expert_counts.double() / expert_counts.sum()
+        metrics.update({f"expert_{i}": f.item() for i, f in enumerate(fractions)})
+    return metrics
```

**Truyền λ, log thời gian epoch và tỉ lệ expert (`experiment.py`)**

```diff
diff --git a/src/dytimenet/experiment.py b/src/dytimenet/experiment.py
index c144b8b..83be04c 100644
--- a/src/dytimenet/experiment.py
+++ b/src/dytimenet/experiment.py
@@ -17,7 +17,7 @@ import yaml
 
 from .config import config_from_dict
 from .data.dataset import TimeSeriesDataset, get_dataloaders
-from .engine.evaluate import evaluate, print_summary
+from .engine.evaluate import evaluate, expert_usage, print_summary
 from .engine.trainer import EarlyStopping, cosine_lr, run_epoch
 from .models import DyTimeNet
 from .utils.seed import set_seed
@@ -57,8 +57,10 @@ def train(config):
         config.data_path, config.seq_len, config.pred_len, config.batch_size,
         config.num_workers, config.target_cols, config.scale, dataset_name=config.dataset)
     model = DyTimeNet(**config.model_kwargs(num_vars)).to(device)
+    moe_info = (f" | MoE E={config.num_experts} k={config.top_k} {config.moe_dispatch} "
+                f"aux_weight={config.moe_aux_weight}" if config.use_moe else "")
     print(f"DyTimeNet v2 | {config.dataset} | L={config.seq_len} S={config.pred_len} | "
-          f"device={device} | parameters={model.count_parameters():,}")
+          f"device={device} | parameters={model.count_parameters():,}{moe_info}")
     # 3. Tạo thư mục riêng và lưu thông tin để tái lập lần chạy.
     stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
     run_dir = Path(config.save_dir) / f"{config.dataset}_L{config.seq_len}_S{config.pred_len}_seed{config.seed}_{stamp}"
@@ -84,19 +86,29 @@ def train(config):
     early = EarlyStopping(config.patience, config.min_delta, checkpoint_path, metadata)
     criterion = nn.MSELoss()
     optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
+    aux_weight = config.moe_aux_weight if config.use_moe else 0.0
     history = []
     start = time.perf_counter()
     # 5. Mỗi epoch: cập nhật LR → train → validation → lưu checkpoint nếu cải thiện.
     for epoch in range(config.epochs):
         lr = cosine_lr(optimizer, epoch, config.lr, config.epochs, config.warmup_epochs)
-        train_m = run_epoch(model, train_loader, criterion, optimizer, device)
+        epoch_start = time.perf_counter()
+        train_m = run_epoch(model, train_loader, criterion, optimizer, device, aux_weight)
+        train_epoch_seconds = time.perf_counter() - epoch_start
         val_m = run_epoch(model, val_loader, criterion, device=device)
         history.append({"epoch": epoch + 1, "lr": lr,
+                        "train_epoch_seconds": train_epoch_seconds,
+                        "epoch_seconds": time.perf_counter() - epoch_start,
                         **{f"train_{k}": v for k, v in train_m.items()},
                         **{f"val_{k}": v for k, v in val_m.items()}})
         pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
+        usage = ""
+        if config.use_moe:
+            usage = " | experts(train)=" + "/".join(
+                f"{train_m[f'expert_{i}']:.2f}" for i in range(config.num_experts))
+            usage += f" | aux={train_m['aux']:.4f}"
         print(f"Epoch {epoch + 1:3}/{config.epochs} | lr={lr:.2e} | "
-              f"train_loss={train_m['loss']:.6f} | val_loss={val_m['loss']:.6f}")
+              f"train_loss={train_m['loss']:.6f} | val_loss={val_m['loss']:.6f}{usage}")
         if early(val_m["loss"], model, epoch + 1):
             print(f"Early stopping at epoch {epoch + 1}.")
             break
@@ -105,10 +117,15 @@ def train(config):
     early.load_best(model)
     metrics = evaluate(model, test_loader, device)
     summary = print_summary(config, metrics, train_seconds)
-    summary.update(best_epoch=early.best_epoch, best_val_loss=early.best_loss,
+    summary.update(best_epoch=early.best_epoch, best_val_loss=early.best_loss, epochs_run=len(history),
+                   mean_epoch_seconds=float(np.mean([h["epoch_seconds"] for h in history])),
                    checkpoint=str(checkpoint_path), data_sha256=metadata["data_sha256"])
+    if config.use_moe:
+        summary["test_expert_fractions"] = expert_usage(model, test_loader, device)
+        print("Test expert usage: " + "/".join(f"{f:.3f}" for f in summary["test_expert_fractions"]))
     write_json(run_dir / "metrics.json", summary)
-    pd.DataFrame([summary]).to_csv(run_dir / "metrics.csv", index=False)
+    pd.DataFrame([{k: v for k, v in summary.items() if not isinstance(v, list)}]).to_csv(
+        run_dir / "metrics.csv", index=False)
     print(f"Checkpoint: {checkpoint_path}")
     return checkpoint_path, summary
```

**Nối μ, log σ từ RevIN vào IntraVBlock (`DyTimeNet.forward`)**

```diff
diff --git a/src/dytimenet/models/dytimenet.py b/src/dytimenet/models/dytimenet.py
index 14754c2..f1710f5 100644
--- a/src/dytimenet/models/dytimenet.py
+++ b/src/dytimenet/models/dytimenet.py
@@ -26,7 +26,8 @@ class DyTimeNet(nn.Module):
                  p1: float = 0.5, p2: float = 0.5,
                  ffn_ratio: float = 2.0, hidden_ratio: float = 0.75,
                  dropout: float = 0.1, affine_revin: bool = True,
-                 decomp_kernel: int = 25):
+                 decomp_kernel: int = 25, use_moe: bool = False,
+                 num_experts: int = 4, top_k: int = 1, moe_dispatch: str = "sparse"):
         super().__init__()
         assert d_model < seq_len, (
             f"d_model ({d_model}) phải < seq_len ({seq_len}) để downsampling."
@@ -37,6 +38,7 @@ class DyTimeNet(nn.Module):
         self.pred_len   = pred_len
         self.d_model    = d_model
         self.num_layers = num_layers
+        self.use_moe    = use_moe
 
         # ── RevIN ────────────────────────────────────────────────────────
         self.revin = RevIN(num_vars, affine=affine_revin)
@@ -60,7 +62,7 @@ class DyTimeNet(nn.Module):
             CrossVBlock(num_vars, d_model, p1, p2, hidden_ratio)
             for _ in range(num_layers)])
         self.intra_v_blocks = nn.ModuleList([
-            IntraVBlock(d_model, ffn_ratio, dropout)
+            IntraVBlock(d_model, ffn_ratio, dropout, use_moe, num_experts, top_k, moe_dispatch)
             for _ in range(num_layers)])
 
         # ── Projection heads ──────────────────────────────────────────────
@@ -79,8 +81,13 @@ class DyTimeNet(nn.Module):
     def forward(self, x: torch.Tensor) -> torch.Tensor:
         """x : [B, C, L] → [B, C, S]"""
 
-        # 1. Chuẩn hóa đầu vào
-        x = self.revin(x, "norm")                          # [B, C, L]
+        # 1. Chuẩn hóa đầu vào (MoE: giữ μ, σ của input gốc làm đặc trưng cho router)
+        if self.use_moe:
+            x, mu, sigma = self.revin(x, "norm", return_stats=True)
+            log_sigma = sigma.log()                         # mu, log_sigma: [B, C, 1]
+        else:
+            x = self.revin(x, "norm")                      # [B, C, L]
+            mu = log_sigma = None
 
         # 2. Tách trend và seasonal
         x_seasonal, x_trend = self.decomp(x)              # mỗi cái [B, C, L]
@@ -98,7 +105,7 @@ class DyTimeNet(nn.Module):
         for cross_v, intra_v in zip(self.cross_v_blocks, self.intra_v_blocks):
             # Preserve the notebook's outer residual in addition to CrossV's inner one.
             s = cross_v(s) + s
-            s = intra_v(s)
+            s = intra_v(s, mu, log_sigma)
 
         seasonal_out = self.projection(s)                  # [B, C, S]
 
@@ -109,5 +116,15 @@ class DyTimeNet(nn.Module):
         out = self.revin(out, "denorm")
         return out
 
+    def moe_aux_loss(self):
+        """Tổng load-balance loss của các IntraVMoE trong forward train gần nhất (None nếu không có)."""
+        losses = [b.moe.aux_loss for b in self.intra_v_blocks if b.use_moe and b.moe.aux_loss is not None]
+        return sum(losses) if losses else None
+
+    def moe_expert_counts(self):
+        """Số lượt gán token → expert của forward gần nhất, cộng qua các layer ([E] hoặc None)."""
+        counts = [b.moe.expert_counts for b in self.intra_v_blocks if b.use_moe]
+        return sum(counts) if counts else None
+
     def count_parameters(self) -> int:
         return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**Các cấu hình** (hidden = tổng độ rộng lớp ẩn mỗi token đi qua; *thực thi* gồm cả expert không được chọn mà dense vẫn tính):

| Tên | E | k | d_ff shared | d_ff mỗi expert | Dispatch | Hidden active | Hidden thực thi | Params |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `baseline` | — | — | 128 (FFN) | — | — (FFN) | 128 | 128 | 38,709 |
| `moe_k1_sparse` | 4 | 1 | 64 | 64 | sparse | 128 | 128 | 64,001 |
| `moe_k1_dense` | 4 | 1 | 64 | 64 | dense | 128 | 320 | 64,001 |
| `moe_k2_sparse` | 4 | 2 | 64 | 32 | sparse | 128 | 128 | 47,489 |
| `moe_lite` | 4 | 1 | 32 | 32 | sparse | 64 | 64 | 43,361 |

**Dense vs sparse** (`model.eval()`, cùng trọng số, toàn bộ tập test; yêu cầu sai khác < 1e-5). Ngoài ra `tests/test_moe.py::test_dense_and_sparse_dispatch_match_in_eval` (k=1, k=2, input ngẫu nhiên) nằm trong bộ test 41/41 pass.

| Trọng số | Số cửa sổ test | Sai khác tuyệt đối lớn nhất | < 1e-5 |
| --- | ---: | ---: | :---: |
| checkpoint `moe_k1_sparse_seed2021` (sparse → dense) | 2,785 | 0.000e+00 | ✅ |
| checkpoint `moe_k1_dense_seed2021` (dense → sparse) | 2,785 | 0.000e+00 | ✅ |
| checkpoint `moe_lite_seed2021` (sparse → dense) | 2,785 | 0.000e+00 | ✅ |
| k=2, trọng số khởi tạo ngẫu nhiên (seed 2021), chưa train | 2,785 | 0.000e+00 | ✅ |

## 3. Bảng kết quả tổng hợp

mean ± std qua các seed đã chạy (n=1 nghĩa là chỉ có 1 seed). FLOPs mỗi mẫu = 2 × MACs, đếm bằng thop trên các module thực sự chạy (*thực thi*); *active* = thực thi trừ phần expert không được chọn mà dense vẫn tính. Với sparse, active = thực thi.

**3a. Độ chính xác và chi phí**

| Cấu hình | MSE | MAE | Params | FLOPs active (M) | FLOPs thực thi (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.3919 ± 0.0050 | 0.4025 ± 0.0017 | 38,709 | 0.5519 | 0.5519 |
| `moe_k1_sparse` | 0.3895 ± 0.0056 | 0.4021 ± 0.0010 | 64,001 | 0.5556 | 0.5556 |
| `moe_k1_dense` ⚠️ | 0.3896 ± 0.0054 | 0.4022 ± 0.0011 | 64,001 | 0.5556 | 0.8997 |
| `moe_k2_sparse` ⚠️ | 0.3894 ± 0.0052 | 0.4038 ± 0.0025 | 47,489 | 0.5556 | 0.5556 |
| `moe_lite` ⚠️ | 0.3880 ± 0.0030 | 0.4006 ± 0.0009 | 43,361 | 0.4409 | 0.4409 |

**3b. Huấn luyện** (s/epoch = train + validation một epoch; số epoch = số epoch đã chạy đến khi early stop)

| Cấu hình | s/epoch | Số epoch | Best epoch | Tổng train (s) |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 1.194 ± 0.007 | 48.3 ± 11.0 | 38.3 ± 11.0 | 57.9 ± 13.1 |
| `moe_k1_sparse` | 2.585 ± 0.524 | 54.3 ± 11.2 | 44.3 ± 11.2 | 139.9 ± 39.7 |
| `moe_k1_dense` ⚠️ | 2.317 ± 0.574 | 54.3 ± 11.2 | 44.3 ± 11.2 | 122.2 ± 12.4 |
| `moe_k2_sparse` ⚠️ | 3.333 ± 0.021 | 36.7 ± 11.2 | 26.7 ± 11.2 | 122.4 ± 37.5 |
| `moe_lite` ⚠️ | 3.136 ± 0.115 | 58.3 ± 15.3 | 48.3 ± 15.3 | 184.3 ± 54.9 |

**3c. Latency suy luận (ms)**: median của 200 lần đo sau 20 lần warm-up, `eval()` + `no_grad()`, input là cửa sổ test thật. IntraV = riêng một lần gọi IntraVBlock (LN + FFN/MoE + residual) với input chụp từ forward thật. bX/tY = batch X, Y luồng.

| Cấu hình | Model b1/t8 | Model b64/t8 | Model b1/t1 | Model b64/t1 | IntraV b1/t8 | IntraV b64/t8 | IntraV b1/t1 | IntraV b64/t1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.613 ± 0.004 | 1.719 ± 0.006 | 0.583 ± 0.005 | 2.952 ± 0.038 | 0.079 ± 0.011 | 0.234 ± 0.003 | 0.080 ± 0.000 | 0.304 ± 0.005 |
| `moe_k1_sparse` | 1.782 ± 0.127 | 4.290 ± 0.045 | 1.592 ± 0.080 | 5.746 ± 0.082 | 0.896 ± 0.088 | 1.683 ± 0.016 | 0.665 ± 0.095 | 1.549 ± 0.274 |
| `moe_k1_dense` ⚠️ | 1.757 ± 0.723 | 4.120 ± 1.423 | 1.400 ± 0.406 | 6.675 ± 2.536 | 0.868 ± 0.371 | 1.495 ± 0.442 | 0.608 ± 0.252 | 1.874 ± 0.767 |
| `moe_k2_sparse` ⚠️ | 2.236 ± 0.295 | 4.810 ± 0.238 | 1.823 ± 0.086 | 7.495 ± 0.100 | 1.241 ± 0.243 | 1.904 ± 0.098 | 0.921 ± 0.184 | 1.952 ± 0.051 |
| `moe_lite` ⚠️ | 1.886 ± 0.194 | 4.457 ± 0.240 | 1.750 ± 0.087 | 6.814 ± 0.923 | 0.965 ± 0.157 | 1.871 ± 0.232 | 0.792 ± 0.062 | 1.521 ± 0.210 |

Tỉ trọng IntraV trong latency toàn model (mean qua seed):

| Cấu hình | b1/t8 | b64/t8 | b1/t1 | b64/t1 |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 13% | 14% | 14% | 10% |
| `moe_k1_sparse` | 50% | 39% | 42% | 27% |
| `moe_k1_dense` ⚠️ | 49% | 37% | 42% | 28% |
| `moe_k2_sparse` ⚠️ | 56% | 40% | 50% | 26% |
| `moe_lite` ⚠️ | 51% | 42% | 45% | 22% |

## 4. So sánh theo cặp với baseline (cùng seed)

% chênh lệch = (MoE − baseline) / baseline × 100, tính riêng từng seed rồi lấy mean và min/max. Dương = lớn hơn (MSE/MAE tệ hơn, thời gian chậm hơn). Tiêu chí đạt: mean ΔMSE ≤ +1% **và** mean Δlatency b64/t8 ≤ 0%.

**`moe_k1_sparse`**

| Seed | MSE | MAE | s/epoch | Latency b1/t8 | Latency b64/t8 | IntraV b64/t8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | +0.59% | -0.26% | +65.28% | +165.25% | +152.90% | +635.16% |
| 2022 | -0.07% | +0.10% | +143.81% | +200.16% | +146.02% | +606.46% |
| 2023 | -2.29% | -0.15% | +140.44% | +206.84% | +149.97% | +620.18% |
| **mean** | -0.59% | -0.10% | +116.51% | +190.75% | +149.63% | +620.60% |
| **min / max** | -2.29% / +0.59% | -0.26% / +0.10% | +65.28% / +143.81% | +165.25% / +206.84% | +146.02% / +152.90% | +606.46% / +635.16% |

**`moe_k1_dense` ⚠️**

| Seed | MSE | MAE | s/epoch | Latency b1/t8 | Latency b64/t8 | IntraV b64/t8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | +0.56% | -0.25% | +58.85% | +61.09% | +45.20% | +326.64% |
| 2022 ⚠️ | -0.06% | +0.07% | +74.88% | +298.93% | +194.46% | +657.31% |
| 2023 ⚠️ | -2.20% | -0.09% | +147.92% | +201.28% | +178.90% | +632.48% |
| **mean** | -0.57% | -0.09% | +93.88% | +187.10% | +139.52% | +538.81% |
| **min / max** | -2.20% / +0.56% | -0.25% / +0.07% | +58.85% / +147.92% | +61.09% / +298.93% | +45.20% / +194.46% | +326.64% / +657.31% |

**`moe_k2_sparse` ⚠️**

| Seed | MSE | MAE | s/epoch | Latency b1/t8 | Latency b64/t8 | IntraV b64/t8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 ⚠️ | +0.29% | +0.50% | +179.82% | +226.83% | +190.62% | +771.76% |
| 2022 ⚠️ | +0.82% | +0.81% | +178.94% | +321.31% | +185.51% | +679.50% |
| 2023 ⚠️ | -2.91% | -0.36% | +178.31% | +246.13% | +163.63% | +695.35% |
| **mean** | -0.60% | +0.32% | +179.02% | +264.76% | +179.92% | +715.54% |
| **min / max** | -2.91% / +0.82% | -0.36% / +0.81% | +178.31% / +179.82% | +226.83% / +321.31% | +163.63% / +190.62% | +679.50% / +771.76% |

**`moe_lite` ⚠️**

| Seed | MSE | MAE | s/epoch | Latency b1/t8 | Latency b64/t8 | IntraV b64/t8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 ⚠️ | -0.57% | -0.74% | +152.16% | +169.39% | +159.41% | +813.33% |
| 2022 ⚠️ | -0.16% | -0.30% | +173.93% | +230.66% | +173.03% | +679.82% |
| 2023 ⚠️ | -2.22% | -0.39% | +161.71% | +223.17% | +145.65% | +610.50% |
| **mean** | -0.98% | -0.48% | +162.60% | +207.74% | +159.36% | +701.22% |
| **min / max** | -2.22% / -0.16% | -0.74% / -0.30% | +152.16% / +173.93% | +169.39% / +230.66% | +145.65% / +173.03% | +610.50% / +813.33% |

**Tổng hợp tiêu chí** (ΔMSE ≤ +1% và Δlatency b64/t8 ≤ 0%)

| Cấu hình | mean ΔMSE | mean Δlatency b64/t8 | Kết quả |
| --- | ---: | ---: | --- |
| `moe_k1_sparse` | -0.59% | +149.63% | ❌ không đạt |
| `moe_k1_dense` ⚠️ | -0.57% | +139.52% | ❌ không đạt (latency bị nhiễu ⚠️, nhưng ngay cả seed sạch cũng chậm hơn baseline) |
| `moe_k2_sparse` ⚠️ | -0.60% | +179.92% | ❌ không đạt (latency bị nhiễu ⚠️) |
| `moe_lite` ⚠️ | -0.98% | +159.36% | ❌ không đạt (latency bị nhiễu ⚠️) |

## 5. Router

Tỉ lệ token trên test: model tốt nhất, `eval()` (không noise), mọi cửa sổ test × 7 kênh. Aux cuối = aux loss trung bình của epoch train cuối cùng (giá trị cân bằng hoàn hảo = 1.0). Với k=2, tỉ lệ tính theo lượt gán chia k, nên tổng = 1. Chỉ số expert **không** tương ứng giữa các seed (mỗi seed học router riêng).

**`moe_k1_sparse`**

| Seed | e0 | e1 | e2 | e3 | Aux cuối | Expert < 5% token |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2021 | 25.4% | 9.6% | 39.3% | 25.7% | 1.0007 | không có |
| 2022 | 22.7% | 12.3% | 25.0% | 40.0% | 1.0008 | không có |
| 2023 | 26.9% | 25.7% | 18.1% | 29.3% | 1.0009 | không có |

**`moe_k1_dense`**

| Seed | e0 | e1 | e2 | e3 | Aux cuối | Expert < 5% token |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2021 | 24.7% | 10.2% | 39.2% | 25.9% | 1.0007 | không có |
| 2022 | 24.0% | 9.5% | 26.5% | 40.0% | 1.0007 | không có |
| 2023 | 29.3% | 24.8% | 17.3% | 28.7% | 1.0009 | không có |

**`moe_k2_sparse`**

| Seed | e0 | e1 | e2 | e3 | Aux cuối | Expert < 5% token |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2021 | 23.4% | 32.1% | 16.1% | 28.5% | 1.0352 | không có |
| 2022 | 24.4% | 23.3% | 24.7% | 27.6% | 1.0129 | không có |
| 2023 | 22.4% | 25.6% | 27.8% | 24.2% | 0.9961 | không có |

**`moe_lite`**

| Seed | e0 | e1 | e2 | e3 | Aux cuối | Expert < 5% token |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2021 | 20.1% | 25.0% | 32.0% | 22.9% | 1.0007 | không có |
| 2022 | 28.9% | 19.4% | 23.6% | 28.1% | 1.0007 | không có |
| 2023 | 19.1% | 26.7% | 12.9% | 41.4% | 1.0008 | không có |

**Expert được chọn theo kênh và theo giờ (k=1)**: % token của mỗi hàng đi vào từng expert. Nhóm giờ theo giờ của bước cuối trong cửa sổ input. Biên logit = logit top-1 − logit top-2 trên test; so với std noise router khi train (1/E = 0.25).

`moe_k1_sparse` seed 2021 (2,785 cửa sổ test). Biên logit: mean 0.021, median 0.017; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | 27.0% | 6.0% | 19.1% | **47.9%** |
| HULL | 31.6% | 4.9% | **44.3%** | 19.2% |
| MUFL | 27.3% | 6.3% | 19.5% | **47.0%** |
| MULL | 30.0% | 4.7% | **47.8%** | 17.6% |
| LUFL | 31.4% | 8.7% | 20.1% | **39.7%** |
| LULL | 30.0% | 5.6% | **56.8%** | 7.6% |
| OT | 0.9% | 31.2% | **67.2%** | 0.7% |
| 0–5h | 11.6% | 7.6% | **46.0%** | 34.8% |
| 6–11h | **39.8%** | 4.1% | 31.1% | 25.0% |
| 12–17h | **30.2%** | 10.8% | 30.1% | 28.9% |
| 18–23h | 20.1% | 16.0% | **49.8%** | 14.1% |

`moe_k1_sparse` seed 2022 (2,785 cửa sổ test). Biên logit: mean 0.028, median 0.023; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | 14.5% | 16.1% | 17.3% | **52.0%** |
| HULL | 23.4% | 14.4% | 23.9% | **38.2%** |
| MUFL | 12.0% | 16.8% | 17.5% | **53.7%** |
| MULL | 28.2% | 8.5% | 28.3% | **35.0%** |
| LUFL | 26.6% | 10.4% | 20.4% | **42.5%** |
| LULL | 25.3% | 11.8% | 25.1% | **37.8%** |
| OT | 28.7% | 7.9% | **42.3%** | 21.0% |
| 0–5h | 17.1% | 24.8% | 26.2% | **31.9%** |
| 6–11h | 27.9% | 9.3% | 27.8% | **35.0%** |
| 12–17h | 29.0% | 6.5% | 22.9% | **41.6%** |
| 18–23h | 16.7% | 8.6% | 23.1% | **51.6%** |

`moe_k1_sparse` seed 2023 (2,785 cửa sổ test). Biên logit: mean 0.030, median 0.024; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | **45.0%** | 40.4% | 3.8% | 10.9% |
| HULL | 17.1% | 29.1% | 17.6% | **36.3%** |
| MUFL | **64.1%** | 16.9% | 17.4% | 1.5% |
| MULL | 32.5% | 10.9% | **48.1%** | 8.5% |
| LUFL | 24.7% | **45.6%** | 9.0% | 20.7% |
| LULL | 4.7% | 12.7% | 18.1% | **64.4%** |
| OT | 0.4% | 24.2% | 12.9% | **62.5%** |
| 0–5h | 28.6% | 23.3% | 18.8% | **29.3%** |
| 6–11h | 25.0% | 24.2% | 17.7% | **33.2%** |
| 12–17h | 26.7% | 22.7% | 19.5% | **31.1%** |
| 18–23h | 27.4% | **32.5%** | 16.5% | 23.5% |

`moe_k1_dense` seed 2021 (2,785 cửa sổ test). Biên logit: mean 0.021, median 0.017; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | 27.4% | 5.9% | 18.6% | **48.2%** |
| HULL | 30.4% | 7.2% | **42.3%** | 20.0% |
| MUFL | 27.5% | 6.0% | 18.4% | **48.1%** |
| MULL | 29.0% | 7.3% | **45.7%** | 18.1% |
| LUFL | 29.7% | 9.5% | 21.2% | **39.6%** |
| LULL | 27.8% | 7.9% | **57.3%** | 7.0% |
| OT | 1.2% | 27.5% | **70.9%** | 0.4% |
| 0–5h | 10.7% | 8.0% | **46.7%** | 34.6% |
| 6–11h | **39.5%** | 5.3% | 31.1% | 24.2% |
| 12–17h | 29.6% | 10.2% | 30.0% | **30.3%** |
| 18–23h | 19.1% | 17.2% | **49.1%** | 14.6% |

`moe_k1_dense` seed 2022 (2,785 cửa sổ test). Biên logit: mean 0.028, median 0.023; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | 15.5% | 13.0% | 19.8% | **51.7%** |
| HULL | 26.2% | 9.9% | 25.4% | **38.4%** |
| MUFL | 12.3% | 13.4% | 19.9% | **54.4%** |
| MULL | 29.5% | 4.8% | 30.4% | **35.2%** |
| LUFL | 28.0% | 8.4% | 22.0% | **41.5%** |
| LULL | 27.7% | 9.5% | 25.2% | **37.6%** |
| OT | 28.8% | 7.1% | **42.8%** | 21.3% |
| 0–5h | 20.0% | 18.1% | 30.2% | **31.7%** |
| 6–11h | 29.4% | 7.3% | 29.1% | **34.2%** |
| 12–17h | 29.0% | 5.3% | 23.5% | **42.3%** |
| 18–23h | 17.6% | 7.2% | 23.3% | **51.9%** |

`moe_k1_dense` seed 2023 (2,785 cửa sổ test). Biên logit: mean 0.029, median 0.023; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | **49.9%** | 34.2% | 5.0% | 10.8% |
| HULL | 18.1% | 27.0% | 18.9% | **36.1%** |
| MUFL | **67.3%** | 19.2% | 11.6% | 1.9% |
| MULL | 37.3% | 13.6% | **39.1%** | 9.9% |
| LUFL | 25.9% | **39.9%** | 15.1% | 19.1% |
| LULL | 6.0% | 11.0% | 23.0% | **59.9%** |
| OT | 0.3% | 28.4% | 8.4% | **62.9%** |
| 0–5h | **30.7%** | 22.4% | 18.1% | 28.7% |
| 6–11h | 28.9% | 23.1% | 14.8% | **33.2%** |
| 12–17h | 28.9% | 21.7% | 19.8% | **29.6%** |
| 18–23h | 28.5% | **31.9%** | 16.5% | 23.2% |

`moe_lite` seed 2021 (2,785 cửa sổ test). Biên logit: mean 0.025, median 0.020; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | 5.0% | 10.4% | **56.2%** | 28.4% |
| HULL | 13.8% | 24.3% | **32.7%** | 29.2% |
| MUFL | 6.0% | 10.7% | **54.0%** | 29.2% |
| MULL | 17.8% | 28.3% | **29.2%** | 24.7% |
| LUFL | 6.4% | 22.1% | **40.9%** | 30.6% |
| LULL | **36.7%** | 36.6% | 10.7% | 15.9% |
| OT | **55.4%** | 42.3% | 0.3% | 2.0% |
| 0–5h | 19.7% | 26.4% | 25.5% | **28.4%** |
| 6–11h | 19.5% | 22.4% | 24.1% | **34.0%** |
| 12–17h | 23.4% | 28.2% | **32.5%** | 16.0% |
| 18–23h | 18.0% | 22.9% | **46.0%** | 13.1% |

`moe_lite` seed 2022 (2,785 cửa sổ test). Biên logit: mean 0.024, median 0.019; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | **60.9%** | 14.4% | 17.4% | 7.3% |
| HULL | 20.3% | 21.9% | 22.5% | **35.3%** |
| MUFL | **62.8%** | 13.8% | 16.1% | 7.4% |
| MULL | 17.1% | 18.3% | 24.5% | **40.1%** |
| LUFL | **35.4%** | 20.2% | 29.0% | 15.4% |
| LULL | 5.1% | 30.9% | 28.3% | **35.7%** |
| OT | 0.6% | 16.4% | 27.4% | **55.5%** |
| 0–5h | 22.5% | 22.2% | 24.9% | **30.4%** |
| 6–11h | 27.8% | 22.9% | 16.6% | **32.7%** |
| 12–17h | **31.5%** | 17.8% | 21.4% | 29.2% |
| 18–23h | **33.8%** | 14.7% | 31.4% | 20.1% |

`moe_lite` seed 2023 (2,785 cửa sổ test). Biên logit: mean 0.028, median 0.022; 100.0% token có biên < 0.25.

| Kênh / giờ | e0 | e1 | e2 | e3 |
| --- | ---: | ---: | ---: | ---: |
| HUFL | 18.6% | 7.5% | 5.9% | **68.1%** |
| HULL | 9.6% | 19.7% | 21.6% | **49.1%** |
| MUFL | **49.3%** | 16.6% | 1.0% | 33.1% |
| MULL | **44.2%** | 17.0% | 3.2% | 35.5% |
| LUFL | 6.5% | 13.7% | 14.3% | **65.6%** |
| LULL | 4.3% | 24.1% | 34.4% | **37.2%** |
| OT | 1.0% | **88.3%** | 9.7% | 1.0% |
| 0–5h | 19.7% | 26.1% | 13.0% | **41.1%** |
| 6–11h | 26.1% | 25.7% | 13.6% | **34.6%** |
| 12–17h | 14.3% | 25.9% | 13.1% | **46.8%** |
| 18–23h | 16.1% | 29.1% | 11.8% | **42.9%** |

## 6. Nhận xét

1. **Độ chính xác:** cả 4 cấu hình MoE đều đạt tiêu chí MSE (mean ΔMSE từ −0,57% đến −0,98% so với baseline cùng seed). Nhưng theo từng seed, ΔMSE dao động từ −2,9% đến +0,8%, và std giữa các seed (~0,005) cùng cỡ với chênh lệch. Với 3 seed, chưa thể khẳng định MoE chính xác hơn.
2. **Tốc độ:** không cấu hình nào đạt tiêu chí. Ngay cả các run không chạy song song (moe_k1_sparse ×3) vẫn có latency b64/t8 chậm hơn baseline 146–153%. Riêng IntraV chậm hơn 7 lần, và tỉ trọng IntraV trong latency toàn model tăng từ ~14% lên ~40%.
3. **FLOPs không phản ánh latency ở quy mô này.** FLOPs active của k1/k2 chỉ hơn baseline 0,7%, moe_lite còn ít hơn 20%, nhưng tất cả đều chậm hơn. Mỗi mẫu chỉ có 7 token, nên chi phí của router, `topk`, `nonzero`/`index_add_` và nhiều phép nhân ma trận nhỏ lớn hơn nhiều so với phần tính toán tiết kiệm được.
4. **Dense và sparse cho output trùng khớp** (sai khác lớn nhất 0 trên toàn tập test). Dense seed 2021 chậm hơn baseline 45% ở b64/t8, sparse seed 2021 chậm hơn 153%. Tuy nhiên run sparse rơi vào giai đoạn máy chậm bất thường (dòng 8), nên chưa kết luận được sparse hay dense nhanh hơn.
5. **Router gần như không học được.** Biên logit top-1 − top-2 trên test có median ~0,02, và 100% token có biên nhỏ hơn std noise khi train (0,25). Vì vậy lúc train, noise quyết định việc chọn expert: tỉ lệ ~25% mỗi expert, aux ≈ 1,0007. Lúc eval (không noise) thì việc route lệch (9,6%–41,4%) theo một cách router chưa từng thấy khi train.
6. Không có expert nào nhận dưới 5% token. Có dấu hiệu chuyên biệt theo kênh: ở 7/9 run k=1, kênh OT dồn ≥ 55% token (tới 88%) vào một expert. Tuy nhiên do biên logit rất nhỏ, sự chuyên biệt này yếu. Nhóm giờ trong ngày ít ảnh hưởng đến việc route.
7. **Lỗi trong quá trình chạy (do mình):** tiến trình `bench_moe.py` được thông báo là đã dừng nhưng thực ra vẫn chạy. Vì vậy moe_lite (cùng một lần chạy thử 1 epoch lúc 09:21:42–09:22:01 UTC, không lưu kết quả) đã chạy song song với nó. Thời gian của moe_k1_dense seed 2022/2023, moe_k2_sparse ×3 và moe_lite ×3 bị nhiễu (⚠️).
8. **Bất thường chưa rõ nguyên nhân:** từ khoảng 09:12 đến 09:18 UTC, s/epoch của moe_k1_sparse tăng từ ~1,7 lên ~3,0 s, dù không có run nào chạy song song. Nguyên nhân nằm ngoài dữ liệu đã lưu.
9. `results/moe_vs_baseline_etth1.md` do tiến trình cũ tự sinh ra, chứa các số thời gian bị nhiễu nói trên và không có cờ cảnh báo. Hãy dùng báo cáo này thay cho file đó.
10. **Đề xuất:** đo lại latency cho cả 15 checkpoint trong một phiên sạch, chạy tuần tự (không cần train lại), và train lại tuần tự moe_k1_dense, moe_k2_sparse, moe_lite nếu cần số s/epoch tin cậy.

## 7. Phụ lục: JSON kết quả thô

Nguyên văn các file kết quả, theo đường dẫn nguồn (cũng lưu tại `results/moe_raw_etth1.json`). Lịch sử từng epoch (loss, thời gian, tỉ lệ expert khi train) nằm trong `results/moe_etth1_histories/` và `results/moe_lite/moe_etth1_histories/`. Đơn vị: latency (`lat_*`, `intra_*`) tính bằng ms; FLOPs là số phép tính mỗi mẫu; thời gian train tính bằng giây.

```json
{
  "results/moe_etth1_runs.json": {
    "baseline_seed2021": {
      "config": "baseline",
      "seed": 2021,
      "mse": 0.3936063030570799,
      "mae": 0.4042254621293429,
      "train_seconds": 43.226,
      "epochs_run": 36,
      "best_epoch": 26,
      "epoch_seconds": 1.1979225389012653,
      "train_epoch_seconds": 1.0585260888813839,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2021_20261015T090804095454Z\\best.pt",
      "params": 38709,
      "active_params": 38709,
      "thop_flops_model": 551936.0,
      "thop_flops_intra": 232960.0,
      "manual_flops_intra_active": 229376,
      "manual_flops_intra_executed": 229376,
      "flops_executed": 551936.0,
      "flops_active": 551936.0,
      "lat_b1_t8": 0.6176999886520207,
      "intra_b1_t8": 0.08599995635449886,
      "lat_b64_t8": 1.7111999914050102,
      "intra_b64_t8": 0.2313999575562775,
      "lat_b1_t1": 0.5892999470233917,
      "intra_b1_t1": 0.08040003012865782,
      "lat_b64_t1": 2.9584000585600734,
      "intra_b64_t1": 0.3089499659836292
    },
    "baseline_seed2022": {
      "config": "baseline",
      "seed": 2022,
      "mse": 0.3861692818105756,
      "mae": 0.400759685649478,
      "train_seconds": 61.93,
      "epochs_run": 52,
      "best_epoch": 42,
      "epoch_seconds": 1.1866684115375392,
      "train_epoch_seconds": 1.050537024998518,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2022_20261015T090850875613Z\\best.pt",
      "params": 38709,
      "active_params": 38709,
      "thop_flops_model": 551936.0,
      "thop_flops_intra": 232960.0,
      "manual_flops_intra_active": 229376,
      "manual_flops_intra_executed": 229376,
      "flops_executed": 551936.0,
      "flops_active": 551936.0,
      "lat_b1_t8": 0.610249990131706,
      "intra_b1_t8": 0.08435000199824572,
      "lat_b64_t8": 1.7235500272363424,
      "intra_b64_t8": 0.23684999905526638,
      "lat_b1_t1": 0.5784500972367823,
      "intra_b1_t1": 0.07950002327561378,
      "lat_b64_t1": 2.986249979585409,
      "intra_b64_t1": 0.29820005875080824
    },
    "baseline_seed2023": {
      "config": "baseline",
      "seed": 2023,
      "mse": 0.3957762738639619,
      "mae": 0.40262782606873093,
      "train_seconds": 68.516,
      "epochs_run": 57,
      "best_epoch": 47,
      "epoch_seconds": 1.1987280824544366,
      "train_epoch_seconds": 1.0594503473686545,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2023_20261015T090954753311Z\\best.pt",
      "params": 38709,
      "active_params": 38709,
      "thop_flops_model": 551936.0,
      "thop_flops_intra": 232960.0,
      "manual_flops_intra_active": 229376,
      "manual_flops_intra_executed": 229376,
      "flops_executed": 551936.0,
      "flops_active": 551936.0,
      "lat_b1_t8": 0.6117000011727214,
      "intra_b1_t8": 0.06564997602254152,
      "lat_b64_t8": 1.7208500066772103,
      "intra_b64_t8": 0.23244996555149555,
      "lat_b1_t1": 0.5827000131830573,
      "intra_b1_t1": 0.07964996621012688,
      "lat_b64_t1": 2.9112999327480793,
      "intra_b64_t1": 0.30519999563694
    },
    "moe_k1_sparse_seed2021": {
      "config": "moe_k1_sparse",
      "seed": 2021,
      "mse": 0.3959371453448729,
      "mae": 0.4031918645226121,
      "train_seconds": 113.038,
      "epochs_run": 57,
      "best_epoch": 47,
      "epoch_seconds": 1.9798775701804021,
      "train_epoch_seconds": 1.7709028403532054,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2021_20261015T091105174685Z\\best.pt",
      "train_expert_last": [
        0.2501860119047619,
        0.249323593073593,
        0.2507440476190476,
        0.2497463474025974
      ],
      "train_expert_max_any_epoch": 0.2648133116883117,
      "test_expert": [
        0.25442421133624005,
        0.09617850730956655,
        0.3925109002308284,
        0.256886381123365
      ],
      "params": 64001,
      "active_params": 39041,
      "thop_flops_model": 555632.0,
      "thop_flops_intra": 236656.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 233072,
      "flops_executed": 555632.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 1.6384500195272267,
      "intra_b1_t8": 0.7946999976411462,
      "lat_b64_t8": 4.327700007706881,
      "intra_b64_t8": 1.7011499730870128,
      "lat_b1_t1": 1.4991999487392604,
      "intra_b1_t1": 0.5578499403782189,
      "lat_b64_t1": 5.840199999511242,
      "intra_b64_t1": 1.865350001025945
    },
    "moe_k1_sparse_seed2022": {
      "config": "moe_k1_sparse",
      "seed": 2022,
      "mse": 0.3858922884751051,
      "mae": 0.40114219211162316,
      "train_seconds": 185.466,
      "epochs_run": 64,
      "best_epoch": 54,
      "epoch_seconds": 2.893216446867882,
      "train_epoch_seconds": 2.593561843752468,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2022_20261015T091303394361Z\\best.pt",
      "train_expert_last": [
        0.2509977002164502,
        0.2493743235930736,
        0.2491037608225108,
        0.2505242153679654
      ],
      "train_expert_max_any_epoch": 0.3007643398268398,
      "test_expert": [
        0.2270325724544755,
        0.12280071813285458,
        0.24975634778148242,
        0.4004103616311875
      ],
      "params": 64001,
      "active_params": 39041,
      "thop_flops_model": 555632.0,
      "thop_flops_intra": 236656.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 233072,
      "flops_executed": 555632.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 1.8316999776288867,
      "intra_b1_t8": 0.9428999619558454,
      "lat_b64_t8": 4.240199981722981,
      "intra_b64_t8": 1.6732499934732914,
      "lat_b1_t1": 1.6312499647028744,
      "intra_b1_t1": 0.6980999605730176,
      "lat_b64_t1": 5.693300045095384,
      "intra_b64_t1": 1.3908499968238175
    },
    "moe_k1_sparse_seed2023": {
      "config": "moe_k1_sparse",
      "seed": 2023,
      "mse": 0.3867017870638486,
      "mae": 0.40202335013307405,
      "train_seconds": 121.244,
      "epochs_run": 42,
      "best_epoch": 32,
      "epoch_seconds": 2.8822257071456296,
      "train_epoch_seconds": 2.5817935309522517,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2023_20261015T091614047628Z\\best.pt",
      "train_expert_last": [
        0.2493743235930736,
        0.2496787067099567,
        0.2506087662337662,
        0.2503382034632034
      ],
      "train_expert_max_any_epoch": 0.2691930465367965,
      "test_expert": [
        0.26914593485509103,
        0.256886381123365,
        0.1813285457809695,
        0.2926391382405745
      ],
      "params": 64001,
      "active_params": 39041,
      "thop_flops_model": 555632.0,
      "thop_flops_intra": 236656.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 233072,
      "flops_executed": 555632.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 1.8769500311464071,
      "intra_b1_t8": 0.950200017541647,
      "lat_b64_t8": 4.301549983210862,
      "intra_b64_t8": 1.674049999564886,
      "lat_b1_t1": 1.644149946514517,
      "intra_b1_t1": 0.7391500403173268,
      "lat_b64_t1": 5.705100018531084,
      "intra_b64_t1": 1.3913500006310642
    },
    "moe_k1_dense_seed2021": {
      "config": "moe_k1_dense",
      "seed": 2021,
      "mse": 0.39580073045228925,
      "mae": 0.4032329627812658,
      "train_seconds": 108.651,
      "epochs_run": 57,
      "best_epoch": 47,
      "epoch_seconds": 1.9028832508892377,
      "train_epoch_seconds": 1.7117563280820225,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2021_20261015T091820522186Z\\best.pt",
      "train_expert_last": [
        0.2501691017316017,
        0.2493743235930736,
        0.2507778679653679,
        0.2496787067099567
      ],
      "train_expert_max_any_epoch": 0.2648640422077922,
      "test_expert": [
        0.24714029238266222,
        0.10176968453449603,
        0.3918953577840472,
        0.25919466529879454
      ],
      "params": 64001,
      "active_params": 39041,
      "thop_flops_model": 899696.0,
      "thop_flops_intra": 580720.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 577136,
      "flops_executed": 899696.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 0.9950500098057091,
      "intra_b1_t8": 0.44110004091635346,
      "lat_b64_t8": 2.484699944034219,
      "intra_b64_t8": 0.9872500086203218,
      "lat_b1_t1": 0.9438000270165503,
      "intra_b1_t1": 0.32620003912597895,
      "lat_b64_t1": 3.7911500548943877,
      "intra_b64_t1": 1.0023500653915107
    },
    "moe_k1_dense_seed2022": {
      "config": "moe_k1_dense",
      "seed": 2022,
      "mse": 0.38594760900125896,
      "mae": 0.40103098022445527,
      "train_seconds": 133.035,
      "epochs_run": 64,
      "best_epoch": 54,
      "epoch_seconds": 2.0751914078045957,
      "train_epoch_seconds": 1.8733515874992008,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2022_20261015T092012345804Z\\best.pt",
      "train_expert_last": [
        0.2510315205627705,
        0.2493574134199134,
        0.2491375811688311,
        0.2504734848484848
      ],
      "train_expert_max_any_epoch": 0.3006966991341991,
      "test_expert": [
        0.24006155424467812,
        0.09453706078481662,
        0.26509361374711465,
        0.4003077712233906
      ],
      "params": 64001,
      "active_params": 39041,
      "thop_flops_model": 899696.0,
      "thop_flops_intra": 580720.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 577136,
      "flops_executed": 899696.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 2.4344500270672143,
      "intra_b1_t8": 1.1106000165455043,
      "lat_b64_t8": 5.07524999557063,
      "intra_b64_t8": 1.7936999793164432,
      "lat_b1_t1": 1.7200500587932765,
      "intra_b1_t1": 0.810599944088608,
      "lat_b64_t1": 8.559449983295053,
      "intra_b64_t1": 2.4437999818474054
    },
    "moe_k1_dense_seed2023": {
      "config": "moe_k1_dense",
      "seed": 2023,
      "mse": 0.38705041556212805,
      "mae": 0.40228237595335586,
      "train_seconds": 125.007,
      "epochs_run": 42,
      "best_epoch": 32,
      "epoch_seconds": 2.971892702384918,
      "train_epoch_seconds": 2.6751192190373936,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2023_20261015T092232062864Z\\best.pt",
      "train_expert_last": [
        0.2493743235930736,
        0.2496956168831169,
        0.2505580357142857,
        0.2503720238095238
      ],
      "train_expert_max_any_epoch": 0.2689732142857143,
      "test_expert": [
        0.2926391382405745,
        0.24780713003334187,
        0.17286483713772763,
        0.286688894588356
      ],
      "params": 64001,
      "active_params": 39041,
      "thop_flops_model": 899696.0,
      "thop_flops_intra": 580720.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 577136,
      "flops_executed": 899696.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 1.8429500050842762,
      "intra_b1_t8": 1.0517999762669206,
      "lat_b64_t8": 4.799499933142215,
      "intra_b64_t8": 1.7026500427164137,
      "lat_b1_t1": 1.5372000634670258,
      "intra_b1_t1": 0.6883500027470291,
      "lat_b64_t1": 7.67405005171895,
      "intra_b64_t1": 2.1766999852843583
    },
    "moe_k2_sparse_seed2021": {
      "config": "moe_k2_sparse",
      "seed": 2021,
      "mse": 0.39473583132189094,
      "mae": 0.4062413978619464,
      "train_seconds": 90.685,
      "epochs_run": 27,
      "best_epoch": 17,
      "epoch_seconds": 3.352055033328253,
      "train_epoch_seconds": 3.0081307407494426,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2021_20261015T092442999773Z\\best.pt",
      "train_expert_last": [
        0.2113010687229437,
        0.3014914772727273,
        0.1896729572510822,
        0.2975344967532468
      ],
      "train_expert_max_any_epoch": 0.3025737283549783,
      "test_expert": [
        0.23426519620415492,
        0.32056937676327263,
        0.16050269299820466,
        0.28466273403436776
      ],
      "params": 47489,
      "active_params": 39105,
      "thop_flops_model": 555632.0,
      "thop_flops_intra": 236656.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 233072,
      "flops_executed": 555632.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 2.018850005697459,
      "intra_b1_t8": 1.0217000381089747,
      "lat_b64_t8": 4.973099974449724,
      "intra_b64_t8": 2.0172499353066087,
      "lat_b1_t1": 1.732999982777983,
      "intra_b1_t1": 0.7077499758452177,
      "lat_b64_t1": 7.489450043067336,
      "intra_b64_t1": 1.9744000164791942
    },
    "moe_k2_sparse_seed2022": {
      "config": "moe_k2_sparse",
      "seed": 2022,
      "mse": 0.38933277496101615,
      "mae": 0.4040062296754368,
      "train_seconds": 112.764,
      "epochs_run": 34,
      "best_epoch": 24,
      "epoch_seconds": 3.3101095941040994,
      "train_epoch_seconds": 2.965684605868506,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2022_20261015T092619797964Z\\best.pt",
      "train_expert_last": [
        0.2457809117965368,
        0.2228507169913419,
        0.2374272862554112,
        0.2939410849567099
      ],
      "train_expert_max_any_epoch": 0.2994622564935065,
      "test_expert": [
        0.24360092331367017,
        0.2330597589125417,
        0.24703770197486535,
        0.2763016157989228
      ],
      "params": 47489,
      "active_params": 39105,
      "thop_flops_model": 555632.0,
      "thop_flops_intra": 236656.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 233072,
      "flops_executed": 555632.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 2.571050019469112,
      "intra_b1_t8": 1.1979500413872302,
      "lat_b64_t8": 4.920849984046072,
      "intra_b64_t8": 1.8462499720044434,
      "lat_b1_t1": 1.9050499540753663,
      "intra_b1_t1": 1.0293000377714634,
      "lat_b64_t1": 7.397999987006187,
      "intra_b64_t1": 1.893749984446913
    },
    "moe_k2_sparse_seed2023": {
      "config": "moe_k2_sparse",
      "seed": 2023,
      "mse": 0.38426430426551367,
      "mae": 0.4011634315044704,
      "train_seconds": 163.778,
      "epochs_run": 49,
      "best_epoch": 39,
      "epoch_seconds": 3.3361800653023685,
      "train_epoch_seconds": 2.987377797951922,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2023_20261015T092818611536Z\\best.pt",
      "train_expert_last": [
        0.2349584009740259,
        0.2521137716450216,
        0.2819179518398268,
        0.2310098755411255
      ],
      "train_expert_max_any_epoch": 0.3150872564935065,
      "test_expert": [
        0.22405745062836624,
        0.25609130546293923,
        0.27753270069248526,
        0.2423185432162093
      ],
      "params": 47489,
      "active_params": 39105,
      "thop_flops_model": 555632.0,
      "thop_flops_intra": 236656.0,
      "manual_flops_intra_active": 233072,
      "manual_flops_intra_executed": 233072,
      "flops_executed": 555632.0,
      "flops_active": 555632.0,
      "lat_b1_t8": 2.1172499982640147,
      "intra_b1_t8": 1.5020000282675028,
      "lat_b64_t8": 4.536650027148426,
      "intra_b64_t8": 1.8487999914214015,
      "lat_b1_t1": 1.8320500385016203,
      "intra_b1_t1": 1.0246500023640692,
      "lat_b64_t1": 7.597850053571165,
      "intra_b64_t1": 1.9880000036209822
    }
  },
  "results/moe_lite/moe_etth1_runs.json": {
    "moe_lite_seed2021": {
      "config": "moe_lite",
      "seed": 2021,
      "mse": 0.39135937745310034,
      "mae": 0.40121617539780785,
      "train_seconds": 148.28,
      "epochs_run": 49,
      "best_epoch": 39,
      "epoch_seconds": 3.0206586163329456,
      "train_epoch_seconds": 2.7088166938866585,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2021_20261015T092210362099Z\\best.pt",
      "train_expert_last": [
        0.2516233766233766,
        0.2500338203463203,
        0.2491714015151515,
        0.2491714015151515
      ],
      "train_expert_max_any_epoch": 0.2659970238095238,
      "test_expert": [
        0.20143626570915618,
        0.24980764298538086,
        0.3199794819184406,
        0.22877660938702232
      ],
      "params": 43361,
      "active_params": 30785,
      "thop_flops_model": 440944.0,
      "thop_flops_intra": 121968.0,
      "manual_flops_intra_active": 118384,
      "manual_flops_intra_executed": 118384,
      "flops_executed": 440944.0,
      "flops_active": 440944.0,
      "lat_b1_t8": 1.6640499816276133,
      "intra_b1_t8": 0.7859999896027148,
      "lat_b64_t8": 4.438999982085079,
      "intra_b64_t8": 2.113450027536601,
      "lat_b1_t1": 1.6927500255405903,
      "intra_b1_t1": 0.7257500546984375,
      "lat_b64_t1": 7.379299961030483,
      "intra_b64_t1": 1.6252999776042998
    },
    "moe_lite_seed2022": {
      "config": "moe_lite",
      "seed": 2022,
      "mse": 0.38553860180039806,
      "mae": 0.3995698030696951,
      "train_seconds": 247.507,
      "epochs_run": 76,
      "best_epoch": 66,
      "epoch_seconds": 3.250600934213971,
      "train_epoch_seconds": 2.909937815792785,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2022_20261015T092446143042Z\\best.pt",
      "train_expert_last": [
        0.2495434253246753,
        0.2505073051948052,
        0.2503212932900432,
        0.2496279761904762
      ],
      "train_expert_max_any_epoch": 0.2624966179653679,
      "test_expert": [
        0.2888432931520903,
        0.19430623236727365,
        0.23595793793280329,
        0.2808925365478328
      ],
      "params": 43361,
      "active_params": 30785,
      "thop_flops_model": 440944.0,
      "thop_flops_intra": 121968.0,
      "manual_flops_intra_active": 118384,
      "manual_flops_intra_executed": 118384,
      "flops_executed": 440944.0,
      "flops_active": 440944.0,
      "lat_b1_t8": 2.0178499980829656,
      "intra_b1_t8": 1.03055004728958,
      "lat_b64_t8": 4.705850034952164,
      "intra_b64_t8": 1.8470000359229743,
      "lat_b1_t1": 1.7084000282920897,
      "intra_b1_t1": 0.8476499351672828,
      "lat_b64_t1": 7.31309998081997,
      "intra_b64_t1": 1.658349996432662
    },
    "moe_lite_seed2023": {
      "config": "moe_lite",
      "seed": 2023,
      "mse": 0.38699376000964236,
      "mae": 0.401049227571145,
      "train_seconds": 157.157,
      "epochs_run": 50,
      "best_epoch": 40,
      "epoch_seconds": 3.1372390460036694,
      "train_epoch_seconds": 2.8023804639908487,
      "checkpoint": "checkpoints\\moe_bench\\ETTh1_L96_S96_seed2023_20261015T092859457531Z\\best.pt",
      "train_expert_last": [
        0.2516233766233766,
        0.2492897727272727,
        0.2496617965367965,
        0.2494250541125541
      ],
      "train_expert_max_any_epoch": 0.269243777056277,
      "test_expert": [
        0.19066427289048474,
        0.2670941266991536,
        0.12864837137727622,
        0.41359322903308543
      ],
      "params": 43361,
      "active_params": 30785,
      "thop_flops_model": 440944.0,
      "thop_flops_intra": 121968.0,
      "manual_flops_intra_active": 118384,
      "manual_flops_intra_executed": 118384,
      "flops_executed": 440944.0,
      "flops_active": 440944.0,
      "lat_b1_t8": 1.976850035134703,
      "intra_b1_t8": 1.0787499486468732,
      "lat_b64_t8": 4.227249999530613,
      "intra_b64_t8": 1.651550002861768,
      "lat_b1_t1": 1.8502500024624169,
      "intra_b1_t1": 0.8015500498004258,
      "lat_b64_t1": 5.7490000035613775,
      "intra_b64_t1": 1.2795999646186829
    }
  }
}
```
