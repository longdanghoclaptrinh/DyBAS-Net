# DyTimeNet: IntraVMoE so với baseline trên ETTh1

ETTh1, seq_len=96, pred_len=96, d_model=64, batch_size=64, lr=0.0005, p1=0.1, p2=0.1, epochs≤100, patience=10. Seed 2021, 2022, 2023; số liệu là mean ± std qua 3 seed.
Thiết bị: CPU (AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD), torch 2.6.0+cpu, `torch.set_num_threads(8)` khi train và đo; latency đo thêm với 1 luồng. Các cấu hình chạy tuần tự trên cùng máy.
Chia dữ liệu ETTh1 theo 12/4/4 tháng (8.640 / 2.880 / 2.880 dòng, như code hiện có), không phải 7:1:2. MSE/MAE trên thang StandardScaler.

## 1. Kích thước và FLOPs (mỗi mẫu)

FLOPs = 2 × MACs. *Thực thi* = thop đếm trên các module thực sự chạy; *active* = thực thi trừ phần expert không được chọn mà chế độ dense vẫn tính. IntraV (tay) chỉ đếm phép nhân ma trận của FFN/MoE (gồm router), để đối chiếu với thop.

| Cấu hình | Dispatch | Params | Params active | FLOPs active | FLOPs thực thi | IntraV FLOPs thop | IntraV FLOPs tay (active / thực thi) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (FFN) | — | 38,709 | 38,709 | 0.552 M | 0.552 M | 233.0 K | 229.4 K / 229.4 K |
| MoE E=4 k=1 | sparse | 64,001 | 39,041 | 0.556 M | 0.556 M | 236.7 K | 233.1 K / 233.1 K |
| MoE E=4 k=1 | dense | 64,001 | 39,041 | 0.556 M | 0.900 M | 580.7 K | 233.1 K / 577.1 K |
| MoE E=4 k=2 | sparse | 47,489 | 39,105 | 0.556 M | 0.556 M | 236.7 K | 233.1 K / 233.1 K |

## 2. Huấn luyện và độ chính xác

s/epoch = train + validation của một epoch. Số epoch = số epoch đã chạy đến khi early stop.

| Cấu hình | Dispatch | s/epoch | Số epoch | Best epoch | Tổng train (s) | MSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (FFN) | — | 1.19 ± 0.01 | 48.3 ± 11.0 | 38.3 ± 11.0 | 57.9 ± 13.1 | 0.3919 ± 0.0050 | 0.4025 ± 0.0017 |
| MoE E=4 k=1 | sparse | 2.59 ± 0.52 | 54.3 ± 11.2 | 44.3 ± 11.2 | 139.9 ± 39.7 | 0.3895 ± 0.0056 | 0.4021 ± 0.0010 |
| MoE E=4 k=1 | dense | 2.32 ± 0.57 | 54.3 ± 11.2 | 44.3 ± 11.2 | 122.2 ± 12.4 | 0.3896 ± 0.0054 | 0.4022 ± 0.0011 |
| MoE E=4 k=2 | sparse | 3.33 ± 0.02 | 36.7 ± 11.2 | 26.7 ± 11.2 | 122.4 ± 37.5 | 0.3894 ± 0.0052 | 0.4038 ± 0.0025 |

## 3. Latency suy luận (ms, median của 200 lần sau 20 lần warm-up, `eval()` + `no_grad()`)

| Cấu hình | Dispatch | Model b1 (8 luồng) | Model b64 (8 luồng) | Model b1 (1 luồng) | Model b64 (1 luồng) | IntraV b1 (8) | IntraV b64 (8) | IntraV b1 (1) | IntraV b64 (1) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (FFN) | — | 0.613 ± 0.004 | 1.719 ± 0.006 | 0.583 ± 0.005 | 2.952 ± 0.038 | 0.079 ± 0.011 | 0.234 ± 0.003 | 0.080 ± 0.000 | 0.304 ± 0.005 |
| MoE E=4 k=1 | sparse | 1.782 ± 0.127 | 4.290 ± 0.045 | 1.592 ± 0.080 | 5.746 ± 0.082 | 0.896 ± 0.088 | 1.683 ± 0.016 | 0.665 ± 0.095 | 1.549 ± 0.274 |
| MoE E=4 k=1 | dense | 1.757 ± 0.723 | 4.120 ± 1.423 | 1.400 ± 0.406 | 6.675 ± 2.536 | 0.868 ± 0.371 | 1.495 ± 0.442 | 0.608 ± 0.252 | 1.874 ± 0.767 |
| MoE E=4 k=2 | sparse | 2.236 ± 0.295 | 4.810 ± 0.238 | 1.823 ± 0.086 | 7.495 ± 0.100 | 1.241 ± 0.243 | 1.904 ± 0.098 | 0.921 ± 0.184 | 1.952 ± 0.051 |

## 4. Tỉ lệ token vào mỗi expert

Train = epoch cuối (có noise router). Test = model tốt nhất trên toàn tập test (không noise). Với k=2 tính theo lượt gán (chia k), nên tổng = 1. Cân bằng hoàn hảo là 0,25 mỗi expert. Lịch sử từng epoch nằm trong `results/moe_etth1_histories/`.

| Cấu hình | Dispatch | Train epoch cuối (e0/e1/e2/e3) | Test (e0/e1/e2/e3) | Max tỉ lệ 1 expert, mọi epoch |
| --- | --- | --- | --- | ---: |
| MoE E=4 k=1 | sparse | 0.250 ± 0.001 / 0.249 ± 0.000 / 0.250 ± 0.001 / 0.250 ± 0.000 | 0.250 ± 0.021 / 0.159 ± 0.086 / 0.275 ± 0.108 / 0.317 ± 0.075 | 0.301 |
| MoE E=4 k=1 | dense | 0.250 ± 0.001 / 0.249 ± 0.000 / 0.250 ± 0.001 / 0.250 ± 0.000 | 0.260 ± 0.029 / 0.148 ± 0.086 / 0.277 ± 0.110 / 0.315 ± 0.075 | 0.301 |
| MoE E=4 k=2 | sparse | 0.231 ± 0.018 / 0.259 ± 0.040 / 0.236 ± 0.046 / 0.274 ± 0.037 | 0.234 ± 0.010 / 0.270 ± 0.045 / 0.228 ± 0.061 / 0.268 ± 0.022 | 0.315 |

## 5. Thay đổi so với baseline (theo mean; dương = lớn hơn / chậm hơn)

| Cấu hình | Dispatch | s/epoch | Tổng train | Latency b1 (8) | Latency b64 (8) | Latency b1 (1) | Latency b64 (1) | MSE | MAE | MSE ≤ +1%? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| MoE E=4 k=1 | sparse | +116.4% | +141.7% | +190.7% | +149.6% | +172.8% | +94.7% | -0.6% | -0.1% | ✅ |
| MoE E=4 k=1 | dense | +94.0% | +111.1% | +186.6% | +139.7% | +140.0% | +126.1% | -0.6% | -0.1% | ✅ |
| MoE E=4 k=2 | sparse | +179.0% | +111.4% | +264.6% | +179.9% | +212.5% | +153.9% | -0.6% | +0.3% | ✅ |
