# Residual quanh CrossVBlock: double vs single vs norm (ETTh1)

- `double`: `s ← cross_v(s) + s = 2s + f(LN s)` — code hiện tại (giữ từ notebook).
- `single`: `s ← s + f(LN s)` — Fig. 5 của paper.
- `norm`: `s ← LN(s) + f(LN s)` — eq. 2.14 trong báo cáo đồ án, eq. 25 của paper.

Mọi thứ khác giữ nguyên `configs/ETTh1.yaml` (seq_len=96, d_model=64, batch=64, lr=0.0005, p1=p2=0.1, tối đa 100 epoch, patience=10). CPU, `torch.set_num_threads(8)`, chạy tuần tự, seed 2021, 2022, 2023. MSE/MAE trên tập test, thang StandardScaler. Ba biến thể có cùng số tham số.

## 1. MSE / MAE (mean ± std qua 3 seed)

| pred_len | double MSE | single MSE | norm MSE | double MAE | single MAE | norm MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.3919 ± 0.0050 | 0.3887 ± 0.0058 | 0.3839 ± 0.0014 | 0.4025 ± 0.0017 | 0.4019 ± 0.0036 | 0.4009 ± 0.0002 |
| 192 | 0.4462 ± 0.0011 | 0.4439 ± 0.0040 | 0.4432 ± 0.0013 | 0.4331 ± 0.0017 | 0.4334 ± 0.0030 | 0.4348 ± 0.0006 |
| 336 | 0.4918 ± 0.0093 | 0.4882 ± 0.0156 | 0.4851 ± 0.0106 | 0.4568 ± 0.0041 | 0.4567 ± 0.0092 | 0.4581 ± 0.0058 |
| 720 | 0.5005 ± 0.0085 | 0.5037 ± 0.0094 | 0.5040 ± 0.0140 | 0.4807 ± 0.0107 | 0.4852 ± 0.0109 | 0.4839 ± 0.0098 |
| **AVG 4 horizon** | 0.4576 ± 0.0017 | 0.4562 ± 0.0041 | 0.4541 ± 0.0049 | 0.4433 ± 0.0013 | 0.4443 ± 0.0017 | 0.4444 ± 0.0037 |

## 2. Chênh lệch theo cặp so với `double` (cùng horizon, cùng seed)

Δ = (biến thể − double) / double × 100. Âm = tốt hơn `double`. Cột "thắng" = số cặp (horizon, seed) mà biến thể có MSE thấp hơn `double`.

| Biến thể | pred_len | ΔMSE mean | ΔMSE min / max | ΔMAE mean | Thắng (MSE) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single` | 96 | -0.79% | -1.68% / +0.19% | -0.15% | 2/3 |
| `single` | 192 | -0.50% | -1.37% / +0.28% | +0.07% | 2/3 |
| `single` | 336 | -0.75% | -2.63% / +0.69% | -0.02% | 2/3 |
| `single` | 720 | +0.66% | +0.38% / +0.98% | +0.93% | 0/3 |
| `single` | **tất cả** | **-0.35%** | -2.63% / +0.98% | | **6/12** |
| `norm` | 96 | -2.02% | -2.90% / -0.97% | -0.41% | 3/3 |
| `norm` | 192 | -0.66% | -0.84% / -0.40% | +0.40% | 3/3 |
| `norm` | 336 | -1.34% | -4.86% / +1.34% | +0.30% | 2/3 |
| `norm` | 720 | +0.70% | -0.34% / +1.88% | +0.67% | 1/3 |
| `norm` | **tất cả** | **-0.83%** | -4.86% / +1.88% | | **9/12** |

## 3. Huấn luyện (mean ± std qua 3 seed)

| pred_len | double epoch (best) | single epoch (best) | norm epoch (best) | double train (s) | single train (s) | norm train (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 38.3 ± 11.0 | 38.3 ± 11.0 | 52.7 ± 15.1 | 77.6 ± 15.7 | 75.8 ± 12.0 | 97.9 ± 21.5 |
| 192 | 23.0 ± 7.2 | 21.7 ± 5.0 | 24.7 ± 8.6 | 53.3 ± 11.4 | 50.4 ± 7.8 | 55.1 ± 14.0 |
| 336 | 12.7 ± 3.1 | 18.3 ± 9.7 | 20.0 ± 14.8 | 38.6 ± 5.4 | 47.7 ± 16.2 | 50.1 ± 24.9 |
| 720 | 6.0 ± 0.0 | 6.0 ± 0.0 | 10.7 ± 7.4 | 28.8 ± 1.3 | 28.3 ± 0.3 | 36.5 ± 13.4 |

## 4. Số liệu từng run

| Run | MSE | MAE | best epoch | epoch đã chạy |
| --- | ---: | ---: | ---: | ---: |
| `double_S96_seed2021` | 0.3936 | 0.4042 | 26 | 36 |
| `single_S96_seed2021` | 0.3944 | 0.4060 | 26 | 36 |
| `norm_S96_seed2021` | 0.3851 | 0.4009 | 46 | 56 |
| `double_S96_seed2022` | 0.3862 | 0.4008 | 42 | 52 |
| `single_S96_seed2022` | 0.3828 | 0.3991 | 42 | 52 |
| `norm_S96_seed2022` | 0.3824 | 0.4007 | 42 | 52 |
| `double_S96_seed2023` | 0.3958 | 0.4026 | 47 | 57 |
| `single_S96_seed2023` | 0.3891 | 0.4007 | 47 | 57 |
| `norm_S96_seed2023` | 0.3843 | 0.4011 | 70 | 80 |
| `double_S192_seed2021` | 0.4456 | 0.4322 | 31 | 41 |
| `single_S192_seed2021` | 0.4468 | 0.4339 | 27 | 37 |
| `norm_S192_seed2021` | 0.4418 | 0.4341 | 23 | 33 |
| `double_S192_seed2022` | 0.4455 | 0.4319 | 21 | 31 |
| `single_S192_seed2022` | 0.4394 | 0.4302 | 21 | 31 |
| `norm_S192_seed2022` | 0.4437 | 0.4352 | 34 | 44 |
| `double_S192_seed2023` | 0.4475 | 0.4351 | 17 | 27 |
| `single_S192_seed2023` | 0.4456 | 0.4360 | 17 | 27 |
| `norm_S192_seed2023` | 0.4442 | 0.4350 | 17 | 27 |
| `double_S336_seed2021` | 0.5017 | 0.4604 | 16 | 26 |
| `single_S336_seed2021` | 0.5001 | 0.4600 | 16 | 26 |
| `norm_S336_seed2021` | 0.4773 | 0.4524 | 13 | 23 |
| `double_S336_seed2022` | 0.4833 | 0.4523 | 12 | 22 |
| `single_S336_seed2022` | 0.4706 | 0.4464 | 29 | 39 |
| `norm_S336_seed2022` | 0.4808 | 0.4580 | 37 | 47 |
| `double_S336_seed2023` | 0.4905 | 0.4577 | 10 | 20 |
| `single_S336_seed2023` | 0.4939 | 0.4638 | 10 | 20 |
| `norm_S336_seed2023` | 0.4971 | 0.4640 | 10 | 20 |
| `double_S720_seed2021` | 0.4913 | 0.4706 | 6 | 16 |
| `single_S720_seed2021` | 0.4943 | 0.4746 | 6 | 16 |
| `norm_S720_seed2021` | 0.4896 | 0.4733 | 8 | 18 |
| `double_S720_seed2022` | 0.5080 | 0.4918 | 6 | 16 |
| `single_S720_seed2022` | 0.5130 | 0.4965 | 6 | 16 |
| `norm_S720_seed2022` | 0.5175 | 0.4928 | 19 | 29 |
| `double_S720_seed2023` | 0.5021 | 0.4798 | 6 | 16 |
| `single_S720_seed2023` | 0.5040 | 0.4844 | 6 | 16 |
| `norm_S720_seed2023` | 0.5049 | 0.4856 | 5 | 15 |

## 5. Kết luận

**Nguồn gốc của `double`:** không phải lựa chọn thiết kế có ghi lại. Code cũ chỉ chú thích "giữ residual ngoài của notebook". Paper (Fig. 5, eq. 25) và báo cáo đồ án (eq. 2.13–2.14) đều chỉ có **một** residual. Trong báo cáo, `X_cross` là đầu ra MLP thuần, không kèm residual, và `X_intra = X_cross + X'_enc`.

**Kết quả trên ETTh1 (12 cặp horizon × seed mỗi biến thể):**

- **`single` ≈ `double`.** ΔMSE trung bình −0,35%, chỉ thắng 6/12 cặp, ngang mức ngẫu nhiên. Bỏ residual ngoài không làm mô hình tệ đi.
- **`norm` (đúng công thức trong báo cáo) nhỉnh hơn một chút về MSE.** ΔMSE trung bình −0,83%, thắng 9/12 cặp. Rõ nhất ở horizon 96 (−2,0%, thắng 3/3) và 192 (−0,7%, thắng 3/3). Ở 720 thì tệ hơn (+0,7%), và MAE tệ hơn nhẹ ở 192/336/720 (+0,3% đến +0,7%).
- **Chênh lệch chưa có ý nghĩa thống kê.** Nếu `norm` và `double` thật sự ngang nhau, xác suất thắng từ 9/12 cặp trở lên do ngẫu nhiên là 7,3% (kiểm định dấu một phía). Các chênh lệch cũng cùng cỡ với std giữa các seed (0,001–0,014).
- `norm` train lâu hơn: best epoch ở horizon 96 là 52,7 so với 38,3 của `double`.

**Khuyến nghị:** residual thừa không mang lại lợi ích nào đo được, nên có thể bỏ. Nếu muốn code khớp với công thức trong báo cáo đồ án, dùng `--cross_residual norm`: độ chính xác không kém, MSE nhỉnh hơn nhẹ ở horizon ngắn. Tuy nhiên, chỉ với ETTh1 và 3 seed thì chưa đủ để khẳng định `norm` tốt hơn. Mặc định trong code vẫn để `double` để không làm thay đổi các kết quả cũ.

**Kiểm tra tái lập:** 3 run `double` ở horizon 96 trùng hoàn toàn (MSE, MAE, best epoch) với 3 run baseline đã chạy trước đó trên branch `feat/moe-topk`.
