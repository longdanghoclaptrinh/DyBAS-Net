# Weather: DyTimeNet, DyTimeNet-RCF và baseline tuyến tính (cùng pipeline)

L = 96, Adam lr 5e-4, cosine, tối đa 100 epoch, patience 10, batch 64, seed 2021–2023, CPU 8 luồng. DyTimeNet dùng residual `norm`; RCF thay MovingAvg bằng chu kỳ học được (CycleNet). CycleNet/Linear = RevIN không affine + RCF + Linear L→S.

## 1. Độ chính xác (MSE, mean ± std qua 3 seed)

| Mô hình | S=96 | S=192 | S=336 | S=720 | AVG MSE | AVG MAE | epoch TB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DyTimeNet (norm) | 0.1550 ± 0.0004 | 0.2044 ± 0.0016 | 0.2677 ± 0.0023 | 0.3439 ± 0.0058 | 0.2427 ± 0.0009 | 0.2719 ± 0.0008 | 43.3 |
| DyTimeNet-RCF | 0.1536 ± 0.0015 | 0.2032 ± 0.0024 | 0.2611 ± 0.0012 | 0.3467 ± 0.0039 | 0.2411 ± 0.0020 | 0.2707 ± 0.0007 | 39.2 |
| RLinear | 0.1910 ± 0.0001 | 0.2357 ± 0.0004 | 0.2868 ± 0.0005 | 0.3586 ± 0.0006 | 0.2680 ± 0.0002 | 0.2879 ± 0.0001 | 27.1 |
| CycleNet/Linear | 0.1777 ± 0.0003 | 0.2276 ± 0.0004 | 0.2792 ± 0.0003 | 0.3528 ± 0.0002 | 0.2593 ± 0.0002 | 0.2813 ± 0.0001 | 36.4 |

Số tham số (S = 96): DyTimeNet (norm) 39,363, DyTimeNet-RCF 42,387, RLinear 9,354, CycleNet/Linear 12,336.

## 2. Latency suy luận S = 96 (ms, median 5 vòng xen kẽ × 300 lần, eager, inference_mode)

| Mô hình | b1/t8 | b64/t8 | b1/t1 | b64/t1 |
| --- | ---: | ---: | ---: | ---: |
| DyTimeNet (norm) | 0.541 | 2.097 | 0.466 | 6.732 |
| DyTimeNet (norm) — đã gộp | 0.282 | 1.376 | 0.286 | 4.971 |
| DyTimeNet-RCF | 0.528 | 1.956 | 0.457 | 5.578 |
| DyTimeNet-RCF — đã gộp | 0.285 | 1.442 | 0.318 | 5.152 |
| RLinear | 0.107 | 0.528 | 0.107 | 2.684 |
| CycleNet/Linear | 0.134 | 0.684 | 0.136 | 3.031 |

## 3. Latency suy luận S = 720 (ms, median 5 vòng xen kẽ × 300 lần, eager, inference_mode)

| Mô hình | b1/t8 | b64/t8 | b1/t1 | b64/t1 |
| --- | ---: | ---: | ---: | ---: |
| DyTimeNet (norm) | 0.524 | 4.489 | 0.517 | 10.488 |
| DyTimeNet (norm) — đã gộp | 0.381 | 3.606 | 0.511 | 9.006 |
| DyTimeNet-RCF | 0.520 | 4.482 | 0.530 | 11.026 |
| DyTimeNet-RCF — đã gộp | 0.392 | 4.015 | 0.490 | 9.859 |
| RLinear | 0.128 | 1.304 | 0.142 | 5.019 |
| CycleNet/Linear | 0.202 | 1.725 | 0.191 | 6.656 |

## 4. Nhận xét

- **Trên Weather (21 kênh), nhánh phi tuyến của DyTimeNet có ích rõ rệt**, ngược hẳn với ETTh1 (7 kênh). AVG MSE: DyTimeNet-RCF 0.2411 so với CycleNet/Linear 0.2593 (−7.0%) và RLinear 0.2680 (−10.0%). Khoảng cách lớn nhất ở S = 96 (0.1536 so với 0.1777, −13.6%) và nhỏ dần khi horizon tăng (S = 720: 0.3467 so với 0.3528, −1.7%). Mọi khoảng cách đều lớn hơn nhiều lần std giữa các seed.
- **RCF trong DyTimeNet chỉ giúp ít:** 0.2411 so với 0.2427 (−0.7%), tốt hơn ở S = 96/192/336 nhưng kém hơn ở S = 720. RCF giúp rõ ở mô hình tuyến tính (RLinear → CycleNet/Linear: −3.2%).
- **Gộp chính xác vẫn giữ tác dụng** ở 21 kênh: S = 96, b1/t8 nhanh hơn khoảng 48%, b64/t8 nhanh hơn khoảng 26–34%. Bản đã gộp vẫn chậm hơn RLinear/CycleNet/Linear khoảng 2–2.6 lần ở b1. Đây là cái giá của phần trộn kênh và FFN, đổi lại MSE giảm 7–10%.
- **Độ chính xác số của bản gộp:** Weather có cửa sổ hằng số (σ = 0, ví dụ lượng mưa = 0; 6.6% cặp cửa sổ–kênh có σ < 1e-3). Bản gộp nay trừ μ trước khi nhân với [M; K] và khớp tham chiếu float64 tới 7e-6. Bản gốc chạy float32 lệch tới 0.027 so với float64 trên các cửa sổ này, do RevIN chia cho σ ≈ 1e-5. MSE toàn tập test của bản gốc và bản gộp trùng nhau tới 4–5 chữ số.
- Latency giữa hai lần đo khác nhau khoảng 10–20% do trạng thái máy. Chỉ nên so sánh các số trong cùng một bảng.
