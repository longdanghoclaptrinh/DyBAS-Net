# ETTh1: DyTimeNet-RCF so với DyTimeNet, RLinear, DLinear, CycleNet/Linear (cùng pipeline)

Cùng split 12/4/4 tháng, StandardScaler, L = 96, Adam lr 5e-4, cosine, tối đa 100 epoch, patience 10, batch 64, `drop_last=False` ở test, seed 2021–2023, CPU 8 luồng. RLinear = RevIN + Linear L→S; DLinear = MovingAvg(25) + 2 Linear L→S; CycleNet/Linear = RevIN (không affine) + RCF (W = 24) + Linear L→S. DyTimeNet-RCF = DyTimeNet residual `norm` với MovingAvg thay bằng RCF (W = 24).

## 1. Độ chính xác (MSE / MAE, mean ± std qua 3 seed)

| Mô hình | S=96 MSE | S=192 MSE | S=336 MSE | S=720 MSE | AVG MSE | AVG MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DyTimeNet norm | 0.3839 ± 0.0014 | 0.4432 ± 0.0013 | 0.4851 ± 0.0106 | 0.5040 ± 0.0140 | 0.4541 ± 0.0049 | 0.4444 ± 0.0037 |
| DyTimeNet-RCF | 0.3839 ± 0.0013 | 0.4391 ± 0.0018 | 0.4825 ± 0.0035 | 0.4926 ± 0.0051 | 0.4495 ± 0.0021 | 0.4369 ± 0.0006 |
| RLinear | 0.3840 ± 0.0006 | 0.4344 ± 0.0010 | 0.4761 ± 0.0017 | 0.4768 ± 0.0002 | 0.4428 ± 0.0001 | 0.4304 ± 0.0001 |
| DLinear | 0.3846 ± 0.0009 | 0.4397 ± 0.0038 | 0.4835 ± 0.0034 | 0.5142 ± 0.0012 | 0.4555 ± 0.0010 | 0.4509 ± 0.0010 |
| CycleNet/Linear | 0.3821 ± 0.0010 | 0.4291 ± 0.0034 | 0.4598 ± 0.0001 | 0.4618 ± 0.0004 | 0.4332 ± 0.0011 | 0.4265 ± 0.0003 |

Số tham số (S = 96): DyTimeNet norm 38,709, DyTimeNet-RCF 38,877, RLinear 9,326, DLinear 18,624, CycleNet/Linear 9,480.

## 2. Latency suy luận S = 96 (ms, median 5 vòng xen kẽ × 300 lần, eager, inference_mode)

| Mô hình | b1/t8 | b64/t8 | b1/t1 | b64/t1 |
| --- | ---: | ---: | ---: | ---: |
| DyTimeNet norm | 0.924 | 2.597 | 0.704 | 4.777 |
| DyTimeNet norm (đã gộp) | 0.475 | 1.920 | 0.369 | 3.527 |
| DyTimeNet-RCF | 0.962 | 2.537 | 0.726 | 4.252 |
| DyTimeNet-RCF (đã gộp) | 0.523 | 2.004 | 0.408 | 3.664 |
| RLinear | 0.185 | 0.511 | 0.139 | 1.702 |
| DLinear | 0.133 | 0.548 | 0.114 | 1.307 |
| CycleNet/Linear | 0.245 | 0.752 | 0.196 | 2.136 |

## 2. Latency suy luận S = 720 (ms, median 5 vòng xen kẽ × 300 lần, eager, inference_mode)

| Mô hình | b1/t8 | b64/t8 | b1/t1 | b64/t1 |
| --- | ---: | ---: | ---: | ---: |
| DyTimeNet norm | 0.950 | 3.148 | 0.780 | 6.581 |
| DyTimeNet norm (đã gộp) | 0.522 | 2.377 | 0.451 | 5.208 |
| DyTimeNet-RCF | 1.029 | 3.439 | 0.863 | 7.457 |
| DyTimeNet-RCF (đã gộp) | 0.578 | 2.554 | 0.493 | 5.662 |
| RLinear | 0.205 | 0.776 | 0.183 | 2.724 |
| DLinear | 0.164 | 0.949 | 0.177 | 2.951 |
| CycleNet/Linear | 0.289 | 1.325 | 0.251 | 4.352 |

## 3. Kết luận

- **RCF cải thiện DyTimeNet, nhưng ít.** AVG MSE 0,4541 → 0,4495 (−1,0%), AVG MAE 0,4444 → 0,4369 (−1,7%). Rõ nhất ở S = 720 (0,504 → 0,493), và độ lệch giữa các seed nhỏ hơn.
- **CycleNet/Linear tốt nhất trên ETTh1:** AVG MSE 0,4332, khớp với số đã công bố (0,432 ở L = 96), nên pipeline đáng tin. Mô hình này thắng DyTimeNet-RCF 3,6%, và nhanh hơn bản gộp của DyTimeNet-RCF khoảng 2 lần.
- **DyTimeNet-RCF tương đương CycleNet/Linear cộng thêm nhánh phi tuyến (CrossV + IntraV).** Việc thêm nhánh này làm MSE *tăng*. Một biến số chưa được tách riêng: DyTimeNet-RCF dùng RevIN có affine, còn CycleNet/Linear thì không.
- **Về tốc độ:** RCF không làm mô hình nhanh hơn. Bản gộp của DyTimeNet-RCF chậm hơn bản gộp không RCF 5–10%, do phải tra hai bảng theo pha. So với DyTimeNet-RCF chưa gộp, bản gộp vẫn nhanh hơn 21–46%.
- Latency tuyệt đối của lần đo này cao hơn các lần trước. Chỉ nên so sánh giữa các dòng trong cùng một bảng.
