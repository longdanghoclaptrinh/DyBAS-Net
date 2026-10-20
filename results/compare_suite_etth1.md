# ETTh1: DyTimeNet, DyTimeNet-RCF và baseline tuyến tính (cùng pipeline)

L = 96, Adam lr 5e-4, cosine, tối đa 100 epoch, patience 10, batch 64, seed 2021–2023, CPU 8 luồng. DyTimeNet dùng residual `norm`; RCF thay MovingAvg bằng chu kỳ học được (CycleNet). CycleNet/Linear = RevIN không affine + RCF + Linear L→S.

## 1. Độ chính xác (MSE, mean ± std qua 3 seed)

| Mô hình | S=96 | S=192 | S=336 | S=720 | AVG MSE | AVG MAE | epoch TB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DyTimeNet-RCF | 0.3839 ± 0.0013 | 0.4391 ± 0.0018 | 0.4825 ± 0.0035 | 0.4926 ± 0.0051 | 0.4495 ± 0.0021 | 0.4369 ± 0.0006 | 26.8 |
| DyTimeNet-RCF (RevIN không affine) | 0.3837 ± 0.0021 | 0.4393 ± 0.0017 | 0.4849 ± 0.0026 | 0.4940 ± 0.0054 | 0.4504 ± 0.0023 | 0.4376 ± 0.0013 | 27.3 |
| CycleNet/Linear | 0.3821 ± 0.0010 | 0.4291 ± 0.0034 | 0.4598 ± 0.0001 | 0.4618 ± 0.0004 | 0.4332 ± 0.0011 | 0.4265 ± 0.0003 | 54.9 |

Số tham số (S = 96): DyTimeNet-RCF 38,877, DyTimeNet-RCF (RevIN không affine) 38,863, CycleNet/Linear 9,480.

