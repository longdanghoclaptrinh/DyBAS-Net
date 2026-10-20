# ETTh1, L = 720: DyTimeNet, DyTimeNet-RCF và baseline (cùng pipeline)

L = 720, Adam lr 5e-4, cosine, tối đa 100 epoch, patience 10, batch 64, seed 2021–2023, CPU 8 luồng. DyTimeNet dùng residual `norm`; RCF thay MovingAvg bằng chu kỳ học được (CycleNet). CycleNet/Linear = RevIN không affine + RCF + Linear L→S.

## 1. Độ chính xác (MSE, mean ± std qua 3 seed)

| Mô hình | S=96 | S=192 | S=336 | S=720 | AVG MSE | AVG MAE | epoch TB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DyTimeNet-RCF | 0.4042 ± 0.0104 | 0.4494 ± 0.0110 | 0.4900 ± 0.0109 | 0.5074 ± 0.0021 | 0.4627 ± 0.0036 | 0.4644 ± 0.0033 | 17.0 |
| CycleNet/Linear | 0.3831 ± 0.0035 | 0.4184 ± 0.0053 | 0.4474 ± 0.0011 | 0.4765 ± 0.0038 | 0.4314 ± 0.0019 | 0.4408 ± 0.0011 | 23.3 |
| TimeBase (lr chọn theo val) | 0.3637 ± 0.0033 | 0.4103 ± 0.0014 | 0.4379 ± 0.0104 | 0.4507 ± 0.0009 | 0.4156 ± 0.0036 | 0.4223 ± 0.0026 | 18.1 |
| DyTimeBase (lr chọn theo val) | 0.3678 ± 0.0017 | 0.4041 ± 0.0012 | 0.4365 ± 0.0058 | 0.4784 ± 0.0094 | 0.4217 ± 0.0019 | 0.4322 ± 0.0026 | 13.2 |
| TimeBase (siêu tham số chính thức) | 0.3575 ± 0.0013 | 0.3963 ± 0.0026 | 0.4226 ± 0.0006 | 0.4656 ± 0.0080 | 0.4105 ± 0.0028 | 0.4199 ± 0.0016 | 12.6 |
| DyTimeBase (siêu tham số TimeBase) | 0.3474 ± 0.0014 | 0.3882 ± 0.0014 | 0.4204 ± 0.0051 | 0.4558 ± 0.0213 | 0.4030 ± 0.0065 | 0.4208 ± 0.0039 | 14.7 |

TimeBase (lr chọn theo val): P theo paper TimeBase, R = 6, không loss trực giao; lr backbone chọn theo val trong {0.005, 0.02, 0.08}: S=96: 0.08, S=192: 0.02, S=336: 0.08, S=720: 0.02.

DyTimeBase (lr chọn theo val): P theo paper TimeBase, R = 6, không loss trực giao; lr backbone chọn theo val trong {0.005, 0.02, 0.08}, mix_lr = 5e-4: S=96: 0.005, S=192: 0.02, S=336: 0.02, S=720: 0.08.

Số tham số (S = 96): DyTimeNet-RCF 138,717, CycleNet/Linear 69,384, TimeBase (lr chọn theo val) 214, DyTimeBase (lr chọn theo val) 35,768, TimeBase (siêu tham số chính thức) 214, DyTimeBase (siêu tham số TimeBase) 35,768.

