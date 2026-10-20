# DyTimeNet so với TimeBase/SparseTSF/baseline tuyến tính: latency CPU và MACs (cùng máy)

CPU AMD Ryzen 7 7840HS, PyTorch eager, `inference_mode`, model khởi tạo ngẫu nhiên (latency không phụ thuộc trọng số). Median 5 vòng xen kẽ. MACs đếm bằng `torch.utils.flop_counter` (FLOPs/2), mỗi mẫu. TimeBase: P = chu kỳ, R = 6, không loss trực giao (không ảnh hưởng suy luận).

## paper_C321_L720_S720 (P = 24, W = 24)

| Mô hình | Tham số | MACs/mẫu | Buffer (MB) | b1/t8 (ms) | b1/t1 (ms) | b12/t8 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeBase | 396 | 2.77M | 0.0 | 0.505 | 0.840 | 5.247 |
| SparseTSF | 925 | 12.71M | 0.0 | 1.081 | 2.750 | 19.729 |
| DLinear | 1,038,240 | 332.81M | 0.0 | 2.443 | 9.244 | 30.458 |
| RLinear | 519,762 | 166.41M | 0.0 | 1.582 | 7.534 | 22.334 |
| CycleNet/Linear | 526,824 | 166.41M | 0.0 | 1.800 | 8.089 | 22.977 |
| DyTimeNet (norm) | 785,142 | 211.80M | 0.0 | 3.629 | 13.627 | 40.468 |
| DyTimeNet-RCF | 792,846 | 211.80M | 0.0 | 2.845 | 10.170 | 34.795 |
| DyTimeNet (norm) — đã gộp | (như trên) | 377.52M | 2540.0 | 35.453 | 77.420 | 86.813 |
| DyTimeNet-RCF — đã gộp | (như trên) | 377.52M | 2582.4 | 35.342 | 79.135 | 89.076 |

## ours_C21_L96_S96 (P = 4, W = 144)

| Mô hình | Tham số | MACs/mẫu | Buffer (MB) | b1/t8 (ms) | b1/t1 (ms) | b64/t8 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeBase | 318 | 0.02M | 0.0 | 0.074 | 0.051 | 0.519 |
| SparseTSF | 581 | 0.06M | 0.0 | 0.141 | 0.130 | 0.542 |
| DLinear | 18,624 | 0.39M | 0.0 | 0.093 | 0.091 | 0.753 |
| RLinear | 9,354 | 0.19M | 0.0 | 0.149 | 0.119 | 0.633 |
| CycleNet/Linear | 12,336 | 0.19M | 0.0 | 0.189 | 0.152 | 0.821 |
| DyTimeNet (norm) | 39,363 | 0.84M | 0.0 | 0.609 | 0.490 | 2.329 |
| DyTimeNet-RCF | 42,387 | 0.84M | 0.0 | 0.635 | 0.492 | 2.085 |
| DyTimeNet (norm) — đã gộp | (như trên) | 1.03M | 3.0 | 0.344 | 0.296 | 1.654 |
| DyTimeNet-RCF — đã gộp | (như trên) | 1.03M | 5.2 | 0.375 | 0.322 | 1.727 |

## ours_C21_L96_S720 (P = 4, W = 144)

| Mô hình | Tham số | MACs/mẫu | Buffer (MB) | b1/t8 (ms) | b1/t1 (ms) | b64/t8 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeBase | 1,410 | 0.10M | 0.0 | 0.106 | 0.072 | 0.722 |
| SparseTSF | 4,325 | 0.37M | 0.0 | 0.207 | 0.154 | 1.003 |
| DLinear | 139,680 | 2.90M | 0.0 | 0.123 | 0.163 | 1.538 |
| RLinear | 69,882 | 1.45M | 0.0 | 0.171 | 0.163 | 1.227 |
| CycleNet/Linear | 72,864 | 1.45M | 0.0 | 0.228 | 0.220 | 1.659 |
| DyTimeNet (norm) | 140,451 | 2.94M | 0.0 | 0.638 | 0.555 | 4.676 |
| DyTimeNet-RCF | 143,475 | 2.94M | 0.0 | 0.684 | 0.581 | 4.745 |
| DyTimeNet (norm) — đã gộp | (như trên) | 3.13M | 12.6 | 0.394 | 0.420 | 3.703 |
| DyTimeNet-RCF — đã gộp | (như trên) | 3.13M | 22.0 | 0.426 | 0.465 | 4.165 |

