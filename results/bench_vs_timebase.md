# DyTimeNet so với TimeBase/SparseTSF/baseline tuyến tính: latency CPU và MACs (cùng máy)

CPU AMD Ryzen 7 7840HS, PyTorch eager, `inference_mode`, model khởi tạo ngẫu nhiên (latency không phụ thuộc trọng số). Median 5 vòng xen kẽ. MACs đếm bằng `torch.utils.flop_counter` (FLOPs/2), mỗi mẫu. TimeBase: P = chu kỳ, R = 6, không loss trực giao (không ảnh hưởng suy luận). Gộp channel: ma trận riêng từng kênh (bộ nhớ O(C·L·(L+S))); gộp shared: Conv depthwise + Linear trend dùng chung (bộ nhớ O(L·S)).

## paper_C321_L720_S720 (P = 24, W = 24)

| Mô hình | Tham số | MACs/mẫu | Buffer (MB) | b1/t8 (ms) | b1/t1 (ms) | b12/t8 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeBase | 396 | 2.77M | 0.0 | 0.480 | 0.845 | 5.215 |
| SparseTSF | 925 | 12.71M | 0.0 | 1.056 | 2.732 | 20.067 |
| DLinear | 1,038,240 | 332.81M | 0.0 | 2.376 | 9.306 | 30.726 |
| RLinear | 519,762 | 166.41M | 0.0 | 1.578 | 7.658 | 22.296 |
| CycleNet/Linear | 526,824 | 166.41M | 0.0 | 1.759 | 8.229 | 22.912 |
| DyTimeNet (norm) | 785,142 | 211.80M | 0.0 | 3.369 | 12.763 | 40.728 |
| DyTimeNet-RCF | 792,846 | 211.80M | 0.0 | 2.918 | 10.217 | 34.950 |
| DyTimeNet (norm) — gộp channel | (như trên) | 377.52M | 2540.0 | 34.908 | 78.946 | 91.348 |
| DyTimeNet-RCF — gộp channel | (như trên) | 377.52M | 2582.4 | 34.963 | 78.336 | 93.784 |
| DyTimeNet (norm) — gộp shared | (như trên) | 211.80M | 2.9 | 3.294 | 12.791 | 38.400 |
| DyTimeNet-RCF — gộp shared | (như trên) | 211.80M | 44.3 | 2.642 | 9.964 | 31.516 |

## ours_C21_L96_S96 (P = 4, W = 144)

| Mô hình | Tham số | MACs/mẫu | Buffer (MB) | b1/t8 (ms) | b1/t1 (ms) | b64/t8 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeBase | 318 | 0.02M | 0.0 | 0.075 | 0.051 | 0.518 |
| SparseTSF | 581 | 0.06M | 0.0 | 0.176 | 0.131 | 0.547 |
| DLinear | 18,624 | 0.39M | 0.0 | 0.092 | 0.091 | 0.776 |
| RLinear | 9,354 | 0.19M | 0.0 | 0.149 | 0.120 | 0.642 |
| CycleNet/Linear | 12,336 | 0.19M | 0.0 | 0.188 | 0.152 | 0.824 |
| DyTimeNet (norm) | 39,363 | 0.84M | 0.0 | 0.506 | 0.492 | 2.310 |
| DyTimeNet-RCF | 42,387 | 0.84M | 0.0 | 0.635 | 0.491 | 2.074 |
| DyTimeNet (norm) — gộp channel | (như trên) | 1.03M | 3.0 | 0.349 | 0.299 | 1.645 |
| DyTimeNet-RCF — gộp channel | (như trên) | 1.03M | 5.2 | 0.363 | 0.324 | 1.709 |
| DyTimeNet (norm) — gộp shared | (như trên) | 0.84M | 0.0 | 0.507 | 0.422 | 2.339 |
| DyTimeNet-RCF — gộp shared | (như trên) | 0.84M | 2.3 | 0.474 | 0.364 | 1.841 |

## ours_C21_L96_S720 (P = 4, W = 144)

| Mô hình | Tham số | MACs/mẫu | Buffer (MB) | b1/t8 (ms) | b1/t1 (ms) | b64/t8 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeBase | 1,410 | 0.10M | 0.0 | 0.107 | 0.071 | 0.717 |
| SparseTSF | 4,325 | 0.37M | 0.0 | 0.207 | 0.155 | 0.896 |
| DLinear | 139,680 | 2.90M | 0.0 | 0.125 | 0.163 | 1.555 |
| RLinear | 69,882 | 1.45M | 0.0 | 0.142 | 0.163 | 1.260 |
| CycleNet/Linear | 72,864 | 1.45M | 0.0 | 0.230 | 0.217 | 1.696 |
| DyTimeNet (norm) | 140,451 | 2.94M | 0.0 | 0.641 | 0.551 | 4.631 |
| DyTimeNet-RCF | 143,475 | 2.94M | 0.0 | 0.684 | 0.578 | 4.683 |
| DyTimeNet (norm) — gộp channel | (như trên) | 3.13M | 12.6 | 0.396 | 0.426 | 3.742 |
| DyTimeNet-RCF — gộp channel | (như trên) | 3.13M | 22.0 | 0.429 | 0.455 | 4.138 |
| DyTimeNet (norm) — gộp shared | (như trên) | 2.94M | 0.3 | 0.548 | 0.480 | 3.967 |
| DyTimeNet-RCF — gộp shared | (như trên) | 2.94M | 9.7 | 0.509 | 0.439 | 4.019 |

