# Tốc độ trên cùng một máy: DyTimeBase so với TimeBase, baseline tuyến tính và SOTA

CPU: AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD; PyTorch 2.6.0+cpu, eager, `inference_mode`. Latency = median của 3 vòng đo xen kẽ (ms/lần gọi). bB/tT: batch B, T luồng. Train = một bước forward + backward + Adam (ms). Kiến trúc PatchTST/TimeMixer/iTransformer/TimeXer theo script chính thức của Time-Series-Library (cấu hình S = 96; PatchTST theo repo chính thức của PatchTST), seq_len/pred_len đổi theo cấu hình đo. MACs mỗi mẫu.

## weather_C21_L720_S96 (P = 4, individual = True)

| Mô hình | Tham số | MACs/mẫu | b1/t8 (ms) | b1/t1 (ms) | b32/t8 (ms) | train_b32 (ms) | b1/t8 so với DyTimeBase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DLinear | 138.4K | 2.90M | 0.127 | 0.280 | 2.029 | 4.449 | 0.31× |
| SparseTSF | 4.3K | 438.5K | 0.159 | 0.166 | 2.810 | 7.017 | 0.39× |
| TimeBase | 26.3K | 102.8K | 0.163 | 0.173 | 0.868 | 3.020 | 0.40× |
| RLinear | 69.3K | 1.45M | 0.337 | 0.324 | 1.518 | 4.166 | 0.82× |
| CycleNet/Linear | 72.2K | 1.45M | 0.373 | 0.370 | 2.667 | 7.993 | 0.91× |
| DyTimeBase | 47.0K | 551.7K | 0.410 | 0.336 | 1.862 | 8.479 | 1.00× |
| DyTimeNet (cũ) — đã gộp | (như trên) | 2.98M | 0.668 | 0.759 | 6.081 | — | 1.63× |
| DyTimeNet (cũ) | 139.2K | 2.98M | 0.716 | 0.822 | 6.164 | 18.048 | 1.74× |
| iTransformer | 5.15M | 130.33M | 4.322 | 5.630 | 35.261 | 76.916 | 10.54× |
| TimeXer | 2.12M | 567.73M | 5.055 | 14.484 | 206.921 | 614.340 | 12.32× |
| PatchTST | 1.51M | 900.91M | 18.421 | 58.290 | 713.811 | 2973.766 | 44.90× |
| TimeMixer | 4.74M | 1.68G | 20.100 | 48.368 | 718.688 | 1738.537 | 48.99× |

## weather_C21_L720_S720 (P = 4, individual = True)

| Mô hình | Tham số | MACs/mẫu | b1/t8 (ms) | b1/t1 (ms) | b32/t8 (ms) | train_b32 (ms) | b1/t8 so với DyTimeBase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeBase | 49.3K | 181.4K | 0.177 | 0.193 | 1.393 | 3.187 | 0.39× |
| SparseTSF | 32.4K | 2.80M | 0.186 | 0.227 | 4.068 | 8.169 | 0.41× |
| DLinear | 1.04M | 21.77M | 0.252 | 0.736 | 6.881 | 9.088 | 0.56× |
| RLinear | 519.2K | 10.89M | 0.400 | 0.563 | 4.557 | 8.805 | 0.88× |
| CycleNet/Linear | 522.1K | 10.89M | 0.434 | 0.649 | 5.297 | 12.357 | 0.96× |
| DyTimeBase | 69.9K | 630.3K | 0.453 | 0.369 | 2.457 | 6.725 | 1.00× |
| DyTimeNet (cũ) — đã gộp | (như trên) | 13.25M | 0.751 | 1.026 | 9.399 | — | 1.66× |
| DyTimeNet (cũ) | 629.7K | 13.25M | 0.807 | 1.098 | 9.668 | 18.269 | 1.78× |
| iTransformer | 5.47M | 138.32M | 3.954 | 10.526 | 37.901 | 83.948 | 8.74× |
| TimeXer | 9.46M | 722.04M | 6.558 | 28.095 | 237.051 | 698.644 | 14.49× |
| PatchTST | 8.69M | 1.05G | 19.527 | 66.428 | 914.724 | 2590.756 | 43.15× |
| TimeMixer | 5.58M | 1.96G | 21.599 | 89.895 | 787.413 | 1936.050 | 47.73× |

## electricity_C321_L720_S720 (P = 24, individual = False)

| Mô hình | Tham số | MACs/mẫu | b1/t8 (ms) | b1/t1 (ms) | b1/t8 so với DyTimeBase |
| --- | ---: | ---: | ---: | ---: | ---: |
| TimeBase | 0.4K | 2.77M | 0.763 | 0.957 | 0.54× |
| SparseTSF | 0.9K | 12.71M | 1.246 | 2.312 | 0.89× |
| DyTimeBase | 190.5K | 23.81M | 1.408 | 1.543 | 1.00× |
| RLinear | 519.8K | 166.41M | 2.018 | 6.161 | 1.43× |
| CycleNet/Linear | 526.8K | 166.41M | 2.285 | 6.589 | 1.62× |
| DLinear | 1.04M | 332.81M | 3.045 | 7.737 | 2.16× |
| DyTimeNet (cũ) — đã gộp | (như trên) | 211.80M | 4.086 | 10.377 | 2.90× |
| DyTimeNet (cũ) | 785.1K | 211.80M | 4.264 | 10.474 | 3.03× |
| iTransformer | 5.47M | 2.10G | 21.077 | 46.046 | 14.97× |
| PatchTST | 8.69M | 16.08G | 347.934 | 935.177 | 247.13× |
| TimeMixer | 5.58M | 29.97G | 355.239 | 792.368 | 252.32× |
| TimeXer | 28.02M | 103.15G | 864.490 | 2100.218 | 614.03× |

