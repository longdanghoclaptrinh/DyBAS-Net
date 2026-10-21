# Hiệu năng trên máy này (kiểu Table 3 của TimeBase): Electricity, L = S = 720

CPU AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD, PyTorch 2.6.0+cpu, eager. Max Mem: đỉnh bộ nhớ tiến trình khi train batch 12 (CPU RAM, không phải GPU). Epoch Time: bước train batch 12 đo thật × 1414 bước. Infer Time: 1 mẫu. MACs mỗi mẫu (FFT không được đếm).

| Mô hình | Params | MACs | Max Mem (MB) | Epoch Time (s) | Infer t8 (ms) | Infer t1 (ms) | Ghi chú |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Informer | 12.45M | 7.91G | 2052.20 | 2926.21 | 66.16 | 187.97 |  |
| Autoformer | 12.14M | 8.97G | 5056.55 | 5302.34 | 83.39 | 270.43 |  |
| FEDformer | 18.44M | 8.99G | 3289.15 | 5367.20 | 135.93 | 310.97 |  |
| TimesNet | 151.32M | 1081.75G | 4078.06 | 170982.88 | 3449.01 | 13928.58 |  |
| PatchTST | 8.69M | 16.08G | 10346.23 | 82399.99 | 285.51 | 932.27 |  |
| DLinear | 1.04M | 332.81M | 149.52 | 54.69 | 1.93 | 7.81 |  |
| FITS | 133.64K | 42.73M | 209.99 | 78.29 | 1.70 | 7.63 |  |
| iTransformer | 5.47M | 2.10G | 996.75 | 695.44 | 10.66 | 46.37 |  |
| TimeMixer | 5.58M | 29.97G | > 11264 | 16716.10 | 247.46 | 793.30 | train batch 12 vượt 11 GB RAM; Epoch Time đo ở batch 4 × 4243 bước |
| SparseTSF | 0.93K | 12.71M | 275.82 | 56.09 | 0.76 | 2.38 |  |
| TimeBase | 0.40K | 2.77M | 117.16 | 20.45 | 0.34 | 1.03 |  |
| DyTimeNet (cũ) | 785.14K | 211.80M | 285.32 | 131.23 | 2.14 | 10.54 |  |
| DyTimeBase | 190.51K | 23.81M | 146.79 | 38.19 | 0.66 | 1.58 |  |
