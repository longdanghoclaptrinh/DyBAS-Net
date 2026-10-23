# Hiệu năng trên máy này (kiểu Table 3 của TimeBase): Electricity, L = S = 720

CPU AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD, PyTorch 2.6.0+cpu, eager. Max Mem: đỉnh bộ nhớ tiến trình khi train batch 12 (CPU RAM, không phải GPU). Epoch Time: bước train batch 12 đo thật × 1414 bước. Infer Time: 1 mẫu. MACs mỗi mẫu (FFT không được đếm).

| Mô hình | Params | MACs | Max Mem (MB) | Epoch Time (s) | Infer t8 (ms) | Infer t1 (ms) | Ghi chú |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Informer | 12.45M | 7.91G | 2053.00 | 2819.07 | 53.54 | 176.13 |  |
| Autoformer | 12.14M | 8.97G | 5056.06 | 5579.04 | 74.30 | 257.63 |  |
| FEDformer | 18.44M | 8.99G | 3295.10 | 5079.54 | 115.75 | 301.83 |  |
| TimesNet | 151.32M | 1081.75G | 4093.46 | 147659.83 | 3072.66 | 13700.21 |  |
| PatchTST | 8.69M | 16.08G | 10949.17 | 76888.19 | 278.86 | 928.74 |  |
| DLinear | 1.04M | 332.81M | 149.58 | 57.54 | 2.13 | 7.88 | trung vị 5 vòng xen kẽ |
| FITS | 133.64K | 42.73M | 209.59 | 81.49 | 1.72 | 7.48 | trung vị 5 vòng xen kẽ |
| iTransformer | 5.47M | 2.10G | 996.44 | 746.68 | 14.47 | 47.30 |  |
| TimeMixer | 5.58M | 29.97G | 10942.14 | 25556.66 | 284.72 | 812.01 |  |
| SparseTSF | 0.93K | 12.71M | 275.23 | 54.15 | 0.77 | 2.39 | trung vị 5 vòng xen kẽ |
| TimeBase | 0.40K | 2.77M | 116.86 | 23.55 | 0.44 | 1.01 | trung vị 5 vòng xen kẽ |
| DyTimeNet (cũ) | 785.14K | 211.80M | 285.48 | 136.27 | 2.88 | 10.63 | trung vị 5 vòng xen kẽ |
| DyTimeBase | 190.51K | 23.81M | 146.46 | 40.06 | 0.87 | 1.63 | trung vị 5 vòng xen kẽ |
| CycleNet/Linear | 573.05K | 166.41M | 152.64 | 73.74 | 1.47 | 6.71 | trung vị 5 vòng xen kẽ |
| CycleNet/MLP | 792.44K | 236.67M | 162.02 | 92.46 | 1.84 | 7.93 | trung vị 5 vòng xen kẽ |
| TQNet | 3.39M | 1.22G | 416.90 | 308.60 | 7.14 | 25.04 |  |
