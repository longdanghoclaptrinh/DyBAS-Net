# ETTh1: DyTimeNet so với RLinear / DLinear trong cùng pipeline

Cùng split 12/4/4 tháng, StandardScaler, L = 96, Adam lr 5e-4, cosine, tối đa 100 epoch, patience 10, batch 64, `drop_last=False` ở test, seed 2021–2023, CPU 8 luồng. RLinear = RevIN + Linear L→S; DLinear = MovingAvg(25) + 2 Linear L→S (đều dùng chung cho mọi kênh).

## 1. Độ chính xác (MSE / MAE, mean ± std qua 3 seed)

| Mô hình | S=96 MSE | S=192 MSE | S=336 MSE | S=720 MSE | AVG MSE | AVG MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DyTimeNet double | 0.3919 ± 0.0050 | 0.4462 ± 0.0011 | 0.4918 ± 0.0093 | 0.5005 ± 0.0085 | 0.4576 ± 0.0017 | 0.4433 ± 0.0013 |
| DyTimeNet norm | 0.3839 ± 0.0014 | 0.4432 ± 0.0013 | 0.4851 ± 0.0106 | 0.5040 ± 0.0140 | 0.4541 ± 0.0049 | 0.4444 ± 0.0037 |
| RLinear | 0.3840 ± 0.0006 | 0.4344 ± 0.0010 | 0.4761 ± 0.0017 | 0.4768 ± 0.0002 | 0.4428 ± 0.0001 | 0.4304 ± 0.0001 |
| DLinear | 0.3846 ± 0.0009 | 0.4397 ± 0.0038 | 0.4835 ± 0.0034 | 0.5142 ± 0.0012 | 0.4555 ± 0.0010 | 0.4509 ± 0.0010 |

Số tham số (S = 96): DyTimeNet double 38,709, DyTimeNet norm 38,709, RLinear 9,326, DLinear 18,624.

## 2. Latency suy luận S = 96 (ms, median 5 vòng xen kẽ × 300 lần, eager, inference_mode)

| Mô hình | b1/t8 | b64/t8 | b1/t1 | b64/t1 |
| --- | ---: | ---: | ---: | ---: |
| DyTimeNet double | 0.742 | 2.291 | 0.374 | 4.022 |
| DyTimeNet norm | 0.733 | 2.130 | 0.360 | 4.143 |
| DyTimeNet norm (đã gộp) | 0.384 | 1.599 | 0.191 | 3.181 |
| RLinear | 0.153 | 0.422 | 0.075 | 1.514 |
| DLinear | 0.085 | 0.462 | 0.057 | 1.145 |

## 2. Latency suy luận S = 720 (ms, median 5 vòng xen kẽ × 300 lần, eager, inference_mode)

| Mô hình | b1/t8 | b64/t8 | b1/t1 | b64/t1 |
| --- | ---: | ---: | ---: | ---: |
| DyTimeNet double | 0.798 | 3.036 | 0.656 | 5.859 |
| DyTimeNet norm | 0.805 | 3.292 | 0.861 | 5.828 |
| DyTimeNet norm (đã gộp) | 0.448 | 2.400 | 0.383 | 4.461 |
| RLinear | 0.192 | 0.644 | 0.149 | 2.245 |
| DLinear | 0.136 | 0.798 | 0.147 | 2.448 |

## 3. Kết luận

- **Pipeline đáng tin:** RLinear trong pipeline này đạt 0,384 / 0,434 / 0,476 / 0,477, sát với số đã công bố trong Table 10 của iTransformer (0,386 / 0,437 / 0,479 / 0,481, cùng L = 96).
- **Trên ETTh1, RLinear thắng DyTimeNet cả về độ chính xác lẫn tốc độ.**
  - AVG MSE: 0,4428 so với 0,4541 (`norm`) và 0,4576 (`double`). Ở S = 96 thì `norm` ngang RLinear (0,3839 so với 0,3840), từ S = 192 trở lên RLinear tốt hơn rõ, nhất là S = 720 (0,477 so với 0,504).
  - Ngay cả bản đã gộp vẫn chậm hơn RLinear 2–4 lần.
  - Với 7 kênh, phần trộn kênh phi tuyến của DyTimeNet chưa đem lại lợi ích.
- **Bản gộp chính xác nhanh hơn chính DyTimeNet 21–48%** ở cả S = 96 và S = 720 (cùng lần đo xen kẽ), với MSE không đổi.
- **Lưu ý về số đo:** latency tuyệt đối của lần đo này cao hơn các lần trước (máy ở trạng thái khác). Chỉ nên so sánh giữa các dòng trong cùng một bảng.
- **Việc cần làm tiếp:** phải kiểm tra DyTimeNet có thắng RLinear ở dataset nhiều kênh không (Weather 21 kênh, Electricity, Traffic). Nếu không thắng, câu chuyện paper không thể là "DyTimeNet nhanh và chính xác", mà phải là "phép gộp chính xác dùng làm plug-in cho các mô hình lai".
