# Dữ liệu

Thư mục này chứa sẵn 9 bộ dữ liệu dùng trong thí nghiệm, nên không cần tải thêm.

| File | Bộ dữ liệu | Nguồn gốc |
| --- | --- | --- |
| `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv` | ETT | [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset), bản benchmark của [Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library) |
| `weather.csv` | Weather | [Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/weather) |
| `electricity.csv` | Electricity | [Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/electricity) |
| `tetouan.csv` | Tetouan | [UCI #849 – Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849) |
| `panama.csv` | Panama | [Mendeley Data byx7sztj59](https://data.mendeley.com/datasets/byx7sztj59) |
| `load_new.csv` | Load | Phụ tải điện Hà Nội theo giờ (01/2020–05/2025), kèm dữ liệu thời tiết |

`tetouan.csv` và `panama.csv` được tạo từ file gốc (`powerconsumption.csv`, `continuous dataset.csv`) bằng
`python scripts/prepare_load_data.py`: đổi tên cột thời gian thành `date`, và với Panama bỏ 3 cột lịch dạng phân loại.

## Cách loader xử lý dữ liệu

- Bỏ các cột ngày/giờ không phải số, giữ mọi cột số còn lại theo thứ tự trong CSV, và dự báo tất cả các biến.
- ETTh chia 8 640 / 2 880 / 2 880 dòng, ETTm chia 34 560 / 11 520 / 11 520 dòng. Các bộ còn lại chia 70% / 10% / 20%
  theo thời gian.
- StandardScaler chỉ fit trên tập train. Sai số được tính trên thang đã chuẩn hóa.
- Checkpoint lưu SHA-256 của CSV, và `evaluate.py` yêu cầu đúng file đó. Không chỉnh sửa các file trong thư mục này.
