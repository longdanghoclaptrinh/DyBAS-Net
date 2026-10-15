# Chuẩn bị dữ liệu

Repo không kèm CSV. Tải riêng 5 bộ dữ liệu rồi đặt trực tiếp trong thư mục này:

```text
data/
├── README.md
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
└── weather.csv
```

Nguồn ETT gốc: [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset).
Bản benchmark được THUML cung cấp trên Hugging Face:
[ETT-small](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/ETT-small)
và [Weather](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/weather).
Weather dùng bản CSV đã chuẩn bị của benchmark để thống nhất tên cột và khoảng thời gian.

| Dataset | Tệp tải trực tiếp |
| --- | --- |
| ETTh1 | [ETTh1.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh1.csv?download=true) |
| ETTh2 | [ETTh2.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh2.csv?download=true) |
| ETTm1 | [ETTm1.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm1.csv?download=true) |
| ETTm2 | [ETTm2.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm2.csv?download=true) |
| Weather | [weather.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv?download=true) |

Có thể tải bằng trình duyệt hoặc chạy đoạn Python sau từ gốc repo (chỉ tải khi bạn chạy):

```python
from pathlib import Path
from urllib.request import urlretrieve

root = Path("data")
root.mkdir(exist_ok=True)
base = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"
files = {f"{name}.csv": f"ETT-small/{name}.csv"
         for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")}
files["weather.csv"] = "weather/weather.csv"
for local, remote in files.items():
    destination = root / local
    if not destination.exists():
        urlretrieve(f"{base}/{remote}?download=true", destination)
```

Giữ nguyên CSV theo thứ tự thời gian, không xáo trộn các dòng hay thay giá trị thiếu
âm thầm. Loader bỏ cột có tên chứa `date`, `time`, `timestamp` và dùng các cột số
còn lại theo thứ tự trong CSV. ETT có 7 biến; Weather benchmark có 21 biến.
Mặc định dự báo mọi biến, bao gồm `OT` trên ETT.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| ETTh1 / ETTh2 | 8.640 | 2.880 | 2.880 |
| ETTm1 / ETTm2 | 34.560 | 11.520 | 11.520 |
| Weather | `int(0.7 * N)` | `int(0.1 * N)` | phần còn lại |

ETT dùng 12/4/4 tháng quy ước 30 ngày như notebook; các dòng sau mốc test không
tham gia thí nghiệm. Validation và test lấy thêm `seq_len` dòng quá khứ làm đầu vào,
nhưng toàn bộ nhãn dự báo vẫn nằm trong split tương ứng. StandardScaler chỉ fit
trên train và được dùng lại ở validation/test. Sai số mặc định tính trên thang
StandardScaler, không phải đơn vị đo ban đầu sau inverse StandardScaler.

Dữ liệu ở nơi khác có thể được chỉ định bằng `--data_path "đường/dẫn/ETTm2.csv"`.
Tên dataset trong config quyết định cách chia ngay cả khi CSV được đổi tên.
Checkpoint lưu SHA-256 của CSV; evaluate yêu cầu cùng nội dung để giữ nguyên thí nghiệm.
