# Chuẩn bị dữ liệu

Repo không kèm file CSV. Tải các bộ cần dùng rồi đặt vào thư mục này:

```text
data/
├── ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
├── weather.csv  electricity.csv  traffic.csv
├── tetouan.csv  panama.csv          (tạo bằng scripts/prepare_load_data.py)
└── README.md
```

## Bộ benchmark

Lấy từ bản đã chuẩn bị sẵn của Time-Series-Library trên
[Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main). Đoạn Python sau, chạy từ gốc repo,
tải các file còn thiếu:

```python
from pathlib import Path
from urllib.request import urlretrieve

base = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"
files = {f"{n}.csv": f"ETT-small/{n}.csv" for n in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")}
files |= {"weather.csv": "weather/weather.csv", "electricity.csv": "electricity/electricity.csv",
          "traffic.csv": "traffic/traffic.csv"}
for local, remote in files.items():
    dest = Path("data") / local
    if not dest.exists():
        urlretrieve(f"{base}/{remote}?download=true", dest)
```

## Bộ phụ tải công khai

1. **Tetouan:** tải `powerconsumption.csv` từ
   [UCI #849 – Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849).
2. **Panama:** tải `continuous dataset.csv` từ
   [Mendeley Data byx7sztj59](https://data.mendeley.com/datasets/byx7sztj59).
3. Đặt hai file vào `data/` rồi chạy `python scripts/prepare_load_data.py`. Script ghi ra `data/tetouan.csv` và
   `data/panama.csv`, và bỏ qua những bộ chưa có file gốc.

## Bộ Load (phụ tải Hà Nội)

Đây là số liệu phụ tải điện Hà Nội do Tổng Công ty Điện lực Việt Nam thu thập (01/2020–05/2025, mỗi giờ một lần, kèm
dữ liệu thời tiết). Bộ này **không công khai** nên không có trong repo. Nếu được cấp, đặt file vào
`configs/load_new.csv`.

## Cách loader xử lý dữ liệu

- Bỏ các cột ngày/giờ không phải số, giữ mọi cột số còn lại theo thứ tự trong CSV, và dự báo tất cả các biến.
- ETTh chia 8 640 / 2 880 / 2 880 dòng, ETTm chia 34 560 / 11 520 / 11 520 dòng. Các bộ còn lại chia 70% / 10% / 20%
  theo thời gian.
- StandardScaler chỉ fit trên tập train. Sai số được tính trên thang đã chuẩn hóa.
- Checkpoint lưu SHA-256 của CSV, và `evaluate.py` yêu cầu đúng file đó.
