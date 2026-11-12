# DyBAS-Net

**DyBAS-Net** (*Dynamic cross-variable dependency on BASis segments*) là mô hình dự báo chuỗi thời gian đa biến dài hạn,
viết bằng PyTorch. Mô hình gồm hai phần:

- **Backbone TimeBase** (ICML 2025): cắt chuỗi thành các đoạn dài một chu kỳ, nén $N$ đoạn quá khứ thành $R$ đoạn cơ
  sở rồi dự báo theo đoạn. Phần này rất nhẹ và xử lý từng biến độc lập.
- **Nhánh trộn biến** lấy từ DyTimeNet (Neurocomputing 2026), gồm CrossVBlock, DropFusion và IntraVBlock, chạy trên
  $P\cdot R$ hệ số cơ sở của mỗi biến thay vì $L$ điểm gốc. Lớp ra của nhánh này khởi tạo bằng 0, nên lúc bắt đầu
  huấn luyện mô hình đúng bằng TimeBase.

Mô hình được huấn luyện hai giai đoạn. Giai đoạn 1 huấn luyện TimeBase. Giai đoạn 2 đóng băng backbone và chỉ huấn
luyện nhánh trộn biến.

Trong mã nguồn, mô hình có tên `DyTimeBase` (`src/dytimenet/models/dytimebase.py`, model `dytimebase`); slide và paper
gọi là DyBAS-Net.

## Cài đặt

Dùng Python 3.10–3.13 và chạy mọi lệnh từ **thư mục gốc repo**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install matplotlib        # chỉ cần khi vẽ hình
```

Nếu muốn chạy trên GPU, cài bản PyTorch phù hợp với CUDA theo hướng dẫn tại
[pytorch.org](https://pytorch.org/get-started/previous-versions/).

Kiểm tra cài đặt (test dùng dữ liệu giả, không cần tải CSV):

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Dữ liệu

Repo không kèm file CSV. Xem cách tải và chuẩn bị từng bộ trong [data/README.md](data/README.md).

| Bộ dữ liệu | Số biến | Tần suất | Nguồn |
| --- | ---: | --- | --- |
| ETTh1, ETTh2 | 7 | 1 giờ | benchmark công khai |
| ETTm1, ETTm2 | 7 | 15 phút | benchmark công khai |
| Weather | 21 | 10 phút | benchmark công khai |
| Tetouan, Panama | 8, 13 | 10 phút, 1 giờ | phụ tải công khai, chuẩn bị bằng `scripts/prepare_load_data.py` |

## Chạy lại thí nghiệm

Mọi thí nghiệm đều dùng $L = 720$, 4 horizon $S \in \{96, 192, 336, 720\}$ và 3 seed (2021, 2022, 2023). Các script
bên dưới ghi kết quả từng lần chạy vào `results/*.json` ngay khi chạy xong. Khi chạy lại, script tự bỏ qua các lần đã
có, nên bị ngắt giữa chừng thì cứ chạy lại để tiếp tục.

### 1. Giai đoạn 1: TimeBase (siêu tham số chính thức)

```bash
python scripts/run_suite.py --dataset ETTh1 --seq_len 720 --variants timebase_off
```

Kết quả nằm ở `results/suite_ETTh1_L720.json`, checkpoint ở `checkpoints/suite_ETTh1_L720/`.
Riêng Weather, thêm `--results results/suite_Weather_L720_official.json`.

### 2. Giai đoạn 2: DyBAS-Net

```bash
python scripts/run_fix.py --dataset ETTh1 --variants dytbase_2stage
```

Script nạp checkpoint TimeBase của giai đoạn 1 (cùng horizon và seed), đóng băng backbone rồi huấn luyện nhánh trộn
biến. Kết quả ghi vào `results/fix_ETTh1_L720.json`. Giá trị $p_1, p_2$ của mỗi bộ nằm trong `TUNED` của
[scripts/run_fix.py](scripts/run_fix.py). Chúng được chọn bằng lưới
$\{0.1,\dots,0.6\}^2$ theo val loss với seed 2021 (biến thể `dytbase_2stage_p<p1>_<p2>`, tổng hợp bằng
[scripts/summarize_grid.py](scripts/summarize_grid.py)).

Trên CPU, mỗi lần chạy (một horizon, một seed) mất vài phút với các bộ ETT.

### 3. Baseline trong cùng pipeline

```bash
python scripts/run_suite.py --dataset ETTh1 --seq_len 720 --variants cyclenet_off cyclenet_mlp_off tqnet_off
```

Các biến thể CycleNet/Linear, CycleNet/MLP và TQNet được cài lại trong `src/dytimenet/models/`, dùng siêu tham số theo
script chính thức của từng bài.

### 4. Baseline chạy bằng code chính thức

[scripts/run_external.py](scripts/run_external.py) huấn luyện mô hình từ repo gốc của tác giả trên đúng cách chia và
chuẩn hóa dữ liệu của repo này: MS-FLOW, GTR, FITS, và các mô hình của Time-Series-Library (PatchTST, iTransformer,
TimeMixer, …). Trước khi chạy, cần tự clone các repo đó rồi truyền đường dẫn vào script:

```bash
python scripts/run_external.py --dataset ETTh1 --models MSFLOW GTR --device cuda \
    --msflow <repo MS-FLOW>/models/MS-FLOW.py --gtr <repo GTR>
python scripts/run_external.py --help      # các tùy chọn --tslib, --fits, ...
```

Ba script `scripts/run_*.sh` là các chuỗi chạy GPU đã dùng. Đặt biến `PY` trỏ tới Python của venv có CUDA, ví dụ
`PY=/path/to/python bash scripts/run_sparsetsf.sh`.

### 5. Tổng hợp bảng và hình

```bash
python scripts/compare_all.py            # bảng tổng hợp 5 bộ chuẩn → results/compare_all_L720.md
python scripts/make_slide_tables.py      # mọi bảng LaTeX của slide → slide/tables/*.tex
python scripts/make_tradeoff_plots.py    # đồ thị MSE – thời gian suy luận → slide/tradeoff/
python scripts/bench_efficiency.py --tslib <TSLib> --fits <FITS.py>   # Params, MACs, bộ nhớ, thời gian (CPU)
```

Bảng không bao giờ được chép tay: mọi số đều đọc từ `results/*.json`. Số tự chạy được ghi dưới dạng $\mu \pm \sigma$
qua 3 seed. Số lấy từ bài báo khác (`results/published_timebase_table10.json`, Table 10 của TimeBase) có trích dẫn bên
cạnh tên mô hình.

### 6. Biên dịch slide và paper

```bash
cd slide && pdflatex main && biber main && pdflatex main && pdflatex main
cd paper && pdflatex DyBAS-Net && bibtex DyBAS-Net && pdflatex DyBAS-Net && pdflatex DyBAS-Net
```

Paper dùng lớp `IEEEtran` và có thể mở trực tiếp trên Overleaf: nén `DyBAS-Net.tex`, `references.bib` và thư mục
`figures/` thành file zip rồi chọn *New Project → Upload Project*.

## Huấn luyện một lần chạy đơn lẻ

[scripts/train.py](scripts/train.py) huấn luyện một cấu hình bất kỳ. Thứ tự ưu tiên của tham số là
**mặc định trong `config.py` < YAML trong `configs/` < tham số dòng lệnh**.

```bash
python scripts/train.py --config configs/ETTh1.yaml --model timebase --seq_len 720 --pred_len 96
python scripts/train.py --help                                   # mọi tham số
python scripts/evaluate.py --checkpoint checkpoints/<run_dir>/best.pt
```

Mỗi lần chạy tạo một thư mục riêng trong `checkpoints/`, gồm `best.pt` (trọng số, cấu hình, scaler, SHA-256 của CSV),
`config.yaml`, `history.csv` và `metrics.json`. Sai số MSE/MAE được tính trên thang đã chuẩn hóa (StandardScaler fit
trên tập train), giống các bài báo được so sánh.

## Cấu trúc repo

| Đường dẫn | Nội dung |
| --- | --- |
| [src/dytimenet/models/dytimebase.py](src/dytimenet/models/dytimebase.py) | DyBAS-Net |
| [src/dytimenet/models/linear_baselines.py](src/dytimenet/models/linear_baselines.py) | TimeBase, SparseTSF, DLinear, RLinear (CycleNet/Linear) |
| [src/dytimenet/models/periodic_baselines.py](src/dytimenet/models/periodic_baselines.py) | CycleNet/MLP, TQNet |
| [src/dytimenet/models/dytimenet.py](src/dytimenet/models/dytimenet.py) | DyTimeNet (mô hình của đồ án trước) |
| [src/dytimenet/layers/](src/dytimenet/layers/) | CrossVBlock + DropFusion, IntraVBlock, RevIN, chu kỳ học được |
| [src/dytimenet/data/dataset.py](src/dytimenet/data/dataset.py) | Đọc CSV, chia tập, chuẩn hóa, cắt cửa sổ |
| [src/dytimenet/engine/](src/dytimenet/engine/) | Vòng huấn luyện, lịch learning rate, early stopping, đánh giá |
| [src/dytimenet/config.py](src/dytimenet/config.py) | Cấu hình mặc định và kiểm tra tham số |
| `configs/` | Cấu hình riêng từng bộ dữ liệu |
| `scripts/` | Chạy thí nghiệm, tổng hợp bảng, vẽ hình |
| `results/` | Kết quả JSON của mọi lần chạy |
| `slide/`, `paper/` | Slide (beamer) và paper (IEEE) |
| `tests/` | Kiểm thử bằng dữ liệu giả |

## Giấy phép

Mã nguồn dùng [MIT License](LICENSE). Các mô hình so sánh thuộc về tác giả của từng bài báo; repo này chỉ cài lại hoặc
gọi code chính thức của họ để đánh giá trong cùng một pipeline.
