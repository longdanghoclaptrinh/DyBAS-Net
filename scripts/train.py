"""Điểm bắt đầu: CLI → cấu hình → train → đánh giá checkpoint tốt nhất.

Ví dụ: python scripts/train.py --config configs/ETTm2.yaml --pred_len 96
Luồng huấn luyện nằm trong src/dytimenet/experiment.py.
"""

from pathlib import Path
import sys

# Also supports a fresh clone using only pip install -r requirements.txt.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dytimenet.config import load_config, training_parser


def main(argv=None):
    # 1. Đọc CLI; giá trị không được truyền sẽ lấy từ YAML hoặc mặc định Python.
    parser = training_parser()
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    try:
        # 2. Ghép cấu hình và kiểm tra tham số trước khi tải dữ liệu.
        config = load_config(config_path, args)

        # 3. Tải dữ liệu, tạo model, train và test (xem experiment.train).
        from dytimenet.experiment import train
        train(config)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
