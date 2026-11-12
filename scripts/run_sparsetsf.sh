# SparseTSF (siêu tham số scripts/SparseTSF/linear/<bộ>.sh chính thức, run_suite.SPARSETSF_SCRIPT) trên 5 bộ chuẩn,
# L = 720, 3 seed, GPU (venv D:/venvs/dybas-cuda). run_suite.py bỏ qua run đã có, nên chạy lại là chạy tiếp.
# Ví dụ (Git Bash, từ thư mục repo): bash scripts/run_sparsetsf.sh
cd "$(dirname "$0")/.."
PY=${PY:-/d/venvs/dybas-cuda/Scripts/python.exe}   # đổi bằng biến môi trường PY nếu venv ở chỗ khác
mkdir -p logs
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather; do
  extra=""; [ "$ds" = Weather ] && extra="--results results/suite_Weather_L720_official.json"
  PYTHONIOENCODING=utf-8 $PY scripts/run_suite.py --dataset $ds --variants sparsetsf_off --seq_len 720 --device cuda \
    --threads 3 $extra >> logs/gpu_sparsetsf.log 2>&1 && echo "GPU_DONE SparseTSF $ds" || { echo "GPU_FAIL SparseTSF $ds"; exit 1; }
done
echo "SPARSETSF_ALL_DONE"
