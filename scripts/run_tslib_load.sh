# PatchTST, TimeMixer (Time-Series-Library chính thức, script ETTh1 như các mô hình TSLib khác trên bộ phụ tải) trên
# Tetouan, Panama; GPU, venv D:/venvs/dybas-cuda (cần reformer_pytorch). run_external.py bỏ qua run đã có, nên chạy lại là
# chạy tiếp. Chạy MỘT chuỗi tuần tự: 2 chuỗi song song làm đầy 6 GB VRAM của RTX 4050 và chậm hơn hẳn (2026-11-05).
# Log ở logs/gpu_tslib.log; nhiệt độ/xung GPU mỗi 5 phút ở logs/gpu_temp.log.
# Ví dụ (Git Bash, từ thư mục repo): bash scripts/run_tslib_load.sh            # PatchTST rồi TimeMixer
#                                    bash scripts/run_tslib_load.sh PatchTST   # chỉ PatchTST
cd "$(dirname "$0")/.."
PY=${PY:-/d/venvs/dybas-cuda/Scripts/python.exe}   # đổi bằng biến môi trường PY nếu venv ở chỗ khác
MODELS=${*:-PatchTST TimeMixer}
run() {   # $1 mô hình, $2 bộ dữ liệu
  PYTHONIOENCODING=utf-8 $PY scripts/run_external.py --dataset "$2" --models "$1" --device cuda --threads 3 \
    && echo "GPU_DONE $1 $2" || { echo "GPU_FAIL $1 $2"; return 1; }
}
mkdir -p logs
while true; do
  echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader)" >> logs/gpu_temp.log
  sleep 300
done &
MON=$!
for m in $MODELS; do
  for ds in Tetouan Panama; do
    run "$m" "$ds" >> logs/gpu_tslib.log 2>&1 || break 2
  done
done
kill $MON
echo "TSLIB_ALL_DONE"
