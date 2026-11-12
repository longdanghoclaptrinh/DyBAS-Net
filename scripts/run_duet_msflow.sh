# Chạy (hoặc chạy tiếp) MS-FLOW và GTR chính thức trên GPU cho 5 bộ gốc + Tetouan, Panama (L = 720, 4 horizon × 3 seed).
# Cần venv CUDA D:/venvs/dybas-cuda (torch 2.6.0+cu124) và repo D:/MSFLOW-official, D:/GTR-official.
# run_external.py bỏ qua các run đã có trong results/external_<bộ>_L720.json, nên chạy lại script là chạy tiếp;
# run đang dở khi dừng sẽ chạy lại từ đầu. 4 chuỗi song song; log ở logs/gpu_*.log.
# Ví dụ (Git Bash, từ thư mục repo): bash scripts/run_duet_msflow.sh
cd "$(dirname "$0")/.."
PY=${PY:-/d/venvs/dybas-cuda/Scripts/python.exe}   # đổi bằng biến môi trường PY nếu venv ở chỗ khác
chain() {   # $1 mô hình, còn lại: các bộ dữ liệu
  local m=$1; shift
  for ds in "$@"; do
    PYTHONIOENCODING=utf-8 $PY scripts/run_external.py --dataset "$ds" --models "$m" --device cuda --threads 3 \
      || { echo "GPU_FAIL $m $ds"; return 1; }
    echo "GPU_DONE $m $ds"
  done
}
mkdir -p logs
chain MSFLOW Weather >> logs/gpu_msflow_A.log 2>&1 &
chain MSFLOW ETTm1 ETTm2 ETTh1 ETTh2 >> logs/gpu_msflow_B.log 2>&1 &
chain MSFLOW Panama Tetouan >> logs/gpu_msflow_C.log 2>&1 &
# DUET: bỏ theo yêu cầu (2026-11-04), quá nặng.
# chain DUET ETTh1 ETTh2 ETTm1 ETTm2 Weather Tetouan Panama >> logs/gpu_duet.log 2>&1 &
# TimeEmb: đã chạy xong, thay bằng GTR theo yêu cầu (2026-11-04).
# chain TimeEmb ETTh1 ETTh2 ETTm1 ETTm2 Weather Tetouan Panama >> logs/gpu_timeemb.log 2>&1 &
chain GTR ETTh1 ETTh2 ETTm1 ETTm2 Weather Tetouan Panama >> logs/gpu_gtr.log 2>&1 &
wait
echo "ALL_DONE"
