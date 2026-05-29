#!/usr/bin/env bash
set -euo pipefail

BASE=/public/home/ba25001026/3DGS
LOG_DIR="$BASE/task3_official/logs"
ENV_DIR=/public/home/ba25001026/.conda/envs/gauss-sd
PYTHON="$ENV_DIR/bin/python"
CUDA_HOME=/public/software/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"
cd "$BASE"

MEM_LOG="$LOG_DIR/task2_mem_probe_${SLURM_JOB_ID:-manual}.csv"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,name,index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 2
    done
) > "$MEM_LOG" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

"$PYTHON" train.py \
    --colmap_dir data/chair \
    --checkpoint_dir data/chair/checkpoints_task2_mem_probe \
    --num_epochs 1 \
    --device cuda \
    --maximum_points 3000 \
    --downsample_factor 8

kill "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

PEAK_MEM_MB=$(awk -F, '{gsub(/ /, "", $4); if ($4+0 > max) max=$4+0} END {print max+0}' "$MEM_LOG")
SUMMARY="$LOG_DIR/task2_mem_probe_${SLURM_JOB_ID:-manual}.txt"
cat > "$SUMMARY" <<EOF
Task 2 Simplified 3DGS Memory Probe
job_id=${SLURM_JOB_ID:-manual}
partition=${SLURM_JOB_PARTITION:-GPU_H800_8}
host=$(hostname)
num_epochs=1
maximum_points=3000
downsample_factor=8
peak_memory_mb=$PEAK_MEM_MB
memory_log=$MEM_LOG
EOF

cat "$SUMMARY"
