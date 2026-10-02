#!/bin/bash
# 네이티브 로더(C++ ftbench)를 돌리는 동안 NVDEC·GPU 사용률을 샘플링한다 — 남은 병목이 NVDEC 하드웨어인지 보려고.
#   bash /mnt/c/behavior-2026/tools/ft_nvdec_util.sh [threads...]   (기본 4 8)
#   이미 줄 선 자리를 이어받으려면 GPU_QUEUE_TS=<20자리> 를 준다.
# GPU 잠금은 공용 도구(선착순 대기열) src/sim/engine/scripts/gpu_lock.sh, owner=fasttrain.
set -u
REPO=/mnt/c/behavior-2026
FT_WORK=${FT_WORK:-$HOME/fasttrain_work}
source "$REPO/src/sim/engine/scripts/gpu_lock.sh"
TABLE=$(ls -d "$FT_WORK"/cache/pi05_b1k-turning_on_radio-* | head -1)
trap 'gpu_lock_release fasttrain >/dev/null 2>&1; gpu_queue_leave fasttrain' EXIT INT TERM
gpu_lock_acquire fasttrain "NVDEC 사용률 측정 (ftbench)" 6 1
for th in ${*:-4 8}; do
  nvidia-smi --query-gpu=utilization.decoder,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 250 \
    > "$FT_WORK/logs/nvdec_util_$th.csv" &
  mon=$!
  out=$("$FT_WORK/build/native/ftbench" "$TABLE" "$FT_WORK/lut/lut_w720.bin" 32 "$th" 6 250 0)
  kill $mon
  # 측정 구간 가운데 절반만 (시작·끝 제외)
  n=$(wc -l < "$FT_WORK/logs/nvdec_util_$th.csv")
  stats=$(awk -F', ' -v n="$n" 'NR>n/4 && NR<3*n/4 {d+=$1; g+=$2; c++; if ($1>md) md=$1} END {printf "decoder 평균 %.0f%% (최대 %d%%), SM 평균 %.0f%%", d/c, md, g/c}' \
    "$FT_WORK/logs/nvdec_util_$th.csv")
  echo "스레드 $th: $out | $stats"
done
