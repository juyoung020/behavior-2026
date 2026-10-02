#!/usr/bin/env bash
# 종단 한 판: 공식 평가기(네이티브 π0.5 프로세스 안) + simlink(계획기 + scenemap 같은 프로세스). 모두 이 PC 에서.
# (리눅스판; 옛 Windows 판 archive/src/sim/integ/run_eval_integ.ps1 과 같은 선택지·순서·결과 파일)
#   bash src/sim/integ/run_eval_integ.sh [--task turning_on_radio] [--instance 0] [--max-steps 600] [--llm kau|oracle|none]
#       [--scene 1|0] [--wrapper rgbd|default] [--simlink-args '--stage vote'] [--video] [--tag x] [--weights W.pi05w] [--replan 16]
#       [--no-lock] [--vram-warn-mib 3500] [--lock-minutes 30] [--max-wait-min 360] [--robot-config Y] [--dry-run]
# 순서: GPU 잠금(선착순, tools/gpu_lock.sh) → simlink 묶음(simlink_stack.sh: VRAM 기록 → simlink) → 평가기 → 정리 → 요약.
# 결과: outputs/integ/<이름>/ (평가 JSON·영상, native_steps.csv, glue_steps.csv, decisions.jsonl,
#       trace/trace.jsonl(계획기·link·scenemap 통계), VRAM 기록(vram.txt, vram_host.txt, vram_timeline.csv, gpu_mem.csv), summary.json)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
usage() { sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
TASK=turning_on_radio; INSTANCE=0; MAX_STEPS=600; WEIGHTS=$REPO/data/pi05_native/pi05_radio.pi05w; REPLAN=16
LLM=kau; SCENE=1; WRAPPER=rgbd; SIMLINK_ARGS=''; VIDEO=0; TAG=''; NO_LOCK=0; VRAM_WARN=3500; LOCK_MIN=30; MAX_WAIT=360
ROBOT_CONFIG=$REPO/src/sim/configs/r1pro_openpi.yaml; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --task) TASK=$2; shift 2 ;;
    --instance) INSTANCE=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    --weights) WEIGHTS=$2; shift 2 ;;
    --replan) REPLAN=$2; shift 2 ;;
    --llm) LLM=$2; shift 2 ;;
    --scene) SCENE=$2; shift 2 ;;
    --wrapper) WRAPPER=$2; shift 2 ;;
    --simlink-args) SIMLINK_ARGS=$2; shift 2 ;;
    --video) VIDEO=1; shift ;;
    --tag) TAG=$2; shift 2 ;;
    --no-lock) NO_LOCK=1; shift ;;
    --vram-warn-mib) VRAM_WARN=$2; shift 2 ;;
    --lock-minutes) LOCK_MIN=$2; shift 2 ;;
    --max-wait-min) MAX_WAIT=$2; shift 2 ;;
    --robot-config) ROBOT_CONFIG=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; usage >&2; exit 2 ;;
  esac
done
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 OMNI_KIT_ACCEPT_EULA=YES
export KMP_DUPLICATE_LIB_OK=TRUE   # run_eval_radio.sh 와 같은 우회(torch MKL + conda llvm-openmp)
STAMP=$(date +%Y%m%d_%H%M%S)
NAME="integ_${TASK}_$STAMP${TAG:+_$TAG}"
OUT=$REPO/outputs/integ/$NAME
LOG=$OUT/eval.log
read -r -a SL_ARGS <<< "$SIMLINK_ARGS"
STACK=(bash "$HERE/simlink_stack.sh" "$OUT" "$SCENE" "$LLM" --task "$TASK" --max-steps "$MAX_STEPS" ${SL_ARGS[@]+"${SL_ARGS[@]}"})
EVAL_ARGS=(--task-name "$TASK" --mode public_test --instance-indices "$INSTANCE" --num-envs 1
           --output-dir "$OUT/eval" --robot-config "$ROBOT_CONFIG" --headless)
[ "$MAX_STEPS" -gt 0 ] && EVAL_ARGS+=(--max-steps "$MAX_STEPS")
[ $VIDEO = 1 ] && EVAL_ARGS+=(--write-video)
PY_ARGS=(--weights "$WEIGHTS" --replan "$REPLAN" --out "$OUT" --wrapper "$WRAPPER" --robot-config "$ROBOT_CONFIG")
[ "$LLM" = none ] && [ "$SCENE" = 0 ] && PY_ARGS+=(--no-link)
EVAL=(python "$HERE/glue/run_eval_integ.py" "${PY_ARGS[@]}" -- "${EVAL_ARGS[@]}")
if [ $DRY = 1 ]; then
  echo "(dry-run) 묶음: ${STACK[*]}"
  echo "(dry-run) 평가기: cd $REPO/BEHAVIOR-1K/OmniGibson && ${EVAL[*]}"
  echo "(dry-run) 결과: $OUT"
  exit 0
fi

CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u; conda activate behavior; set -u
# shellcheck source=../../../tools/gpu_lock.sh
source "$REPO/tools/gpu_lock.sh"
mkdir -p "$OUT"
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
other_sims() {  # 다른 시뮬레이터(파이썬 + omnigibson 등) 수
  ps -eo pid=,args= | awk -v me=$$ '$1 != me && /python/ && /omnigibson|og_black_repro|isaac_black_repro/' | grep -vc ' awk ' || true
}

LOCKED=0; STACK_PID=''; SAMPLER=''
cleanup() {
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null || true
  if [ -n "$STACK_PID" ] && kill -0 "$STACK_PID" 2>/dev/null; then
    # 평가기가 비정상으로 끝났으면 simlink 쪽 정리(simlink 가 연결 끝을 못 봤을 수 있음)
    bash "$HERE/simlink_cleanup.sh" "$OUT" > /dev/null || true
  fi
  [ $LOCKED = 1 ] && { gpu_lock_exit integ > /dev/null || true; LOCKED=0; }
  return 0
}
trap 'gpu_lock_cancel; cleanup' EXIT
trap 'exit 130' INT TERM

if [ $NO_LOCK = 0 ]; then
  while true; do
    gpu_lock_enter integ "통합 한 판: 평가기+네이티브 pi0.5+scenemap ($TASK, $MAX_STEPS 스텝)" "$LOCK_MIN" 14 "$MAX_WAIT" || { echo 'no GPU lock'; exit 1; }
    LOCKED=1
    SIMS=$(other_sims); USED=$(gpu_used)
    if [ "$SIMS" -eq 0 ] && [ "$USED" -lt "$VRAM_WARN" ]; then echo "다른 프로세스 GPU 사용(시작 전): $USED MiB"; break; fi
    gpu_lock_exit integ > /dev/null || true; LOCKED=0
    echo "기다림: 다른 시뮬레이터 $SIMS, GPU $USED MiB"
    sleep 60
  done
fi

echo "gpu_used_mib_host_before=$(gpu_used)" > "$OUT/vram_host.txt"
# ---- simlink 묶음 ----
"${STACK[@]}" > "$OUT/stack_stdout.log" 2> "$OUT/stack_stderr.log" &
STACK_PID=$!
T0=$(date +%s)
while [ ! -f "$OUT/stack_ready" ]; do
  kill -0 "$STACK_PID" 2>/dev/null || { echo "simlink 묶음이 먼저 끝남 — $OUT/stack.log"; exit 1; }
  [ $(($(date +%s) - T0)) -gt 420 ] && { echo 'simlink 묶음 준비 시간 초과(420 s)'; exit 1; }
  sleep 2
done
USED_READY=$(gpu_used)
echo "gpu_used_mib_host_stack_ready=$USED_READY" >> "$OUT/vram_host.txt"
echo "simlink 준비 ($(($(date +%s) - T0)) s), 평가기 시작 전 GPU 사용 $USED_READY MiB"
[ "$USED_READY" -ge "$VRAM_WARN" ] && echo "경고: 평가기 밖 GPU 사용 $USED_READY MiB >= $VRAM_WARN — 검은 화면 (A) 위험(plan 3절)"
# GPU 메모리 표본(2 s, 최대 1800 s) — 옛 Windows gpu_mem_sampler.ps1 자리
timeout 1800 nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,nounits -l 2 > "$OUT/gpu_mem.csv" 2>/dev/null &
SAMPLER=$!
# ---- 평가기 ----
cd "$REPO/BEHAVIOR-1K/OmniGibson"
TE=$(date +%s)
set +e
"${EVAL[@]}" 2>&1 | tee "$LOG"
set -e
echo "평가기 끝: $(($(date +%s) - TE)) s"
T1=$(date +%s)
while [ ! -f "$OUT/stack_done" ] && kill -0 "$STACK_PID" 2>/dev/null; do
  [ $(($(date +%s) - T1)) -gt 180 ] && { echo 'simlink 묶음 정리 시간 초과 — 계속'; break; }
  sleep 2
done
cleanup
python "$HERE/summarize_run.py" "$OUT" 2>&1 | tee "$OUT/summary.txt"
echo "결과: $OUT"
