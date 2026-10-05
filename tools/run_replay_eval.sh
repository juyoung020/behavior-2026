#!/usr/bin/env bash
# 기록해 둔 행동열(actions.npz)을 재생 서버로 먹이며 공식 평가기를 돌린다 -- 일치 검증·시뮬레이터 시간 측정용.
# (리눅스판; 옛 Windows 판 archive/tools/run_replay_eval.ps1 과 같은 선택지·동작)
#   재생 서버: tools/replay_policy_server.py (기본 conda behavior 환경 파이썬 — REPLAY_PY 로 바꿈, 포트 8010, 받은 관측을 <결과폴더>/server_log.npz 로)
#   평가기:   tools/run_eval_radio.sh --port 8010 (공식 명령 그대로, --timing/--trace/--deep 등은 그대로 넘김)
# 예) tools/run_replay_eval.sh --actions outputs/<기준 실행>/actions.npz --trace --tag nf_a
#   --perturb 'STEP:DIM:DELTA'  음성 대조용으로 한 스텝 행동을 조금 바꾼다
#   --conda-server              재생 서버를 CONDA_BASE 의 behavior 환경 파이썬으로 (옛 -WindowsServer)
#   그 밖: --task, --instances '0', --max-steps 500, --wrapper Default|RGBD, --tag replay, --black-diag, --black-guard,
#          --kit-set (여러 번), --render-iters, --kit-arg (여러 번), --vk-nvidia-only, --dry-run
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() { sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

ACTIONS=''; TASK=turning_on_radio; INSTANCES=0; MAX_STEPS=500; WRAPPER=Default; TAG=replay; PERTURB=''
CONDA_SERVER=0; DRY=0; PASS=()
while [ $# -gt 0 ]; do
  case $1 in
    --actions) ACTIONS=$2; shift 2 ;;
    --task) TASK=$2; shift 2 ;;
    --instances) INSTANCES=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    --wrapper) WRAPPER=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --perturb) PERTURB=$2; shift 2 ;;
    --conda-server) CONDA_SERVER=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --timing|--trace|--deep|--black-diag|--vk-nvidia-only) PASS+=("$1"); shift ;;
    --black-guard|--kit-set|--render-iters|--kit-arg) PASS+=("$1" "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$ACTIONS" ] || { echo "--actions 가 필요하다" >&2; usage >&2; exit 2; }
[ -f "$ACTIONS" ] || { echo "행동열이 없다: $ACTIONS" >&2; exit 2; }
ACTIONS=$(realpath "$ACTIONS")

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/outputs/eval_${TASK}_${STAMP}_$TAG
SRV_LOG=$REPO/logs/replay_server_${STAMP}_$TAG
if [ $CONDA_SERVER = 1 ]; then
  CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
  SRV_PY=$CONDA_BASE/envs/behavior/bin/python
else
  SRV_PY=${REPLAY_PY:-$HOME/miniconda3/envs/behavior/bin/python}
fi
SRV=("$SRV_PY" "$REPO/tools/replay_policy_server.py" --actions "$ACTIONS" --port 8010 --log "$OUT/server_log.npz" --once)
[ -n "$PERTURB" ] && SRV+=(--perturb "$PERTURB")
EV=(--task "$TASK" --instances "$INSTANCES" --max-steps "$MAX_STEPS" --wrapper "$WRAPPER" --port 8010 --out-dir "$OUT"
    ${PASS[@]+"${PASS[@]}"})
if [ $DRY = 1 ]; then
  echo "(dry-run) 재생 서버: ${SRV[*]}  > $SRV_LOG.out.log 2> $SRV_LOG.log"
  "$REPO/tools/run_eval_radio.sh" "${EV[@]}" --dry-run
  exit 0
fi

mkdir -p "$OUT" "$REPO/logs"
"${SRV[@]}" > "$SRV_LOG.out.log" 2> "$SRV_LOG.log" &
SRV_PID=$!
OK=0
for _ in $(seq 60); do
  curl -sf -m 2 http://127.0.0.1:8010/healthz > /dev/null 2>&1 && { OK=1; break; }
  kill -0 $SRV_PID 2>/dev/null || break
  sleep 1
done
if [ $OK = 0 ]; then
  echo "재생 서버가 안 떴다 ($SRV_LOG.log)"
  kill $SRV_PID 2>/dev/null || true
  exit 1
fi
set +e
"$REPO/tools/run_eval_radio.sh" "${EV[@]}"
CODE=$?
set -e
# --once 서버는 연결이 끝나면 스스로 끝난다. 30 초 기다려도 안 끝나면 이 실행의 서버만 끈다
for _ in $(seq 30); do kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done
kill $SRV_PID 2>/dev/null || true
echo "=== 재생 평가 끝: $OUT ==="
exit "$CODE"
