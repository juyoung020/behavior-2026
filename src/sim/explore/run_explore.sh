#!/bin/bash
# 스킬 explore 한 판(시뮬): 평가기(run_explore.py, 지도·move_robot) + 에이전트(explore, LLM 또는 기준선).
#   run_explore.sh <policy llm|frontier> <task> [tag] [extra agent args...]
# 결과: outputs/explore_<ts>_<task>_<policy>[_tag]/ (memory/ 지도, decisions.jsonl, timeline.jsonl, summary.json, sim.log)
# 사전 조건: VRAM 여유 ≥ 9 GB, RAM 여유 ≥ 16 GB (아니면 기다린다). 키는 환경변수로만(kau.env).
set -u
POL=$1; TASK=$2; TAG=${3:-}; shift 3 2>/dev/null || shift $#
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)            # behavior-2026
SUPER=$(cd "$REPO/../.." && pwd)              # robot-agent
TS=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/outputs/explore_${TS}_${TASK}_${POL}${TAG:+_$TAG}
PORT=${PORT:-8771}
MAXSTEPS=${MAXSTEPS:-27000}
mkdir -p "$OUT"
while true; do
  fv=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}')
  fr=$(free -g | awk '/Mem:/{print $7}')
  [ "$fv" -ge 9000 ] && [ "$fr" -ge 16 ] && break
  echo "[run] waiting: VRAM free $fv MiB, RAM avail $fr GB"; sleep 30
done
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES
export SGRT_POSE=${SGRT_POSE:-gt}   # 시뮬 시험: 정답 자세(map = world). 실제 로봇 기본은 slam
export SGRT_LIB=${SGRT_LIB:-$HOME/sgrt_build_explore/libsgrt.so}
cd "$OUT"
python "$HERE/run_explore.py" --listen 127.0.0.1:$PORT --out "$OUT" -- --task-name "$TASK" --mode public_test \
  --instance-indices 0 --num-envs 1 --max-steps $MAXSTEPS --headless \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper > "$OUT/sim.log" 2>&1 &
SIM=$!
echo "[run] sim pid $SIM out $OUT"
# 평가기가 첫 관측을 받을 때까지(지도가 생김)
for i in $(seq 1 120); do
  grep -q "reference set\|Traceback" "$OUT/sim.log" 2>/dev/null && break
  kill -0 $SIM 2>/dev/null || break
  sleep 5
done
grep -q "Traceback" "$OUT/sim.log" && { echo "[run] sim failed"; tail -30 "$OUT/sim.log"; kill $SIM 2>/dev/null; exit 1; }
set -a; . ~/.config/behavior-2026/kau.env; set +a
"$SUPER/src/agent/skills/explore/target/release/explore" --policy "$POL" --addr 127.0.0.1:$PORT --out "$OUT" --task "$TASK" \
  --max-calls ${MAXCALLS:-80} --max-sim-s ${MAXSIM:-880} --max-wall-s ${MAXWALL:-3600} "$@" > "$OUT/agent.log" 2>&1
echo "[run] agent done: $(tail -c 300 $OUT/agent.log | tr '\n' ' ')"
kill -INT $SIM 2>/dev/null; sleep 20; kill $SIM 2>/dev/null; wait $SIM 2>/dev/null
echo "[run] finished $OUT"
