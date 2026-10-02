#!/usr/bin/env bash
# 통합 한 판의 simlink 쪽: simlink(계획기 + scenemap 같은 프로세스) 하나. ROS 없음.
# (리눅스판; 옛 WSL 판 archive/src/sim/integ/wsl_stack.sh 와 같은 순서·기록 파일. 평가기와 같은 PC 에서 돈다)
#   bash src/sim/integ/simlink_stack.sh <출력 폴더> [scene 1|0] [llm kau|oracle|none] [simlink 인자 ...]
# 순서: VRAM 기록 → simlink(--once) → 준비 표시 파일(stack_ready) → 평가기(run_eval_integ.sh)가 붙어 한 판
#       → 연결이 끝나면 simlink 종료 → 끝 표시 파일(stack_done).
# 키는 ~/.config/behavior-2026/kau.env 에서 환경변수로만 읽는다(저장소·기록에 남기지 않음).
# simlink 바이너리: SIMLINK(기본 ~/cargo-target/simlink/release/simlink, src/sim/integ/build_simlink.sh 로 빌드).
# 듣는 주소: SIMLINK_LISTEN(기본 127.0.0.1:7801 — 같은 PC 라 밖으로 열지 않는다).
set -euo pipefail
OUT=${1:?출력 폴더}; SCENE=${2:-1}; LLM=${3:-kau}
shift $(( $# < 3 ? $# : 3 ))
mkdir -p "$OUT"
LOG="$OUT/stack.log"
exec > >(tee -a "$LOG") 2>&1
echo "[stack] $(date '+%F %T') out=$OUT scene=$SCENE llm=$LLM args=$*"
SIMLINK=${SIMLINK:-$HOME/cargo-target/simlink/release/simlink}
LISTEN=${SIMLINK_LISTEN:-127.0.0.1:7801}
[ -x "$SIMLINK" ] || { echo "[stack] simlink 바이너리가 없다: $SIMLINK (src/sim/integ/build_simlink.sh)"; exit 1; }
rm -f "$OUT/stack_ready" "$OUT/stack_done"
if pgrep -x simlink >/dev/null; then echo "[stack] 다른 simlink 가 돌고 있다(pid $(pgrep -x simlink | tr '\n' ' ')) — 끝"; exit 1; fi
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
echo "gpu_used_mib_before=$(gpu)" > "$OUT/vram.txt"
if [ "$LLM" = "kau" ]; then
  set -a
  # shellcheck disable=SC1090
  . ~/.config/behavior-2026/kau.env
  set +a
fi
PLAN=(--llm "$LLM")
[ "$LLM" = "none" ] && PLAN=(--no-planner)
GRAPH=none; [ "$SCENE" = "1" ] && GRAPH=scenemap
"$SIMLINK" --listen "$LISTEN" "${PLAN[@]}" --graph $GRAPH --trace-dir "$OUT/trace" --once "$@" &
SL=$!
echo "SL=$SL STACK=$$" > "$OUT/stack_pids"
PORT=${LISTEN##*:}
for _ in $(seq 1 50); do (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break; sleep 0.2; done
# nvidia-smi 표본(1 s) — 판 전체 VRAM
( while kill -0 $SL 2>/dev/null; do echo "$(date +%T),$(gpu)"; sleep 1; done ) > "$OUT/vram_timeline.csv" &
echo "gpu_used_mib_ready=$(gpu)" >> "$OUT/vram.txt"
touch "$OUT/stack_ready"
echo "[stack] simlink 준비 — 평가기를 기다림"
set +e
wait $SL
CODE=$?
set -e
echo "[stack] simlink 끝 ($CODE)"
echo "gpu_used_mib_after=$(gpu)" >> "$OUT/vram.txt"
touch "$OUT/stack_done"
echo "[stack] 끝 $(date '+%F %T')"
