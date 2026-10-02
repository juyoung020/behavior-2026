#!/bin/bash
# WSL 쪽 통합 한 판: simlink(계획기 + scenemap 같은 프로세스) 하나. ROS 없음.
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/integ/wsl_stack.sh \
#       <출력 폴더(/mnt/c/...)> [scene 1|0] [llm kau|oracle|none] [simlink 인자 ...]
# 순서: VRAM 기록 → simlink(--once) → 준비 표시 파일 → 평가기(Windows)가 붙어 한 판 → 연결이 끝나면 simlink 종료 → 끝 표시 파일.
# 키는 ~/.config/behavior-2026/kau.env 에서 환경변수로만 읽는다(저장소·기록에 남기지 않음).
OUT=${1:?출력 폴더}; SCENE=${2:-1}; LLM=${3:-kau}; shift 3 2>/dev/null
mkdir -p "$OUT"
LOG="$OUT/wsl_stack.log"
exec > >(tee -a "$LOG") 2>&1
echo "[stack] $(date '+%F %T') out=$OUT scene=$SCENE llm=$LLM args=$*"
SIMLINK=${SIMLINK:-$HOME/cargo-target/simlink/release/simlink}
rm -f "$OUT/wsl_ready" "$OUT/wsl_done"
if pgrep -x simlink >/dev/null; then echo "[stack] 다른 simlink 가 돌고 있다(pid $(pgrep -x simlink | tr '\n' ' ')) — 끝"; exit 1; fi
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
echo "gpu_used_mib_before=$(gpu)" > "$OUT/vram.txt"
set -a
[ "$LLM" = "kau" ] && . ~/.config/behavior-2026/kau.env
set +a
PLAN=(--llm "$LLM")
[ "$LLM" = "none" ] && PLAN=(--no-planner)
GRAPH=none; [ "$SCENE" = "1" ] && GRAPH=scenemap
"$SIMLINK" --listen 0.0.0.0:7801 "${PLAN[@]}" --graph $GRAPH --trace-dir "$OUT/trace" --once "$@" &
SL=$!
echo "SL=$SL STACK=$$" > "$OUT/wsl_pids"
for i in $(seq 1 50); do (echo > /dev/tcp/127.0.0.1/7801) 2>/dev/null && break; sleep 0.2; done
# nvidia-smi 표본(1 s) — 판 전체 VRAM
( while kill -0 $SL 2>/dev/null; do echo "$(date +%T),$(gpu)"; sleep 1; done ) > "$OUT/vram_timeline.csv" &
echo "gpu_used_mib_ready=$(gpu)" >> "$OUT/vram.txt"
touch "$OUT/wsl_ready"
echo "[stack] simlink 준비 — 평가기를 기다림"
wait $SL
echo "[stack] simlink 끝 ($?)"
echo "gpu_used_mib_after=$(gpu)" >> "$OUT/vram.txt"
touch "$OUT/wsl_done"
echo "[stack] 끝 $(date '+%F %T')"
