#!/usr/bin/env bash
# 과제 하나의 렌더 장면(scene.rsc) 을 뜬다(렌더 없음, 기하·재질·조명만) -> ~/engine-data/render_scenes/<과제>/
# (리눅스판; 옛 WSL 판 archive/src/sim/engine/capture/render_scene_wsl.sh 와 같은 인자·결과. conda behavior 환경 + 이 저장소 BEHAVIOR-1K)
#   bash src/sim/engine/capture/render_scene.sh <과제> [모드=public_test] [번호=0] [텍스처 상한=512]
#   DRY_RUN=1 이면 명령만 찍는다.
# 같은 과제의 다른 판도 이 장면을 쓴다(움직이는 기준 prim 은 엔진 물리 자세로 그림). 에셋 파생물 -> git 밖.
# GPU 잠금: tools/gpu_lock.sh (owner engine-lead, 10 분).
set -uo pipefail
ulimit -c 0
ENGINE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$ENGINE/../../.." && pwd)
TASK=${1:?task}
MODE=${2:-public_test}
IDX=${3:-0}
TEX=${4:-512}
OUT=~/engine-data/render_scenes/$TASK
DUMP=$OUT/dump
PORT=8014
ZA=~/engine-data/scenes/zero_actions_1.npz
EV=(python "$ENGINE/capture/render_scene.py" --dump-dir "$DUMP" -- --trace --
    --task-name "$TASK" --robot-config "$REPO/src/sim/configs/r1pro_openpi.yaml"
    --env-wrapper omnigibson.eval.wrappers.DefaultWrapper --mode "$MODE"
    --host 127.0.0.1 --port "$PORT" --instance-indices "$IDX" --num-envs 1 --max-steps 1
    --output-dir "$DUMP" --headless)
if [ "${DRY_RUN:-0}" = 1 ]; then echo "(dry-run) 평가기: ${EV[*]}"; echo "(dry-run) 변환: convert_scene.py $DUMP --tex-max $TEX --out $OUT"; exit 0; fi
mkdir -p "$DUMP" "$(dirname "$ZA")"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1
export OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-$REPO/BEHAVIOR-1K/datasets}
CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior
[ -f "$ZA" ] || python -c "import numpy as np; np.savez('$ZA', actions=np.zeros((2,1,23), np.float32))"
"${REPLAY_PY:-$HOME/openpi/.venv/bin/python}" "$REPO/tools/replay_policy_server.py" --actions "$ZA" --port $PORT \
  --log "$DUMP/server_log.npz" --once > "$DUMP/replay_server.log" 2>&1 &
SRV=$!
for _ in $(seq 60); do curl -sf http://127.0.0.1:$PORT/healthz >/dev/null && break; sleep 1; done
# shellcheck source=../../../../tools/gpu_lock.sh
source "$REPO/tools/gpu_lock.sh"
gpu_lock_enter "engine-lead" "렌더 장면 $TASK (약 3분)" 10 6 1440 || { kill $SRV; exit 3; }
cd "$REPO/BEHAVIOR-1K/OmniGibson"
"${EV[@]}" > "$DUMP/eval.log" 2>&1
CODE=$?
gpu_lock_exit "engine-lead"
for _ in $(seq 30); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill $SRV 2>/dev/null || true
[ -f "$DUMP/scene.json" ] || { echo "렌더 덤프 실패 (exit $CODE) — $DUMP/eval.log"; exit 1; }
python "$ENGINE/tests/render/capture/convert_scene.py" "$DUMP" --tex-max "$TEX" --out "$OUT" > "$OUT/convert.log" 2>&1 || { echo "변환 실패 — $OUT/convert.log"; exit 1; }
cp "$DUMP/scene.json" "$OUT/scene.json"
# 덤프 원본(frame npz·geom)은 rsc 로 바뀌었으니 지운다 (크기 줄이기)
rm -f "$DUMP"/frame_*.npz "$DUMP"/geom.npz
echo "$(date +%F_%T) $(du -sm "$OUT" | cut -f1)MB" > "$OUT/render_ok"
echo "=== 렌더 장면 끝: $OUT ($(du -sh "$OUT" | cut -f1)) ==="
