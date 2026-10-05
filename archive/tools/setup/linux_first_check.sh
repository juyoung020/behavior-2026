#!/usr/bin/env bash
# 리눅스 첫 확인 — 공식 평가기로 radio 한 판(0 행동, 영상 저장) + 검은 프레임 검사, 그리고 Windows 기록과 같은 행동열 재생 비교.
#   bash linux_first_check.sh [N판=3]
# 판정:
#   ① 검은 프레임(RGB 전부 0)이 한 장도 없어야 한다 — Windows 에서는 radio 판의 약 60% 에서 3 스텝 주기로 났다((B), docs/평가기_가속설계.md 5.2.2).
#   ② 같은 행동열(nf_a)을 재생한 판의 물리·판정·JSON 을 Windows 기록과 비교해 차이를 적는다(OS·GPU 가 달라 같지 않을 수 있다 — 기록만).
set -euo pipefail
N=${1:-3}
WORK=${WORK:-$HOME/behavior-2026}
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES
cd "$WORK/BEHAVIOR-1K/OmniGibson"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$WORK/outputs/linux_first_$STAMP"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv | tee "$OUT/gpu.txt"

# ① 0 행동 N 판(150 스텝), 검은 프레임은 끝까지 센다(warn)
for i in $(seq 1 "$N"); do
  python "$WORK/tools/eval_instrumented.py" --black-guard=warn --trace -- \
    --task-name turning_on_radio --mode public_test --policy local \
    --robot-config "$WORK/src/sim/configs/r1pro_robot.yaml" --env-wrapper omnigibson.eval.wrappers.DefaultWrapper \
    --instance-indices 0 --num-envs 1 --max-steps 150 --output-dir "$OUT/zero_$i" --write-video --headless \
    > "$OUT/zero_$i.log" 2>&1 || echo "판 $i 실패 — $OUT/zero_$i.log"
  python -c "import json;j=json.load(open('$OUT/zero_$i/black_frames.json'));print('판 $i 검은 프레임', j['black'], '/', list(j['total'].values())[0])"
done
python "$WORK/tools/black_frame_check.py" "$OUT"/zero_* --max-ratio 0 || echo "!! 검은 프레임이 나왔다 — (B) 가 리눅스에서도 난다"

# ② Windows 기록과 같은 행동열 재생(서버: 파이썬판, 같은 PC 안)
A="$WORK/outputs/eval_turning_on_radio_20260929_195500_nf_a"
if [ -f "$A/actions.npz" ]; then
  python "$WORK/tools/replay_policy_server.py" --actions "$A/actions.npz" --port 8110 --log "$OUT/replay/server_log.npz" --once --quickack > "$OUT/replay_server.log" 2>&1 &
  sleep 3
  python "$WORK/tools/eval_instrumented.py" --black-guard=warn --trace -- \
    --task-name turning_on_radio --mode public_test --policy websocket --host 127.0.0.1 --port 8110 \
    --robot-config "$WORK/src/sim/configs/r1pro_robot.yaml" --env-wrapper omnigibson.eval.wrappers.DefaultWrapper \
    --instance-indices 0 --num-envs 1 --max-steps 500 --output-dir "$OUT/replay" --headless > "$OUT/replay.log" 2>&1
  wait || true
  python "$WORK/tools/trace_compare.py" "$A" "$OUT/replay" --pixels-report-only | tail -25 | tee "$OUT/compare_vs_windows.txt"
fi
echo "결과: $OUT"
