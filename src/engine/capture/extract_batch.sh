#!/usr/bin/env bash
# 장면 추출물 일괄 생성 (인스턴스 불러오기 B안): 과제 번호 범위 × 공개 인스턴스 번호 범위를 차례로 뜬다. 다시 부르면 끝난 판은 건너뛴다(재개).
#   bash /mnt/c/behavior-2026/src/engine/capture/extract_batch.sh [과제 시작=0] [과제 끝=49] [인스턴스 끝=9] [모드=public_test]
# 과제 번호 = data/2026-challenge-demos/meta/tasks.jsonl 의 task_index. 결과 ~/engine-data/scenes/<과제>/<모드>_<번호>/
# 판마다 디스크 사용량과 C: 남은 공간(WSL 디스크 파일이 C: 에 있음)을 찍고, C: 여유가 5 GB 밑이면 멈춘다.
set -uo pipefail
T0=${1:-0}
T1=${2:-49}
I1=${3:-9}
MODE=${4:-public_test}
LOG=~/engine-data/scenes/extract_batch.log
mkdir -p ~/engine-data/scenes
TASKS=$(python3 - "$T0" "$T1" <<'EOF'
import json, sys
t0, t1 = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in open("/mnt/c/behavior-2026/data/2026-challenge-demos/meta/tasks.jsonl", encoding="utf-8")]
for r in sorted(rows, key=lambda r: r["task_index"]):
    if t0 <= r["task_index"] <= t1:
        print(r["task_name"])
EOF
)
for TASK in $TASKS; do
  for IDX in $(seq 0 "$I1"); do
    D=~/engine-data/scenes/$TASK/${MODE}_$IDX
    if [ -f "$D/extract_ok" ]; then continue; fi
    FREE=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
    if [ "$FREE" -lt 5 ]; then echo "$(date +%T) C: 여유 ${FREE}G < 5G -> 멈춤" | tee -a "$LOG"; exit 2; fi
    T=$(date +%s)
    bash /mnt/c/behavior-2026/src/engine/capture/extract_instance.sh "$MODE" "$IDX" "$TASK" > "$D.log" 2>&1
    OK=$([ -f "$D/extract_ok" ] && echo ok || echo 실패)
    SZ=$(du -sm "$D" 2>/dev/null | cut -f1)
    TOT=$(du -sm ~/engine-data/scenes | cut -f1)
    echo "$(date +%T) $TASK ${MODE}_$IDX $OK ${SZ}MB $(( $(date +%s) - T ))s | 합계 ${TOT}MB, C: 여유 ${FREE}G" | tee -a "$LOG"
    # 실패한 판의 기록은 남긴다(로그 확인용). Kit 이 GPU 를 오래 붙잡지 않게 판 사이 쉼은 두지 않는다
  done
done
echo "=== 일괄 추출 끝: $(du -sh ~/engine-data/scenes | cut -f1) ===" | tee -a "$LOG"
