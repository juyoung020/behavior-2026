#!/bin/bash
# 8080 뷰어를 이 탐사 판의 memory 로 바꾼다(뷰어는 하나만). 다른 sgviz 는 PID 로만 끈다(pkill 안 씀).
#   viewer_8080.sh <run dir>    — memory/scene.json 이 생길 때까지 기다린 뒤
set -u
RUN=$(cd "$1" && pwd); MEM=$RUN/memory
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
for i in $(seq 1 240); do [ -f "$MEM/scene.json" ] && break; sleep 5; done
[ -f "$MEM/scene.json" ] || { echo "[viewer] no scene.json in $MEM"; exit 1; }
for pid in $(ps -eo pid,args | awk '/sgviz\.py/ && /--port 8080/ && !/awk/ {print $1}'); do kill "$pid"; done
sleep 2
cd "$REPO"
setsid nohup ~/sdsg_venv/bin/python src/scene_graph/viewer/sgviz.py "$MEM" --port 8080 > "$RUN/viewer.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  c=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/ || true)
  [ "$c" = 200 ] && { echo "[viewer] http://localhost:8080 -> $MEM"; exit 0; }
  sleep 1
done
echo "[viewer] not answering"; exit 1
