#!/bin/bash
# Submission-server check (inside a GPU-lock session): start pi05_server, drive it with the official evaluator client
# (server_protocol_test.py), stop it, replay the same observations in-process (pi05_server_ref) and compare bits.
#   server_check.sh radio|pb [envs] [steps] [chunk]
T=/mnt/c/behavior-2026/src/pi05_native/tools
B=~/pi05_native_build
D=/mnt/c/behavior-2026/data/pi05_native
KIND=${1:-radio}; ENVS=${2:-2}; STEPS=${3:-60}; CHUNK=${4:-0}
PORT=$((8700 + RANDOM % 200))
if [ "$KIND" = pb ]; then
  ARGS="--task-map /mnt/c/behavior-2026/refs/behavior-1k-solution/task_checkpoint_mapping.json --weights-dir $D"
  W=$D/pb2025_ckpt2.pi05w   # task 0 -> checkpoint_2 (task_checkpoint_mapping.json)
else
  ARGS="--weights $D/pi05_radio.pi05w"; W=$D/pi05_radio.pi05w
fi
$B/pi05_server $ARGS --port $PORT > /tmp/pi05_server_check.log 2>&1 &
SP=$!
for i in $(seq 120); do grep -q ready /tmp/pi05_server_check.log && break; sleep 1; done
grep -E "ready|loading" /tmp/pi05_server_check.log
bash $T/wsl_py.sh $T/server_protocol_test.py --port $PORT --kind $KIND --envs $ENVS --steps $STEPS --chunk $CHUNK \
  --seq /tmp/srv_seq.pi05d --out /tmp/srv_acts.f32 2>&1 | grep -v -i warn
kill $SP; wait $SP 2>/dev/null
grep -E "error|crosses" /tmp/pi05_server_check.log | head -3
[ "$CHUNK" = 0 ] && $B/pi05_server_ref --weights $W --seq /tmp/srv_seq.pi05d --actions /tmp/srv_acts.f32
rm -f /tmp/srv_seq.pi05d /tmp/srv_acts.f32
