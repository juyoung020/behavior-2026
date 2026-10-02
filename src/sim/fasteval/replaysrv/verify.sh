#!/usr/bin/env bash
# replaysrv(Rust) vs tools/replay_policy_server.py(파이썬): 같은 요청열 -> 같은 응답·같은 관측 기록인지 (시뮬레이터 없이, WSL 에서)
#   wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/sim/fasteval/replaysrv/verify.sh
# 행동열: nf_a 의 actions.npz, --perturb 5:7:0.01 (음성 대조 경로도 같은지), --quickack 켬.
set -u
PY=$HOME/openpi/.venv/bin/python
RS=$HOME/cargo-target/replaysrv/release/replaysrv
SRV=/mnt/c/behavior-2026/tools/replay_policy_server.py
ACT=/mnt/c/behavior-2026/outputs/eval_turning_on_radio_20260929_195500_nf_a/actions.npz
D=$(mktemp -d)
fail=0
for mode in rgb rgbd; do
  extra=""; [ "$mode" = rgbd ] && extra="--rgbd"
  "$PY" "$SRV" --actions "$ACT" --port 8041 --log "$D/py_$mode.npz" --once --perturb 5:7:0.01 --quickack 2> "$D/py_$mode.err" &
  PPID1=$!
  "$RS" --actions "$ACT" --port 8042 --log "$D/rs_$mode.npz" --once --perturb 5:7:0.01 --quickack 2> "$D/rs_$mode.err" &
  PPID2=$!
  "$PY" /mnt/c/behavior-2026/src/sim/fasteval/replaysrv/verify_vs_python.py --py-port 8041 --rs-port 8042 \
      --py-log "$D/py_$mode.npz" --rs-log "$D/rs_$mode.npz" $extra || fail=1
  wait $PPID1 $PPID2 2> /dev/null
done
rm -rf "$D"
exit $fail
