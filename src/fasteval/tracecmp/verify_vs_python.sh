#!/usr/bin/env bash
# tracecmp(Rust) 와 tools/trace_compare.py(파이썬)의 출력·종료 코드가 글자 하나까지 같은지 본다.
#   wsl -d Ubuntu-22.04 -u juyoung -- bash /mnt/c/behavior-2026/src/fasteval/tracecmp/verify_vs_python.sh
# 두 쪽 다 WSL 에서 같은 경로 문자열(/mnt/c/...)로 부른다. 파이썬은 openpi venv(numpy), Rust 는 ~/cargo-target/tracecmp 빌드.
set -u
O=/mnt/c/behavior-2026/outputs
PY=$HOME/openpi/.venv/bin/python
TC=/mnt/c/behavior-2026/tools/trace_compare.py
RS=$HOME/cargo-target/tracecmp/release/tracecmp
A=$O/eval_turning_on_radio_20260929_195500_nf_a
fail=0
run_case() {  # $1 이름, $2 A, $3 B, 나머지 = 선택지
  local name=$1 a=$2 b=$3
  shift 3
  PYTHONIOENCODING=utf-8 "$PY" "$TC" "$a" "$b" "$@" > /tmp/tc_py.txt 2> /tmp/tc_py.err
  local cp=$?
  "$RS" "$a" "$b" "$@" > /tmp/tc_rs.txt 2> /tmp/tc_rs.err
  local cr=$?
  if cmp -s /tmp/tc_py.txt /tmp/tc_rs.txt && [ "$cp" -eq "$cr" ]; then
    echo "같음  $name (exit $cp, $(wc -l < /tmp/tc_rs.txt) 줄)"
  else
    echo "다름  $name (exit py $cp / rs $cr)"
    diff /tmp/tc_py.txt /tmp/tc_rs.txt | head -20
    tail -3 /tmp/tc_py.err /tmp/tc_rs.err
    fail=1
  fi
}
run_case "노이즈 바닥 nf_a vs nf_b --check --strict" "$A" "$O/eval_turning_on_radio_20260929_195757_nf_b" --check --strict
run_case "픽셀 제외 nf_a vs nf_b" "$A" "$O/eval_turning_on_radio_20260929_195757_nf_b" --check --strict --pixels-report-only
run_case "음성 대조 판 nf_a vs nf_neg + show" "$A" "$O/eval_turning_on_radio_20260929_200516_nf_neg" --check --strict --show robot_qpos,action
run_case "도구 음성 대조 --negative" "$A" "$A" --negative
run_case "환경 2 개 nf_a vs nf_n2 env-b 1" "$A" "$O/eval_turning_on_radio_20260929_200830_nf_n2" --env-b 1 --check
run_case "길이 다름 nf_a vs bk_ref(151 스텝)" "$A" "$O/eval_turning_on_radio_20260929_202701_bk_ref" --check --strict --pixels-report-only
run_case "계측 없는 판 nf_a vs nf_plain" "$A" "$O/eval_turning_on_radio_20260929_200102_nf_plain" --check
F=$(ls -d "$O"/exp_ported_dummy_radio_*/ported/replay/turning_on_radio/i0 2> /dev/null | head -1)
if [ -n "$F" ]; then
  run_case "모양 다름 nf_a vs 포팅 dummy" "$A" "$F" --check --strict --pixels-report-only
fi
rm -f /tmp/tc_py.txt /tmp/tc_rs.txt /tmp/tc_py.err /tmp/tc_rs.err
exit $fail
