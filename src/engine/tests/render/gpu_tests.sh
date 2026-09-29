#!/usr/bin/env bash
# render B GPU 시험 묶음 (잠금은 부르는 쪽 gpu_session.ps1 이 잡는다). 순서대로 짧게:
#  1) 합성 장면 층1=층2 (판 64, 720/480/480) + 첫 다름 픽셀 단계별 탐침
#  2) 공식 기록 장면(있으면) 층1=층2 + 처리량: 원 해상도 판 32, 224 판 256
set -uo pipefail
B=~/engine-build/render/render
D=/mnt/c/behavior-2026/src/engine/dumps/render_radio_rgbd
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "== 합성"
timeout 120 $B/test_render_synth --envs 64 --check 4 | grep -v "probe\["
if [ -f "$D/scene.rsc" ]; then
  echo "== 장면 (원 해상도)"
  timeout 200 $B/test_render_scene "$D" --envs 32 --check 3 --frames 0,100,300 2>&1 | sed -n '/== 층 2/,$p'
  echo "== 장면 (224)"
  timeout 200 $B/test_render_scene "$D" --envs 256 --check 3 --res 224 --frames 0,100,300 2>&1 | sed -n '/== 층 2/,$p'
fi
