#!/usr/bin/env bash
# render GPU 시험 묶음 (잠금은 부르는 쪽 gpu_session.ps1 이 잡는다). 순서대로 짧게:
#  1) 합성 장면 층1=층2 (판 64, 720/480/480, 잡음 제거 4·반사 광선) + 첫 다름 픽셀 단계별 탐침
#  2) 공식 기록 장면(있으면) 층1=층2 + 처리량: 원 해상도 판 16·32, 224 판 128·256 (음영 = rsc 기본값, 14.6.3)
set -uo pipefail
B=~/engine-build/render/render
D=/mnt/c/behavior-2026/src/sim/engine/dumps/render_radio_rgbd/rsc
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "== 합성 (잡음 제거 4, 반사 1)"
timeout 150 $B/test_render_synth --envs 64 --check 4 | grep -v "probe\["
echo "== 합성 (잡음 제거 0, 반사 0 = 옛 경로 비교)"
timeout 150 $B/test_render_synth --envs 64 --check 2 --denoise 0 --spec 0 | grep -v "probe\["
echo "== 합성 (인스턴스 6000, 224², 판 512)"
timeout 150 $B/test_render_synth --inst 6000 --envs 512 --check 2 --res 224,224,224 | grep -v "probe\["
if [ -f "$D/scene.rsc" ]; then
  for E in 16 32; do
    echo "== 장면 (원 해상도, 판 $E)"
    timeout 300 $B/test_render_scene "$D" --envs $E --check 3 --frames 0,100,300 --official 0 2>&1 | sed -n '/== 층 2/,$p'
  done
  for E in 128 256; do
    echo "== 장면 (224, 판 $E)"
    timeout 300 $B/test_render_scene "$D" --envs $E --check 3 --res 224 --frames 0,100,300 --official 0 2>&1 | sed -n '/== 층 2/,$p'
  done
fi
