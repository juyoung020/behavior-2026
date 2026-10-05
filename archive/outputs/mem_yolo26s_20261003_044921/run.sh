#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior
cd /home/juyoung/robot-agent/src/behavior-2026/outputs/mem_yolo26s_20261003_044921
export OMNI_KIT_ACCEPT_EULA=YES SGRT_ENGINE=$HOME/ovdet_models/x86_sm120/yolo26s-seg.plan
python ../../src/vla/pi05_native/glue/run_eval_native.py --weights ../../data/pi05_native/comet_pt12.pi05w --replan 32   --native-log native_steps.csv --scene-out memory -- --task-name bringing_water --mode public_test --instance-indices 0   --num-envs 1 --output-dir . --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper   --robot-config ../../src/sim/configs/r1pro_openpi.yaml --headless
