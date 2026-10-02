#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior
cd /home/juyoung/robot-agent/src/behavior-2026/outputs/move_robot_live_051826
export OMNI_KIT_ACCEPT_EULA=YES
python ../../src/sim/move_robot/run_eval_move.py --script calls.jsonl --log move_robot.jsonl -- --task-name turning_on_radio --mode public_test --instance-indices 0 --num-envs 1 --output-dir . --max-steps 1200 --headless
