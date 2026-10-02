#!/bin/bash
# Build and run the tokenizer check in WSL (CPU only).
set -e
D=/mnt/c/behavior-2026/src/vla/pi05_native
B=~/pi05_native_build; mkdir -p $B
cd ~/openpi && .venv/bin/python $D/tools/make_tokenizer_cases.py /mnt/c/behavior-2026/data/2026-challenge-demos/meta/tasks.jsonl $B/tok_cases.tsv 3000
g++ -O2 -std=c++20 -o $B/tok_test $D/tools/tok_test.cpp $D/src/tokenizer.cpp $D/src/weights.cpp
$B/tok_test /mnt/c/behavior-2026/data/pi05_native/pi05_radio.pi05w $B/tok_cases.tsv
