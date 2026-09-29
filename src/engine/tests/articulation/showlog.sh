#!/bin/bash
grep -v '대기' ~/engine-build/articulation/gpu_all.log | grep -n -A8 "R1Pro 1 개 + 무작위" | head -30
