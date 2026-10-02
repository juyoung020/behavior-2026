"""GPU 메모리를 일부러 잡아 두는 도구 (검은 화면이 VRAM 압박 때문인지 가르는 실험용).

    python C:\\behavior-2026\\tools\\gpu_hog.py --gib 8 [--seconds 900]

torch 로 GiB 만큼 GPU 메모리를 잡고, 끝날 때까지(또는 --seconds) 가만히 있는다. 계산은 안 한다(GPU 사용률 0 → 메모리만의 효과).
안전장치: 지금 비어 있는 전용 VRAM(torch.cuda.mem_get_info) - 0.5 GiB 를 넘게는 절대 잡지 않는다
(Windows WDDM 은 넘치면 시스템 RAM 으로 넘겨 받아 주므로, 큰 값을 잘못 주면 RAM 을 다 먹는다 -- 2026-09-29 실제로 한 번 있었음).
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--gib", type=float, required=True)
ap.add_argument("--seconds", type=float, default=900)
a = ap.parse_args()
free, total = torch.cuda.mem_get_info()
cap = max(0.0, free / (1 << 30) - 0.5)
want = min(a.gib, cap)
if want < a.gib:
    print(f"요청 {a.gib} GiB > 비어 있는 VRAM {free / (1 << 30):.1f} GiB -> {want:.1f} GiB 만 잡는다", flush=True)
blocks = []
left = int(want * (1 << 30))
while left > 0:
    n = min(left, 1 << 30)
    blocks.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
    left -= n
torch.cuda.synchronize()
print(f"잡음: {want:.1f} GiB (torch reserved {torch.cuda.memory_reserved() / (1 << 30):.2f} GiB)", flush=True)
time.sleep(a.seconds)
