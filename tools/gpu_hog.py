"""GPU 메모리를 일부러 잡아 두는 도구 (검은 화면이 VRAM 압박 때문인지 가르는 실험용).

    python C:\\behavior-2026\\tools\\gpu_hog.py --gib 8 [--seconds 900]

torch 로 GiB 만큼 GPU 메모리를 잡고, 끝날 때까지(또는 --seconds) 가만히 있는다. 계산은 안 한다(GPU 사용률 0 → 메모리만의 효과).
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--gib", type=float, required=True)
ap.add_argument("--seconds", type=float, default=900)
a = ap.parse_args()
blocks = []
left = int(a.gib * (1 << 30))
while left > 0:
    n = min(left, 1 << 30)
    blocks.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
    left -= n
torch.cuda.synchronize()
print(f"잡음: {a.gib} GiB (torch reserved {torch.cuda.memory_reserved() / (1 << 30):.2f} GiB)", flush=True)
time.sleep(a.seconds)
