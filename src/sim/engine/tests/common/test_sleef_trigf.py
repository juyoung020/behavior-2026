"""core/common/sleef_trigf.h (sinf_u10·cosf_u10) vs torch CPU th.sin/th.cos — |x| < 125 인 float32 전부 비교.
    WSL: bash /mnt/c/behavior-2026/src/sim/engine/tests/common/run_sleef_trigf.sh
"""
import ctypes as C
import sys

import numpy as np
import torch as th

L = C.CDLL(sys.argv[1])
F = C.POINTER(C.c_float)
L.sl_batch.argtypes = [F, F, F, C.c_int64]
bits = np.arange(0, 1 << 32, 1 << 25, dtype=np.uint64)  # 덩어리 시작
lim = np.float32(125.0).view(np.uint32)
bad_s = bad_c = n = 0
step = 1 << 25
for b0 in range(0, 1 << 32, step):
    u = np.arange(b0, b0 + step, dtype=np.uint64).astype(np.uint32)
    x = u.view(np.float32)
    x = x[np.abs(x) < 125.0]
    if x.size == 0:
        continue
    x = np.ascontiguousarray(x)
    ts = th.sin(th.from_numpy(x)).numpy()
    tc = th.cos(th.from_numpy(x)).numpy()
    os_, oc = np.empty_like(x), np.empty_like(x)
    L.sl_batch(x.ctypes.data_as(F), os_.ctypes.data_as(F), oc.ctypes.data_as(F), x.size)
    ds = ts.view(np.uint32) != os_.view(np.uint32)
    dc = tc.view(np.uint32) != oc.view(np.uint32)
    # NaN 끼리는 같다고 본다 (NaN 입력)
    nanx = np.isnan(x)
    ds &= ~nanx
    dc &= ~nanx
    bad_s += int(ds.sum())
    bad_c += int(dc.sum())
    n += x.size
    if (ds.any() or dc.any()) and bad_s + bad_c < 5:
        i = np.where(ds | dc)[0][0]
        print("다름 예", x[i], ts[i], os_[i], tc[i], oc[i])
print(f"float32 {n} 개 (|x|<125): sin 다름 {bad_s}, cos 다름 {bad_c}")
sys.exit(1 if bad_s or bad_c else 0)
