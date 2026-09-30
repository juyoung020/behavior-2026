"""core/omni/gfmat.h (손으로 옮긴 Gf RemoveScaleShear·ExtractRotationQuat·곱) vs 공식 usdrt.Gf — 비트 대조.
    WSL: bash /mnt/c/behavior-2026/src/engine/tests/omni/run_gfmat.sh
표본: (1) 무작위 회전·척도·전단 행렬, (2) PhysX 자세(float32) -> SetRotate 식 링크 행렬 × 카메라 local (실제 카메라 사슬 3개).
"""
import ctypes as C
import glob
import json
import sys

import numpy as np

E = "/home/juyoung/behavior-linux/.venv/lib/python3.11/site-packages/isaacsim/extscache/"
sys.path.insert(0, glob.glob(E + "omni.usd.libs-*")[0])
sys.path.insert(0, glob.glob(E + "usdrt.scenegraph-*")[0])
from usdrt import Gf  # noqa: E402

L = C.CDLL(sys.argv[1])
P = C.POINTER(C.c_double)
L.gf_rss_quat.argtypes = [P, P, P]
L.gf_mul.argtypes = [P, P, P]


def ours(m):
    m = np.ascontiguousarray(m, np.float64).reshape(16)
    q, t = np.zeros(4), np.zeros(3)
    L.gf_rss_quat(m.ctypes.data_as(P), q.ctypes.data_as(P), t.ctypes.data_as(P))
    return q, t


def official(m):
    g = Gf.Matrix4d(*[float(x) for x in np.asarray(m).reshape(16)])
    q = g.RemoveScaleShear().ExtractRotationQuat()
    im = q.GetImaginary()
    t = g.ExtractTranslation()
    return np.array([im[0], im[1], im[2], q.GetReal()]), np.array([t[0], t[1], t[2]])


def gmul(a, b):
    o = np.zeros(16)
    a = np.ascontiguousarray(a, np.float64).reshape(16)
    b = np.ascontiguousarray(b, np.float64).reshape(16)
    L.gf_mul(a.ctypes.data_as(P), b.ctypes.data_as(P), o.ctypes.data_as(P))
    return o.reshape(4, 4)


def quat2mat_pxr(p, q):
    x, y, z, w = [float(v) for v in q]
    m = np.eye(4)
    m[0][0] = 1.0 - 2.0 * (y * y + z * z); m[0][1] = 2.0 * (x * y + z * w); m[0][2] = 2.0 * (z * x - y * w)
    m[1][0] = 2.0 * (x * y - z * w); m[1][1] = 1.0 - 2.0 * (z * z + x * x); m[1][2] = 2.0 * (y * z + x * w)
    m[2][0] = 2.0 * (z * x + y * w); m[2][1] = 2.0 * (y * z - x * w); m[2][2] = 1.0 - 2.0 * (y * y + x * x)
    m[3][:3] = [float(v) for v in p]
    return m


rng = np.random.default_rng(1)
bad = n = 0
first = None
# (1) 무작위: 회전 × (척도·약한 전단) + 이동
for _ in range(20000):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    R = quat2mat_pxr(rng.standard_normal(3) * 5, q)
    S = np.eye(4)
    S[:3, :3] = np.diag(rng.uniform(0.2, 3.0, 3)) + rng.standard_normal((3, 3)) * rng.choice([0, 1e-9, 1e-3])
    m = S @ R if rng.random() < 0.5 else R
    for fn in (lambda a: a,):
        a, b = ours(m), official(m)
        n += 1
        if not (np.array_equal(a[0].view(np.uint64), b[0].view(np.uint64)) and np.array_equal(a[1], b[1])):
            bad += 1
            first = first or (m, a, b)
print(f"무작위 행렬 {n} 개: 쿼터니언·이동 double 비트 다름 {bad}")
if first:
    print("  첫 다름:", first[1][0], first[2][0])
# (2) 실제 카메라 사슬: PhysX float32 자세 -> 링크 행렬 -> local × 링크 (Gf 곱) -> 분해
cams = json.load(open("/home/juyoung/engine-data/linux_official/camchain/scope.json"))[0]["robot"]["camera_chain"]
bad2 = n2 = 0
mulbad = 0
for name, c in cams.items():
    L0 = np.array(c["chain"][0]["local"]).reshape(4, 4)
    for _ in range(20000):
        q = rng.standard_normal(4).astype(np.float32)
        q = (q / np.linalg.norm(q)).astype(np.float32)
        p = (rng.standard_normal(3) * 5).astype(np.float32)
        W = quat2mat_pxr(p, q)
        M = gmul(L0, W)
        # Gf 곱 검산 (usdrt Matrix4d 곱)
        gm = Gf.Matrix4d(*[float(x) for x in L0.reshape(16)]) * Gf.Matrix4d(*[float(x) for x in W.reshape(16)])
        gm = np.array([[gm[i][j] for j in range(4)] for i in range(4)])
        mulbad += int(not np.array_equal(gm, M))
        a, b = ours(M), official(M)
        n2 += 1
        if not np.array_equal(a[0].view(np.uint64), b[0].view(np.uint64)):
            bad2 += 1
print(f"카메라 사슬 {n2} 개: 곱 다름 {mulbad}, 쿼터니언 double 비트 다름 {bad2}")
sys.exit(1 if bad or bad2 or mulbad else 0)
