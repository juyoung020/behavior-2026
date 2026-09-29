"""크기 조정 커널 보정 — 원래 JAX 크기 조정이 **이 GPU 에서** 어떤 순서로 더하는지 알아낸다 (시작 때 몇 초).

왜 필요한가 (docs/학습환경_가속.md 3.3):
  원래 resize_with_pad 는 XLA 가 cuBLAS SGEMM 두 번으로 컴파일한다(행 방향 → 열 방향). SGEMM 은 출력마다 k 오름차순
  FMA 로 더하지만, cuBLAS 가 split-K(k 구간을 나눠 따로 더한 뒤 합침)를 고르면 구간 경계에 걸친 출력의 반올림이 달라진다.
  어느 경계에서 나누는지는 모양·배치·작업공간·GPU 기종·cuBLAS 버전에 따라 바뀐다(실측: 같은 모양의 단독 GEMM 과
  resize 안의 GEMM 이 다르게 나눔). 그래서 고정값을 넣지 않고, 학습을 돌릴 그 GPU 에서 원래 함수를 직접 찔러 알아낸다.

방법: 원래 jit 함수에 float32 영상을 넣는다(uint8 판과 GEMM 설정이 같음을 HLO 로 확인 — 모양·축·stride 동일).
  - 열 방향(2단계) 경계: 입력 행을 16 칸마다 하나만 값이 있게 → 1단계 결과가 곱 하나(정확)라서 2단계 합 순서만 보인다
  - 행 방향(1단계) 경계: 입력 열을 16 칸마다 하나만 → 2단계가 곱 하나라서 1단계 합 순서만 보인다
  출력마다 "몇 번째 탭부터 새 구간인지"(0 = 안 나뉨)를 찾고, 모형(구간마다 순차 FMA, 구간 합)으로 설명 안 되는 출력이
  하나라도 있으면 멈춘다. 마지막에 uint8 무작위 영상으로 원래 함수 vs 커널을 비트 대조한다.
"""
from __future__ import annotations

import numpy as np

SPACING = 16  # 탭 폭(최대 7)보다 넓게


def weights(m: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from jax._src.image import scale as S

    f = jax.jit(lambda: S.compute_weight_mat(m, 224, jnp.float32(224 / m), jnp.float32(0.0), S._fill_triangle_kernel, True))
    return np.asarray(f())


def taps(W: np.ndarray):
    st = np.zeros(224, np.int32)
    ct = np.zeros(224, np.int32)
    wt = np.zeros((224, 8), np.float32)
    for i in range(224):
        nz = np.nonzero(W[:, i])[0]
        assert 0 < len(nz) <= 8 and np.all(np.diff(nz) == 1), "탭이 연속이 아니다"
        st[i], ct[i] = nz[0], len(nz)
        wt[i, : len(nz)] = W[nz, i]
    return st, ct, wt


def emu(V: np.ndarray, st, ct, wt, split, rows=None):
    """out[i, :] = Σ_k w·V[st+k, :]  (구간마다 순차 FMA, 두 구간 합). FMA 는 long double 로 정확히 흉내낸다."""
    L = np.longdouble
    rows = range(len(st)) if rows is None else rows
    out = {}
    for i in rows:
        p = np.zeros(V.shape[1], np.float32)
        q = np.zeros(V.shape[1], np.float32)
        sp = int(split[i])
        for k in range(int(ct[i])):
            prod = L(wt[i, k]) * V[st[i] + k].astype(L)
            if sp and k >= sp:
                q = (prod + q.astype(L)).astype(np.float32)
            else:
                p = (prod + p.astype(L)).astype(np.float32)
        out[i] = (p + q).astype(np.float32) if sp else p
    return out


def _find_splits(V, G, st, ct, wt, what):
    """G[i,:] (원래 결과) 를 설명하는 출력별 분할 위치."""
    split = np.zeros(224, np.int32)
    base = emu(V, st, ct, wt, split)
    for i in range(224):
        if np.array_equal(base[i], G[i]):
            continue
        for sp in range(1, int(ct[i])):
            s = split.copy()
            s[i] = sp
            if np.array_equal(emu(V, st, ct, wt, s, rows=[i])[i], G[i]):
                split[i] = sp
                break
        else:
            raise RuntimeError(f"{what}: 출력 {i} 를 '구간별 순차 FMA' 모형으로 설명할 수 없다 — 커널을 못 맞춘다")
    return split


def calibrate(m: int, seed: int = 0, trials: int = 4):
    """입력 크기 m 에 대해 (st, ct, wt, split_rows, split_cols) — 원래 GPU 크기 조정과 같은 순서."""
    from openpi.shared import image_tools

    rng = np.random.default_rng(seed)
    W = weights(m)
    st, ct, wt = taps(W)
    zero = np.zeros(224, np.int32)
    split_cols = np.zeros(224, np.int32)
    split_rows = np.zeros(224, np.int32)
    for t in range(trials):
        # 2단계(열) 경계
        X = np.zeros((m, m, 3), np.float32)
        r0 = (t * 5) % SPACING
        X[r0::SPACING] = rng.uniform(-0.9, 0.9, X[r0::SPACING].shape).astype(np.float32)
        out = np.asarray(image_tools.resize_with_pad(X, 224, 224))  # [224(i),224(j),3]
        T = emu(X.reshape(m, 3 * m), st, ct, wt, zero)  # 1단계: 곱 하나라 순서 무관
        T = np.stack([T[i] for i in range(224)])  # [224, 3m]
        V = T.reshape(224, m, 3).transpose(1, 0, 2).reshape(m, 224 * 3)  # [w, (i,c)]
        G = out.transpose(1, 0, 2).reshape(224, 224 * 3)  # [j, (i,c)]
        s = _find_splits(V, G, st, ct, wt, "열 방향")
        split_cols = _merge(split_cols, s)
        # 1단계(행) 경계
        X = np.zeros((m, m, 3), np.float32)
        c0 = (t * 7) % SPACING
        X[:, c0::SPACING] = rng.uniform(-0.9, 0.9, X[:, c0::SPACING].shape).astype(np.float32)
        out = np.asarray(image_tools.resize_with_pad(X, 224, 224))
        # 2단계는 곱 하나: O[i,j,c] = f32(T[i,w0,c] * W[w0,j]), w0 = j 의 탭 중 값이 있는 열
        V1 = X.reshape(m, 3 * m)
        cols = np.full(224, -1)
        for j in range(224):
            nz = [w for w in range(st[j], st[j] + ct[j]) if (w - c0) % SPACING == 0]
            if len(nz) == 1:
                cols[j] = nz[0]
        js = np.nonzero(cols >= 0)[0]
        L = np.longdouble

        def predict(Ti_row):  # Ti_row [3m] → [len(js)*3]
            Tw = Ti_row.reshape(m, 3)[cols[js]]  # [nj, 3]
            wj = np.array([W[cols[j], j] for j in js], np.float32)[:, None]
            return (Tw.astype(L) * wj.astype(L)).astype(np.float32).reshape(-1)

        s = np.zeros(224, np.int32)
        base = emu(V1, st, ct, wt, zero)
        for i in range(224):
            g = out[i, js, :].reshape(-1)
            if np.array_equal(predict(base[i]), g):
                continue
            for sp in range(1, int(ct[i])):
                z = zero.copy()
                z[i] = sp
                if np.array_equal(predict(emu(V1, st, ct, wt, z, rows=[i])[i]), g):
                    s[i] = sp
                    break
            else:
                raise RuntimeError(f"행 방향: 출력 {i} 를 모형으로 설명할 수 없다")
        split_rows = _merge(split_rows, s)
    return st, ct, wt, split_rows, split_cols


def _merge(a, b):
    """시도끼리 합친다. 같은 출력에서 서로 다른 위치가 나오면 원래 크기 조정이 결정적이지 않다는 뜻 → 멈춘다."""
    both = (a > 0) & (b > 0)
    if (a[both] != b[both]).any():
        raise RuntimeError("시도마다 다른 분할 위치가 나왔다 — 원래 크기 조정이 결정적이지 않다")
    return np.maximum(a, b)


def boundaries(st, split):
    """출력별 분할 위치 → 입력 좌표의 경계 목록 (문서·로그용)."""
    return sorted({int(st[i] + split[i]) for i in range(len(st)) if split[i]})


def validate(resize, sizes=(720, 480), n: int = 24, seed: int = 1):
    """uint8 무작위 영상으로 원래 함수(영상 한 장씩, 원래처럼) vs 커널. 크기별 다른 값 개수 (0 이어야 한다).
    resize: uint8 [n,W,W,3] 호스트 배열 → [n,224,224,3] (네이티브 엔진, fast.Engine.resize)."""
    from openpi.shared import image_tools

    rng = np.random.default_rng(seed)
    res = {}
    for m in sizes:
        X = rng.integers(0, 256, (n, m, m, 3), dtype=np.uint8)
        ref = np.stack([np.asarray(image_tools.resize_with_pad(x, 224, 224)) for x in X])
        res[m] = int((resize(X) != ref).sum())
    return res


def calibrate_all(sizes=(720, 480)):
    S, C, Wt, SR, SC = [], [], [], [], []
    for m in sizes:
        st, ct, wt, sr, sc = calibrate(m)
        S.append(st), C.append(ct), Wt.append(wt), SR.append(sr), SC.append(sc)
    return tuple(np.stack(a) for a in (S, C, Wt, SR, SC))


if __name__ == "__main__":
    for m in (720, 480):
        st, ct, wt, sr, sc = calibrate(m)
        print(f"{m}: 행 방향 경계 {boundaries(st, sr)}, 열 방향 경계 {boundaries(st, sc)}")
