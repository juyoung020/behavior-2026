# 층 0(particles) 정답지: torch CPU 난수 — 생성기 상태(get_rng_state) → T.random_quaternion(n) 출력·뒤 상태, th.randint 한 원소.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_trng_ref.py --out ~/engine-data/particles/trng
import argparse
import os

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(23)
    # 예열 (평가기처럼 동적 모양 커널) — docs 12.7
    for k in (1, 3, 7, 50):
        T.random_quaternion(k)
    states, ns, quats, after, rint = [], [], [], [], []
    rstates, rvals, rlens = [], [], []
    nstates, nvals, nlens = [], [], []
    maxn = 64
    for i in range(a.n):
        th.manual_seed(int(rng.integers(0, 2**63)))
        # 생성기를 아무 곳으로 흘려 보냄 (left·next 가 여러 값이 되게)
        for _ in range(int(rng.integers(0, 3))):
            th.rand(int(rng.integers(1, 700)))
        st = th.get_rng_state().numpy().copy()
        n = int(rng.integers(1, maxn + 1))
        q = T.random_quaternion(n).numpy()
        after.append(th.get_rng_state().numpy().copy())
        qq = np.zeros((maxn, 4), np.float32)
        qq[:n] = q
        states.append(st)
        ns.append(n)
        quats.append(qq)
        rint.append(int(th.randint(-(2**63), 2**63 - 1, [1])[0]))
        # eager th.rand (뿌리기 표본 th.rand(n, 2) 와 같은 꼴)
        st2 = th.get_rng_state().numpy().copy()
        k = int(rng.integers(1, 3000))
        rr = th.rand(k, 2).numpy().ravel()
        rstates.append(st2)
        rvals.append(np.pad(rr, (0, 6000 - len(rr))))
        rlens.append(len(rr))
        # eager th.randn(3) 여러 번 (캐시가 이어지는 꼴: get_parallel_rays 의 random_vector)
        st3 = th.get_rng_state().numpy().copy()
        rn = np.concatenate([th.randn(3).numpy() for _ in range(int(rng.integers(1, 6)))])
        nstates.append(st3)
        nvals.append(np.pad(rn, (0, 15 - len(rn))))
        nlens.append(len(rn))
    np.save(os.path.join(a.out, "state.npy"), np.stack(states))
    np.save(os.path.join(a.out, "after.npy"), np.stack(after))
    np.save(os.path.join(a.out, "n.npy"), np.array(ns, np.int32))
    np.save(os.path.join(a.out, "quat.npy"), np.stack(quats))
    np.save(os.path.join(a.out, "randint_next.npy"), np.array(rint, np.int64))
    np.save(os.path.join(a.out, "rand_state.npy"), np.stack(rstates))
    np.save(os.path.join(a.out, "rand_vals.npy"), np.stack(rvals).astype(np.float32))
    np.save(os.path.join(a.out, "rand_len.npy"), np.array(rlens, np.int32))
    np.save(os.path.join(a.out, "randn_state.npy"), np.stack(nstates))
    np.save(os.path.join(a.out, "randn_vals.npy"), np.stack(nvals).astype(np.float32))
    np.save(os.path.join(a.out, "randn_len.npy"), np.array(nlens, np.int32))
    print("done", a.n, "state bytes", states[0].shape)


if __name__ == "__main__":
    main()
