# 층 0(omni) 정답지: AttachedTo._attach (object_states/attached_to.py:393) 고정 관절 자세 맞춤 — 공식 식을 그대로 돌린다.
#   rel = T.mat2pose(T.pose2mat(parent) @ T.pose_inv(T.pose2mat(child)));  new_root = T.pose_transform(rel, child_root)
#   WSL: python gen_attach_pose_ref.py --out ~/engine-data/omni/attach --n 20000   (GPU 안 씀)
# 출력: attach_pose.bin float32 행 = parent p3 q4, child p3 q4, root p3 q4 | new_root p3 q4
import argparse
import os

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T


def rq(rng):
    q = rng.standard_normal(4)
    q = (q / np.linalg.norm(q)).astype(np.float32)
    if rng.random() < 0.3:
        q = (q * np.float32(1.0 + rng.uniform(-3e-7, 3e-7))).astype(np.float32)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(13)
    rows = []
    for _ in range(a.n):
        pp = rng.uniform(-2, 2, 3).astype(np.float32)
        cp = (pp + rng.uniform(-0.05, 0.05, 3)).astype(np.float32)
        rp = (cp + rng.uniform(-0.3, 0.3, 3)).astype(np.float32)
        pq, cq, rqq = rq(rng), rq(rng), rq(rng)
        t = lambda x: th.from_numpy(x)
        rel_pos, rel_quat = T.mat2pose(T.pose2mat((t(pp), t(pq))) @ T.pose_inv(T.pose2mat((t(cp), t(cq)))))
        npos, nq = T.pose_transform(rel_pos, rel_quat, t(rp), t(rqq))
        rows.append(np.concatenate([pp, pq, cp, cq, rp, rqq, npos.numpy(), nq.numpy()]).astype(np.float32))
    np.array(rows, np.float32).tofile(os.path.join(a.out, "attach_pose.bin"))
    print("done", a.n)


if __name__ == "__main__":
    main()
