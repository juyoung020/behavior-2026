# 층 0(omni) 정답지: AttachedTo 붙이기 판정 (object_states/attached_to.py:315 _find_attachment_links 의 정렬 시험)
#   pos_diff = th.norm(child_pos - parent_pos), orn_diff = T.get_orientation_diff_in_radian(child_orn, parent_orn)
#   붙음 = pos_diff < m.DEFAULT_POSITION_THRESHOLD and orn_diff < m.DEFAULT_ORIENTATION_THRESHOLD
# T.get_orientation_diff_in_radian 은 Linux 에서 torch.compile(inductor CPU). GPU 안 씀.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_attach_ref.py --out ~/engine-data/omni/attach --n 20000
# 출력: attach.bin float32 행 = child_pos3 child_q4 parent_pos3 parent_q4 | pos_diff orn_diff attached(0/1)
import argparse
import os

import numpy as np
import torch as th

import omnigibson.object_states.attached_to as AT
import omnigibson.utils.transform_utils as T


def rq(rng):
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pt, ot = AT.m.DEFAULT_POSITION_THRESHOLD, AT.m.DEFAULT_ORIENTATION_THRESHOLD
    rows = []
    for i in range(a.n):
        pp = rng.uniform(-2, 2, 3)
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        r = rng.random()
        dist = pt * (1 + rng.integers(-3, 4) * 1e-7) if r < 0.3 else rng.uniform(0, 0.1)
        cp = (pp + d * dist).astype(np.float32)
        pq = rq(rng)
        ax = rng.standard_normal(3)
        ax /= np.linalg.norm(ax)
        ang = ot * (1 + rng.integers(-3, 4) * 1e-7) if rng.random() < 0.3 else rng.choice([0.0, rng.uniform(0, 0.6), np.pi - 1e-4])
        dq = np.concatenate([ax * np.sin(ang / 2), [np.cos(ang / 2)]])
        x1, y1, z1, w1 = dq
        x0, y0, z0, w0 = pq
        cq = np.array([w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0, w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                       w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0, w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0])
        if rng.random() < 0.5:
            cq = -cq
        cq = cq.astype(np.float32)
        pq = pq.astype(np.float32)
        pp = pp.astype(np.float32)
        c_pos, c_orn, p_pos, p_orn = map(th.from_numpy, (cp, cq, pp, pq))
        pos_diff = th.norm(c_pos - p_pos)
        orn_diff = T.get_orientation_diff_in_radian(c_orn, p_orn)
        att = bool(pos_diff < pt and orn_diff < ot)
        rows.append(np.concatenate([cp, cq, pp, pq, [pos_diff.item(), orn_diff.item(), float(att)]]).astype(np.float32))
    np.array(rows, np.float32).tofile(os.path.join(a.out, "attach.bin"))
    print("done", a.n, pt, ot)


if __name__ == "__main__":
    main()
