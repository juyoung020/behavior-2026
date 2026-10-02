# 층 0(omni) 정답지: 보조 잡기 관절 틀 계산 (robots/robot.py:3525 _establish_grasp) 을 공식 파이썬 그대로 돌려 입·출력을 적는다.
#   parent_frame = T.relative_pose_transform(contact, [0,0,0,1], eef_pos, eef_orn); pos / robot.scale
#   child_frame  = T.relative_pose_transform(contact, [0,0,0,1], obj_link_pos, obj_link_orn); pos / obj.scale
# T.relative_pose_transform 은 Linux 에서 torch.compile(inductor, CPU) 이다 (utils/python_utils.py:716). GPU 안 씀.
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_agframe_ref.py --out ~/engine-data/omni/agframe --n 20000
# 출력: agframe.bin (float32 행: 입력 contact3 eef_pos3 eef_q4 scale3 → 출력 pos3 q4), 한 줄 = 한 번의 relative_pose_transform
import argparse
import os

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T


def rand_quat(rng):
    r = rng.random()
    if r < 0.15:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    elif r < 0.3:  # z 축만
        a = rng.uniform(-np.pi, np.pi)
        q = np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])
    elif r < 0.4:  # 180 도 근처 (mat2quat 가지 경계)
        ax = rng.standard_normal(3)
        ax /= np.linalg.norm(ax)
        a = np.pi - rng.uniform(0, 1e-3)
        q = np.concatenate([ax * np.sin(a / 2), [np.cos(a / 2)]])
    else:
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
    q = q.astype(np.float32)
    if rng.random() < 0.3:  # PhysX 가 주는 것처럼 끝비트가 벗어난 단위 사원수
        q = (q * np.float32(1.0 + rng.uniform(-3e-7, 3e-7))).astype(np.float32)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    jq = th.tensor([0, 0, 0, 1.0])  # robot.py 와 같은 만들기 (float32)
    for i in range(a.n):
        base = rng.uniform(-3, 3, 3)
        eef = (base + rng.uniform(-0.2, 0.2, 3)).astype(np.float32)
        contact = (eef + rng.uniform(-0.08, 0.08, 3)).astype(np.float32)
        q = rand_quat(rng)
        scale = np.ones(3, np.float32) if rng.random() < 0.5 else rng.uniform(0.3, 2.0, 3).astype(np.float32)
        pos, orn = T.relative_pose_transform(th.from_numpy(contact), jq, th.from_numpy(eef), th.from_numpy(q))
        pos = pos / th.from_numpy(scale)
        rows.append(np.concatenate([contact, eef, q, scale, pos.numpy().astype(np.float32), orn.numpy().astype(np.float32)]))
    np.array(rows, np.float32).tofile(os.path.join(a.out, "agframe.bin"))
    print("done", a.n, th.__version__)


if __name__ == "__main__":
    main()
