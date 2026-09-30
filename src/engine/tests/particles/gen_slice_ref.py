# 층 0(particles) 정답지: 자르기 전이의 실수 부분 — 공식 함수를 가짜 물체에 붙여 그대로 부른다 (Kit·GPU 안 씀).
#   A. SlicingRule.transition (transition_rules.py:942): 반쪽마다 경계 상자 중심 자세·경계 상자 크기
#        part_bb_pos = pos + T.quat2mat(orn) @ (bb_pos * scale),  part_bb_orn = T.quat_multiply(orn, bb_orn),
#        bounding_box = bb_size * scale  (척도가 고르지 않으면 scale = |T.quat2mat(bb_orn) @ scale|)
#      + 반쪽들에 넘기는 상태가 모두 "마지막으로 자른 물체" 의 것인지 (lambda 늦은 묶임, :985)
#   B. 반쪽 물체 척도 = bounding_box / ig:nativeBB (dataset_object.py:383, 1e-4 넘는 축만)
#   C. set_bbox_center_position_orientation (dataset_object.py:472): 기준 링크 위치 = bb 중심 + T.pose_transform(0, orn, -scale*offsetBaseLink, 단위)[0]
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_slice_ref.py --out ~/engine-data/particles/slice
import argparse
import os

import numpy as np
import torch as th

import omnigibson.transition_rules as TR
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject


class FakeDO:
    """transition_rules.DatasetObject 대역: 만들 때 인자만 적는다"""

    made = []

    def __init__(self, **kw):
        self.kw = kw
        FakeDO.made.append(self)


class FakeSliceable:
    def __init__(self, name, pos, orn, scale, parts):
        self.name = name
        self._pos, self._orn = th.from_numpy(pos), th.from_numpy(orn)
        self.scale = th.from_numpy(scale)
        self.metadata = {"object_parts": parts}

    def get_position_orientation(self):
        return self._pos.clone(), self._orn.clone()

    def dump_state(self):
        return {"from": self.name}


class FakeBBoxObj:
    """set_bbox_center_position_orientation 대역: 척도·offsetBaseLink 만 두고 공식 메서드·property 를 붙인다"""

    set_bbox_center_position_orientation = DatasetObject.set_bbox_center_position_orientation
    scaled_bbox_center_in_base_frame = DatasetObject.__dict__["scaled_bbox_center_in_base_frame"]

    def __init__(self, scale, offset):
        self.scale = scale
        self.base_link_offset = offset
        self.got = None

    def set_position_orientation(self, position=None, orientation=None):
        self.got = (position, orientation)


def rand_quat(rng):
    q = rng.standard_normal(4)
    return (q / np.linalg.norm(q)).astype(np.float32)


def warm_torch_compile():
    """평가기 프로세스처럼 T.quat2mat 을 여러 모양으로 먼저 불러 마지막 축까지 동적인 커널이 쓰이게 한다 (docs 20.5)"""
    for shp in [(4,), (3, 4), (5, 4), (2, 5), (3, 6)]:
        T.quat2mat(th.rand(*shp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    warm_torch_compile()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    TR.DatasetObject = FakeDO
    rule = TR.SlicingRule.__new__(TR.SlicingRule)
    rows_in, rows_out, late = [], [], []
    for i in range(a.n):
        k = int(rng.integers(1, 4))  # 한 스텝에 자른 물체 수
        objs = []
        for j in range(k):
            pos = rng.uniform(-3, 3, 3).astype(np.float32)
            orn = rand_quat(rng)
            u = rng.random()
            if u < 0.4:
                scale = np.full(3, rng.uniform(0.5, 1.5), np.float32)
            elif u < 0.5:
                scale = np.ones(3, np.float32)
            else:
                scale = rng.uniform(0.5, 1.5, 3).astype(np.float32)
            n_parts = 2
            parts = {}
            for p in range(n_parts):
                # 부분 방향은 공식 가정대로 90 도 배수 (고르지 않은 척도일 때 assert)
                ax = rng.integers(3)
                ang = np.pi / 2 * rng.integers(4)
                q = np.zeros(4)
                q[ax] = np.sin(ang / 2)
                q[3] = np.cos(ang / 2)
                parts[str(p)] = dict(category=f"half_x{j}", model=f"m{p}", bb_pos=rng.uniform(-0.2, 0.2, 3).tolist(),
                                     bb_orn=q.tolist(), bb_size=rng.uniform(0.02, 0.4, 3).tolist())
            objs.append(FakeSliceable(f"obj{i}_{j}", pos, orn, scale, parts))
        FakeDO.made = []
        res = rule.transition({"sliceable": objs, "slicer": []})
        m = 0
        for j, o in enumerate(objs):
            for p, part in enumerate(o.metadata["object_parts"].values()):
                at = res.add[m]
                d = FakeDO.made[m]
                m += 1
                pos, orn = o.get_position_orientation()
                rows_in.append(np.concatenate([pos.numpy(), orn.numpy(), o.scale.numpy(), np.float32(part["bb_pos"]),
                                               np.float32(part["bb_orn"]), np.float32(part["bb_size"])]))
                bb = d.kw["bounding_box"]
                # B. 반쪽 척도 (ig:nativeBB 는 USD float3 → 파이썬 float → th.tensor float32)
                native = rng.uniform(0.02, 0.5, 3).astype(np.float32)
                if rng.random() < 0.05:
                    native[rng.integers(3)] = 0.0  # 1e-4 이하 축은 척도 1
                native_t = th.tensor([float(x) for x in native])
                bbt = th.as_tensor(bb, dtype=th.float32)
                scale_h = th.ones(3)
                valid = native_t > 1e-4
                scale_h[valid] = bbt[valid] / native_t[valid]
                # C. 기준 링크 자세
                offset = th.tensor([float(x) for x in rng.uniform(-0.1, 0.1, 3).astype(np.float32)])
                fo = FakeBBoxObj(scale_h, offset)
                fo.set_bbox_center_position_orientation(position=at.bb_pos, orientation=at.bb_orn)
                rows_out.append(np.concatenate([at.bb_pos.numpy(), at.bb_orn.numpy(), bb.numpy(), native, scale_h.numpy(),
                                                offset.numpy().astype(np.float32), fo.got[0].numpy(), fo.got[1].numpy()]).astype(np.float32))
                # 늦은 묶임: 콜백이 넘기는 상태가 누구 것인가
                got = {}
                at.callback(type("X", (), {"load_non_kin_state": lambda self, s: got.update(s)})())
                late.append([j, int(got["from"].split("_")[-1]), len(objs)])
    np.save(os.path.join(a.out, "slice_in.npy"), np.array(rows_in, np.float32))
    np.save(os.path.join(a.out, "slice_out.npy"), np.array(rows_out, np.float32))
    np.save(os.path.join(a.out, "slice_late.npy"), np.array(late, np.int32))
    print("done parts", len(rows_in))


if __name__ == "__main__":
    main()
