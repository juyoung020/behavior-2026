"""카메라 외부 자세(robot2cam)를 proprio 관절값 + R1Pro 순기구학(URDF)으로 만들 수 있는지 확인하고, 링크 → 카메라 prim
고정 변환을 시연 데이터로 뽑는다(일회성 도구, 학습 데이터만 씀 — 평가 때는 뽑은 상수만 쓴다).

    (WSL) ~/meridian_venv/bin/python /mnt/c/behavior-2026/src/integ/fk/fit_cam_fk.py [--episodes 0 200] [--out …json]
    Windows conda behavior 의 numpy 는 3x3 행렬곱에서 BLAS 지연 로드 오류(0xc06d007f)로 죽어서 WSL 에서 돌린다.

이유: 평가기 관측의 `cam_rel_poses` 는 평가기가 시뮬레이터 카메라 자세·로봇 전역 자세 API 로 계산한 값이라, 규칙 해석상
쓰지 않는다(코디네이터 09-30). 대신 proprio 의 trunk_qpos(4)·arm_*_qpos(7) + URDF 순기구학 + 고정 변환으로 만든다.
- 체인: base_link → torso_joint1..4 → zed_joint → zed_link (머리), … → left/right_arm_joint1..7 → *_gripper_joint → *_realsense_joint (손목)
- 시연의 observation.robot2cam_pose.<카메라> 와 비교: 고정 변환 T_link_cam = FK(link)⁻¹ · robot2cam 이 프레임마다 같으면(표준편차 ~0)
  순기구학 + 상수 하나로 robot2cam 을 그대로 재현한다.
출력 JSON(Rust simlink 가 읽음): 체인(관절 원점 xyz·rpy, 축, proprio 번호) + 카메라별 고정 변환 + 재현 오차.
"""
import argparse
import json
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = pathlib.Path("C:/behavior-2026" if __import__("os").name == "nt" else "/mnt/c/behavior-2026")
URDF = ROOT / "BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
DEMOS = ROOT / "data/2026-challenge-demos"
# proprio 번호(OmniGibson eval_utils PROPRIOCEPTION_INDICES["R1Pro"])
QIDX = {**{f"torso_joint{i + 1}": 53 + i for i in range(4)},
        **{f"left_arm_joint{i + 1}": 3 + i for i in range(7)},
        **{f"right_arm_joint{i + 1}": 28 + i for i in range(7)}}
CAMS = {"head": ("zed_link", "zed_link_camera_0"),
        "left_wrist": ("left_realsense_link", "left_realsense_link_camera_0"),
        "right_wrist": ("right_realsense_link", "right_realsense_link_camera_0")}


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(a, t):
    a = np.asarray(a, float)
    a = a / (np.linalg.norm(a) or 1.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def quat_mat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def mat_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def load_urdf():
    root = ET.parse(URDF).getroot()
    by_child = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        ax = j.find("axis")
        axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
        by_child[j.find("child").get("link")] = {"name": j.get("name"), "type": j.get("type"),
                                                 "parent": j.find("parent").get("link"), "xyz": xyz, "rpy": rpy, "axis": axis}
    return by_child


def chain_to(by_child, link):
    out = []
    while link in by_child:
        j = by_child[link]
        out.append(j)
        link = j["parent"]
    return list(reversed(out)), link  # (관절들, 뿌리 링크)


def fk(chain, q):
    R, t = np.eye(3), np.zeros(3)
    for j in chain:
        Rj = rpy_mat(*j["rpy"])
        t = t + R @ np.asarray(j["xyz"])
        R = R @ Rj
        if j["type"] in ("revolute", "continuous"):
            R = R @ axis_angle(j["axis"], q[QIDX[j["name"]]])
        elif j["type"] == "prismatic":
            t = t + R @ (np.asarray(j["axis"]) * q[QIDX[j["name"]]])
    return R, t


def load_frames(episodes):
    files = sorted((DEMOS / "data").glob("chunk-*/file-*.parquet"))
    cols = ["episode_index", "observation.state"] + [f"observation.robot2cam_pose.{c[1]}" for c in CAMS.values()]
    got = {}
    for f in files:
        t = pq.read_table(f, columns=cols)
        m = t.filter(pc.is_in(t["episode_index"], value_set=__import__("pyarrow").array(episodes)))
        for e in set(m["episode_index"].to_pylist()):
            s = m.filter(pc.equal(m["episode_index"], e))
            got[e] = {k: np.asarray(s[k].to_pylist(), float) for k in cols[1:]}
        if len(got) == len(episodes):
            break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 200, 5000, 9000])
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "src/integ/fk/r1pro_cam_fk.json"))
    a = ap.parse_args()
    by_child = load_urdf()
    data = load_frames(a.episodes)
    print("episodes:", {e: len(v["observation.state"]) for e, v in data.items()})
    out = {"source": "URDF " + str(URDF.relative_to(ROOT)).replace("\\", "/") + " + 시연 robot2cam 에서 뽑은 고정 변환(fit_cam_fk.py)",
           "proprio_index": QIDX, "cams": {}}
    for cam, (link, key) in CAMS.items():
        chain, root = chain_to(by_child, link)
        offs, fks, r2cs = [], [], []
        for e, v in data.items():
            st, r2c = v["observation.state"], v[f"observation.robot2cam_pose.{key}"]
            for i in range(0, len(st), a.every):
                R, t = fk(chain, st[i])
                Rc = quat_mat(*r2c[i, 3:7])
                tc = r2c[i, :3]
                # T_link_cam = FK⁻¹ · r2c
                offs.append((R.T @ Rc, R.T @ (tc - t)))
                fks.append((R, t))
                r2cs.append((Rc, tc))
        # 고정 변환 = 평균(회전은 첫 값 기준 투영)
        Ro = np.mean([o[0] for o in offs], axis=0)
        U, _, Vt = np.linalg.svd(Ro)
        Ro = U @ Vt
        to = np.mean([o[1] for o in offs], axis=0)
        perr = [np.linalg.norm(R @ to + t - tc) for (R, t), (Rc, tc) in zip(fks, r2cs)]
        aerr = [math.degrees(math.acos(max(-1.0, min(1.0, (np.trace((R @ Ro).T @ Rc) - 1) / 2)))) for (R, _), (Rc, _) in zip(fks, r2cs)]
        spread = float(np.max([np.linalg.norm(o[1] - to) for o in offs]))
        print(f"{cam}: 뿌리 {root}, 관절 {[j['name'] for j in chain if j['type'] != 'fixed']}, 표본 {len(offs)}, "
              f"위치 오차 최대 {max(perr) * 1000:.3f} mm (중앙 {np.median(perr) * 1000:.3f}), 각 오차 최대 {max(aerr):.4f}°, "
              f"고정 변환 흩어짐 {spread * 1000:.3f} mm")
        out["cams"][cam] = {"link": link, "root": root,
                            "chain": [{k: j[k] for k in ("name", "type", "xyz", "rpy", "axis")} | {"q": QIDX.get(j["name"], -1)} for j in chain],
                            "link_to_cam": {"xyz": to.tolist(), "xyzw": mat_quat(Ro)},
                            "check": {"samples": len(offs), "pos_err_max_mm": max(perr) * 1000, "pos_err_p50_mm": float(np.median(perr)) * 1000,
                                      "ang_err_max_deg": max(aerr), "episodes": a.episodes}}
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", a.out)


if __name__ == "__main__":
    main()
