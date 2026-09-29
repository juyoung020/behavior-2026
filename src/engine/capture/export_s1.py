"""Linux 공식 기록에서 S1(제어기 오프라인 비교, 문서 15절) 입력을 만든다. 저장소 밖에 쓴다(에셋 파생물 폴더 안).

    python export_s1.py <기록 폴더>      -> <기록 폴더>/s1_setup.txt, s1_actions.bin

s1_setup.txt (한 줄에 한 항목, 공백 구분)
    episode_start_post N       에피소드 첫 스텝 직전까지의 simulate 수 (meta.json)
    post_step_count N          프로세스 전체 simulate 수 (meta.json) -> OVD 파일 앞 몫 계산에 쓴다
    substeps K                 스텝당 물리 서브스텝 수 ((post_step_count - episode_start_post) / 스텝 수)
    base_link PATH             robot.get_position_orientation() 의 링크 (robot.py:3879 base_footprint_link)
    root_link PATH             robot.get_root_position_orientation() 의 링크 (관절체 뿌리)
    n_dof N
    group NAME d0 d1 ...       base(3: x,y,rz) trunk(4) arm_left(7) gripper_left(2) arm_right(7) gripper_right(2) -- 제어기 dof_idx
    base_limits in_lo(3) in_hi(3) out_lo(3) out_hi(3)
    dof d CHILD_LINK_PATH sign pos_lo pos_hi vel_lo vel_hi has_limit   -- 뷰의 dof 순서(= 로봇 관절 순서), 한계는 그 dof 를 가진 제어기 설정의 값
s1_actions.bin: u32 T, u32 A, f32[T*A]  (이 실행이 실제로 쓴 행동, trace 의 actions.npz)
실수는 float32 값을 repr 로 적어 C++ 에서 strtof 로 같은 비트로 읽는다.
"""
import json
import os
import struct
import sys

import numpy as np

GROUPS = ["base", "trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"]


def f32s(x):
    return repr(float(np.float32(x)))


def main():
    d = sys.argv[1]
    meta = json.load(open(os.path.join(d, "meta.json")))
    scope = json.load(open(os.path.join(d, "scope.json")))[0]
    views = json.load(open(os.path.join(d, "views.json")))
    rob = scope["robot"]
    cc = rob["controller_config"]
    acts = np.load(os.path.join(d, "actions.npz"))["actions"].astype(np.float32)  # (T, N, A)
    T, A = acts.shape[0], acts.shape[2]
    # 로봇 관절체 뷰(가장 마지막 것): dof 경로와 부호
    rv = [v for v in views.values() if v.get("kind") == "art" and any(p.endswith(rob["prim_path"].split("/")[-1]) for p in v.get("prims", []))]
    if not rv:
        sys.exit("로봇 관절체 뷰를 views.json 에서 못 찾음")
    v = rv[-1]
    dof_paths = v["dof_paths"][0]
    signs = v["signs"][0]
    n = len(dof_paths)
    joint_by_path = {j["prim_path"]: j for j in rob["joints"].values()}
    owner = {}
    for g in GROUPS:
        for k in cc[g]["dof_idx"]:
            owner[k] = g
    sub = (meta["post_step_count"] - meta["episode_start_post"]) // T
    lines = [f"episode_start_post {meta['episode_start_post']}", f"post_step_count {meta['post_step_count']}", f"substeps {sub}",
             f"base_link {rob['base_footprint_link']}", f"root_link {rob['root_link']}", f"n_dof {n}"]
    for g in GROUPS:
        lines.append("group " + g + " " + " ".join(str(k) for k in cc[g]["dof_idx"]))
    b = cc["base"]
    lines.append("base_limits " + " ".join(f32s(x) for x in (*b["command_input_limits"][0], *b["command_input_limits"][1],
                                                             *b["command_output_limits"][0], *b["command_output_limits"][1])))
    for k, p in enumerate(dof_paths):
        j = joint_by_path.get(p)
        if j is None:
            sys.exit(f"관절 {p} 를 scope.json 에서 못 찾음")
        cl = cc[owner.get(k, "base")]["control_limits"]
        lines.append(f"dof {k} {j['body1']} {int(signs[k])} {f32s(cl['position'][0][k])} {f32s(cl['position'][1][k])} "
                     f"{f32s(cl['velocity'][0][k])} {f32s(cl['velocity'][1][k])} {int(bool(cl['has_limit'][k]))}")
    with open(os.path.join(d, "s1_setup.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "s1_actions.bin"), "wb") as f:
        f.write(struct.pack("<II", T, A))
        f.write(np.ascontiguousarray(acts[:, 0, :]).tobytes())
    print(f"S1 입력: 스텝 {T}, 행동 {A}, dof {n}, 서브스텝 {sub} -> {d}")


if __name__ == "__main__":
    main()
