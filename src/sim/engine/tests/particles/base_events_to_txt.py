# harvest_base.json (harvest_spawn.py HARVEST_BASE=1: removing_objects 의 상태 뜨기·되돌리기 정답) → test_base_state 가 읽는 글줄
#   python3 base_events_to_txt.py <수확 기록 폴더> <출력 파일>
# 줄: W 스텝 물체수 / R 지운 물체 이름 / O 이름 종류(0 강체 1 관절체 2 운동학) 관절수 뿌리링크 링크수 링크... / D 이름 잠 p3 q4 선3 각3 관절위치수 ... 관절속도수 ...
#     C 호출 토큰... (obj 이름 | read 경로 p3 q4 | set_position_orientation 경로 p3 q4 | set_linear_velocity 경로 v3 | set_angular_velocity 경로 v3
#                    | set_joint_positions 경로 n v... | set_joint_velocities 경로 n v... | rigid_sleep 경로 | rigid_wake 경로 | art_sleep 이름 | art_wake 이름)
#     E (창 끝)
import json
import os
import sys

d, out = sys.argv[1], sys.argv[2]
rec = json.load(open(os.path.join(d, "harvest_base.json")))
hmap = json.load(open(os.path.join(d, "harvest_map.json")))  # 전이마다 원본(= 지운 물체)과 스텝


def flat(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        r = []
        for y in x:
            r += flat(y)
        return r
    return [x]


def num(v):
    return repr(float(v))


lines = []
for w in rec["windows"]:
    objs = w["objects"]
    lines.append(f"W {w['step']} {len(objs)}")
    for o in objs:
        kind = 1 if o["articulated"] else (2 if o["kinematic_only"] else 0)
        lines.append(" ".join(["O", o["name"], str(kind), str(o["n_joints"]), o["root_link"], str(len(o["dynamic_links"]))] + o["dynamic_links"]))
    for e in hmap:
        if e["step"] == w["step"]:
            lines.append(f"R {e['src']}")  # removing_objects 가 상태에서 뺀 물체
    for name, s in w["dump"].items():
        jp, jv = flat(s.get("jpos")), flat(s.get("jvel"))
        lin, ang = flat(s.get("lin")) or [0.0] * 3, flat(s.get("ang")) or [0.0] * 3
        toks = ["D", name, str(int(s["asleep"]))] + [num(v) for v in flat(s["pos"]) + flat(s["ori"]) + lin + ang]
        toks += [str(len(jp))] + [num(v) for v in jp] + [str(len(jv))] + [num(v) for v in jv]
        lines.append(" ".join(toks))
    for c in w["calls"]:
        k = c[0]
        if k in ("obj", "rigid_sleep", "rigid_wake", "art_sleep", "art_wake"):
            lines.append(f"C {k} {c[1]}")
        elif k in ("set_joint_positions", "set_joint_velocities"):
            v = flat(c[2])
            lines.append(" ".join(["C", k, c[1], str(len(v))] + [num(x) for x in v]))
        else:
            lines.append(" ".join(["C", k, c[1]] + [num(x) for x in flat(c[2:])]))
    lines.append("E")
open(out, "w").write("\n".join(lines) + "\n")
print("창", len(rec["windows"]), "호출", sum(len(w["calls"]) for w in rec["windows"]))
