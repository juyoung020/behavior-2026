# 동기화 두 벌 정답 뽑기 (공식 K+1 창의 omni.physx USD→PhysX 쓰기): 물체 넣기(add_object)의 update_handles → psi.flush_changes 가
# 모든 강체 행위자(정적·동적)에 자세를 다시 쓰고 동적엔 속도 0 을 쓴다 (OVD API set 으로만 남음, 곁기록엔 없음).
#   python3 sync_sweep_to_txt.py <수확 기록 폴더(HARVEST_BASE=1, 척도 포함)> <새 물체 이름(예 half_log_174_0)> <출력 txt>
# 출력 줄:
#   S <벌 번호> <행위자 이름> <종류 static|dynamic> q4 p3            — 벌 안 차례대로 (자세 쓰기)
#   V <벌 번호> <행위자 이름> lin3 ang3                               — 그 행위자의 속도 쓰기 (자세 바로 뒤)
#   L <행위자 이름> q4 p3                                             — 벌 앞 마지막 자세 쓰기 (load_state 값 = 벌 입력)
#   O <물체 이름> <뿌리 링크> <척도 3> <동적 링크 수> <링크...>        — 그 창 dump 때 물체 등록부 차례
import json
import os
import re
import subprocess
import sys

rec, newname, out = sys.argv[1], sys.argv[2], sys.argv[3]
D = os.path.expanduser("~/engine-build/replay-checked/ovd_dump")
ovd = sorted([f for f in os.listdir(rec) if f.endswith("_rec.ovd")], key=lambda f: os.path.getsize(os.path.join(rec, f)))[-1]
F = os.path.join(rec, ovd)


def trace(arg):
    return subprocess.run([D, F, "--trace", arg, "--limit", "100000000"], capture_output=True, text=True, errors="replace").stdout.splitlines()


# 새 물체의 첫 사건 번호 → 그 앞뒤 프레임 경계
first = None
for ln in trace(f"/World/scene_0/{newname}/base_link"):
    m = re.match(r"\s*#(\d+)", ln)
    if m:
        first = int(m.group(1))
        break
assert first is not None, "새 물체 사건 없음"
lines = trace(f"#{max(0, first - 3000)}-{first + 3000}")
ev = []
for ln in lines:
    m = re.match(r"\s*#(\d+) (\w+) obj=(0x[0-9a-f]+) ?(\S*) ?(.*)$", ln.rstrip())
    if m:
        ev.append((int(m.group(1)), m.group(2), m.group(3), m.group(4), m.group(5)))
start = max(i for i, c, *_ in ev if c == "startFrame" and i < first)
stop = min(i for i, c, *_ in ev if c == "stopFrame" and i > first)
blk = [e for e in ev if start < e[0] < stop]
# 이름·종류 (창 앞까지 + 창 안)
names, cls = {}, {}
for ln in trace(f"#0-{stop}"):
    m = re.search(r"#(\d+) create obj=(0x[0-9a-f]+) (\S+)", ln)
    if m:
        cls[m.group(2)] = m.group(3)
    m = re.search(r"obj=(0x[0-9a-f]+) PxActor\.name \"(.*)$", ln.rstrip())
    if m:
        names[m.group(1)] = m.group(2).rstrip('"')
# 벌 = 정적 행위자의 자세 쓰기 묶음: 창 안 정적 자세 쓰기를 차례로 보고, 같은 정적이 다시 나오면 다음 벌
sweeps, cur, seen, last_pose, load_pose = [], [], set(), {}, {}
blocks = []
for i, c, h, a, v in blk:
    if a == "PxRigidActor.globalPose":
        k = cls.get(h, "?")
        if k == "PxRigidStatic":
            if h in seen:
                sweeps.append(cur)
                cur, seen = [], set()
            seen.add(h)
        blocks.append((i, h, k, v))
# 벌 경계: 정적 첫 쓰기 번호들
stat_first = []
seen = {}
for i, h, k, v in blocks:
    if k == "PxRigidStatic":
        seen.setdefault(h, []).append(i)
# 벌 = 정적 행위자의 마지막 두 쓰기 (앞의 쓰기는 운동학 물체 되돌리기 등)
multi = [L for L in seen.values() if len(L) >= 2]
n_sweep = 2
bounds = [min(L[-2] for L in multi), min(L[-1] for L in multi)]
bounds.append(stop)
rows = []
for s in range(n_sweep):
    lo, hi = bounds[s], bounds[s + 1]
    # 벌 s 의 끝 = 다음 벌 시작 앞의 마지막 자세 쓰기
    for idx, (i, c, h, a, v) in enumerate(blk):
        if not (lo <= i < hi):
            continue
        if a == "PxRigidActor.globalPose":
            k = "static" if cls.get(h) == "PxRigidStatic" else "dynamic"
            rows.append(f"S {s} {names.get(h, '?' + h)} {k} " + v.strip("[]"))
        elif a in ("PxRigidBody.linearVelocity",):
            nxt = blk[idx + 1]
            rows.append(f"V {s} {names.get(h, '?' + h)} " + v.strip("[]") + " " + nxt[4].strip("[]"))
# 벌 앞 마지막 자세 (벌 입력)
for i, c, h, a, v in blk:
    if a == "PxRigidActor.globalPose" and i < bounds[0]:
        load_pose[h] = v
for h, v in load_pose.items():
    rows.append(f"L {names.get(h, '?' + h)} " + v.strip("[]"))
bp = os.path.join(rec, "harvest_base.json")
base = json.load(open(bp)) if os.path.exists(bp) else {"windows": []}
hm = json.load(open(os.path.join(rec, "harvest_map.json")))
step = [e["step"] for e in hm if any(n["name"] == newname for n in e.get("new", []))][0]
win = ([w for w in base["windows"] if w["step"] == step] + [{"objects": []}])[0]
for o in win["objects"]:
    rows.append(" ".join(["O", o["name"], o["root_link"]] + [repr(x) for x in o.get("scale", [1, 1, 1])] + [str(len(o["dynamic_links"]))] + o["dynamic_links"]))
open(out, "w").write("\n".join(rows) + "\n")
print("창", start, stop, "벌", n_sweep, "경계", bounds, "줄", len(rows))
