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
for ln in trace(newname if newname.startswith("/") else f"/World/scene_0/{newname}/base_link"):
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
# 벌 나누기: 정적 자세 쓰기를 차례로 보며 같은 정적이 다시 나오면 새 묶음. 정적 수의 절반 넘게 든 묶음만 벌
# (앞쪽 작은 묶음 = 운동학 물체 되돌리기의 XForm 쓰기 등)
runs, cur_run, cur_set = [], [], set()
for i, h, k, v in blocks:
    if k != "PxRigidStatic":
        continue
    if h in cur_set:
        runs.append(cur_run)
        cur_run, cur_set = [], set()
    cur_set.add(h)
    cur_run.append(i)
runs.append(cur_run)
n_static = len(seen)
big = [r for r in runs if len(r) > n_static // 2]
n_sweep = len(big)
bounds = [r[0] for r in big]
bounds.append(stop)
# 벌 안 차례는 실행마다 달라질 수 있어(omni.physx 안쪽 용기) 행위자별로 가른다: 창 앞부터 있던 행위자의 마지막 n_sweep 번 쓰기 = 벌 0..n-1,
# 그 앞 쓰기 = 벌 입력. 창 안에서 새로 생긴 행위자(반쪽)는 뺀다. 차례는 벌 번호별 창 안 사건 번호 차례로 적는다(참고용).
created_in = {h for i, c, h, a, v in blk if c == "create"}
writes = {}
for idx, (i, c, h, a, v) in enumerate(blk):
    if a == "PxRigidActor.globalPose":
        writes.setdefault(h, []).append(idx)
rows = []
per_sweep = {}
for h, L in writes.items():
    if h in created_in or len(L) < n_sweep:
        continue
    k = "static" if cls.get(h) == "PxRigidStatic" else "dynamic"
    for sw in range(n_sweep):
        idx = L[len(L) - n_sweep + sw]
        per_sweep.setdefault(sw, []).append((blk[idx][0], h, k, idx))
    if len(L) > n_sweep:
        rows.append(f"L {names.get(h, '?' + h)} " + blk[L[len(L) - n_sweep - 1]][4].strip("[]"))
for sw, lst in sorted(per_sweep.items()):
    for ev_i, h, k, idx in sorted(lst):
        rows.append(f"S {sw} {names.get(h, '?' + h)} {k} " + blk[idx][4].strip("[]"))
        if k == "dynamic" and idx + 2 < len(blk) and blk[idx + 1][3] == "PxRigidBody.linearVelocity":
            rows.append(f"V {sw} {names.get(h, '?' + h)} " + blk[idx + 1][4].strip("[]") + " " + blk[idx + 2][4].strip("[]"))
bp = os.path.join(rec, "harvest_base.json")
base = json.load(open(bp)) if os.path.exists(bp) else {"windows": []}
hm = json.load(open(os.path.join(rec, "harvest_map.json")))
steps = [e["step"] for e in hm if any(n["name"] == newname for n in e.get("new", []))]
step = steps[0] if steps else -1
win = ([w for w in base["windows"] if w["step"] == step] + [{"objects": []}])[0]
for o in win["objects"]:
    for lp, ls in (o.get("link_scale") or {}).items():
        if ls:
            rows.append(" ".join(["K", lp] + [repr(float(x)) for x in ls]))  # 행위자 prim 세계 척도 (USD double 행 길이)
    rows.append(" ".join(["O", o["name"], o["root_link"]] + [repr(x) for x in o.get("scale", [1, 1, 1])] + [str(len(o["dynamic_links"]))] + o["dynamic_links"]))
open(out, "w").write("\n".join(rows) + "\n")
print("창", start, stop, "벌", n_sweep, "경계", bounds, "줄", len(rows), "정적 묶음 크기", [(r[0], len(r)) for r in runs], "정적 수", n_static)
if len(sys.argv) > 4:  # 진단: 이 이름이 든 행위자의 창 안 자세 쓰기
    for i, c, h, a, v in blk:
        if a == "PxRigidActor.globalPose" and sys.argv[4] in names.get(h, ""):
            print(i, names.get(h), v)
if os.environ.get("SWEEP_LAYOUT"):  # 진단: 창 안 행위자 생성·정적 벌 구간·동적 쓰기 수
    cr = [i for i, c, h, a, v in blk if c == "create" and cls.get(h) == "PxRigidDynamic"]
    print("동적 생성", cr[:3], "...", cr[-3:], len(cr))
    for r in runs:
        print("정적 묶음", r[0], r[-1], len(r))
    dyn_w = [i for i, c, h, a, v in blk if a == "PxRigidActor.globalPose" and cls.get(h) == "PxRigidDynamic"]
    print("동적 자세 쓰기", len(dyn_w), "처음", dyn_w[:3], "끝", dyn_w[-3:])
if os.environ.get("SWEEP_PATTERN"):  # 진단: 창 안 사건 흐름을 (종류) 연속 묶음으로
    def tag(c, h, a):
        k = cls.get(h, "?")
        n = names.get(h, "")
        if c == "create" and k == "PxRigidDynamic":
            return "생성"
        if a == "PxRigidActor.globalPose":
            if "Particle" in n:
                return "입자자세"
            return "정적자세" if k == "PxRigidStatic" else "동적자세"
        if a in ("PxRigidBody.linearVelocity", "PxRigidBody.angularVelocity"):
            return "입자속도" if "Particle" in n else "동적속도"
        return None
    seq = []
    for i, c, h, a, v in blk:
        t = tag(c, h, a)
        if t is None:
            continue
        if seq and seq[-1][0] == t:
            seq[-1][1] += 1
        else:
            seq.append([t, 1, i])
    print(" ".join(f"{t}×{n}" for t, n, i in seq))
