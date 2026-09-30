# 닫힌 고리 편집 계획 (g1_edit_hook.cpp 가 읽음): 수확 기록(HARVEST_BASE=1) 의 전이마다 편집 창 simulate·지울 물체·넣을 반쪽(틀·부분 메타)·base 물체 표.
#   python3 build_edit_plan.py <수확 기록 폴더> <출력 plan.txt> [다지기 npy 폴더 = <기록>/dice_npy]   (틀 폴더 = <기록>/tmpl, 리드 G1_SC_TEMPLATES 로 뜬 것)
#   P <K+1> <원본> <원본 뿌리> <계 이름> <틀 파일> <계 틀 prim 행위자 또는 -1> <입자 수> <틀 안 입자 행위자...> <다지기 ev 폴더>   (다지기)
# 줄:
#   T <K+1> <원본 이름> <원본 뿌리 링크 경로> <원본 척도 3>   — 지우기 창 = simulate K+1 앞 (K = step_physics), 무덤·뜨기 = K 앞
#   H <틀 파일> <틀 안 행위자> <새 물체 이름> <부분> bb_pos3 bb_orn4 bb_size3 native_bb3 base_link_offset3
#   O <이름> <종류 0 강체 1 관절체 2 운동학> <관절 수> <뿌리 링크 경로> <동적 링크 수> <링크...>   (그 창 dump 때 등록부 차례)
#   C <이름> <척도 3>   (있으면 — 물체 척도)
#   X <이름> p3 q4      (운동학 물체: dump 가 읽은 XForm 자세)
#   K <행위자 prim 경로> <세계 척도 3>   (있으면 — USD 세계 행렬 행 길이, 동기화 벌 척도)
#   E
# 다지기(입자)는 아직 (자르기만).
import json
import os
import sys

rec, out = sys.argv[1], sys.argv[2]
hm = json.load(open(os.path.join(rec, "harvest_map.json")))
assets = json.load(open(os.path.join(rec, "harvest_assets.json")))
base = json.load(open(os.path.join(rec, "harvest_base.json")))
tdir = os.path.join(rec, "tmpl")
tl = []
for ln in open(os.path.join(tdir, "templates.txt"), encoding="utf-8"):
    ln = ln.strip()
    if not ln:
        continue
    left, right = ln.split(" | ")
    lt, rt = left.split(), right.split()
    # 틀 파일 경로는 뜬 때의 폴더(새 이름 .tmp 뒤 옮김)라 이 틀 폴더 기준으로 다시 붙인다
    tl.append(dict(sim=int(lt[1]), removed=lt[4:], template=None if rt[2] == "-" else os.path.join(tdir, os.path.basename(rt[2])), added=rt[3:]))


def num(v):
    return repr(v) if isinstance(v, float) else str(v)


lines = []
dice_i = 0
for ev in hm:
    if ev["rule"] == "DicingRule":  # P 줄: 입자 넣기 창(K) = 틀 줄 simulate, 지우기 창(K+1) = 원본을 뺀 줄 simulate
        src = ev["src"]
        R = [x for x in tl if any(f"/{src}/" in r for r in x["removed"])]
        A = [x for x in tl if any(ev["system"] in a for a in x["added"]) and x["template"]]
        if not R or len(A) <= dice_i:
            print("다지기 틀 줄 없음", src)
            continue
        L, Ad = R[0], A[dice_i]
        dice_i += 1
        root = [r for r in L["removed"] if f"/{src}/" in r][0]
        part = [k for k, a in enumerate(Ad["added"]) if "Particle" in a]
        tmpl_prim = ([k for k, a in enumerate(Ad["added"]) if "/template/" in a] + [-1])[0]
        ddir = os.path.join(sys.argv[3] if len(sys.argv) > 3 else os.path.join(rec, "dice_npy"), f"ev_{dice_i - 1:03d}")
        lines.append(" ".join(["P", str(L["sim"]), src, root, ev["system"], Ad["template"], str(tmpl_prim), str(len(part))] + [str(k) for k in part] + [ddir]))
        win = [w for w in base["windows"] if w["step"] == ev["step"]]
        for o in (win[0]["objects"] if win else []):
            kind = 1 if o["articulated"] else (2 if o["kinematic_only"] else 0)
            lines.append(" ".join(["O", o["name"], str(kind), str(o["n_joints"]), o["root_link"], str(len(o["dynamic_links"]))] + o["dynamic_links"]))
            if "scale" in o:
                lines.append(" ".join(["C", o["name"]] + [repr(float(x)) for x in o["scale"]]))
            for lp, ls in (o.get("link_scale") or {}).items():
                if ls:
                    lines.append(" ".join(["K", lp] + [repr(float(x)) for x in ls]))
            dd = (win[0].get("dump") or {}).get(o["name"]) if win else None
            if o["kinematic_only"] and dd:  # 운동학 물체: dump 가 읽은 XForm 자세 (되돌리기가 이 값을 정적 자세로 씀)
                lines.append(" ".join(["X", o["name"]] + [repr(float(x)) for x in list(dd["pos"]) + list(dd["ori"])]))
        lines.append("E")
        continue
    if ev["rule"] != "SlicingRule":
        continue
    src = ev["src"]
    L = [x for x in tl if any(f"/{src}/" in r for r in x["removed"])]
    if not L:
        print("틀 줄 없음", src)
        continue
    L = L[0]
    root = [r for r in L["removed"] if f"/{src}/" in r][0]
    lines.append(" ".join(["T", str(L["sim"]), src, root] + [num(v) for v in ev["src_scale"]]))
    for i, n in enumerate(ev["new"]):
        p = ev["parts"][n["part"]]
        a = assets[n["name"]]
        idx = [k for k, x in enumerate(L["added"]) if f"/{n['name']}/" in x][0]
        V = lambda x: json.loads(x) if isinstance(x, str) else list(x)  # 메타데이터는 float32 최단 표기 글 (그대로 float32 로 읽힘)
        vals = V(p["bb_pos"]) + V(p["bb_orn"]) + V(p["bb_size"]) + V(a["native_bbox"]) + V(a["offset"])
        lines.append(" ".join(["H", L["template"], str(idx), n["name"], str(n["part"])] + [num(v) for v in vals]))
    win = [w for w in base["windows"] if w["step"] == ev["step"]]
    if not win:
        print("base 창 없음", src)
    else:
        for o in win[0]["objects"]:
            kind = 1 if o["articulated"] else (2 if o["kinematic_only"] else 0)
            lines.append(" ".join(["O", o["name"], str(kind), str(o["n_joints"]), o["root_link"], str(len(o["dynamic_links"]))] + o["dynamic_links"]))
            if "scale" in o:
                lines.append(" ".join(["C", o["name"]] + [repr(float(x)) for x in o["scale"]]))  # 물체 척도 (동기화 벌 USD 왕복)
            for lp, ls in (o.get("link_scale") or {}).items():
                if ls:
                    lines.append(" ".join(["K", lp] + [repr(float(x)) for x in ls]))
            dd = (win[0].get("dump") or {}).get(o["name"]) if win else None
            if o["kinematic_only"] and dd:  # 운동학 물체: dump 가 읽은 XForm 자세 (되돌리기가 이 값을 정적 자세로 씀)
                lines.append(" ".join(["X", o["name"]] + [repr(float(x)) for x in list(dd["pos"]) + list(dd["ori"])]))
    lines.append("E")
open(out, "w").write("\n".join(lines) + "\n")
print("전이", sum(1 for l in lines if l.startswith("T ")))
