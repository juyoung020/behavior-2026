# 새 행위자 틀 키 대응 (P2·P3 물리 결합 (c)): 수확 기록의 harvest_map.json(공식 전이: 원본·부분·새 이름) 과
# 리드 g1_sc 의 templates.txt(simulate 마다 뺀 행위자·넣은 틀 파일·넣은 행위자 이름) 를 맞춰 spawn_table.json 을 만든다.
#   python3 build_spawn_table.py <수확 기록 폴더> <틀 폴더(G1_SC_TEMPLATES)>
# spawn_table.json:
#   slices: [{src, src_category, parts: [{part, name, category, model, template: <틀 파일>, actor: <틀 안 행위자 번호>}]}]
#   dices:  [{src, system, n_new, template, actors: [<틀 안 행위자 번호> ...]}]
# 행위자 이름은 omni 가 붙인 prim 경로(/World/scene_0/<물체 이름>/base_link 또는 입자 prim)라 물체 이름으로 찾는다.
import json
import os
import sys

rec, tdir = sys.argv[1], sys.argv[2]
hm = json.load(open(os.path.join(rec, "harvest_map.json")))
lines = []
for ln in open(os.path.join(tdir, "templates.txt"), encoding="utf-8"):
    ln = ln.strip()
    if not ln:
        continue
    left, right = ln.split(" | ")
    lt, rt = left.split(), right.split()
    sim, removed = int(lt[1]), lt[4:]
    tmpl = None if rt[2] == "-" else rt[2]
    added = rt[3:]
    lines.append(dict(sim=sim, removed=removed, template=tmpl, added=added))


def find_actor(name):
    """물체 이름이 들어간 넣은 행위자를 찾는다 → (틀 파일, 틀 안 번호)"""
    for L in lines:
        for i, a in enumerate(L["added"]):
            if f"/{name}/" in a or a.endswith("/" + name) or a == name:
                return L["template"], i, L["sim"]
    return None, -1, -1


out = dict(slices=[], dices=[], unmatched=[])
for ev in hm:
    if ev["rule"] == "SlicingRule":
        parts = []
        for n in ev["new"]:
            t, i, sim = find_actor(n["name"])
            if t is None:
                out["unmatched"].append(n["name"])
            parts.append(dict(n, template=t, actor=i, simulate=sim))
        out["slices"].append(dict(src=ev["src"], src_category=ev["src_category"], src_scale=ev["src_scale"], parts=parts))
    elif ev["rule"] == "DicingRule":
        # 다진 입자: 원본이 빠진 줄(같은 창)에 넣은 입자 행위자들
        cand = [L for L in lines if any(f"/{ev['src']}/" in r or r.endswith("/" + ev["src"]) for r in L["removed"])]
        ent = dict(src=ev["src"], system=ev["system"], n_new=ev["n_new"], template=None, actors=[])
        for L in cand + lines:
            idx = [i for i, a in enumerate(L["added"]) if ev["system"] in a]
            if idx:
                ent["template"], ent["actors"], ent["simulate"] = L["template"], idx, L["sim"]
                break
        if ent["template"] is None:
            out["unmatched"].append(ev["src"])
        out["dices"].append(ent)
json.dump(out, open(os.path.join(rec, "spawn_table.json"), "w"), indent=1)
print("자르기", len(out["slices"]), "다지기", len(out["dices"]), "못 맞춤", out["unmatched"])
