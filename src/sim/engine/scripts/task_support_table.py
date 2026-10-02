"""100 과제별 "엔진이 아직 못 하는 것" 표 (docs/엔진_자체구현.md 15절). CPU, 에셋 안 씀 (BDDL·메타데이터만).

    python task_support_table.py [--md out.md] [--csv out.csv]

과제마다: 장면, 목표(goal)에 나오는 술어, 범위(:objects)의 물질(substance, 입자·유체)·천(cloth)·연체(softBody)·붙이기(attachable)
물체 여부, 초기(init)·목표에 쓰인 상태 술어를 core/omni 지원 여부와 대조한다.
지원(09-30): inside ontop nextto under touching open toggled_on cooked frozen on_fire hot(온도 사슬) inroom(정적) attached(부분)
물리 지원: 강체·관절체(PhysX 비계 재생·엔진 자유 실행). 입자(PBD)·천·연체는 재생기에서 빠짐("지원 안 한 것 physics list:PBDMaterials").
09-30 정정(문서 20.2, particles 작업자): OG 는 softBody 를 강체로 불러온다 -> 천은 `cloth` 능력(과 rope)만 센다.
자르개가 범위에 있어도 목표에 sliced/diced·future/real 이 없으면 막지 않는다(주의만). 입자 물질·술어가 초기(init)에만 있고
목표에 없으면 막지 않는다(주의: 초기 입자 렌더). 막힘 판단은 목표 쪽만.
"""
import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
B = os.path.join(ROOT, "BEHAVIOR-1K")
DEF = os.path.join(B, "bddl3", "bddl", "activity_definitions")
GEN = os.path.join(B, "bddl3", "bddl", "generated_data")
META = os.path.join(B, "datasets", "2026-challenge-task-instances", "metadata")

SUPPORTED = {"inside", "ontop", "nextto", "under", "touching", "open", "toggled_on", "cooked", "frozen", "on_fire", "hot", "inroom"}
PARTIAL = {"attached"}
PARTICLE = {"filled", "contains", "covered", "saturated", "insource"}
CLOTH = {"folded", "unfolded", "draped", "overlaid"}
TRANSITION = {"sliced", "diced", "cooked", "future", "real", "broken", "melted", "assembled"}  # cooked 은 온도 값은 되지만 요리 전이 규칙(새 물체)은 별개
QUANT = {"forall", "exists", "forn", "forpairs", "fornpairs", "and", "or", "not", "imply", "if"}


def scan(text):
    text = re.sub(r";[^\n]*", "", text).lower()
    return re.findall(r"\(|\)|[^\s()]+", text)


def section(tokens, name):
    """(:name ...) 안의 토큰"""
    for i, t in enumerate(tokens):
        if t == ":" + name or (t.startswith(":") and t[1:] == name):
            depth, j, out = 1, i + 1, []
            while j < len(tokens) and depth > 0:
                if tokens[j] == "(":
                    depth += 1
                elif tokens[j] == ")":
                    depth -= 1
                if depth > 0:
                    out.append(tokens[j])
                j += 1
            return out
    return []


def preds(tok):
    out = set()
    for i, t in enumerate(tok[:-1]):
        if t == "(" and tok[i + 1] not in ("(", ")") and not tok[i + 1].startswith("?") and tok[i + 1] not in QUANT:
            out.add(tok[i + 1])
    return {p for p in out if p != "-" and not p.isdigit()}


def objects(tok):
    """:objects 의 범주 synset 들 (a b - cat.n.01)"""
    cats = set()
    for i, t in enumerate(tok):
        if t == "-" and i + 1 < len(tok):
            cats.add(tok[i + 1])
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    ap.add_argument("--csv")
    a = ap.parse_args()
    props = json.load(open(os.path.join(GEN, "properties_to_synsets.json"), encoding="utf-8"))
    sets = {k: set(v) for k, v in props.items()}
    tasks = [json.loads(l) for l in open(os.path.join(META, "task.jsonl"), encoding="utf-8")]
    scene_of = {}
    cur = None
    for line in open(os.path.join(META, "available_tasks.yaml"), encoding="utf-8"):
        if line and not line.startswith(" ") and line.rstrip().endswith(":"):
            cur = line.strip()[:-1]
        m = re.match(r"\s+scene_model:\s*(\S+)", line)
        if m and cur and cur not in scene_of:
            scene_of[cur] = m.group(1)
    rows = []
    for t in sorted(tasks, key=lambda r: r["task_index"]):
        name = t["task_name"]
        p = os.path.join(DEF, name, "problem0.bddl")
        if not os.path.exists(p):
            rows.append(dict(idx=t["task_index"], task=name, scene=scene_of.get(name, "?"), status="BDDL 없음"))
            continue
        tok = scan(open(p, encoding="utf-8").read())
        cats = objects(section(tok, "objects"))
        g = preds(section(tok, "goal"))
        ini = preds(section(tok, "init"))
        goal_tok = section(tok, "goal")
        in_goal = lambda c: any(x == c or x.startswith(c + "_") or x.startswith("?" + c) for x in goal_tok)
        subst_all = sorted(c for c in cats if c in sets.get("substance", set()))
        subst = [c for c in subst_all if in_goal(c)]
        cloth = sorted(c for c in cats if c in sets.get("cloth", set()) or c in sets.get("rope", set()))
        attach = sorted(c for c in cats if c in sets.get("attachable", set()))
        heat = sorted(c for c in cats if c in sets.get("heatSource", set()) or c in sets.get("coldSource", set()))
        slic = sorted(c for c in cats if c in sets.get("slicer", set()))
        need, note = [], []
        unsup = sorted(g - SUPPORTED - PARTIAL - {"agent"})
        unsup_init = sorted(ini - g - SUPPORTED - PARTIAL - {"agent"})
        if unsup_init or (set(subst_all) - set(subst)):
            note.append("초기만(" + ",".join(unsup_init + [c.split(".")[0] for c in subst_all if c not in subst]) + ", 렌더)")
        if subst:
            need.append("입자·유체(" + ",".join(s.split(".")[0] for s in subst) + ")")
        if cloth:
            need.append("천·연체(" + ",".join(s.split(".")[0] for s in cloth) + ")")
        if unsup:
            need.append("술어(" + ",".join(unsup) + ")")
        if "attached" in g or attach:
            need.append("붙이기(부분 지원)")
        if {"sliced", "diced"} & g:
            need.append("자르기 전이")
        elif slic:
            note.append("자르개(" + ",".join(c.split(".")[0] for c in slic) + ", 우연히 자를 때만)")
        status = "됨(radio 와 같은 부류)" if not need else ("부분" if need == ["붙이기(부분 지원)"] else "막힘")
        if note and status.startswith("됨"):
            status = "됨(주의: " + "; ".join(note) + ")"
        rows.append(dict(idx=t["task_index"], task=name, scene=scene_of.get(name, "?"), goal=" ".join(sorted(g)),
                         need="; ".join(need), heat="열원" if heat else "", status=status))
    # 출력
    n_ok = sum(r["status"].startswith("됨") for r in rows)
    n_part = sum(r["status"] == "부분" for r in rows)
    n_block = sum(r["status"] == "막힘" for r in rows)
    scenes = sorted(set(r["scene"] for r in rows))
    lines = [f"과제 {len(rows)}: 됨 {n_ok}, 부분 {n_part}, 막힘 {n_block} | 장면 {len(scenes)}: " + ", ".join(
        f"{s} {sum(r['scene'] == s for r in rows)}" for s in scenes), "",
        "| # | 과제 | 장면 | 목표 술어 | 엔진에 없는 것 | 판정 |", "|---:|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['idx']} | {r['task']} | {r['scene']} | {r.get('goal', '')} | {r.get('need', '') or '-'} | {r['status']} |")
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(txt + "\n")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "task", "scene", "goal", "need", "heat", "status"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})


if __name__ == "__main__":
    sys.exit(main())
