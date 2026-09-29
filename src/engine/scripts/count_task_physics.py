"""대회 100과제의 BDDL 을 읽어 과제마다 필요한 물리 종류를 센다 (엔진 자체 구현 범위 결정용).

    python count_task_physics.py [--json out.json]

분류 (BDDL 지식 베이스 기준, 추측 없이 synset 속성으로)
  - 입자(particle) 계열: 과제에 나오는 synset 중 substance(물질) 인 것 -> OmniGibson 이 입자계(물·얼룩·가루 등)로 만든다
      * 유체(fluid): substance 이면서 liquid 속성
      * 그 밖 입자: 얼룩(stain)·먼지(dust)·가루 등 (시각 입자/거시 입자)
  - 천(cloth): synset 이 cloth 속성 -> OmniGibson 이 PhysX 입자 천(PBD cloth)으로 만든다
  - 전이 규칙이 필요한 술어: cooked / frozen / sliced / diced / 섞기 등
대회 과제 목록: data\\2026-challenge-demos\\meta\\tasks.jsonl
"""
import argparse
import json
import os
import re
import sys

ROOT = r"C:\behavior-2026"
sys.path.insert(0, os.path.join(ROOT, "BEHAVIOR-1K", "bddl3"))

from bddl.object_taxonomy import ObjectTaxonomy  # noqa: E402

OT = ObjectTaxonomy()

TASKS = os.path.join(ROOT, "data", "2026-challenge-demos", "meta", "tasks.jsonl")
DEFS = os.path.join(ROOT, "BEHAVIOR-1K", "bddl3", "bddl", "activity_definitions")
TRANSITION_PREDS = {"cooked", "frozen", "sliced", "diced", "real", "hot", "burnt", "open", "toggled_on", "filled",
                    "covered", "saturated", "contains", "insource", "folded", "unfolded", "draped", "attached"}


def synset_props(name):
    try:
        return set(OT.get_abilities(name))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(TASKS, encoding="utf-8") if l.strip()]
    rows = []
    for t in tasks:
        name = t.get("task_name") or t.get("name")
        idx = t.get("task_index", len(rows))
        p = os.path.join(DEFS, name, "problem0.bddl")
        if not os.path.exists(p):
            rows.append({"idx": idx, "task": name, "missing": True})
            continue
        text = open(p, encoding="utf-8").read()
        syns = sorted(set(re.findall(r"\b([a-z_]+\.n\.\d\d)", text)))
        preds = sorted(set(re.findall(r"\(\s*(?:not\s*\(\s*)?([a-z_]+)\s+\?", text)))
        liquid, substance, visual, cloth, unknown = [], [], [], [], []
        for s in syns:
            pr = synset_props(s)
            if pr is None:
                unknown.append(s)
                continue
            if "substance" in pr:
                if "liquid" in pr:
                    liquid.append(s)          # 유체 (PBD 입자 유체)
                elif "visualSubstance" in pr:
                    visual.append(s)          # 얼룩·먼지 같은 표면 입자 (시각용, 동역학 없음)
                else:
                    substance.append(s)       # 가루·잘린 조각 같은 물리 입자 (physical/macroPhysical)
            if "cloth" in pr:
                cloth.append(s)
        rows.append({"idx": idx, "task": name, "liquid": liquid, "substance": substance, "visual": visual, "cloth": cloth,
                     "preds": [x for x in preds if x in TRANSITION_PREDS], "unknown": unknown})
    ok = [r for r in rows if not r.get("missing")]
    n_liq = sum(1 for r in ok if r["liquid"])
    n_phys = sum(1 for r in ok if r["substance"])
    n_vis = sum(1 for r in ok if r["visual"])
    n_cloth = sum(1 for r in ok if r["cloth"])
    n_dyn = sum(1 for r in ok if r["liquid"] or r["substance"] or r["cloth"])
    n_any = sum(1 for r in ok if r["liquid"] or r["substance"] or r["visual"] or r["cloth"])
    print(f"과제 {len(rows)} (BDDL 없음 {len(rows) - len(ok)})")
    print(f"  유체(액체) 입자: {n_liq}")
    print(f"  물리 입자(가루·조각 등, 액체 아님): {n_phys}")
    print(f"  시각 입자(얼룩·먼지 등, 동역학 없음): {n_vis}")
    print(f"  천: {n_cloth}")
    print(f"  동역학이 있는 입자·천 필요: {n_dyn}  / 강체·관절체(+시각 입자)만: {len(ok) - n_dyn}  / 입자·천 전혀 없음: {len(ok) - n_any}")
    for r in ok:
        tags = []
        if r["liquid"]: tags.append("유체:" + ",".join(r["liquid"]))
        if r["substance"]: tags.append("물리입자:" + ",".join(r["substance"]))
        if r["visual"]: tags.append("시각입자:" + ",".join(r["visual"]))
        if r["cloth"]: tags.append("천:" + ",".join(r["cloth"]))
        print(f"  {r['idx']:>3} {r['task']:<45} {' | '.join(tags) if tags else '강체만'}")
    if a.json:
        json.dump(rows, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
