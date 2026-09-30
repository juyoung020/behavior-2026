# 층 0(particles) 정답지: 입자 익히기 규칙(CookingPhysicalParticleRule) 선택 — 공식 step() 을 가짜 용기·계에 붙여 그대로.
#   조건 Heated == True (StateCondition) → 용기마다(후보 순서) 활성 레시피를 순서대로 보고, 용기 범주가 맞고 입력 계를 모두 Contains 하면
#   그 레시피 하나만 실행(_execute_recipe) — transition_rules.py:1651
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_cook_ref.py --out ~/engine-data/particles/cook
import argparse
import os

import numpy as np

import omnigibson.transition_rules as TR
from omnigibson.object_states import Contains, Heated

MC, MR, MS = 5, 6, 8  # 용기·레시피·계 최대


class St:
    def __init__(self, f):
        self.f = f

    def get_value(self, *a, **k):
        return self.f(*a, **k)


class Cont:
    def __init__(self, i, cat, heated, contains):
        self.name, self.category = f"c{i}", f"cat{cat}"
        self.states = {Heated: St(lambda: heated), Contains: St(lambda system: bool(contains[int(system[1:])]))}


class Scene:
    def get_system(self, name):
        return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(37)
    rows = []
    for t in range(a.n):
        nc, nr = int(rng.integers(0, MC + 1)), int(rng.integers(1, MR + 1))
        heated = rng.random(MC) < 0.6
        cat = rng.integers(0, 3, MC)
        cont = rng.random((MC, MS)) < rng.uniform(0.1, 0.7)
        rin = rng.integers(0, MS, (MR, 2))
        rn2 = rng.random(MR) < 0.3  # 입력 계 둘 (xyz + cooked__water 꼴)
        rfc = rng.integers(-1, 3, MR)  # -1 = 범주 제한 없음
        recipes = {}
        for r in range(nr):
            ins = [f"s{rin[r, 0]}"] + ([f"s{rin[r, 1]}"] if rn2[r] else [])
            recipes[f"r{r}"] = dict(input_systems=ins, output_systems=["out"], input_objects={}, output_objects={},
                                    fillable_categories=None if rfc[r] < 0 else [f"cat{rfc[r]}"])
        rule = TR.CookingPhysicalParticleRule.__new__(TR.CookingPhysicalParticleRule)
        rule.scene = Scene()
        rule.conditions = None
        rule._active_recipes = recipes
        conts = [Cont(i, cat[i], bool(heated[i]), cont[i]) for i in range(nc)]
        rule.candidates = {"container": conts}
        rule._compute_global_rule_info = lambda: {}
        rule._compute_container_info = lambda object_candidates, container, global_info: {"in_volume": None}
        rule._validate_recipe_objects_are_contained_and_states_satisfied = lambda recipe, container_info: True
        got = {}

        def ex(container, recipe, container_info):
            got[container.name] = [k for k, v in recipes.items() if v is recipe][0]
            return TR.TransitionResults(add=[], remove=[])

        rule._execute_recipe = ex
        if nc > 0:
            rule.step()
        sel = [int(got[f"c{i}"][1:]) if f"c{i}" in got else -1 for i in range(MC)]
        rows.append(np.concatenate([[nc, nr], heated.astype(int), cat, cont.ravel().astype(int), rin.ravel(), rn2.astype(int), rfc, sel]))
    np.save(os.path.join(a.out, "cook_rows.npy"), np.array(rows, np.int32))
    print("done", a.n, "executed", sum(int((np.array(r[-MC:]) >= 0).sum()) for r in rows))


if __name__ == "__main__":
    main()
