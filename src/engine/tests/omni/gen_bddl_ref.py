# 층 0(omni) 정답지: 공식 bddl3 로 100과제 목표 판정·q_score 를 무작위 술어값에서 계산해 적는다.
# 우리 C++ (core/omni/bddl.h) 는 같은 문제 문자열·같은 술어값으로 같은 결과를 내야 한다 (참/거짓 완전 일치, q_score 비트 동일).
#   WSL: source ~/behavior-linux/.venv/bin/activate
#        python gen_bddl_ref.py --out ~/engine-data/omni/bddl --trials 300
# 원본 경로: bddl3/bddl/condition_evaluation.py (compile/evaluate/ground options),
#           OmniGibson/omnigibson/metrics/task_metric.py:6 compute_q_score,
#           OmniGibson/omnigibson/tasks/behavior_task.py:176 get_goal_option_satisfaction
import argparse
import json
import os
import random
import re
import struct
import sys

from bddl.knowledge_base import KnowledgeBase
from bddl.condition_evaluation import evaluate_state
from bddl.wildcard import expand_wildcards
from bddl.predicates import Predicate

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_JSONL = "/mnt/c/behavior-2026/data/2026-challenge-demos/meta/tasks.jsonl"


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def wildcard_layout(definition, kb, rng):
    """평가기 _build_scene_layout_from_rooms 대신: 와일드카드마다 방에 max(최소 개수,1) + 0~3 개 있는 가짜 배치."""
    layout = {}
    info = {}
    for line in definition.splitlines():
        if "*" not in line:
            continue
        if " - " in line:
            insts, synset = line.strip(" \n\t").split(" - ")
            insts = insts.split(" ")
            syn = kb.get_synset(synset)
            cats = set()
            for s in [syn] + sorted(syn.descendants, key=lambda x: x.name):
                if s.is_leaf:
                    for c in s.categories:
                        cats.add(c.name)
            info[insts[-1]] = (len(insts) - 1, sorted(cats))
        else:
            toks = line.strip(" ()\n\t").split(" ")
            _, inst, room = toks
            n_min, cats = info[inst]
            layout.setdefault(room, {})
            cat = cats[0]
            layout[room][cat] = layout[room].get(cat, 0) + max(n_min, 1) + rng.randint(0, 3)
    return layout


def leaves(node, out):
    if isinstance(node, list):
        for c in node:
            leaves(c, out)
        return
    if isinstance(node, Predicate):
        out.append((node.STATE_NAME, tuple(node.inputs)))
        return
    for c in node.children:
        leaves(c, out)


def lit_key(cond):
    # ground option 원소: ["pred", a, (b)] 또는 ["not", [...]]
    if cond[0] == "not":
        return (1, cond[1][0], tuple(cond[1][1:]))
    return (0, cond[0], tuple(cond[1:]))


def q_score(success, now_opts, init_opts):  # task_metric.py:6 그대로
    if success:
        return 1.0
    if not now_opts:
        return 0.0
    scores = []
    for now_opt, init_opt in zip(now_opts, init_opts):
        if len(now_opt) == 0:
            scores.append(0.0)
            continue
        newly = sum(int((not i) and n) for n, i in zip(now_opt, init_opt))
        scores.append(newly / len(now_opt))
    return max(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-options", type=int, default=2_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kb = KnowledgeBase(verbose=False)
    tasks = [json.loads(l) for l in open(TASKS_JSONL, encoding="utf-8") if l.strip()]
    rng = random.Random(a.seed)
    index = []
    for t in tasks:
        name = t["task_name"]
        task = kb.get_task(f"{name}-0")
        layout = wildcard_layout(task.definition, kb, rng) if task.has_wildcards else {}
        compiled = task.compile(scene_layout=layout)
        # 확장된 문제 문자열 = CompiledTask 가 파싱한 그 문자열 (models.py:1108 Task.compile)
        problem = expand_wildcards(task.definition, layout, kb) if task.has_wildcards else task.definition
        heads = compiled.goal_conditions
        opts = compiled.ground_goal_state_options
        # 원자(술어+인자) 목록: 목표 트리 잎 + ground option 원소 (같은 것)
        lv = []
        leaves(heads, lv)
        for opt in opts:
            for h in opt:
                leaves(h, lv)
        atoms = sorted(set(lv))
        aid = {x: i for i, x in enumerate(atoms)}
        # ground option 정규형: 각 선택지 = (부정, 원자번호) 정렬 목록, 선택지 목록도 정렬 (중복 유지)
        canon = []
        for opt in opts:
            lits = []
            for h in opt:
                k = lit_key(h.body)
                lits.append((k[0], aid[(k[1], k[2])]))
            canon.append(tuple(sorted(lits)))
        canon.sort()
        ctext = ";".join(",".join(f"{n}:{i}" for n, i in o) for o in canon).encode()
        # 시행: 원자마다 참/거짓. 일부는 ground option 하나를 만족하게 맞춰 성공 경우도 나오게
        trials = []
        prev_masks = None
        for k in range(a.trials):
            p = rng.random()
            bits = [rng.random() < p for _ in atoms]
            if k % 3 == 1 and opts:
                o = opts[rng.randrange(len(opts))]
                for h in o:
                    ng, pn, args = lit_key(h.body)
                    bits[aid[(pn, args)]] = not ng
                # 가끔 하나 뒤집어 거의-성공
                if rng.random() < 0.5 and atoms:
                    j = rng.randrange(len(atoms))
                    bits[j] = not bits[j]
            if k == 0:
                bits = [False] * len(atoms)
            if k == 2:
                bits = [True] * len(atoms)
            val = {atoms[i]: bits[i] for i in range(len(atoms))}

            # bddl 의 evaluate_fn 은 (predicate_cls, *inputs) (predicates.py:86). 클래스→토큰 역표로 원자를 찾는다
            def evaluate_fn(pred_cls, *ents):
                return val[(CLS2TOK[pred_cls], tuple(ents))]

            success, res = evaluate_state(heads, evaluate_fn)
            sat_heads = set(res["satisfied"])
            masks = []
            for opt in opts:
                _, r = evaluate_state(opt, evaluate_fn)
                s = set(r["satisfied"])
                masks.append([i in s for i in range(len(opt))])
            init = masks if prev_masks is None else prev_masks
            q = q_score(success, masks, init)
            prev_masks = masks
            trials.append(("".join("1" if b else "0" for b in bits), int(success),
                           "".join("1" if i in sat_heads else "0" for i in range(len(heads))),
                           struct.pack("<d", q).hex()))
        with open(os.path.join(a.out, f"{name}.txt"), "w") as f:
            f.write(f"TASK {name}\n")
            f.write("PROBLEM_BEGIN\n" + problem.rstrip("\n") + "\nPROBLEM_END\n")
            f.write(f"ATOMS {len(atoms)}\n")
            for pn, args in atoms:
                f.write(pn + " " + " ".join(args) + "\n")
            f.write(f"HEADS {len(heads)}\n")
            f.write(f"NOPTIONS {len(opts)}\n")
            f.write(f"OPTHASH {fnv1a64(ctext):016x}\n")
            f.write(f"TRIALS {len(trials)}\n")
            for b, s, hs, q in trials:
                f.write(f"{b} {s} {hs} {q}\n")
        index.append(name)
        print(f"{name}: atoms {len(atoms)} heads {len(heads)} options {len(opts)}", flush=True)
    with open(os.path.join(a.out, "index.txt"), "w") as f:
        f.write("\n".join(index) + "\n")


from bddl.predicates import TOKEN_TO_PREDICATE
CLS2TOK = {v: k for k, v in TOKEN_TO_PREDICATE.items()}

if __name__ == "__main__":
    main()
