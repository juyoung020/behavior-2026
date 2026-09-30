# 층 0(particles) 정답지: 전이 규칙의 논리 부분 — 공식 코드를 그대로 부른다 (Kit·GPU 안 씀).
#   A. SlicerActive warp 커널 3 개 (object_states/slicer_active.py) 를 warp CPU 로: 닿음 무작위 열 → 스텝마다 value·delay
#   B. SlicingRule.step (transition_rules.py:699) : TouchingAnyCondition(RigidContactAPI.is_in_contact 를 가짜 행렬로) + StateCondition
#   C. BDDL 범위 칸 채우기 (tasks/behavior_task.py:737 _update_bddl_scope_from_added_obj / :759 _removed_obj / :775 system_init)
#   WSL: source ~/behavior-linux/.venv/bin/activate && python gen_rule_ref.py --out ~/engine-data/particles/rule
import argparse
import os

import numpy as np
import warp as wp

import omnigibson.object_states.slicer_active as SA
import omnigibson.tasks.behavior_task as BT
import omnigibson.transition_rules as TR


def part_a(rng, out, n_seq=400, steps=300):
    """자르개 n_seq 개, 스텝 steps. 자르개마다 dt 가 다르므로 dt 값별로 따로 돌린다 (커널은 칸마다 독립)"""
    wp.init()
    O = n_seq
    touch = (rng.random((steps, O)) < rng.uniform(0.02, 0.6, O)).astype(np.int32)
    for o in range(O):  # 긴 무접촉 구간 (2 초 뒤 다시 켜짐)
        if rng.random() < 0.5:
            a0 = rng.integers(0, steps - 100)
            touch[a0 : a0 + rng.integers(40, 100), o] = 0
    dts = rng.choice(np.float32([1.0 / 30.0, 1.0 / 60.0, 0.05, 1.0 / 15.0]), O).astype(np.float32)
    rec_v = np.zeros((steps, O), np.uint8)
    rec_d = np.zeros((steps, O), np.float32)
    for dtv in np.unique(dts):
        idx = np.nonzero(dts == dtv)[0]
        k = len(idx)
        vals = wp.array(np.ones((1, k), np.uint8), dtype=wp.uint8, device="cpu")
        delay = wp.zeros((1, k), dtype=wp.float32, device="cpu")
        prev = wp.zeros((1, k), dtype=wp.int32, device="cpu")
        cur = wp.zeros((1, k), dtype=wp.int32, device="cpu")
        dt = wp.array(np.array([dtv], np.float32), dtype=wp.float32, device="cpu")
        for s in range(steps):
            wp.launch(SA._slicer_pre_clear_kernel, dim=(1, k), inputs=[vals, delay, prev], device="cpu")
            wp.launch(SA._slicer_zero_currently_touching_kernel, dim=(1, k), inputs=[cur], device="cpu")
            cur.assign(touch[s : s + 1, idx])
            wp.launch(SA._slicer_post_update_kernel, dim=(1, k),
                      inputs=[vals, delay, cur, prev, wp.float32(SA.m.REACTIVATION_DELAY), dt], device="cpu")
            rec_v[s, idx], rec_d[s, idx] = vals.numpy()[0], delay.numpy()[0]
    np.save(os.path.join(out, "sa_touch.npy"), touch)
    np.save(os.path.join(out, "sa_dt.npy"), dts)
    np.save(os.path.join(out, "sa_value.npy"), rec_v)
    np.save(os.path.join(out, "sa_delay.npy"), rec_d)


class FakeState:
    def __init__(self, v):
        self.v = v

    def get_value(self):
        return self.v


class FakeObj:
    def __init__(self, name, scene, active=None):
        self.name = name
        self.scene = scene
        self.states = {SA.SlicerActive: FakeState(active)} if active is not None else {}

    def __repr__(self):
        return self.name


class FakeScene:
    idx = 0


def part_b(rng, out, n=20000):
    TM = {}

    class FakeRCA:
        @staticmethod
        def is_in_contact(scene_idx, query_set, with_set, ignore_set, current_only):
            assert current_only is False
            q = query_set[0]
            return any(TM[(q.name, w.name)] for w in with_set)

    TR.RigidContactAPI = FakeRCA
    sc = FakeScene()
    rows = []
    maxs, maxk = 6, 4
    for i in range(n):
        ns, nk = int(rng.integers(0, maxs + 1)), int(rng.integers(0, maxk + 1))
        sl = [FakeObj(f"s{j}", sc) for j in range(ns)]
        act = rng.random(nk) < rng.uniform(0, 1)
        kn = [FakeObj(f"k{j}", sc, bool(act[j])) for j in range(nk)]
        tmat = rng.random((maxs, maxk)) < rng.uniform(0, 0.5)
        TM.clear()
        for a in range(ns):
            for b in range(nk):
                TM[(f"s{a}", f"k{b}")] = bool(tmat[a, b])
        rule = TR.SlicingRule.__new__(TR.SlicingRule)
        rule.scene = sc
        rule.conditions = None
        rule.candidates = {"sliceable": sl, "slicer": kn}
        got = []
        rule.transition = lambda object_candidates: got.extend(o.name for o in object_candidates["sliceable"]) or TR.TransitionResults(add=[], remove=[])
        # 규칙 활성 여부는 TransitionRuleAPI.get_rule_candidates (거르개 둘 다 비어 있지 않아야) → 비면 step 자체가 안 불림
        if ns > 0 and nk > 0:
            rule.step()
        sel = np.zeros(maxs, np.uint8)
        for g in got:
            sel[int(g[1:])] = 1
        rows.append(np.concatenate([[ns, nk], act.astype(np.uint8).tolist() + [0] * (maxk - nk), tmat.astype(np.uint8).ravel(), sel]))
    np.save(os.path.join(out, "rule_rows.npy"), np.array(rows, np.int32))


def part_c(rng, out, n=5000):
    """범위 칸: dict 순서(agent 먼저, 그다음 과제 범위). 물체 추가·삭제·계 초기화 사건열 → 칸 값.
    정수로 적는다: 칸 = (범주 번호, 계인가), 사건 = (종류 0 추가 1 삭제 2 계 초기화, 범주 또는 물체 번호),
    물체 번호: 처음부터 있던 것 = 1000 + 칸, 추가된 것 = 사건 번호"""
    cats = ["half_log", "half_apple", "apple", "log", "diced__apple", "cooked__diced__apple"]
    systems = {"diced__apple", "cooked__diced__apple"}
    BT.is_system_bddl_inst = lambda inst: inst.split(".")[0] in systems
    BT.og_categories_from_bddl_inst = lambda inst: [inst.split(".")[0]]

    class Obj:
        def __init__(self, oid, cat):
            self.oid, self.name, self.category, self.scene = oid, f"o{oid}", cat, None

    class FakeEnv:
        scenes = [None]

    MS, ME = 16, 12
    slots = np.full((n, MS, 2), -1, np.int32)
    init = np.full((n, MS), -1, np.int32)
    evs = np.full((n, ME, 2), -1, np.int32)
    final = np.full((n, MS), -1, np.int32)
    for t in range(n):
        insts = []
        for ci, c in enumerate(cats):
            for k in range(int(rng.integers(0, 3))):
                insts.append((f"{c}.n.01_{k + 1}", ci))
        rng.shuffle(insts)
        insts = insts[:MS]
        fake = BT.BehaviorTask.__new__(BT.BehaviorTask)
        fake._env = FakeEnv()
        scope = {"agent.n.01_1": Obj(-5, "agent")}
        live = []
        for j, (s_, ci) in enumerate(insts):
            slots[t, j] = [ci, int(cats[ci] in systems)]
            if rng.random() < 0.4 and cats[ci] not in systems:
                o = Obj(1000 + j, cats[ci])
                scope[s_] = o
                live.append(o)
                init[t, j] = o.oid
            else:
                scope[s_] = None
        fake.object_scopes = [scope]
        for e in range(int(rng.integers(1, ME + 1))):
            u = rng.random()
            if u < 0.5:
                ci = int(rng.integers(len(cats) - 2))
                o = Obj(e, cats[ci])
                fake._update_bddl_scope_from_added_obj(o)
                live.append(o)
                evs[t, e] = [0, ci]
            elif u < 0.8 and live:
                o = live.pop(int(rng.integers(len(live))))
                fake._update_bddl_scope_from_removed_obj(o)
                evs[t, e] = [1, o.oid]
            else:
                ci = int(rng.integers(len(cats) - 2, len(cats)))
                sy = Obj(2000 + ci, cats[ci])
                sy.name = cats[ci]
                fake._update_bddl_scope_from_system_init(sy)
                evs[t, e] = [2, ci]
        for j, (s_, ci) in enumerate(insts):
            v = fake.object_scopes[0][s_]
            final[t, j] = -1 if v is None else v.oid
    for k, v in dict(scope_slots=slots, scope_init=init, scope_evs=evs, scope_final=final).items():
        np.save(os.path.join(out, k + ".npy"), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    part_a(rng, a.out)
    part_b(rng, a.out)
    part_c(rng, a.out)
    print("done")


if __name__ == "__main__":
    main()
