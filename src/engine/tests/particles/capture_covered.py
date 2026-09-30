"""층 0(particles) 실제 과제 기록: 공식 평가기(BatchedEvaluator, 무수정)로 Covered 과제 한 판을 불러 제거기를 먼지 묻은 물체 위로
매 스텝 순간이동시키며, 공식 입자 제거 사건을 가로채 적는다. 엔진 쪽 test_covered_capture 가 같은 입력으로 같은 결정을 내는지 본다.

가로채는 곳 (값만 읽음, 결과는 바꾸지 않음):
  MacroVisualParticleSystem._modify_batch_particles_position_orientation(local=True)  — 불러올 때 국소 자세 입력과 배치 크기
  ParticleRemover._modify_particles                                                  — 제거기·계·그 순간 링크/투영 메시 행렬·전후 생존
또 스텝마다 입자 붙은 링크의 scaled_transform(float32 4x4), PhysX 자세, Fabric 세계 행렬(double)과 척도를 적어
"scaled_transform = f(PhysX 자세, 척도)" 식을 엔진이 세울 수 있게 한다.

  WSL (GPU 대기열 안에서): bash run_capture_covered.sh <과제> [스텝=60] [인스턴스=0]
출력: ~/engine-data/particles/covered/<과제>_<인스턴스>/*.npy
"""
import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "capture")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remover", default=None, help="순간이동시킬 제거기 물체 이름 (없으면 과제 범위의 첫 제거기)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch as th
    from omegaconf import OmegaConf

    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.eval import evaluator as E
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
    from omnigibson.eval.evaluator import resolve_instance_ids
    import omnigibson.object_states.particle_modifier as PM
    import omnigibson.systems.macro_particle_system as MPS
    from omnigibson.object_states.saturated import ModifiedParticles
    from omnigibson.object_states import Covered
    from physx_capture import install_no_render  # 리드 도구: 렌더 없는 WSL 에서 카메라만 0 영상 (물리 무관)

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    install_no_render()
    seed = seed_everything(DEFAULT_EVAL_SEED)

    # ---- 가로채기 ----
    Vis = MPS.MacroVisualParticleSystem
    load_calls = []  # (계, [이름], 위치 n×3, 방향 n×4)
    orig_mod = Vis._modify_batch_particles_position_orientation

    def mod_wrap(self, particles, positions=None, orientations=None, local=False):
        if local and positions is not None and orientations is not None:
            load_calls.append((self.name, list(particles), positions.clone().numpy(), orientations.clone().numpy()))
        return orig_mod(self, particles, positions=positions, orientations=orientations, local=local)

    Vis._modify_batch_particles_position_orientation = mod_wrap

    # 국소 행렬이 마지막으로 어디서 정해졌나 (입자별 호출 출처 사슬)
    import inspect

    mat_origin = {}
    orig_set_mat = Vis._modify_particle_local_mat

    def set_mat_wrap(self, name, mat):
        st = inspect.stack()
        mat_origin[name] = " < ".join(f.function for f in st[1:6])
        return orig_set_mat(self, name, mat)

    Vis._modify_particle_local_mat = set_mat_wrap

    events = []  # 제거 사건
    cur_step = [-1]
    orig_rm = PM.ParticleRemover._modify_particles

    def rm_wrap(self, system):
        before = list(system.particles.keys()) if system.particles else []
        ev = dict(step=cur_step[0], obj=self.obj.name, system=system.name, method=int(self.method.value) if hasattr(self.method, "value") else str(self.method),
                  before=before, modified_before=int(self.obj.states[ModifiedParticles].get_value(system)),
                  limit=int(self.visual_particle_modification_limit))
        if self.method == PM.ParticleModifyMethod.ADJACENCY:
            ev["link"] = self.link.name
            ev["tf"] = self.link.scaled_transform.clone().numpy()
            ev["hull"] = self.link.visual_boundary_points_local.clone().numpy()
        else:
            ms = list(self.link.visual_meshes.values())
            ev["tf"] = ms[0].scaled_transform.clone().numpy()
            ev["mesh_type"] = ms[0]._mesh_type
            ev["attr"] = [float(ms[0].get_attribute(k) or 0.0) if k in ms[0].prim.GetPropertyNames() else 0.0
                          for k in ("radius", "height", "size")]
        # 입자 붙은 링크의 그 순간 행렬
        ev["ptf"] = {}
        for nm in before:
            lk = system._particles_info[nm]["link"]
            if lk.prim_path not in ev["ptf"]:
                ev["ptf"][lk.prim_path] = lk.scaled_transform.clone().numpy()
        r = orig_rm(self, system)
        ev["after"] = list(system.particles.keys()) if system.particles else []
        ev["modified_after"] = int(self.obj.states[ModifiedParticles].get_value(system))
        events.append(ev)
        return r

    PM.ParticleRemover._modify_particles = rm_wrap

    # ---- 평가기 (eval.py main 과 같은 설정, 정책 = LocalPolicy 0 행동) ----
    inst = resolve_instance_ids(a.task, [a.instance], mode="public_test")
    cfg = OmegaConf.create(dict(
        env_wrapper={"_target_": "omnigibson.eval.wrappers.DefaultWrapper"}, policy_name="local",
        model={"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}, headless=True,
        partial_scene_load=True, max_steps=a.steps + 10, write_video=False, mode="public_test", seed=seed, num_envs=1,
        task={"name": a.task}, robot=OmegaConf.load("/mnt/c/behavior-2026/src/configs/r1pro_openpi.yaml")))
    ev_ = E.BatchedEvaluator(cfg)
    try:
        ev_.load_batch({0: int(inst[0])})
        scene = og.sim.scenes[0]
        # 입자 계·무리
        vis = [s for s in scene.active_systems.values() if isinstance(s, Vis)]
        print("[particles] 시각 계:", [(s.name, s.n_particles, sorted(s.groups)) for s in vis], flush=True)
        removers = [o for o in scene.objects if PM.ParticleRemover in o.states]
        print("[particles] 제거기:", [(o.name, o.category, str(o.states[PM.ParticleRemover].method)) for o in removers], flush=True)
        scope = ev_.instance_eval_states[0].env_accessor.object_scope
        covered_objs = []
        for s in vis:
            for g in sorted(s.groups):
                covered_objs.append(s._group_objects[g])
        target = covered_objs[0]
        if a.remover:
            rem = scene.object_registry("name", a.remover)
        else:
            cands = [e for k, e in scope.items() if e is not None and hasattr(e, "states") and PM.ParticleRemover in e.states
                     and e is not target]
            rem = cands[0] if cands else [o for o in removers if o is not target][0]
        print(f"[particles] 대상 {target.name}, 움직일 제거기 {rem.name}", flush=True)

        # 입자 표 (불러온 뒤 순서 = 공식 dict 순서)
        P = []
        for s in vis:
            for nm in s.particles.keys():
                info = s._particles_info[nm]
                P.append((s.name, nm, info["obj"].name, info["link"].prim_path, s._particles_local_mat[nm].numpy()))
        links = sorted({p[3] for p in P})
        # 불러오기 입력 (계·이름 → 위치·방향·배치 크기). 마지막 호출이 최종값
        inp = {}
        for sysn, names, pos, orn in load_calls:
            for i, nm in enumerate(names):
                inp[(sysn, nm)] = (pos[i], orn[i], len(names))
        q2m_check = {}
        # 평가기 프로세스 안의 T.quat2mat 이 마지막 불러오기 입력에서 같은 행렬을 내는가 (컴파일 판 / 원본 eager)
        import omnigibson.utils.transform_utils as T
        import torch._dynamo.utils as DU

        if load_calls:
            q_last = th.from_numpy(load_calls[-1][3])
            names_last = load_calls[-1][1]
            lm_at_load = {p[1]: p[4] for p in P}  # 불러온 직후 스냅숏 (스텝 동안 입자가 지워지므로)
            want = np.stack([lm_at_load[nm][:3, :3] for nm in names_last if nm in lm_at_load])
            if len(want) == len(names_last):
                rc = T.quat2mat(q_last).numpy()
                orig_f = getattr(T.quat2mat, "_torchdynamo_orig_callable", None)
                re_ = orig_f(q_last).numpy() if orig_f else rc
                q2m_check["q2m_inproc_compiled_diff"] = int(sum(not np.array_equal(rc[i].view(np.uint32), want[i].view(np.uint32)) for i in range(len(want))))
                q2m_check["q2m_inproc_eager_diff"] = int(sum(not np.array_equal(re_[i].view(np.uint32), want[i].view(np.uint32)) for i in range(len(want))))
                np.save(os.path.join(a.out, "q2m_last_in.npy"), load_calls[-1][3])
        lo, hi = target.aabb
        lo, hi = lo.numpy(), hi.numpy()
        link_objs = {p[3]: s._particles_info[p[1]]["link"] for s in vis for p in P if p[1] in s._particles_info}
        rec_ltf, rec_pose, rec_fab, rec_scale, rec_cov = [], [], [], [], []
        import omnigibson.lazy as lazy
        for step in range(a.steps):
            cur_step[0] = step
            f = (step % 20) / 19.0
            row = (step // 20) % 3
            pos = th.tensor([lo[0] + (hi[0] - lo[0]) * f, lo[1] + (hi[1] - lo[1]) * (row + 0.5) / 3.0, hi[2] + 0.02], dtype=th.float32)
            rem.set_position_orientation(position=pos, orientation=th.tensor([0, 0, 0, 1.0]))
            rem.set_linear_velocity(th.zeros(3))
            rem.set_angular_velocity(th.zeros(3))
            ev_._step_fn([0])
            ltf, pose, fab, scl = [], [], [], []
            for lp in links:
                lk = link_objs[lp]
                ltf.append(lk.scaled_transform.numpy())
                p_, q_ = lk.get_position_orientation()
                pose.append(np.r_[p_.numpy(), q_.numpy()])
                fh = og.sim.fabric_hierarchy
                mats = []
                par = lp.rsplit("/", 1)[0]
                for M in (fh.get_world_xform(lazy.usdrt.Sdf.Path(lp)), fh.get_local_xform(lazy.usdrt.Sdf.Path(lp)),
                          fh.get_world_xform(lazy.usdrt.Sdf.Path(par)), fh.get_local_xform(lazy.usdrt.Sdf.Path(par)),
                          fh.get_world_xform(lazy.usdrt.Sdf.Path(par.rsplit("/", 1)[0]))):
                    mats.append(np.array([[M[i][j] for j in range(4)] for i in range(4)], np.float64))
                fab.append(np.stack(mats))  # 링크 세계·국소, 부모(물체) 세계·국소, 장면 세계
                scl.append(np.r_[lk.scale.numpy(), lk.obj.scale.numpy() if hasattr(lk, "obj") else np.ones(3)])
            rec_ltf.append(ltf)
            rec_pose.append(pose)
            rec_fab.append(fab)
            rec_scale.append(scl)
            rec_cov.append([bool(o.states[Covered].get_value(s)) for s in vis for o in covered_objs if s._group_objects.get(o.name) is o])
        # ---- 내보내기 ----
        names = [p[1] for p in P]
        idx = {(p[0], p[1]): i for i, p in enumerate(P)}
        sys_names = sorted({p[0] for p in P})
        grp_names = sorted({p[2] for p in P})
        N = len(P)
        out = dict(
            p_sys=np.array([sys_names.index(p[0]) for p in P], np.int32),
            p_group=np.array([grp_names.index(p[2]) for p in P], np.int32),
            p_link=np.array([links.index(p[3]) for p in P], np.int32),
            p_lm=np.stack([p[4] for p in P]).astype(np.float32) if N else np.zeros((0, 4, 4), np.float32),
            p_in_pos=np.stack([inp[(p[0], p[1])][0] for p in P]).astype(np.float32),
            p_in_quat=np.stack([inp[(p[0], p[1])][1] for p in P]).astype(np.float32),
            p_in_batch=np.array([inp[(p[0], p[1])][2] for p in P], np.int32),
            step_link_tf=np.array(rec_ltf, np.float32), step_link_pose=np.array(rec_pose, np.float32),
            step_link_fabric=np.array(rec_fab, np.float64), step_link_scale=np.array(rec_scale, np.float32),
            step_covered=np.array(rec_cov, np.uint8),
        )
        E_ = len(events)
        H = max([len(e.get("hull", [])) for e in events] + [1])
        ev_arr = dict(ev_step=np.zeros(E_, np.int32), ev_sys=np.zeros(E_, np.int32), ev_kind=np.zeros(E_, np.int32),
                      ev_tf=np.zeros((E_, 4, 4), np.float32), ev_hull=np.zeros((E_, H, 3), np.float32), ev_nh=np.zeros(E_, np.int32),
                      ev_attr=np.zeros((E_, 3), np.float64), ev_mod_before=np.zeros(E_, np.int32), ev_mod_after=np.zeros(E_, np.int32),
                      ev_limit=np.zeros(E_, np.int32), ev_before=np.zeros((E_, N), np.uint8), ev_after=np.zeros((E_, N), np.uint8),
                      ev_ptf=np.zeros((E_, len(links), 4, 4), np.float32), ev_has_ptf=np.zeros((E_, len(links)), np.uint8))
        kinds = {"Cylinder": 1, "Cone": 2, "Cube": 3, "Sphere": 4, "Mesh": 5}
        for k, e in enumerate(events):
            ev_arr["ev_step"][k] = e["step"]
            ev_arr["ev_sys"][k] = sys_names.index(e["system"]) if e["system"] in sys_names else -1
            ev_arr["ev_kind"][k] = 0 if "hull" in e else kinds.get(e.get("mesh_type"), 9)
            ev_arr["ev_tf"][k] = e["tf"]
            if "hull" in e:
                ev_arr["ev_hull"][k, : len(e["hull"])] = e["hull"]
                ev_arr["ev_nh"][k] = len(e["hull"])
            else:
                ev_arr["ev_attr"][k] = e["attr"]
            ev_arr["ev_mod_before"][k], ev_arr["ev_mod_after"][k], ev_arr["ev_limit"][k] = e["modified_before"], e["modified_after"], e["limit"]
            for nm in e["before"]:
                if (e["system"], nm) in idx:
                    ev_arr["ev_before"][k, idx[(e["system"], nm)]] = 1
            for nm in e["after"]:
                if (e["system"], nm) in idx:
                    ev_arr["ev_after"][k, idx[(e["system"], nm)]] = 1
            for lp, m in e["ptf"].items():
                ev_arr["ev_ptf"][k, links.index(lp)] = m
                ev_arr["ev_has_ptf"][k, links.index(lp)] = 1
        out.update(ev_arr)
        for k, v in out.items():
            np.save(os.path.join(a.out, k + ".npy"), v)
        # 장면 제거기 사양 (엔진 입력용): 능력 인자 원문(조건·방식·기본 조건), 링크, 볼록 껍질 점 수, 계 순서
        import json

        spec = dict(active_systems=list(scene.active_systems.keys()), removers=[])
        for o in removers:
            st = o.states[PM.ParticleRemover]
            ab = o._abilities.get("particleRemover", {}) if hasattr(o, "_abilities") else {}
            spec["removers"].append(dict(
                name=o.name, category=o.category, method=str(st.method), link=st.link.name if st.link is not None else None,
                link_path=st.link.prim_path if st.link is not None else None,
                ability=json.loads(json.dumps(ab, default=str)),
                parsed_systems=list(st.conditions.keys()),
                n_conds={k: (None if v is None else len(v)) for k, v in st.conditions.items()},
                limit_visual=int(st.visual_particle_modification_limit)))
        spec["mat_origin"] = {p[1]: mat_origin.get(p[1]) for p in P}
        spec.update(q2m_check)
        spec["dynamo_counters"] = {k: dict(v) for k, v in DU.counters.items() if k in ("stats", "recompiles", "unimplemented", "frames")}
        spec["torch_threads"] = th.get_num_threads()
        spec["load_calls"] = [(c[0], len(c[1]), c[1][:3]) for c in load_calls]
        with open(os.path.join(a.out, "scene_spec.json"), "w") as f:
            json.dump(spec, f, indent=1, ensure_ascii=False)
        with open(os.path.join(a.out, "meta.txt"), "w") as f:
            f.write(f"task {a.task} instance {a.instance} steps {a.steps}\nsystems {sys_names}\ngroups {grp_names}\nlinks {links}\n")
            f.write(f"remover {rem.name} target {target.name}\nremovers {[o.name for o in removers]}\nevents {E_}\n")
        n_rm = sum(len(e["before"]) - len(e["after"]) for e in events)
        print(f"[particles] 기록 끝: 입자 {N}, 사건 {E_}, 지운 입자 {n_rm}, 끝 Covered {rec_cov[-1] if rec_cov else None}", flush=True)
    finally:
        td = getattr(og, "tempdir", None)
        if td and os.path.isdir(td) and os.path.basename(td).startswith("tmp"):
            shutil.rmtree(td, ignore_errors=True)
            print(f"[particles] 임시 장면 폴더 지움: {td}", flush=True)
        sys.stdout.flush()
        os._exit(0)  # Kit 종료 segfault 덤프 방지 (physx_capture 와 같은 이유)


if __name__ == "__main__":
    main()
