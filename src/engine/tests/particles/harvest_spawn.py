"""새 행위자 틀 수확 (P2·P3 물리 결합 (a), 조정자 결정 09-30): physx_capture.py --script 로 부른다(평가 실행과 별개, 인스턴스마다 1 회).
판의 자를 것(sliceable)·다질 것(diceable)을 하나씩 공식 경로로 자르고 다진다: 자르개를 대상 위 공중에 두어 SlicerActive 가 켜지게(닿지 않고 2 초)
기다린 뒤 대상에 겹치게 한다. 잘려서 생긴 반쪽 중 다질 수 있는 것은 줄 끝에 더해 이어서 다진다. OVD 에는 모든 새 행위자 생성이 남고,
리드의 g1_sc 틀 쓰기가 그것을 "새 행위자 틀" 파일로 만든다. 이 스크립트는 키 대응(harvest_map.json)을 적는다:
  전이마다 {스텝, 규칙, 원본 물체 이름·범주·척도, 새 물체 이름·범주·모델·부분 번호·prim 경로, 다진 계 이름·만든 입자 수}
  bash /mnt/c/behavior-2026/src/engine/tests/particles/run_harvest.sh <과제> <모드> <인스턴스 번호> [최대 스텝]
조작은 OmniGibson API 로만 한다(PhysX 쓰기는 곁기록·OVD 에 남음). 환경 변수 HARVEST_WAIT(기본 75), HARVEST_HOLD(기본 40).
"""
import json
import os

import torch as th

STATE = {"queue": None, "cur": None, "phase_start": 0, "slicer": None, "map": [], "done": False, "installed": False}


def _install(cap):
    import omnigibson.transition_rules as TR

    out = os.path.join(cap.dump_dir, "harvest_map.json")

    def dump():
        with open(out, "w") as f:
            json.dump(STATE["map"], f, indent=1)

    orig_slice = TR.SlicingRule.transition

    def slice_wrap(self, object_candidates):
        srcs = [(o.name, o.category, o.scale.tolist(), [dict(p) for p in o.metadata["object_parts"].values()]) for o in object_candidates["sliceable"]]
        res = orig_slice(self, object_candidates)
        k = 0
        for name, cat, scale, parts in srcs:
            news = []
            for i, p in enumerate(parts):
                o = res.add[k].obj
                k += 1
                news.append(dict(part=i, name=o.name, category=o.category, model=o.model))  # prim 경로는 장면에 넣은 뒤 /World/scene_0/<이름>
            STATE["map"].append(dict(step=STATE.get("step", -1), rule="SlicingRule", src=name, src_category=cat, src_scale=scale, new=news))
        dump()
        return res

    TR.SlicingRule.transition = slice_wrap

    # 다지기 검증 자료 (test_dice_capture): 격자 입력(링크 aabb·충돌 메시 점·틀), 난수 상태, 생성 중심, 원점 자세
    import omnigibson.systems.macro_particle_system as MPS
    import omnigibson.systems.system_base as SB

    Lt = lambda t: t.detach().cpu().numpy().tolist()
    gen_rec = STATE.setdefault("gen", [])
    orig_gfl = SB.PhysicalParticleSystem.generate_particles_from_link

    def gfl_wrap(self, obj, link, use_visual_meshes=True, **kw):
        lo, hi = link.visual_aabb
        meshes = link.visual_meshes if use_visual_meshes else link.collision_meshes
        gen_rec.append(dict(system=self.name, radius=float(self.particle_radius), lo=Lt(lo), hi=Lt(hi), offset=Lt(self._particle_offset),
                            meshes=[dict(type=m._mesh_type, points=Lt(m.points) if m._mesh_type == "Mesh" else None, tf=Lt(m.scaled_transform))
                                    for m in meshes.values()],
                            rng=th.get_rng_state().numpy().tolist(), n_before=self.n_particles))
        return orig_gfl(self, obj, link, use_visual_meshes=use_visual_meshes, **kw)

    SB.PhysicalParticleSystem.generate_particles_from_link = gfl_wrap
    orig_gp = MPS.MacroPhysicalParticleSystem.generate_particles

    def gp_wrap(self, positions, orientations=None, **kw):
        if gen_rec and "centers" not in gen_rec[-1]:
            gen_rec[-1]["centers"] = Lt(positions)
        r = orig_gp(self, positions, orientations=orientations, **kw)
        if gen_rec and "frames" not in gen_rec[-1]:
            gen_rec[-1]["frames"] = Lt(self.particles_view.get_transforms())
        return r

    MPS.MacroPhysicalParticleSystem.generate_particles = gp_wrap

    def dump_gen():
        with open(os.path.join(cap.dump_dir, "harvest_dice.json"), "w") as f:
            json.dump(gen_rec, f)

    orig_dice = TR.DicingRule.transition

    def dice_wrap(self, object_candidates):
        before = {}
        objs = list(object_candidates["diceable"])
        for o in objs:
            sname = "diced__" + o.category.removeprefix("half_")
            try:
                from omnigibson.object_states import Cooked

                if Cooked in o.states and o.states[Cooked].get_value():
                    sname = "cooked__" + sname
            except Exception:
                pass
            sysm = self.scene.get_system(sname)
            before[o.name] = (sname, o.category, o.scale.tolist(), sysm.n_particles)
        res = orig_dice(self, object_candidates)
        for o in objs:
            sname, cat, scale, n0 = before[o.name]
            n1 = self.scene.get_system(sname).n_particles
            STATE["map"].append(dict(step=STATE.get("step", -1), rule="DicingRule", src=o.name, src_category=cat, src_scale=scale, system=sname,
                                     n_new=n1 - n0))
        dump()
        dump_gen()
        return res

    TR.DicingRule.transition = dice_wrap
    STATE["installed"] = True


def _alive(o):
    """장면 등록부에 아직 있는가 (지운 물체는 prim 이 잠시 남아도 등록부에서 빠진다)"""
    import omnigibson as og

    try:
        return o is not None and og.sim.scenes[0].object_registry("name", o.name) is o
    except Exception:
        return False


def on_before_apply(ev, step, cap):
    import omnigibson as og

    if not STATE["installed"]:
        _install(cap)
    STATE["step"] = step
    if STATE["done"]:
        return
    scene = og.sim.scenes[0]
    wait = int(os.environ.get("HARVEST_WAIT", "75"))
    hold = int(os.environ.get("HARVEST_HOLD", "40"))
    if STATE["queue"] is None:
        slicers = scene.object_registry("abilities", "slicer", [])
        slicers = sorted(slicers, key=lambda o: o.name)
        if not slicers:
            print("[harvest] 자르개 없음 — 수확할 것 없음", flush=True)
            STATE["done"] = True
            return
        STATE["slicer"] = slicers[0]
        cands = [o for o in scene.objects if ("sliceable" in o._abilities or "diceable" in o._abilities) and not o.fixed_base]
        STATE["queue"] = sorted(cands, key=lambda o: o.name)
        print(f"[harvest] 자르개 {STATE['slicer'].name}, 대상 {[o.name for o in STATE['queue']]}", flush=True)
    slicer = STATE["slicer"]
    zero = th.zeros(3)
    # 지금 대상이 없어졌으면 다음으로 (새로 생긴 다질 반쪽을 줄에 더함)
    if STATE["cur"] is not None and not _alive(STATE["cur"]):
        print(f"[harvest] 스텝 {step}: {STATE['cur'].name} 전이됨", flush=True)
        for o in scene.objects:
            if "diceable" in o._abilities and o not in STATE["queue"] and o.name.startswith("half_") and _alive(o):
                STATE["queue"].append(o)
        STATE["cur"] = None
    if STATE["cur"] is None:
        while STATE["queue"] and not _alive(STATE["queue"][0]):
            STATE["queue"].pop(0)
        if not STATE["queue"]:
            print(f"[harvest] 스텝 {step}: 끝 (전이 {len(STATE['map'])})", flush=True)
            STATE["done"] = True
            pos, _ = slicer.get_position_orientation()
            slicer.set_position_orientation(position=pos + th.tensor([0.0, 0.0, 2.0]))
            slicer.set_linear_velocity(zero)
            slicer.set_angular_velocity(zero)
            return
        STATE["cur"] = STATE["queue"].pop(0)
        STATE["phase_start"] = step
    t = STATE["cur"]
    k = step - STATE["phase_start"]
    tpos, _ = t.get_position_orientation()
    if k < wait:
        slicer.set_position_orientation(position=tpos + th.tensor([0.0, 0.0, 0.8]))
    elif k < wait + hold:
        slicer.set_position_orientation(position=tpos + th.tensor([0.0, 0.0, 0.05]))
    else:  # 안 잘림 — 건너뜀
        print(f"[harvest] 스텝 {step}: {t.name} 안 잘림 (건너뜀)", flush=True)
        STATE["cur"] = None
        return
    slicer.set_linear_velocity(zero)
    slicer.set_angular_velocity(zero)
