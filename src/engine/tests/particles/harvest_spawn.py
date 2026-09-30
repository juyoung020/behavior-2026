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
        srcs = []
        for o in object_candidates["sliceable"]:
            pp, qq = o.get_position_orientation()
            parts = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in dict(p).items()} for p in o.metadata["object_parts"].values()]
            srcs.append((o.name, o.category, o.scale.tolist(), parts, pp.tolist(), qq.tolist()))
        res = orig_slice(self, object_candidates)
        k = 0
        for name, cat, scale, parts, spos, sorn in srcs:
            news = []
            for i, p in enumerate(parts):
                o = res.add[k].obj
                k += 1
                news.append(dict(part=i, name=o.name, category=o.category, model=o.model))  # prim 경로는 장면에 넣은 뒤 /World/scene_0/<이름>
            STATE["map"].append(dict(step=STATE.get("step", -1), rule="SlicingRule", src=name, src_category=cat, src_scale=scale, new=news,
                                     src_pos=spos, src_orn=sorn, parts=json.loads(json.dumps(parts, default=str))))
        dump()
        return res

    TR.SlicingRule.transition = slice_wrap

    # 반쪽 에셋 값 (ig:nativeBB·ig:offsetBaseLink) — set_bbox_center 에서
    from omnigibson.objects.dataset_object import DatasetObject

    orig_set = DatasetObject.set_bbox_center_position_orientation
    STATE.setdefault("assets", {})

    def set_wrap(self, position=None, orientation=None):
        STATE["assets"][self.name] = dict(native_bbox=self.native_bbox.tolist(), offset=self.base_link_offset.tolist(), scale=self.scale.tolist())
        with open(os.path.join(cap.dump_dir, "harvest_assets.json"), "w") as f:
            json.dump(STATE["assets"], f)
        return orig_set(self, position=position, orientation=orientation)

    DatasetObject.set_bbox_center_position_orientation = set_wrap

    # USD 경로 자세 쓰기 중간값 (XFormPrim.set_position_orientation, 막 넣은 물체): 부모 세계 행렬(float32·double), 입력, USD 에 쓴 값
    from omnigibson.prims.xform_prim import XFormPrim
    import omnigibson.lazy as lazy
    from omnigibson.utils.usd_utils import get_world_pose_with_scale

    orig_x = XFormPrim.set_position_orientation
    STATE.setdefault("xform", [])

    def xwrap(self, position=None, orientation=None, frame="world"):
        rec = None
        try:
            if self.name.split(":")[0].startswith(("half_", "diced__")) or "half_" in self.prim_path:
                par = str(lazy.isaacsim.core.utils.prims.get_prim_parent(self._prim).GetPath())
                import omnigibson as og

                M = og.sim.fabric_hierarchy.get_world_xform(lazy.usdrt.Sdf.Path(par))
                rec = dict(path=self.prim_path, parent=par, frame=frame, pos=None if position is None else th.as_tensor(position).tolist(),
                           orn=None if orientation is None else th.as_tensor(orientation).tolist(),
                           parent_f32=get_world_pose_with_scale(par).tolist(), parent_d=[[M[i][j] for j in range(4)] for i in range(4)])
        except Exception as e:
            rec = dict(err=repr(e))
        r = orig_x(self, position=position, orientation=orientation, frame=frame)
        if rec is not None and "err" not in rec:
            t = self._prim.GetAttribute("xformOp:translate").Get()
            o = self._prim.GetAttribute("xformOp:orient").Get()
            rec.update(usd_translate=list(t), usd_orient=[o.GetImaginary()[0], o.GetImaginary()[1], o.GetImaginary()[2], o.GetReal()],
                       orient_type=self._prim.GetAttribute("xformOp:orient").GetTypeName().type.typeName)
        if rec is not None:
            STATE["xform"].append(rec)
            with open(os.path.join(cap.dump_dir, "harvest_xform.json"), "w") as f:
                json.dump(STATE["xform"], f)
        return r

    XFormPrim.set_position_orientation = xwrap

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

    # 재적재 왕복 검증 자료 (test_reload): removing_objects 의 dump_state 순간 입자 원점 자세(뷰) → load_state 가 set 에 준 값
    rl_rec = STATE.setdefault("reload", [])
    MP = MPS.MacroPhysicalParticleSystem
    orig_ds, orig_ls, orig_sp = MP._dump_state, MP._load_state, MP.set_particles_position_orientation

    def ds_wrap(self):
        if self.n_particles > 0:
            rl_rec.append(dict(step=STATE.get("step", -1), system=self.name, offset=Lt(self._particle_offset), dump_tfs=Lt(self.particles_view.get_transforms()),
                               pose=Lt(self.scene.pose), pose_inv=Lt(self.scene.pose_inv)))
        return orig_ds(self)

    def ls_wrap(self, state):
        STATE["loading"] = self.name
        try:
            return orig_ls(self, state)
        finally:
            STATE["loading"] = None
            with open(os.path.join(cap.dump_dir, "harvest_reload.json"), "w") as f:
                json.dump(rl_rec, f)

    def sp_wrap(self, positions=None, orientations=None):
        r = orig_sp(self, positions=positions, orientations=orientations)
        if STATE.get("loading") == self.name and rl_rec and rl_rec[-1]["system"] == self.name and "load_set" not in rl_rec[-1]:
            rl_rec[-1]["load_set"] = Lt(th.cat([positions, orientations], dim=1))
            rl_rec[-1]["load_step"] = STATE.get("step", -1)
        return r

    MP._dump_state, MP._load_state, MP.set_particles_position_orientation = ds_wrap, ls_wrap, sp_wrap
    if os.environ.get("HARVEST_BASE"):
        _install_base(cap)
    STATE["installed"] = True


def _install_base(cap):
    """base 창 정답 (core/particles/base_state.h, test_base_state): removing_objects 의 og.sim.dump_state / load_state 동안
    물체마다 뜬 값(잠·뿌리 자세·속도·관절)과 되돌리기 때의 호출 열(읽기·자세·속도·관절 쓰기·sleep/wake)을 공식 호출 자리에서 적는다.
    물체 표(등록부 차례)도 함께. 결과 harvest_base.json."""
    import omnigibson as og
    from omnigibson.prims.entity_prim import EntityPrim
    from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim

    Lt = lambda t: t.detach().cpu().numpy().tolist() if hasattr(t, "detach") else t
    rec = STATE.setdefault("base", {"objects": None, "windows": []})
    cur = {"win": None, "phase": None}
    out = os.path.join(cap.dump_dir, "harvest_base.json")

    def table():
        scene = og.sim.scenes[0]
        t = []
        for o in scene.objects:
            links = [l.prim_path for l in o.links.values() if isinstance(l, RigidDynamicPrim)]
            t.append(dict(name=o.name, articulated=bool(o.articulated), n_joints=int(o.n_joints), kinematic_only=bool(o.kinematic_only),
                          root_link=o.root_link.prim_path, dynamic_links=links))
        return t

    def log(*a):
        if cur["win"] is not None and cur["phase"] == "load":
            cur["win"]["calls"].append(list(a))

    S = type(og.sim)
    orig_dump, orig_load = S.dump_state, S.load_state

    def dump_wrap(self, serialized=False):
        cur["win"] = dict(step=STATE.get("step", -1), objects=table(), dump={}, calls=[])
        cur["phase"] = "dump"
        try:
            return orig_dump(self, serialized=serialized)
        finally:
            cur["phase"] = None

    def load_wrap(self, state, serialized=False):
        if cur["win"] is None:
            return orig_load(self, state, serialized=serialized)
        cur["phase"] = "load"
        try:
            return orig_load(self, state, serialized=serialized)
        finally:
            cur["phase"] = None
            rec["windows"].append(cur["win"])
            cur["win"] = None
            with open(out, "w") as f:
                json.dump(rec, f)

    S.dump_state, S.load_state = dump_wrap, load_wrap

    oed, oel = EntityPrim._dump_state, EntityPrim._load_state

    def ed_wrap(self):
        st = oed(self)
        if cur["win"] is not None and cur["phase"] == "dump":
            rl = st["root_link"]
            cur["win"]["dump"][self.name] = dict(asleep=bool(st["is_asleep"]), pos=Lt(rl["pos"]), ori=Lt(rl["ori"]), lin=Lt(rl.get("lin_vel")),
                                                 ang=Lt(rl.get("ang_vel")), jpos=Lt(st.get("joint_pos")), jvel=Lt(st.get("joint_vel")))
        return st

    def el_wrap(self, state):
        log("obj", self.name)
        return oel(self, state)

    EntityPrim._dump_state, EntityPrim._load_state = ed_wrap, el_wrap

    def wrap(cls, name, kind):
        orig = getattr(cls, name)

        def w(self, *a, **kw):
            r = orig(self, *a, **kw)
            if kind == "read":
                log("read", self.prim_path, Lt(r[0]), Lt(r[1]))
            elif kind == "set":
                vals = [Lt(x) for x in a] + [Lt(v) for v in kw.values()]
                log(name, self.prim_path, *vals)
            return r

        setattr(cls, name, w)

    wrap(RigidDynamicPrim, "get_position_orientation", "read")
    for n in ("set_position_orientation", "set_linear_velocity", "set_angular_velocity"):
        wrap(RigidDynamicPrim, n, "set")
    for n in ("set_joint_positions", "set_joint_velocities"):
        wrap(EntityPrim, n, "set")
    osl, owk = RigidDynamicPrim.sleep, RigidDynamicPrim.wake
    RigidDynamicPrim.sleep = lambda self: (log("rigid_sleep", self.prim_path), osl(self))[1]
    RigidDynamicPrim.wake = lambda self: (log("rigid_wake", self.prim_path), owk(self))[1]
    oes, oew = EntityPrim.sleep, EntityPrim.wake

    def es(self):
        if self.articulated:
            log("art_sleep", self.name)
        return oes(self)

    def ew(self):
        if self.articulated:
            log("art_wake", self.name)
        return oew(self)

    EntityPrim.sleep, EntityPrim.wake = es, ew


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
