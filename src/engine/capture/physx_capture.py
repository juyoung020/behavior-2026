"""공식 평가기를 한 줄도 고치지 않고 돌리면서, 물리 층 검증에 쓸 기록을 뜬다 (엔진 자체 구현 층 0).

    python C:\\behavior-2026\\src\\engine\\capture\\physx_capture.py --dump-dir <폴더> [--no-ovd] [--no-convex] [--no-sidelog] [--no-render] \\
        -- <tools\\eval_instrumented.py 인자 그대로, 예: --trace -- --task-name turning_on_radio ...>

남기는 것 (<폴더> 안, 에셋 파생물이라 git 에 올리지 않는다 -- src\\engine\\.gitignore 의 dumps/)
    *_rec.ovd        OmniPVD 기록. PhysX SDK 가 모든 객체 생성·설정값과 매 서브스텝 결과(자세·속도·관절값)를
                     float 원본 그대로 쓴 것. Kit 설정 두 개만 켠다(평가기 코드 무수정):
                       /persistent/physics/omniPvdOvdRecordingDirectory, /physics/omniPvdOutputEnabled
                     (omni.physx 소스 omni/extensions/runtime/source/omni.physx/plugins/Setup.cpp:444-560)
    convex.npz       장면의 모든 볼록 충돌 메시 원본(꼭짓점·인덱스·다각형 평면식). OVD 에는 꼭짓점·삼각형만 있어서
                     면 평면식까지 비트 그대로 다시 만들려면 이게 필요하다. omni.physx 공개 API
                     request_convex_collision_representation 로 읽는다.
    sidelog.npz      OVD 에 안 남는 PhysX 쓰기(관절체 상태를 한꺼번에 덮는 applyCache 계열 = omni.physics.tensors 의
                     set_dof_positions / set_root_transforms / set_dof_actuation_forces 등)를 '몇 번째 물리 스텝 뒤'
                     번호와 함께 적는다. 번호 = 물리 스텝 뒤 콜백(post-step, 우선순위 0)이 불린 횟수 = OVD 프레임 번호.
    meta.json        물리 설정(스레드 수 등), 스텝 수, 파일 목록

원리
- OmniPVD 설정이 바뀌면 omni.physx 가 PxPhysics 를 다시 만들며 OVD 기록을 붙인다. 앱이 막 뜬 직후(장면이 없을 때)에
  켜서, 장면 로딩부터 끝까지 PhysX 호출 전체가 기록되게 한다.
- 기록을 켜도 물리 결과가 안 바뀌는지는 tools\\trace_compare.py 로 기존 실행(nf_a)과 비트 비교해 확인한다.
- 나머지(시간 계측, trace.npz)는 평가기 가속 에이전트의 tools\\eval_instrumented.py 를 그대로 부른다.
"""
from __future__ import annotations

import functools
import json
import os
import runpy
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INSTRUMENTED = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools", "eval_instrumented.py"))


class Capture:
    def __init__(self, dump_dir: str, ovd: bool, convex: bool, sidelog: bool):
        self.dump_dir = os.path.abspath(dump_dir)
        os.makedirs(self.dump_dir, exist_ok=True)
        self.ovd, self.convex, self.sidelog = ovd, convex, sidelog
        self.n_post = 0  # post-step 콜백 횟수 = 끝난 PhysX simulate 수
        self.n_pre = 0
        self.log = []  # (post_count, kind, method, view_id, indices, data)
        self.views = {}  # id(view) -> (kind, prim_paths)
        self.view_extra = {}  # id(view) -> {dof_paths, signs, max_dofs}
        self.filters = {}  # 태그 -> 거르개 표 재료
        self.meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._subs = []
        self.convex_done = False
        self.dump_at_pre = 0
        self.dump_prim = ""

    # ------------------------------------------------------------------ 앱이 뜬 직후
    def after_launch(self):
        import carb
        import omni.physx

        cs = carb.settings.get_settings()
        keys = ["/persistent/physics/numThreads", "/physics/physxDispatcher", "/physics/suppressReadback",
                "/physics/updateToUsd", "/physics/useFastCache", "/persistent/physics/useLocalMeshCache",
                "/physics/collisionConeCustomGeometry", "/physics/collisionCylinderCustomGeometry"]
        self.meta["settings"] = {k: repr(cs.get(k)) for k in keys}
        if self.ovd:
            d = self.dump_dir.replace("\\", "/").rstrip("/") + "/"
            cs.set_string("/persistent/physics/omniPvdOvdRecordingDirectory", d)
            cs.set_bool("/physics/omniPvdOutputEnabled", True)
            self.meta["ovd_recording"] = bool(cs.get("/physics/omniPvdIsRecording"))
            print(f"[capture] OmniPVD 기록 시작: {d} (recording={self.meta['ovd_recording']})", flush=True)
        iface = omni.physx.get_physx_interface()

        def pre(_dt):
            self.n_pre += 1
            if self.dump_at_pre and self.n_pre == self.dump_at_pre:
                try:
                    self.dump_prim_state()
                except Exception as e:  # 진단 실패가 평가를 바꾸면 안 된다
                    print(f"[capture] 상태 덤프 실패: {e!r}", flush=True)

        def post(_dt):
            self.n_post += 1

        # order 0 = 가장 먼저. OmniGibson 의 콜백(order 0)보다 먼저 구독하므로 같은 단계에서 앞에 불린다.
        self._subs.append(iface.subscribe_physics_on_step_events(pre, pre_step=True, order=0))
        self._subs.append(iface.subscribe_physics_on_step_events(post, pre_step=False, order=0))

    # ------------------------------------------------------------------ OVD 에 안 남는 쓰기
    def install_sidelog(self):
        if not self.sidelog:
            return
        import omni.physics.tensors.impl.api as api

        def to_np(x, dt):
            if x is None:
                return np.zeros(0, dt)
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            elif hasattr(x, "numpy") and not isinstance(x, np.ndarray):
                x = x.numpy()
            return np.ascontiguousarray(np.asarray(x), dtype=dt)

        def wrap(cls, name, kind):
            orig = getattr(cls, name)

            @functools.wraps(orig)
            def f(view, data, indices, *a, **kw):
                vid = id(view)
                if vid not in self.views:
                    try:
                        paths = list(view.prim_paths)
                    except Exception:
                        paths = []
                    self.views[vid] = (kind, paths)
                    if kind == "art":
                        self.view_extra[vid] = self.view_meta(view) or {}
                        self.view_extra[vid]["max_dofs"] = int(getattr(view, "max_dofs", 0))
                ret = orig(view, data, indices, *a, **kw)
                # 쓴 값이 실제로 들어갔는지 같은 뷰의 get_* 로 다시 읽어 본다 (읽기만). 곁기록에는 효과 없는 쓰기가 섞인다
                # (옛 PhysX 인스턴스의 뷰, post 43 에 새로 만든 로봇 뷰 등 — 문서 15절 S0 ②). 1 = 들어감, 0 = 안 들어감, -1 = 확인 못함
                eff = -1
                try:
                    getter = getattr(view, "get_" + name[4:], None)
                    if getter is not None:
                        got = to_np(getter(), np.float32).reshape(-1)
                        want = to_np(data, np.float32).reshape(-1)
                        if got.size == want.size:
                            eff = int(np.array_equal(got.view(np.uint32), want.view(np.uint32)))
                        else:
                            idx = to_np(indices, np.int64).reshape(-1)
                            n = view.count if hasattr(view, "count") else 0
                            if n and got.size % n == 0 and want.size % max(len(idx), 1) == 0 and len(idx):
                                per = got.size // n
                                sel = got.reshape(n, per)[idx].reshape(-1)
                                w = want.reshape(-1)[: sel.size] if want.size >= sel.size else want
                                if sel.size == w.size:
                                    eff = int(np.array_equal(sel.view(np.uint32), w.view(np.uint32)))
                except Exception:
                    eff = -1
                self.log.append((self.n_post, self.n_pre, kind, name, vid, to_np(indices, np.uint32),
                                 to_np(data, np.float32), eff))
                return ret

            setattr(cls, name, f)

        n = 0
        for cls, kind in ((api.ArticulationView, "art"), (api.RigidBodyView, "rb")):
            for name in dir(cls):
                if name.startswith("set_") and callable(getattr(cls, name)):
                    wrap(cls, name, kind)
                    n += 1
        # OmniGibson 의 묶음 제어(BatchControlViewAPIImpl)는 _backend 를 직접 부른다 -> 그 함수들도 감싼다
        from omnigibson.utils import usd_utils

        B = usd_utils.BatchControlViewAPIImpl
        for name in ("_set_dof_position_targets", "_set_dof_velocity_targets", "_set_dof_actuation_forces"):
            orig = getattr(B, name)

            def mk(orig, name):
                @functools.wraps(orig)
                def f(self_, data, indices, cast=True):
                    vid = id(self_._view)
                    if vid not in self.views:
                        try:
                            self.views[vid] = ("art", list(self_._view.prim_paths))
                        except Exception:
                            self.views[vid] = ("art", [])
                        self.view_extra[vid] = self.view_meta(self_._view) or {}
                        self.view_extra[vid]["max_dofs"] = int(getattr(self_._view, "max_dofs", 0))
                    self.log.append((self.n_post, self.n_pre, "batch", name, vid, to_np(indices, np.uint32),
                                     to_np(data, np.float32), -1))
                    return orig(self_, data, indices, cast=cast)

                return f

            setattr(B, name, mk(orig, name))
            n += 1
        print(f"[capture] 텐서 쓰기 {n} 개 함수 감쌈", flush=True)
        self.install_psi_proxy()

    def install_psi_proxy(self):
        """og.sim.psi (IPhysxSimulation) 의 wake_up / put_to_sleep / apply_force_at_pos / apply_torque 도 OVD 에 안 남는다.
        Simulator.psi 속성을 기록 대리 객체로 바꿔 끼운다 (호출은 그대로 넘긴다)."""
        import omnigibson as og
        from pxr import PhysicsSchemaTools

        cap = self
        Sim = type(og.sim)
        orig_prop = Sim.psi

        class PsiProxy:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                attr = getattr(self._inner, name)
                if name not in ("wake_up", "put_to_sleep", "apply_force_at_pos", "apply_torque", "apply_force"):
                    return attr

                def logged(stage_id, prim_id, *a):
                    path = str(PhysicsSchemaTools.intToSdfPath(prim_id))
                    vid = ("psi", path)
                    if vid not in cap.views:
                        cap.views[vid] = ("psi", [path])
                    data = np.concatenate([np.asarray(x, np.float32).ravel() for x in a]) if a else np.zeros(0, np.float32)
                    cap.log.append((cap.n_post, cap.n_pre, "psi", name, vid, np.zeros(0, np.uint32), data, -1))
                    return attr(stage_id, prim_id, *a)

                return logged

        def psi(sim_self):
            return PsiProxy(orig_prop.fget(sim_self))

        Sim.psi = property(psi)
        print("[capture] og.sim.psi 대리 객체 설치 (wake_up/put_to_sleep/힘)", flush=True)

    # ------------------------------------------------------------------ 충돌 거르개 표 (omni.physx 내부 표의 재료)
    def dump_filters(self, tag):
        """omni.physx 거르개(PhysXScene.cpp:48)가 보는 두 표의 재료를 USD 에서 뜬다.
        - 충돌 그룹: CollisionGroup prim 의 filteredGroups 와 includes (그룹 번호 = 모양 filterData.word2, OVD 에서 찾음)
        - 거른 쌍: FilteredPairsAPI 의 filteredPairs (쌍 번호 = 모양 포인터 해시 = word1, InternalFilteredPairs.cpp:157)"""
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        groups, rels, reports = [], [], 0
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.CollisionGroup):
                g = UsdPhysics.CollisionGroup(prim)
                inc = g.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
                groups.append({"path": str(prim.GetPath()),
                               "filtered": [str(t) for t in g.GetFilteredGroupsRel().GetTargets()],
                               "includes": [str(t) for t in inc]})
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                for t in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                    rels.append([str(prim.GetPath()), str(t)])
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                reports += 1
        self.filters[tag] = {"groups": groups, "rels": rels, "contact_report_prims": reports, "post": self.n_post}
        print(f"[capture] 거르개 표({tag}): 그룹 {len(groups)}, 거른 쌍 {len(rels)}, 접촉 보고 prim {reports}", flush=True)

    def view_meta(self, view):
        """관절체 뷰의 dof 부호(isDofBody0Parent)를 USD 로 계산: dof 의 조인트 body0 가 부모 링크면 +1, 아니면 -1."""
        from pxr import UsdPhysics
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        signs = []
        try:
            dof_paths = list(view.dof_paths)
            link_paths = list(view.link_paths)
        except Exception:
            return None
        for a, dofs in enumerate(dof_paths):
            try:
                mt = view.get_metatype(a)
                names = list(mt.link_names)
                parents = list(mt.link_parents)
                lp = list(link_paths[a])
                name2path = dict(zip(names, lp))
                parent_of = {name2path.get(n): name2path.get(p) for n, p in zip(names, parents)}
            except Exception:
                parent_of = {}
            s = []
            for jp in dofs:
                j = UsdPhysics.Joint(stage.GetPrimAtPath(str(jp)))
                b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
                b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
                b0 = b0[0] if b0 else None
                b1 = b1[0] if b1 else None
                s.append(-1 if (b0 is not None and parent_of.get(b0) == b1) else 1)
            signs.append(s)
        return {"dof_paths": [[str(x) for x in d] for d in dof_paths], "signs": signs}

    # ------------------------------------------------------------------ 볼록 메시 원본
    def dump_convex(self):
        if not self.convex or self.convex_done:
            return
        self.convex_done = True
        t0 = time.perf_counter()
        import omni.usd
        from omni.physx import get_physx_cooking_interface
        from pxr import PhysicsSchemaTools, UsdPhysics

        ctx = omni.usd.get_context()
        stage, stage_id = ctx.get_stage(), ctx.get_stage_id()
        cooking = get_physx_cooking_interface()
        paths, nconv, verts, idx, polys, poly_counts, vert_counts, idx_counts, results = [], [], [], [], [], [], [], [], []
        n_prims = 0
        for prim in stage.Traverse():
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) and prim.HasAPI(UsdPhysics.MeshCollisionAPI)):
                continue
            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approx not in ("convexHull", "convexDecomposition"):
                continue
            n_prims += 1
            got = {}

            def cb(result, convexes, got=got):
                got["r"] = int(result)
                got["c"] = convexes

            cooking.request_convex_collision_representation(
                stage_id, PhysicsSchemaTools.sdfPathToInt(prim.GetPath()), False, cb)
            convs = got.get("c") or []
            paths.append(str(prim.GetPath()))
            results.append(got.get("r", -1))
            nconv.append(len(convs))
            for c in convs:
                v = np.array([[p.x, p.y, p.z] for p in c.vertices], np.float32).reshape(-1, 3)
                ii = np.array(list(c.indices), np.uint8)
                pp = np.array([[q.plane.x, q.plane.y, q.plane.z, q.plane.w, q.num_vertices, q.index_base]
                               for q in c.polygons], np.float64).reshape(-1, 6)
                verts.append(v); idx.append(ii); polys.append(pp)
                vert_counts.append(len(v)); idx_counts.append(len(ii)); poly_counts.append(len(pp))
        out = os.path.join(self.dump_dir, "convex.npz")
        np.savez_compressed(
            out, paths=np.array(paths), result=np.array(results, np.int32), nconv=np.array(nconv, np.int32),
            vert_counts=np.array(vert_counts, np.int32), idx_counts=np.array(idx_counts, np.int32),
            poly_counts=np.array(poly_counts, np.int32),
            verts=np.concatenate(verts) if verts else np.zeros((0, 3), np.float32),
            indices=np.concatenate(idx) if idx else np.zeros(0, np.uint8),
            polygons=np.concatenate(polys) if polys else np.zeros((0, 6)))
        self.meta["convex"] = {"prims": n_prims, "meshes": len(verts), "sec": round(time.perf_counter() - t0, 2),
                               "at_post_count": self.n_post}
        print(f"[capture] 볼록 메시 {len(verts)} 개 (충돌 prim {n_prims}) -> {out}", flush=True)

    # ------------------------------------------------------------------ 진단: 지정 simulate 직전 관절체의 PhysX 쪽 실제 값 (읽기만)
    def dump_prim_state(self):
        """--dump-at-pre N --dump-prim PATH: N 번째 simulate 직전(전체 번호, meta 의 post 번호와 같은 셈)에 관절체 텐서 뷰의 get_* 를 전부 읽어
        state_pre<N>.npz 로. 재생기 쪽 같은 덤프(ovd_replay --dump-art)와 비교해 OVD 밖에서 달라진 값을 찾는다 (문서 15절 S0 로봇)."""
        import omni.physics.tensors as T
        import omni.usd

        stage_id = omni.usd.get_context().get_stage_id()
        sv = T.create_simulation_view("numpy", stage_id)
        av = sv.create_articulation_view(self.dump_prim)
        out = {}
        for name in sorted(dir(av)):
            if not name.startswith("get_"):
                continue
            try:
                v = getattr(av, name)()
            except Exception:
                continue
            try:
                out[name[4:]] = np.asarray(v)
            except Exception:
                pass
        for name in ("link_paths", "dof_paths", "dof_names", "link_names", "shared_metatype"):
            try:
                v = getattr(av, name)
                out["attr_" + name] = np.asarray(v if not hasattr(v, "link_names") else v.link_names, dtype=object)
            except Exception:
                pass
        try:
            mt = av.shared_metatype
            out["meta_link_names"] = np.asarray(mt.link_names, dtype=object)
            out["meta_dof_names"] = np.asarray(mt.dof_names, dtype=object)
        except Exception:
            pass
        path = os.path.join(self.dump_dir, f"state_pre{self.n_pre}.npz")
        np.savez(path, **out)
        print(f"[capture] 상태 덤프 {len(out)} 항목 -> {path}", flush=True)

    # ------------------------------------------------------------------ 이름 대응 (엔진이 BDDL 이름·로봇을 물리 몸체에 잇는 데 필요)
    def dump_scope(self, ev):
        """판마다 BDDL 물체 이름 -> prim 경로·링크 경로, 로봇 관절 순서·팔/손끝 링크·센서·제어기 설정을 scope.json 에.
        prim 경로는 OVD 의 PxActor.name 과 같은 문자열이라 엔진이 OVD 몸체 번호로 잇는다. 읽기만 한다."""

        def plain(v):
            try:
                import torch as th

                if isinstance(v, th.Tensor):
                    return v.detach().cpu().tolist()
            except Exception:
                pass
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {str(k): plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            return repr(v)

        def links_of(ent):
            try:
                return {n: l.prim_path for n, l in ent.links.items()}
            except Exception:
                return {}

        out = []
        for st in ev.instance_eval_states:
            acc = st.env_accessor
            objs = {}
            for name, ent in acc.object_scope.items():
                if ent is None:
                    objs[name] = None
                    continue
                objs[name] = {"cls": type(ent).__name__, "name": getattr(ent, "name", None),
                              "prim_path": getattr(ent, "prim_path", None), "links": links_of(ent)}
            r = acc.robot
            def lp(attr):
                try:
                    return getattr(r, attr).prim_path
                except Exception:
                    return None

            rob = {"name": r.name, "model": r.model, "prim_path": r.prim_path, "action_dim": int(r.action_dim),
                   # get_position_orientation = base_footprint_link 자세 (robot.py:3879), get_root_... = root_link 자세
                   "base_footprint_link": lp("base_footprint_link"), "root_link": lp("root_link"),
                   "arm_names": list(getattr(r, "arm_names", [])),
                   "eef_link_names": plain(getattr(r, "eef_link_names", {})),
                   "joints": {n: {"prim_path": j.prim_path, "body0": plain(getattr(j, "body0", None)),
                                  "body1": plain(getattr(j, "body1", None)), "type": plain(getattr(j, "joint_type", None)),
                                  "n_dof": plain(getattr(j, "n_dof", None))} for n, j in r.joints.items()},
                   "links": links_of(r),
                   "sensors": {n: getattr(s, "prim_path", None) for n, s in r.sensors.items()},
                   "controller_order": plain(getattr(r, "controller_order", [])),
                   "controller_action_idx": plain(getattr(r, "controller_action_idx", {})),
                   "controller_config": plain(getattr(r, "_controller_config", {}))}
            out.append({"env_idx": st.env_idx, "instance_id": st.instance_id, "objects": objs, "robot": rob})
        with open(os.path.join(self.dump_dir, "scope.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        # OVD 에 안 남는 단일 액터 플래그(중력 끔): USD physxRigidBody:disableGravity 가 참인 강체 prim 목록 -> gravity_off.txt
        # (omni 가 setActorFlag(eDISABLE_GRAVITY) 로 넣는데 PhysX 는 단일 플래그 쓰기를 OVD 에 기록하지 않는다 — 문서 15절)
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            names = []
            for prim in stage.Traverse():
                a = prim.GetAttribute("physxRigidBody:disableGravity")
                if a and a.IsValid() and a.Get():
                    names.append(str(prim.GetPath()))
            with open(os.path.join(self.dump_dir, "gravity_off.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(names) + ("\n" if names else ""))
            print(f"[capture] 중력 끔 prim {len(names)} 개 -> gravity_off.txt", flush=True)
        except Exception as e:
            print(f"[capture] 중력 끔 목록 실패: {e!r}", flush=True)
        print(f"[capture] 이름 대응 scope.json: 판 {len(out)}, 물체 {sum(len(o['objects']) for o in out)}", flush=True)

    # ------------------------------------------------------------------ 끝
    def finish(self):
        import carb

        self.meta["post_step_count"] = self.n_post
        self.meta["pre_step_count"] = self.n_pre
        if self.sidelog:
            kinds = np.array([e[2] for e in self.log])
            np.savez_compressed(
                os.path.join(self.dump_dir, "sidelog.npz"),
                post=np.array([e[0] for e in self.log], np.int64), pre=np.array([e[1] for e in self.log], np.int64),
                kind=kinds, method=np.array([e[3] for e in self.log]), view=np.array([str(e[4]) for e in self.log]),
                idx_len=np.array([len(e[5]) for e in self.log], np.int64),
                idx=np.concatenate([e[5] for e in self.log]) if self.log else np.zeros(0, np.uint32),
                data_len=np.array([e[6].size for e in self.log], np.int64),
                data=np.concatenate([e[6].ravel() for e in self.log]) if self.log else np.zeros(0, np.float32),
                eff=np.array([e[7] if len(e) > 7 else -1 for e in self.log], np.int8),
                view_ids=np.array([str(k) for k in self.views]),
                view_kind=np.array([v[0] for v in self.views.values()]),
                view_paths=np.array(["|".join(v[1]) for v in self.views.values()]))
            # 뷰 id 는 str(id) 로 맞춘다 (psi 는 튜플)
            with open(os.path.join(self.dump_dir, "views.json"), "w", encoding="utf-8") as f:
                json.dump({str(k): {"kind": v[0], "prims": v[1], **(self.view_extra.get(k) or {})}
                           for k, v in self.views.items()}, f, ensure_ascii=False)
            self.meta["sidelog_calls"] = len(self.log)
        with open(os.path.join(self.dump_dir, "filters.json"), "w", encoding="utf-8") as f:
            json.dump(self.filters, f, ensure_ascii=False, indent=1)
        if self.ovd:
            cs = carb.settings.get_settings()
            cs.set_bool("/physics/omniPvdOutputEnabled", False)  # -> omni.physx 가 SDK 를 정리하며 tmp.ovd 를 *_rec.ovd 로
            self.meta["ovd_files"] = sorted(f for f in os.listdir(self.dump_dir) if f.endswith(".ovd"))
        self.meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.dump_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=1)
        print(f"[capture] 끝: 물리 스텝 {self.n_post}, 곁기록 {len(self.log)} 건, {self.meta.get('ovd_files')}", flush=True)


def install_no_render():
    """렌더 장치가 없는 곳(WSL, RTX 없음)용: 카메라 읽기만 같은 모양·형의 0 영상으로 바꾼다.
    물리는 안 건드린다(읽기 전용 경로). 재생 서버는 관측을 안 보므로 행동열·물리 결과는 그대로다.
    원래 경로: VisionSensor._get_obs -> replicator annotator.get_data (렌더 장치 없으면 data=None 으로 죽는다,
    omni.replicator.core annotator_utils.py:454). 모양·형은 VisionSensor._obs_space_mapping 과
    _preprocess_cpu_obs/_preprocess_gpu_obs(seg_ 는 int64/int32) 를 따른다."""
    import omnigibson as og
    import torch as th
    from omnigibson.sensors import vision_sensor as VS
    from omnigibson.sensors.sensor_base import BaseSensor

    def get_obs_zero(self):
        obs, info = BaseSensor._get_obs(self)
        space = self._obs_space_mapping
        dev = og.sim.device if "cuda" in str(og.sim.device) else "cpu"
        for m in self._modalities:
            spec = space[m]
            if not isinstance(spec, tuple):  # bbox (Sequence 공간) -> 빈 목록
                obs[m], info[m] = [], {}
                continue
            shape, _lo, _hi, dt = spec
            if "seg_" in m:
                tdt = th.int32 if dev != "cpu" else th.int64
                info[m] = {}
            else:
                tdt = {np.dtype(np.uint8): th.uint8, np.dtype(np.float32): th.float32}[np.dtype(dt)]
            obs[m] = th.zeros(shape, dtype=tdt, device=dev)
        return obs, info

    VS.VisionSensor._get_obs = get_obs_zero
    print("[capture] 렌더 없음: 카메라 관측을 0 영상으로 (물리 무관)", flush=True)


def install(cap: Capture):
    import omnigibson as og
    from omnigibson import simulator as S

    if getattr(cap, "no_render", False):
        install_no_render()

    orig_launch_app = S._launch_app

    @functools.wraps(orig_launch_app)
    def launch_app(*a, **kw):
        app = orig_launch_app(*a, **kw)
        cap.after_launch()
        return app

    S._launch_app = launch_app

    from omnigibson.eval import evaluator as E

    Ev = E.BatchedEvaluator

    # og.sim(Simulator 클래스)은 og.launch 안에서 만들어진다 -> launch 가 끝난 직후에 곁기록 겉싸개를 건다
    orig_og_launch = og.launch

    @functools.wraps(orig_og_launch)
    def og_launch(*a, **kw):
        r = orig_og_launch(*a, **kw)
        if not getattr(cap, "_sidelog_installed", False):
            cap._sidelog_installed = True
            cap.install_sidelog()
        return r

    og.launch = og_launch

    orig_load = Ev.load_batch

    @functools.wraps(orig_load)
    def load_batch(self, *a, **kw):
        r = orig_load(self, *a, **kw)
        try:
            cap.dump_filters("episode_start")
            cap.meta["episode_start_post"] = cap.n_post  # 이 번호 뒤부터가 정책 롤아웃
        except Exception as e:
            print(f"[capture] 거르개 표 기록 실패: {e!r}", flush=True)
        try:
            cap.dump_scope(self)
        except Exception as e:  # 기록 실패가 평가를 바꾸면 안 된다
            print(f"[capture] 이름 대응 기록 실패: {e!r}", flush=True)
        return r

    Ev.load_batch = load_batch

    orig_exit = Ev.__exit__

    @functools.wraps(orig_exit)
    def ev_exit(self, *a, **kw):
        try:
            cap.dump_convex()  # 장면이 아직 살아 있을 때 (물리에 쓰지 않는 읽기 전용 요청)
        except Exception as e:  # 기록 실패가 평가 결과를 바꾸면 안 된다
            print(f"[capture] 볼록 메시 기록 실패: {e!r}", flush=True)
        try:
            cap.dump_filters("end")
        except Exception as e:
            print(f"[capture] 거르개 표 기록 실패: {e!r}", flush=True)
        return orig_exit(self, *a, **kw)

    Ev.__exit__ = ev_exit

    orig_shutdown = og.shutdown

    @functools.wraps(orig_shutdown)
    def shutdown(*a, **kw):
        try:
            cap.finish()
        except Exception as e:
            print(f"[capture] 마무리 실패: {e!r}", flush=True)
        # Kit 종료(og.shutdown)는 렌더 장치 없는 WSL 에서 늘 segfault 하고(139), WSL 이 판마다 약 11 GB 덤프를 %TEMP%\wsl-crashes 에 남긴다
        # (core_pattern 이 파이프라 ulimit -c 0 이 안 먹는다). 결과(JSON·trace·OVD·곁기록)는 이 시점에 모두 써졌으므로 여기서 바로 끝낸다.
        # 렌더 있는 장비에서는 원래 종료를 쓴다 (--no-render 일 때만).
        if getattr(cap, "no_render", False):
            import sys as _sys

            print("[capture] 결과 저장 끝 — Kit 종료를 건너뛰고 끝냄 (segfault 덤프 방지)", flush=True)
            _sys.stdout.flush()
            _sys.stderr.flush()
            os._exit(0)
        return orig_shutdown(*a, **kw)

    og.shutdown = shutdown


def main():
    argv = sys.argv[1:]
    ours, rest = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    dump_dir = None
    if "--dump-dir" in ours:
        dump_dir = ours[ours.index("--dump-dir") + 1]
    if not dump_dir:
        sys.exit("--dump-dir 가 필요하다")
    cap = Capture(dump_dir, ovd="--no-ovd" not in ours, convex="--no-convex" not in ours,
                  sidelog="--no-sidelog" not in ours)
    cap.no_render = "--no-render" in ours
    cap.dump_at_pre = int(ours[ours.index("--dump-at-pre") + 1]) if "--dump-at-pre" in ours else 0
    cap.dump_prim = ours[ours.index("--dump-prim") + 1] if "--dump-prim" in ours else "/World/scene_0/controllable__r1pro__robot"
    cap.meta["no_render"] = cap.no_render
    install(cap)
    sys.argv = [INSTRUMENTED] + rest
    runpy.run_path(INSTRUMENTED, run_name="__main__")


if __name__ == "__main__":
    main()
