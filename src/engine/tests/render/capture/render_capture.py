"""렌더 모듈 기준 자료 뜨기 -- 공식 평가기를 한 줄도 고치지 않고 돌리면서, 정해 둔 스텝마다
   (1) 장면의 보이는 메시 전부(기하·재질·조명)와 그 순간의 월드 자세, (2) 카메라 3대의 자세·내부 변수,
   (3) 공식 RTX 가 그린 RGB·depth_linear 원본, (4) 같은 상태에서 다시 그린 RGB(공식 자신의 잡음 폭) 를 뜬다.

    python render_capture.py --dump-dir <폴더> [--steps 0,1,10,...] [--noise-renders 2] \\
        -- <tools\\eval_instrumented.py 인자 그대로>

남기는 것 (<폴더>, 에셋 파생물 -> git 에 절대 올리지 않는다. src/engine/.gitignore 의 dumps/ 아래에 둘 것)
    geom.npz        고유 기하(점·삼각형·모서리 법선·모서리 UV·삼각형별 재질 칸)
    scene.json      메시 prim 목록(기하 번호·기준 prim·기준 prim 에 대한 상대 행렬·재질), 재질 입력 전부, 조명, 렌더 설정
    frame_<k>.npz   스텝 k: 기준 prim 월드 행렬(Fabric), 메시 보임 여부, 카메라 월드 행렬·투영, 공식 영상(rgb/depth), 잡음 영상
행렬은 USD 규약(행 벡터, p_world = p_local * M) 그대로 float64 로 둔다.

원리
- OmniGibson 은 물리 결과를 USD 가 아니라 Fabric 에만 쓴다(simulator.py:680 /physics/updateToUsd=False). RTX 도 Fabric 을 그린다.
  그래서 움직이는 것의 월드 행렬은 og.sim.fabric_hierarchy.get_world_xform 으로 읽는다(usd_utils.py:2155 와 같은 경로).
- 메시마다 "기준 prim" = 자기 또는 가장 가까운 강체(RigidBodyAPI) 조상 (없으면 인스턴스 프록시가 아닌 가장 가까운 조상).
  기준 prim 아래 상대 행렬은 USD 에서 한 번 계산(강체 아래 시각 메시는 강체와 같이 움직임), 스텝마다 기준 prim 의 Fabric 행렬만 뜬다.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import runpy
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INSTRUMENTED = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "tools", "eval_instrumented.py"))
DEFAULT_STEPS = [0, 1, 2, 10, 50, 100, 101, 150, 200, 300, 400, 500]  # eval_instrumented IMAGE_STEPS 와 같게(+224 기록과 맞춤)


def _np(v):
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _mat(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], np.float64)


def _jsonable(v):
    from pxr import Gf, Sdf

    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Sdf.AssetPath):
        return {"asset": v.path, "resolved": v.resolvedPath}
    if isinstance(v, (Gf.Vec2f, Gf.Vec3f, Gf.Vec4f, Gf.Vec2d, Gf.Vec3d, Gf.Vec4d, Gf.Vec2i, Gf.Vec3i, Gf.Vec4i)):
        return [float(x) for x in v]
    if isinstance(v, (Gf.Quatf, Gf.Quatd, Gf.Quath)):
        return [float(v.GetReal())] + [float(x) for x in v.GetImaginary()]
    if isinstance(v, (Gf.Matrix4d, Gf.Matrix4f, Gf.Matrix3d, Gf.Matrix3f)):
        return np.array(v).tolist()
    try:
        a = np.asarray(v)
        if a.dtype.kind in "fiub" and a.size <= 64:
            return a.tolist()
        if a.dtype.kind in "fiub":
            return {"array_len": int(a.size)}
    except Exception:
        pass
    return repr(v)


class RenderCapture:
    def __init__(self, dump_dir, steps, noise_renders):
        self.dump_dir = os.path.abspath(dump_dir)
        os.makedirs(self.dump_dir, exist_ok=True)
        self.steps = set(steps)
        self.noise_renders = noise_renders
        self.k = 0
        self.scene = None  # 첫 기록 때 채움
        self.meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": sorted(steps), "frames": []}

    # ------------------------------------------------------------------ 장면 한 번
    def dump_scene(self, ev):
        import carb
        import omni.usd
        from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        t0 = time.perf_counter()
        stage = omni.usd.get_context().get_stage()
        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

        geoms, geom_key = [], {}
        meshes, anchors, anchor_idx = [], [], {}
        materials, mat_idx = [], {}
        skipped = {"invisible": 0, "purpose": 0, "empty": 0}

        def material_id(mat):
            if not mat:
                return -1
            p = str(mat.GetPath())
            if p in mat_idx:
                return mat_idx[p]
            info = {"path": p, "shaders": []}
            for sh_prim in Usd.PrimRange(mat.GetPrim(), pred):
                if not sh_prim.IsA(UsdShade.Shader):
                    continue
                sh = UsdShade.Shader(sh_prim)
                d = {"path": str(sh_prim.GetPath()), "id": _jsonable(sh.GetIdAttr().Get()) if sh.GetIdAttr() else None,
                     "mdl_asset": _jsonable(sh_prim.GetAttribute("info:mdl:sourceAsset").Get())
                     if sh_prim.HasAttribute("info:mdl:sourceAsset") else None,
                     "mdl_sub": _jsonable(sh_prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get())
                     if sh_prim.HasAttribute("info:mdl:sourceAsset:subIdentifier") else None,
                     "inputs": {}}
                for inp in sh.GetInputs():
                    v = None
                    try:
                        attrs = inp.GetValueProducingAttributes()
                        v = attrs[0].Get() if attrs else inp.Get()
                    except Exception:
                        v = inp.Get()
                    d["inputs"][inp.GetBaseName()] = _jsonable(v)
                info["shaders"].append(d)
            # 재질 prim 자신의 입력(인터페이스)
            info["material_inputs"] = {i.GetBaseName(): _jsonable(i.Get()) for i in UsdShade.Material(mat).GetInputs()}
            mat_idx[p] = len(materials)
            materials.append(info)
            return mat_idx[p]

        def anchor_of(prim):
            q = prim
            while q and not q.IsPseudoRoot():
                if q.HasAPI(UsdPhysics.RigidBodyAPI) and not q.IsInstanceProxy():
                    return q
                q = q.GetParent()
            q = prim
            while q and q.IsInstanceProxy():
                q = q.GetParent()
            return q

        def geometry(prim, mesh):
            pts = mesh.GetPointsAttr().Get()
            cnt = mesh.GetFaceVertexCountsAttr().Get()
            idx = mesh.GetFaceVertexIndicesAttr().Get()
            if pts is None or cnt is None or idx is None or len(pts) == 0 or len(cnt) == 0:
                return None
            pts = np.asarray(pts, np.float32).reshape(-1, 3)
            cnt = np.asarray(cnt, np.int64)
            idx = np.asarray(idx, np.int64)
            # 부채꼴 삼각분할 (Hydra 기본과 같은 방식: 면의 첫 꼭짓점 기준)
            starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            ntri = np.maximum(cnt - 2, 0)
            face_of_tri = np.repeat(np.arange(len(cnt)), ntri)
            k_in_face = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri)
            c0 = starts[face_of_tri]
            corners = np.stack([c0, c0 + k_in_face + 1, c0 + k_in_face + 2], 1)  # 면-꼭짓점(모서리) 번호
            tri = idx[corners].astype(np.int32)

            def corner_attr(values, interp, indices=None):
                if values is None:
                    return None
                v = np.asarray(values, np.float32)
                if v.ndim == 1:
                    v = v.reshape(len(v), -1)
                if indices is not None and len(indices):
                    v = v[np.asarray(indices, np.int64)]
                if interp in ("vertex", "varying"):
                    return v[tri] if len(v) == len(pts) else None
                if interp == "faceVarying":
                    return v[corners] if len(v) == len(idx) else None
                if interp == "uniform":
                    return np.repeat(v[face_of_tri][:, None, :], 3, 1) if len(v) == len(cnt) else None
                if interp == "constant":
                    return np.broadcast_to(v[0], (len(tri), 3, v.shape[1])).copy()
                return None

            pv = UsdGeom.PrimvarsAPI(prim)
            nrm = None
            npv = pv.GetPrimvar("normals")
            if npv and npv.HasValue():
                nrm = corner_attr(npv.Get(), npv.GetInterpolation(), npv.GetIndices() if npv.IsIndexed() else None)
            elif mesh.GetNormalsAttr().HasValue():
                nrm = corner_attr(mesh.GetNormalsAttr().Get(), mesh.GetNormalsInterpolation())
            uv = None
            for name in ("st", "st0", "UVMap", "uv"):
                p = pv.GetPrimvar(name)
                if p and p.HasValue():
                    uv = corner_attr(p.Get(), p.GetInterpolation(), p.GetIndices() if p.IsIndexed() else None)
                    if uv is not None:
                        uv = uv[..., :2]
                        break
            return {"points": pts, "tri": tri, "face_of_tri": face_of_tri.astype(np.int32),
                    "normals": nrm, "uv": uv, "nface": len(cnt)}

        for prim in Usd.PrimRange.Stage(stage, pred):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            img = UsdGeom.Imageable(prim)
            if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                skipped["purpose"] += 1
                continue
            vis = img.ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible
            mesh = UsdGeom.Mesh(prim)
            proto = prim.GetPrimInPrototype().GetPath() if prim.IsInstanceProxy() else None
            g = geometry(prim, mesh)
            if g is None:
                skipped["empty"] += 1
                continue
            h = hashlib.blake2b(g["points"].tobytes() + g["tri"].tobytes() +
                                (g["normals"].tobytes() if g["normals"] is not None else b"") +
                                (g["uv"].tobytes() if g["uv"] is not None else b""), digest_size=16).hexdigest()
            if h not in geom_key:
                geom_key[h] = len(geoms)
                geoms.append(g)
            gid = geom_key[h]
            # 재질: 메시 전체 + GeomSubset(면 묶음별)
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            subsets = []
            for s in UsdGeom.Subset.GetAllGeomSubsets(mesh):
                if s.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                    continue
                sm, _ = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()
                subsets.append({"faces": [int(x) for x in (s.GetIndicesAttr().Get() or [])], "material": material_id(sm),
                                "family": s.GetFamilyNameAttr().Get()})
            anc = anchor_of(prim)
            ap = str(anc.GetPath())
            if ap not in anchor_idx:
                anchor_idx[ap] = len(anchors)
                anchors.append({"path": ap, "rigid": bool(anc.HasAPI(UsdPhysics.RigidBodyAPI)),
                                "usd_world": _mat(xc.GetLocalToWorldTransform(anc)).tolist()})
            w_mesh = _mat(xc.GetLocalToWorldTransform(prim))
            w_anc = _mat(xc.GetLocalToWorldTransform(anc))
            rel = w_mesh @ np.linalg.inv(w_anc)  # 행 벡터 규약: M_mesh = rel * M_anchor
            meshes.append({"path": str(prim.GetPath()), "proto": str(proto) if proto else None, "geom": gid,
                           "anchor": anchor_idx[ap], "rel": rel.tolist(), "visible0": bool(vis),
                           "material": material_id(mat), "subsets": subsets,
                           "double_sided": bool(mesh.GetDoubleSidedAttr().Get()),
                           "orientation": str(mesh.GetOrientationAttr().Get()),
                           "subdiv": str(mesh.GetSubdivisionSchemeAttr().Get())})
        # 조명
        lights = []
        for prim in Usd.PrimRange.Stage(stage, pred):
            if not (prim.HasAPI(UsdLux.LightAPI) or "Light" in prim.GetTypeName()):
                continue
            attrs = {}
            for a in prim.GetAttributes():
                n = a.GetName()
                if n.startswith(("inputs:", "shaping:", "treatAsPoint", "visibility", "light:", "intensity", "exposure",
                                 "color", "radius", "width", "height", "length", "angle", "texture", "enableColorTemperature",
                                 "colorTemperature", "normalize", "diffuse", "specular", "visibleInPrimaryRay")):
                    try:
                        attrs[n] = _jsonable(a.Get())
                    except Exception:
                        pass
            anc = prim
            while anc and anc.IsInstanceProxy():
                anc = anc.GetParent()
            lights.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(), "attrs": attrs,
                           "anchor_path": str(anc.GetPath()),
                           "rel": (_mat(xc.GetLocalToWorldTransform(prim)) @ np.linalg.inv(_mat(xc.GetLocalToWorldTransform(anc)))).tolist(),
                           "visible0": UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
                           if prim.IsA(UsdGeom.Imageable) else True})
        cs = carb.settings.get_settings()
        rtx = {}
        for key in ("/rtx/rendermode", "/rtx/post", "/rtx/sceneDb", "/rtx/directLighting", "/rtx/indirectDiffuse",
                    "/rtx/ambientOcclusion", "/rtx/reflections", "/rtx/domeLight", "/rtx/pathtracing", "/rtx/raytracing",
                    "/rtx/rtpt", "/rtx/materialDb", "/rtx/hydra", "/rtx/shadows", "/rtx/translucency", "/rtx/newDenoiser",
                    "/rtx/rt2", "/rtx/lightspeed", "/rtx/viewTile", "/rtx/sceneDB"):
            try:
                rtx[key] = cs.get(key)
            except Exception as e:
                rtx[key] = repr(e)
        # 기하 저장 (모서리 단위 배열을 이어 붙임)
        G = len(geoms)
        np.savez_compressed(
            os.path.join(self.dump_dir, "geom.npz"),
            n_points=np.array([len(g["points"]) for g in geoms], np.int64),
            n_tri=np.array([len(g["tri"]) for g in geoms], np.int64),
            n_face=np.array([g["nface"] for g in geoms], np.int64),
            has_normals=np.array([g["normals"] is not None for g in geoms]),
            has_uv=np.array([g["uv"] is not None for g in geoms]),
            points=np.concatenate([g["points"] for g in geoms]) if G else np.zeros((0, 3), np.float32),
            tri=np.concatenate([g["tri"] for g in geoms]) if G else np.zeros((0, 3), np.int32),
            face_of_tri=np.concatenate([g["face_of_tri"] for g in geoms]) if G else np.zeros(0, np.int32),
            normals=np.concatenate([g["normals"] for g in geoms if g["normals"] is not None])
            if any(g["normals"] is not None for g in geoms) else np.zeros((0, 3, 3), np.float32),
            uv=np.concatenate([g["uv"] for g in geoms if g["uv"] is not None])
            if any(g["uv"] is not None for g in geoms) else np.zeros((0, 3, 2), np.float32))
        self.scene = {"meshes": meshes, "anchors": anchors, "materials": materials, "lights": lights, "rtx": rtx,
                      "skipped": skipped, "n_geom": G}
        with open(os.path.join(self.dump_dir, "scene.json"), "w", encoding="utf-8") as f:
            json.dump(self.scene, f, ensure_ascii=False, default=str)
        ntri = int(sum(len(g["tri"]) for g in geoms))
        ntri_inst = int(sum(len(geoms[m["geom"]]["tri"]) for m in meshes))
        self.meta["scene"] = {"meshes": len(meshes), "geoms": G, "tri_unique": ntri, "tri_instanced": ntri_inst,
                              "anchors": len(anchors), "materials": len(materials), "lights": len(lights),
                              "skipped": skipped, "sec": round(time.perf_counter() - t0, 2)}
        print(f"[render_capture] 장면: {self.meta['scene']}", flush=True)

    # ------------------------------------------------------------------ 스텝마다
    def world(self, path):
        import omnigibson as og
        import usdrt

        try:
            return _mat(og.sim.fabric_hierarchy.get_world_xform(usdrt.Sdf.Path(path))), True
        except Exception:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            return _mat(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(stage.GetPrimAtPath(path))), False

    def dump_frame(self, ev, k):
        import omnigibson as og
        import omni.usd
        from pxr import Usd, UsdGeom

        t0 = time.perf_counter()
        if self.scene is None:
            self.dump_scene(ev)
        og.sim.fabric_hierarchy.update_world_xforms()
        st = ev.instance_eval_states[0]
        robot = st.env_accessor.robot
        out = {}
        aw, fab = [], []
        for a in self.scene["anchors"]:
            m, ok = self.world(a["path"])
            aw.append(m)
            fab.append(ok)
        out["anchor_world"] = np.stack(aw)
        out["anchor_from_fabric"] = np.array(fab)
        stage = omni.usd.get_context().get_stage()
        vis = []
        for m in self.scene["meshes"]:
            p = stage.GetPrimAtPath(m["path"])
            vis.append(bool(p) and UsdGeom.Imageable(p).ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible)
        out["mesh_visible"] = np.array(vis)
        lw = []
        for L in self.scene["lights"]:
            m, _ = self.world(L["anchor_path"])
            lw.append(np.array(L["rel"]) @ m)
        out["light_world"] = np.stack(lw) if lw else np.zeros((0, 4, 4))
        # 메시 prim 직접 Fabric 행렬 (프록시 아닌 것만, 검산용: rel*anchor 와 같아야 함)
        chk = []
        for i, m in enumerate(self.scene["meshes"][:: max(1, len(self.scene["meshes"]) // 200)]):
            if m["proto"]:
                continue
            w, ok = self.world(m["path"])
            if ok:
                pred = np.array(m["rel"]) @ out["anchor_world"][m["anchor"]]
                chk.append(float(np.abs(w - pred).max()))
        out["check_rel_maxdiff"] = np.array(chk)
        # 카메라
        cams = {}
        for role, cam_name in ev.robot_camera_names.items():
            sname = cam_name.split("::")[1]
            s = robot.sensors[sname]
            w, ok = self.world(s.prim_path)
            prim = stage.GetPrimAtPath(s.prim_path)
            params = {}
            for an in ("focalLength", "horizontalAperture", "verticalAperture", "horizontalApertureOffset",
                       "verticalApertureOffset", "clippingRange", "projection", "fStop", "focusDistance"):
                a = prim.GetAttribute(an)
                if a:
                    params[an] = _jsonable(a.Get())
            params["image_height"] = int(s.image_height)
            params["image_width"] = int(s.image_width)
            pos, quat = s.get_position_orientation()
            cams[role] = {"sensor": sname, "prim": s.prim_path, "world": w.tolist(), "from_fabric": ok, "params": params,
                          "og_pos": _np(pos).tolist(), "og_quat_xyzw": _np(quat).tolist()}
            for mod in ("rgb", "depth_linear", "depth"):
                key = f"{cam_name}::{mod}"
                if st.obs is not None and key in st.obs:
                    out[f"img::{role}::{mod}"] = _np(st.obs[key])
        rp, rq = robot.get_position_orientation()
        out["robot_pose"] = np.concatenate([_np(rp), _np(rq)]).astype(np.float64)
        # 같은 상태에서 다시 그리기 -> 공식 자신의 잡음 폭 (물리는 안 움직인다: 평가기도 조명 맞출 때 og.sim.render() 3 번)
        for r in range(self.noise_renders):
            og.sim.render()
            for role, cam_name in ev.robot_camera_names.items():
                s = robot.sensors[cam_name.split("::")[1]]
                try:
                    o, _ = s.get_obs()
                    for mod in ("rgb", "depth_linear"):
                        if mod in o:
                            out[f"noise{r}::{role}::{mod}"] = _np(o[mod])
                except Exception as e:
                    print(f"[render_capture] 잡음 렌더 실패 {role}: {e!r}", flush=True)
        # 투영 행렬 원본(RTX 가 쓴 것) -- 주석기를 새로 만들며 렌더를 부르므로 마지막에
        if k == min(self.steps):
            for role, cam_name in ev.robot_camera_names.items():
                s = robot.sensors[cam_name.split("::")[1]]
                try:
                    cp = s.camera_parameters
                    cams[role]["camera_parameters"] = {kk: _jsonable(np.asarray(v).tolist() if hasattr(v, "__len__")
                                                                     and not isinstance(v, str) else v)
                                                       for kk, v in cp.items()}
                except Exception as e:
                    cams[role]["camera_parameters_error"] = repr(e)
        out["cams_json"] = np.array(json.dumps(cams))
        np.savez_compressed(os.path.join(self.dump_dir, f"frame_{k:04d}.npz"), **out)
        info = {"k": k, "sec": round(time.perf_counter() - t0, 2), "fabric_anchors": int(np.sum(fab)),
                "anchors": len(fab), "check_rel_maxdiff": float(np.max(chk)) if chk else None,
                "images": sorted(x for x in out if x.startswith("img::"))}
        self.meta["frames"].append(info)
        print(f"[render_capture] 스텝 {k}: {info}", flush=True)

    def finish(self):
        self.meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.dump_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=1, default=str)


def install(cap: RenderCapture):
    import tempfile

    import omnigibson as og
    from omnigibson.eval import evaluator as E

    # og.tempdir(%TEMP%\tmpXXXX, omnigibson/__init__.py:70)가 09-29 23:57 첫 시도에서 장면 로딩 중 사라져
    # (다른 작업의 %TEMP%\tmp* 정리로 추정) LightObject._build_usd 가 FileNotFoundError 로 죽었다.
    # -> %TEMP% 밖 전용 폴더로 옮긴다. 저장소 밖이라 여기 풀린 USD 가 git 에 들어갈 일도 없다. og.shutdown 의 cleanup 이 지운다.
    base = os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), "og_render_capture")
    os.makedirs(base, exist_ok=True)
    og.tempdir = tempfile.mkdtemp(prefix="og_", dir=base)
    print(f"[render_capture] og.tempdir -> {og.tempdir}", flush=True)

    Ev = E.BatchedEvaluator
    orig_apply = Ev._apply_actions

    @functools.wraps(orig_apply)
    def apply_actions(self, *a, **kw):
        r = orig_apply(self, *a, **kw)
        k = cap.k
        cap.k += 1
        if k in cap.steps:
            try:
                cap.dump_frame(self, k)
            except Exception as e:  # 기록 실패가 평가를 멈추지 않게
                import traceback

                traceback.print_exc()
                print(f"[render_capture] 스텝 {k} 기록 실패: {e!r}", flush=True)
        return r

    Ev._apply_actions = apply_actions

    orig_shutdown = og.shutdown

    @functools.wraps(orig_shutdown)
    def shutdown(*a, **kw):
        try:
            cap.finish()
        except Exception as e:
            print(f"[render_capture] 마무리 실패: {e!r}", flush=True)
        return orig_shutdown(*a, **kw)

    og.shutdown = shutdown


def main():
    argv = sys.argv[1:]
    ours, rest = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    dump_dir = ours[ours.index("--dump-dir") + 1] if "--dump-dir" in ours else None
    if not dump_dir:
        sys.exit("--dump-dir 가 필요하다")
    steps = DEFAULT_STEPS
    if "--steps" in ours:
        steps = [int(x) for x in ours[ours.index("--steps") + 1].split(",") if x.strip()]
    noise = int(ours[ours.index("--noise-renders") + 1]) if "--noise-renders" in ours else 2
    cap = RenderCapture(dump_dir, steps, noise)
    install(cap)
    sys.argv = [INSTRUMENTED] + rest
    runpy.run_path(INSTRUMENTED, run_name="__main__")


if __name__ == "__main__":
    main()
