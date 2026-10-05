"""렌더 기준 자료(render_capture.py 가 뜬 덤프) -> 렌더러 입력 묶음(.npy 여러 장)과 공식 영상.

    python export_render_scene.py <덤프 폴더> [--out <폴더>] [--tex-max 512] [--assets <behavior-1k-assets>]

출력(<덤프>/export, 에셋 파생물 -> git 에 절대 올리지 않는다). 형식은 docs/엔진_자체구현.md 14.3 절 표가 기준이다.
  정적  geom_*.npy tri_*.npy inst_*.npy slot_mat.npy mat_f.npy mat_i.npy tex_info.npy texels.npy light_*.npy meta.json
  프레임 frame_<k>/anchor_world.npy inst_visible.npy light_world.npy cam_<역할>.npy
        frame_<k>/off_rgb_<역할>.npy off_depth_<역할>.npy noise<r>_rgb_<역할>.npy noise<r>_depth_<역할>.npy (공식 RTX)
행렬(Aff) = float32 12 개, 행 우선 3x4, p' = A p + t. USD 행 벡터 행렬 M 에서 A[r][c] = M[c][r], t[r] = M[3][r].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DEFAULT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "..", "BEHAVIOR-1K", "datasets", "behavior-1k-assets"))
ROBOT_ASSETS_DEFAULT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "..", "BEHAVIOR-1K", "datasets", "omnigibson-robot-assets"))

# mat_f 열 (float32). 바꾸면 docs 14.3 과 렌더러 불러오기(core/render 쪽)를 같이 바꾼다.
MAT_F_COLS = ["albedo_r", "albedo_g", "albedo_b", "tint_r", "tint_g", "tint_b", "uv_scale_u", "uv_scale_v", "uv_off_u", "uv_off_v",
              "uv_rot_rad", "roughness", "metallic", "opacity", "emis_r", "emis_g", "emis_b", "albedo_add", "albedo_brightness",
              "albedo_desaturation", "rough_tex_influence", "metal_tex_influence", "opacity_threshold", "specular_level"]
# mat_i 열 (int32)
MAT_I_COLS = ["tex_albedo", "tex_rough", "tex_metal", "tex_normal", "tex_opacity", "tex_emissive", "flags", "kind"]
MAT_FLAG = {"enable_opacity": 1, "opacity_texture": 2, "emission": 4, "skip": 8, "unknown_shader": 16}
MAT_KIND = {"OmniPBR": 0, "OmniSurface": 1, "OmniGlass": 2, "UsdPreviewSurface": 3, "other": 9}
LIGHT_TYPES = {"SphereLight": 0, "RectLight": 1, "DiskLight": 2, "DistantLight": 3, "DomeLight": 4, "CylinderLight": 5}
# light_f 열
LIGHT_F_COLS = ["rad_r", "rad_g", "rad_b", "radius", "width", "height", "length", "angle_deg", "cone_angle_deg", "cone_softness",
                "treat_as_point", "normalize", "intensity", "exposure", "diffuse", "specular"]


def aff_from_usd(M):
    M = np.asarray(M, np.float64)
    A = np.zeros(12, np.float64)
    for r in range(3):
        for c in range(3):
            A[r * 4 + c] = M[c][r]
        A[r * 4 + 3] = M[3][r]
    return A


def blackbody_rgb(t_kelvin):
    """색온도 -> 선형 RGB (최대 성분 1). Tanner Helland 근사를 sRGB 로 보고 선형화한다(RTX 식은 닫힘 -> 근사, 통계 비교용)."""
    t = t_kelvin / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
        b = 255.0
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin / max(lin.max(), 1e-9)


class TexStore:
    def __init__(self, assets, robot_assets, tex_max):
        self.tex_max = tex_max
        self.index = {}  # 파일 이름 -> 전체 경로 (모델 id 폴더에서)
        self.model_dir = {}
        self.assets = assets
        objs = os.path.join(assets, "objects")
        if os.path.isdir(objs):
            for cat in os.scandir(objs):
                if not cat.is_dir():
                    continue
                for mdl in os.scandir(cat.path):
                    if mdl.is_dir():
                        self.model_dir[mdl.name] = mdl.path
        self.robot_assets = robot_assets
        self.info, self.chunks, self.key = [], [], {}
        self.n_texels = 0
        self.missing = []

    def find(self, asset):
        if asset is None:
            return None
        if isinstance(asset, dict):
            res, path = asset.get("resolved") or "", asset.get("asset") or ""
        else:
            res, path = "", str(asset)
        if res and os.path.isfile(res):
            return res
        base = os.path.basename(path.replace("\\", "/"))
        if not base:
            return None
        mid = base.split("__")[0]
        d = self.model_dir.get(mid)
        if d:
            for sub in ("material", ""):
                p = os.path.join(d, sub, base)
                if os.path.isfile(p):
                    return p
        if base in self.index:
            return self.index[base]
        # 로봇 에셋 등: 한 번 훑어서 이름표 만든다
        if not getattr(self, "_robot_indexed", False):
            self._robot_indexed = True
            for root, _, files in os.walk(self.robot_assets):
                for fn in files:
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".tga")):
                        self.index.setdefault(fn, os.path.join(root, fn))
        return self.index.get(base)

    def add(self, asset, srgb):
        p = self.find(asset)
        if p is None:
            if asset:
                self.missing.append(asset if isinstance(asset, str) else asset.get("asset"))
            return -1
        k = (p, srgb)
        if k in self.key:
            return self.key[k]
        from PIL import Image

        try:
            im = Image.open(p)
            im.load()
        except Exception as e:
            self.missing.append(f"{p}: {e!r}")
            return -1
        if im.mode in ("I;16", "I;16B", "I", "F"):
            a = np.asarray(im, np.float64)
            a = (np.clip(a / max(a.max(), 1), 0, 1) * 255).astype(np.uint8)
            im = Image.fromarray(a)
        im = im.convert("RGBA")
        w, h = im.size
        s = max(w, h)
        if s > self.tex_max:
            f = self.tex_max / s
            im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BOX)
        a = np.ascontiguousarray(np.asarray(im, np.uint8).reshape(-1, 4))
        idx = len(self.info)
        self.info.append([im.size[0], im.size[1], self.n_texels, 1 if srgb else 0])
        self.chunks.append(a)
        self.n_texels += len(a)
        self.key[k] = idx
        return idx


def material_row(m, tex):
    """재질 prim 의 셰이더 입력 -> (mat_f, mat_i, 설명). OmniPBR 기본값은 OmniPBR.mdl 기본값(작성 안 된 입력)."""
    f = dict(albedo_r=0.2, albedo_g=0.2, albedo_b=0.2, tint_r=1.0, tint_g=1.0, tint_b=1.0, uv_scale_u=1.0, uv_scale_v=1.0,
             uv_off_u=0.0, uv_off_v=0.0, uv_rot_rad=0.0, roughness=0.5, metallic=0.0, opacity=1.0, emis_r=0.0, emis_g=0.0,
             emis_b=0.0, albedo_add=0.0, albedo_brightness=1.0, albedo_desaturation=0.0, rough_tex_influence=1.0,
             metal_tex_influence=1.0, opacity_threshold=0.0, specular_level=0.5)
    i = dict(tex_albedo=-1, tex_rough=-1, tex_metal=-1, tex_normal=-1, tex_opacity=-1, tex_emissive=-1, flags=0, kind=MAT_KIND["other"])
    shaders = m.get("shaders", [])
    sh = None
    for s in shaders:
        if s.get("mdl_sub") or s.get("mdl_asset") or s.get("id"):
            sh = s
            break
    if sh is None:
        i["flags"] |= MAT_FLAG["unknown_shader"]
        return f, i, "none"
    sub = sh.get("mdl_sub") or ""
    asset = sh.get("mdl_asset") or ""
    asset = asset.get("asset") if isinstance(asset, dict) else str(asset)
    name = sub or os.path.splitext(os.path.basename(asset))[0] or str(sh.get("id"))
    inp = dict(sh.get("inputs", {}))
    inp.update({k: v for k, v in m.get("material_inputs", {}).items() if k not in inp})

    def g(k, d=None):
        v = inp.get(k)
        return d if v is None else v

    def col(k, d):
        v = g(k)
        return list(v)[:3] if isinstance(v, (list, tuple)) and len(v) >= 3 else d

    if "OmniGlass" in name or "Glass" in name:
        i["kind"] = MAT_KIND["OmniGlass"]
        i["flags"] |= MAT_FLAG["skip"]
        c = col("glass_color", [1, 1, 1])
        f.update(albedo_r=c[0], albedo_g=c[1], albedo_b=c[2], roughness=float(g("frosting_roughness", 0.0)), opacity=0.0)
        return f, i, name
    if "OmniSurface" in name:
        i["kind"] = MAT_KIND["OmniSurface"]
        c = col("diffuse_reflection_color", [0.8, 0.8, 0.8])
        w = float(g("diffuse_reflection_weight", 0.8))
        f.update(albedo_r=c[0] * w, albedo_g=c[1] * w, albedo_b=c[2] * w,
                 roughness=float(g("specular_reflection_roughness", 0.2)), metallic=float(g("metalness", 0.0)))
        i["tex_albedo"] = tex.add(g("diffuse_reflection_color_image"), True)
        if bool(g("enable_specular_transmission", False)) and float(g("specular_transmission_weight", 0.0)) > 0.5:
            i["flags"] |= MAT_FLAG["skip"]
        if float(g("emission_weight", 0.0)) > 0:
            e = col("emission_color", [1, 1, 1])
            s = float(g("emission_intensity", 1.0)) * float(g("emission_weight", 0.0))
            f.update(emis_r=e[0] * s, emis_g=e[1] * s, emis_b=e[2] * s)
            i["flags"] |= MAT_FLAG["emission"]
        return f, i, name
    if "UsdPreviewSurface" in str(sh.get("id")):
        i["kind"] = MAT_KIND["UsdPreviewSurface"]
        c = col("diffuseColor", [0.18, 0.18, 0.18])
        f.update(albedo_r=c[0], albedo_g=c[1], albedo_b=c[2], roughness=float(g("roughness", 0.5)), metallic=float(g("metallic", 0.0)),
                 opacity=float(g("opacity", 1.0)))
        return f, i, name
    # OmniPBR (및 변형)
    i["kind"] = MAT_KIND["OmniPBR"] if "OmniPBR" in name else MAT_KIND["other"]
    c = col("diffuse_color_constant", [0.2, 0.2, 0.2])
    t = col("diffuse_tint", [1, 1, 1])
    sc = g("texture_scale", [1, 1])
    tr = g("texture_translate", [0, 0])
    f.update(albedo_r=c[0], albedo_g=c[1], albedo_b=c[2], tint_r=t[0], tint_g=t[1], tint_b=t[2],
             uv_scale_u=float(sc[0]), uv_scale_v=float(sc[1]), uv_off_u=float(tr[0]), uv_off_v=float(tr[1]),
             uv_rot_rad=math.radians(float(g("texture_rotate", 0.0))),
             roughness=float(g("reflection_roughness_constant", 0.5)), metallic=float(g("metallic_constant", 0.0)),
             albedo_add=float(g("albedo_add", 0.0)), albedo_brightness=float(g("albedo_brightness", 1.0)),
             albedo_desaturation=float(g("albedo_desaturation", 0.0)),
             rough_tex_influence=float(g("reflection_roughness_texture_influence", 1.0)),
             metal_tex_influence=float(g("metallic_texture_influence", 1.0)),
             specular_level=float(g("specular_level", 0.5)))
    i["tex_albedo"] = tex.add(g("diffuse_texture"), True)
    i["tex_rough"] = tex.add(g("reflectionroughness_texture"), False)
    i["tex_metal"] = tex.add(g("metallic_texture"), False)
    i["tex_normal"] = tex.add(g("normalmap_texture"), False)
    if bool(g("enable_opacity", False)):
        i["flags"] |= MAT_FLAG["enable_opacity"]
        f["opacity"] = float(g("opacity_constant", 1.0))
        f["opacity_threshold"] = float(g("opacity_threshold", 0.0))
        if g("opacity_texture") is not None:
            i["tex_opacity"] = tex.add(g("opacity_texture"), False)
            i["flags"] |= MAT_FLAG["opacity_texture"]
    if bool(g("enable_emission", False)):
        e = col("emissive_color", [1, 0.1, 0.1])
        s = float(g("emissive_intensity", 40.0))
        f.update(emis_r=e[0] * s, emis_g=e[1] * s, emis_b=e[2] * s)
        i["tex_emissive"] = tex.add(g("emissive_color_texture"), True)
        i["flags"] |= MAT_FLAG["emission"]
    return f, i, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex-max", type=int, default=512)
    ap.add_argument("--assets", default=ASSETS_DEFAULT)
    ap.add_argument("--robot-assets", default=ROBOT_ASSETS_DEFAULT)
    ap.add_argument("--robot-name", default="robot", help="로봇 prim 이름 (r1pro_robot.yaml = robot, 공식 r1pro.yaml = robot_r1)")
    ap.add_argument("--ref224", default="", help="쉼표로 가른 결과 폴더들: 같은 행동열 재생의 224 공식 영상(trace_images.npz)도 프레임에 붙인다")
    a = ap.parse_args()
    out = a.out or os.path.join(a.dump, "export")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(a.dump, "scene.json"), encoding="utf-8") as fh:
        S = json.load(fh)
    G = np.load(os.path.join(a.dump, "geom.npz"))
    n_pts, n_tri, n_face = G["n_points"], G["n_tri"], G["n_face"]
    has_n, has_uv = G["has_normals"], G["has_uv"]
    pt0 = np.concatenate([[0], np.cumsum(n_pts)[:-1]])
    tr0 = np.concatenate([[0], np.cumsum(n_tri)[:-1]])
    n_idx = np.cumsum(has_n) - 1
    uv_idx = np.cumsum(has_uv) - 1
    nrm_tr0 = np.concatenate([[0], np.cumsum(np.where(has_n, n_tri, 0))[:-1]])
    uv_tr0 = np.concatenate([[0], np.cumsum(np.where(has_uv, n_tri, 0))[:-1]])

    # ---- 재질·텍스처
    tex = TexStore(a.assets, a.robot_assets, a.tex_max)
    mat_f, mat_i, mat_names = [], [], []
    for m in S["materials"]:
        f, i, name = material_row(m, tex)
        mat_f.append([f[c] for c in MAT_F_COLS])
        mat_i.append([i[c] for c in MAT_I_COLS])
        mat_names.append(name)
    default_mat = len(mat_f)  # 재질 없는 메시용 회색
    f, i, _ = material_row({"shaders": [{"mdl_sub": "OmniPBR", "inputs": {}}]}, tex)
    mat_f.append([f[c] for c in MAT_F_COLS])
    mat_i.append([i[c] for c in MAT_I_COLS])
    mat_names.append("default")

    # ---- 기하 (재질 칸 나눔까지 같은 것끼리 한 번만)
    anchors = S["anchors"]
    geo_key, geo_list = {}, []  # (원 기하 번호, 칸 해시) -> 내보낸 번호
    tri_pos, tri_nrm, tri_uv, tri_slot, geom_ntri, geom_flags = [], [], [], [], [], []
    inst_geom, inst_anchor, inst_slot_base, inst_flags, inst_rel, slot_mat, inst_paths = [], [], [], [], [], [], []
    for m in S["meshes"]:
        g = m["geom"]
        nt = int(n_tri[g])
        fot = G["face_of_tri"][tr0[g]:tr0[g] + nt]
        slot = np.zeros(nt, np.int32)
        mats = [m["material"] if m["material"] >= 0 else default_mat]
        if m["subsets"]:
            face_slot = np.zeros(int(n_face[g]), np.int32)
            for k, s in enumerate(m["subsets"]):
                fl = np.asarray(s["faces"], np.int64)
                fl = fl[(fl >= 0) & (fl < len(face_slot))]
                face_slot[fl] = k + 1
                mats.append(s["material"] if s["material"] >= 0 else mats[0])
            slot = face_slot[fot]
        key = (g, hashlib.blake2b(slot.tobytes(), digest_size=8).hexdigest())
        if key not in geo_key:
            geo_key[key] = len(geom_ntri)
            tri = G["tri"][tr0[g]:tr0[g] + nt].astype(np.int64)
            P = G["points"][pt0[g]:pt0[g] + n_pts[g]]
            tri_pos.append(P[tri].reshape(nt, 9).astype(np.float32))
            fl = 0
            if has_n[g]:
                tri_nrm.append(G["normals"][nrm_tr0[g]:nrm_tr0[g] + nt].reshape(nt, 9).astype(np.float32))
                fl |= 1
            else:
                tri_nrm.append(np.zeros((nt, 9), np.float32))
            if has_uv[g]:
                tri_uv.append(G["uv"][uv_tr0[g]:uv_tr0[g] + nt].reshape(nt, 6).astype(np.float32))
                fl |= 2
            else:
                tri_uv.append(np.zeros((nt, 6), np.float32))
            tri_slot.append(slot.astype(np.int32))
            geom_ntri.append(nt)
            geom_flags.append(fl)
        inst_geom.append(geo_key[key])
        inst_anchor.append(m["anchor"])
        inst_slot_base.append(len(slot_mat))
        slot_mat.extend(mats)
        flg = 1 if m.get("double_sided") else 0
        segs = anchors[m["anchor"]]["path"].split("/")
        if any(x == a.robot_name or x.endswith("__" + a.robot_name) for x in segs):  # 예: controllable__r1pro__robot
            flg |= 2
        if str(m.get("orientation")) == "leftHanded":
            flg |= 4
        inst_flags.append(flg)
        inst_rel.append(aff_from_usd(m["rel"]))
        inst_paths.append(m["path"])

    def save(name, arr):
        np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(arr))

    save("geom_ntri", np.array(geom_ntri, np.int32))
    save("geom_flags", np.array(geom_flags, np.int32))
    save("tri_pos", np.concatenate(tri_pos) if tri_pos else np.zeros((0, 9), np.float32))
    save("tri_nrm", np.concatenate(tri_nrm) if tri_nrm else np.zeros((0, 9), np.float32))
    save("tri_uv", np.concatenate(tri_uv) if tri_uv else np.zeros((0, 6), np.float32))
    save("tri_slot", np.concatenate(tri_slot) if tri_slot else np.zeros(0, np.int32))
    save("inst_geom", np.array(inst_geom, np.int32))
    save("inst_anchor", np.array(inst_anchor, np.int32))
    save("inst_slot_base", np.array(inst_slot_base, np.int32))
    save("inst_flags", np.array(inst_flags, np.int32))
    save("inst_rel", np.array(inst_rel, np.float64).astype(np.float32).reshape(-1, 12))
    save("slot_mat", np.array(slot_mat, np.int32))
    save("mat_f", np.array(mat_f, np.float32).reshape(-1, len(MAT_F_COLS)))
    save("mat_i", np.array(mat_i, np.int32).reshape(-1, len(MAT_I_COLS)))
    save("tex_info", np.array(tex.info, np.int64).reshape(-1, 4))
    save("texels", np.concatenate(tex.chunks) if tex.chunks else np.zeros((0, 4), np.uint8))

    # ---- 조명 (anchor = -1: 첫 프레임 월드 행렬을 rel 로. 프레임마다 light_world 도 따로 낸다)
    frames = sorted(fn for fn in os.listdir(a.dump) if fn.startswith("frame_") and fn.endswith(".npz"))
    F0 = np.load(os.path.join(a.dump, frames[0])) if frames else None
    light_i, light_f, light_rel, light_desc = [], [], [], []
    for li, L in enumerate(S["lights"]):
        t = LIGHT_TYPES.get(L["type"], -1)
        at = L["attrs"]

        def ga(*names, d=None):
            for n in names:
                if n in at and at[n] is not None:
                    return at[n]
            return d

        inten = float(ga("inputs:intensity", "intensity", d=1.0))
        expo = float(ga("inputs:exposure", "exposure", d=0.0))
        colr = np.array(ga("inputs:color", "color", d=[1, 1, 1]), np.float64)[:3]
        if bool(ga("inputs:enableColorTemperature", "enableColorTemperature", d=False)):
            colr = colr * blackbody_rgb(float(ga("inputs:colorTemperature", "colorTemperature", d=6500.0)))
        rad = colr * inten * (2.0 ** expo)
        vis = 1 if L.get("visible0", True) else 0
        row = [rad[0], rad[1], rad[2], float(ga("inputs:radius", "radius", d=0.5)), float(ga("inputs:width", "width", d=1.0)),
               float(ga("inputs:height", "height", d=1.0)), float(ga("inputs:length", "length", d=1.0)),
               float(ga("inputs:angle", "angle", d=0.53)), float(ga("inputs:shaping:cone:angle", "shaping:cone:angle", d=180.0)),
               float(ga("inputs:shaping:cone:softness", "shaping:cone:softness", d=0.0)),
               1.0 if bool(ga("treatAsPoint", "inputs:treatAsPoint", d=False)) else 0.0,
               1.0 if bool(ga("inputs:normalize", "normalize", d=False)) else 0.0, inten, expo,
               float(ga("inputs:diffuse", "diffuse", d=1.0)), float(ga("inputs:specular", "specular", d=1.0))]
        light_i.append([t, -1, vis, 0])
        light_f.append(row)
        W0 = F0["light_world"][li] if F0 is not None and len(F0["light_world"]) > li else np.eye(4)
        light_rel.append(aff_from_usd(W0))
        light_desc.append({"path": L["path"], "type": L["type"],
                           "texture": ga("inputs:texture:file", "texture:file")})
    save("light_i", np.array(light_i, np.int32).reshape(-1, 4))
    save("light_f", np.array(light_f, np.float32).reshape(-1, len(LIGHT_F_COLS)))
    save("light_rel", np.array(light_rel, np.float64).astype(np.float32).reshape(-1, 12))

    # ---- 프레임
    ref224 = []
    for rd in (a.ref224.split(",") if a.ref224 else []):
        p = os.path.join(rd, "trace_images.npz")
        if os.path.isfile(p):
            tag = "".join(os.path.basename(os.path.normpath(rd)).split("_")[6:])  # eval_<과제3>_<날짜>_<시각>_<꼬리> -> 꼬리(밑줄 없앰)
            ref224.append((tag, np.load(p)))
    frame_meta = []
    for fn in frames:
        k = int(fn[6:10])
        Z = np.load(os.path.join(a.dump, fn), allow_pickle=False)
        d = os.path.join(out, f"frame_{k:04d}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "anchor_world.npy"), np.stack([aff_from_usd(M) for M in Z["anchor_world"]]).astype(np.float32))
        np.save(os.path.join(d, "inst_visible.npy"), Z["mesh_visible"].astype(np.uint8))
        np.save(os.path.join(d, "light_world.npy"),
                np.stack([aff_from_usd(M) for M in Z["light_world"]]).astype(np.float32) if len(Z["light_world"]) else np.zeros((0, 12), np.float32))
        cams = json.loads(str(Z["cams_json"]))
        fm = {"k": k, "cams": {}, "black": {}, "robot_pose": Z["robot_pose"].tolist(),
              "check_rel_maxdiff": float(Z["check_rel_maxdiff"].max()) if Z["check_rel_maxdiff"].size else None}
        for role, c in cams.items():
            A = aff_from_usd(c["world"])
            for col in range(3):  # 축 정규화 (열 = 카메라 x·y·z 축)
                v = A[[col, 4 + col, 8 + col]]
                A[[col, 4 + col, 8 + col]] = v / np.linalg.norm(v)
            p = c["params"]
            fl = float(p.get("focalLength", 24.0))
            ha = float(p.get("horizontalAperture", 20.955))
            W, H = int(p["image_width"]), int(p["image_height"])
            va_usd = float(p.get("verticalAperture", ha * H / W))
            tanx = ha / 2 / fl
            tany = tanx * H / W  # RTX: 세로 조리개는 해상도 비율로 (usd verticalAperture 무시, 정사각 픽셀) -- cameraProjection 으로 검산
            cr = p.get("clippingRange", [0.001, 1e7])
            proj = None
            cp = c.get("camera_parameters") or (frame_meta[0]["cams"].get(role, {}).get("camera_parameters") if frame_meta else None)
            if cp and "cameraProjection" in cp:
                P = np.asarray(cp["cameraProjection"], np.float64).reshape(4, 4)
                proj = P.tolist()
            np.save(os.path.join(d, f"cam_{role}.npy"),
                    np.concatenate([A, [tanx, tany, float(cr[0]), float(cr[1])]]).astype(np.float32))
            fm["cams"][role] = {"w": W, "h": H, "focal": fl, "haperture": ha, "vaperture_usd": va_usd, "tanx": tanx, "tany": tany,
                                "clip": cr, "prim": c["prim"], "camera_parameters": cp, "projection": proj,
                                "from_fabric": c.get("from_fabric")}
            for src, pre in (("img", "off"), ("noise0", "noise0"), ("noise1", "noise1"), ("noise2", "noise2")):
                for mod, tag in (("rgb", "rgb"), ("depth_linear", "depth")):
                    key = f"{src}::{role}::{mod}"
                    if key not in Z.files:
                        continue
                    im = Z[key]
                    if mod == "rgb":
                        im = np.ascontiguousarray(im[..., :3]).astype(np.uint8)
                        fm["black"][f"{pre}_{role}"] = bool(im.mean() < 2.0)
                    else:
                        im = np.ascontiguousarray(im.reshape(im.shape[0], im.shape[1])).astype(np.float32)
                    np.save(os.path.join(d, f"{pre}_{tag}_{role}.npy"), im)
        # 224 공식 영상(Default 래퍼) -- 같은 행동열 재생이라 물리 상태가 비트 같은 다른 실행의 trace_images.npz
        for tag, zi in ref224:
            for role, sub in (("left_wrist", "left_realsense"), ("right_wrist", "right_realsense"), ("head", "zed")):
                keys = [x for x in zi.files if x.startswith(f"{k}|0|") and sub in x and x.endswith("::rgb")]
                if not keys:
                    continue
                im = np.ascontiguousarray(zi[keys[0]][..., :3]).astype(np.uint8)
                np.save(os.path.join(d, f"ref224_{tag}_rgb_{role}.npy"), im)
                fm["black"][f"ref224_{tag}_{role}"] = bool(im.mean() < 2.0)
        frame_meta.append(fm)

    meta = {"source": os.path.abspath(a.dump), "tex_max": a.tex_max,
            "counts": {"geoms": len(geom_ntri), "tris": int(sum(geom_ntri)), "insts": len(inst_geom),
                       "tris_instanced": int(sum(geom_ntri[g] for g in inst_geom)), "anchors": len(anchors),
                       "mats": len(mat_f), "texs": len(tex.info), "texel_bytes": int(tex.n_texels * 4), "lights": len(light_i),
                       "robot_insts": int(sum(1 for f in inst_flags if f & 2))},
            "mat_f_cols": MAT_F_COLS, "mat_i_cols": MAT_I_COLS, "mat_flags": MAT_FLAG, "mat_kinds": MAT_KIND,
            "light_types": LIGHT_TYPES, "light_f_cols": LIGHT_F_COLS, "light_desc": light_desc,
            "mat_names": mat_names, "shader_kinds_count": {n: mat_names.count(n) for n in sorted(set(mat_names))},
            "anchor_paths": [x["path"] for x in anchors], "inst_paths": inst_paths, "robot_name": a.robot_name,
            "tex_missing": sorted(set(str(x) for x in tex.missing))[:200], "n_tex_missing": len(set(str(x) for x in tex.missing)),
            "rtx": S.get("rtx"), "frames": frame_meta,
            "inst_flags": {"double_sided": 1, "robot": 2, "left_handed": 4}}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(meta["counts"], ensure_ascii=False))
    print("재질 종류:", meta["shader_kinds_count"])
    print("텍스처 못 찾음:", meta["n_tex_missing"])
    for fm in frame_meta:
        print(f"프레임 {fm['k']}: 검은 칸 {[k for k, v in fm['black'].items() if v]}, rel 검산 {fm['check_rel_maxdiff']}")


if __name__ == "__main__":
    main()
