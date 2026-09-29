"""render_capture.py 가 뜬 공식 기록 -> 우리 렌더 파일(scene.rsc, frame_<k>.rfr). 형식은 core/render/rsc_io.h 머리말.

    python convert_scene.py <덤프 폴더> [--tex-max 1024] [--out <폴더>]

- 한 번 도는 자료 준비 도구(파이썬 접착부). 결과는 에셋 파생물 -> git 에 올리지 않는다(기본 출력 = 덤프 폴더, dumps/ 아래).
- 기하: 고유 기하 × (GeomSubset 으로 나눈 재질 칸 배치) 마다 하나. 삼각형은 지역 좌표 그대로(굽기는 C++ 가 한다).
- 인스턴스 월드 = anchor ∘ rel 을 렌더러가 float 로 계산한다(USD 행 벡터 행렬 -> 열 벡터 3x4 로 전치해 넣음).
- 조명: 조명마다 기준 prim 칸을 하나 더 만들어 프레임의 조명 월드 행렬을 그대로 넣는다(rel = 항등).
- 재질(OmniPBR 입력): albedo = diffuse_texture(있으면) × diffuse_tint, 없으면 diffuse_color_constant × diffuse_tint.
  (albedo + albedo_add) × albedo_brightness 는 렌더러가 한다. 방출 = emissive_color × emissive_intensity (enable_emission).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys

import numpy as np

LIGHT_TYPES = {"SphereLight": 0, "RectLight": 1, "DiskLight": 2, "DistantLight": 3, "DomeLight": 4, "CylinderLight": 5}


def usd_to_aff(M):
    """USD 행 벡터 4x4 (p_w = p * M) -> 열 벡터 3x4 행 우선 12 개"""
    M = np.asarray(M, np.float64)
    out = np.zeros(12, np.float64)
    for r in range(3):
        for c in range(3):
            out[r * 4 + c] = M[c][r]
        out[r * 4 + 3] = M[3][r]
    return out.astype(np.float32)


def blackbody_rgb(t):
    """색온도 -> 선형 RGB (Tanner Helland 근사, 최대값 1 로 정규화) — RTX 의 enableColorTemperature 흉내(추정)"""
    t = max(1000.0, min(40000.0, t)) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c / max(c.max(), 1e-6)


def get_in(sh, name, default=None):
    v = sh["inputs"].get(name, default)
    return default if v is None else v


class TexCache:
    def __init__(self, tex_max):
        self.tex_max = tex_max
        self.idx = {}
        self.infos = []
        self.chunks = []
        self.n = 0
        self.missing = []

    def get(self, asset):
        if not asset:
            return -1
        from tex_lookup import find_texture  # 같은 폴더: 풀린 경로가 없으면 에셋 폴더에서 모델 id 로 찾는다

        path = find_texture(asset)
        if not path:
            if asset and isinstance(asset, dict) and asset.get("asset"):
                self.missing.append(asset.get("asset"))
            return -1
        if path in self.idx:
            return self.idx[path]
        from PIL import Image

        im = Image.open(path)
        im = im.convert("RGBA")
        if max(im.size) > self.tex_max:
            s = self.tex_max / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BOX)
        a = np.asarray(im, np.uint8)
        u = (a[..., 0].astype(np.uint32) | (a[..., 1].astype(np.uint32) << 8) | (a[..., 2].astype(np.uint32) << 16)
             | (a[..., 3].astype(np.uint32) << 24)).reshape(-1)
        self.idx[path] = len(self.infos)
        self.infos.append((im.width, im.height, self.n))
        self.chunks.append(u)
        self.n += u.size
        return self.idx[path]


def material_row(m, tex: TexCache, stats):
    """재질 -> f32[24] (rsc_io.h mat_from 순서)"""
    row = np.zeros(24, np.float32)
    row[0:3] = 0.5
    row[3] = -1
    row[4:6] = 1.0
    row[11] = 1.0
    row[17] = 1.0
    if m is None:
        return row
    sh = next((s for s in m["shaders"] if s.get("mdl_asset") or s.get("id")), None)
    if sh is None:
        stats["no_shader"] += 1
        return row
    mdl = (sh.get("mdl_asset") or {}).get("asset", "") if isinstance(sh.get("mdl_asset"), dict) else str(sh.get("mdl_asset"))
    kind = os.path.basename(mdl) or str(sh.get("id"))
    stats.setdefault("kinds", {})
    stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
    tint = np.array(get_in(sh, "diffuse_tint", [1, 1, 1]), np.float32)[:3]
    if "OmniGlass" in kind:  # 유리: 렌더러가 색 광선은 지나가게 한다(깊이는 맞음). flags 2 (render B, rsc_io.h)
        row[15] = 2
        return row
    if "UsdPreviewSurface" in kind:
        base = np.array(get_in(sh, "diffuseColor", [0.18, 0.18, 0.18]), np.float32)[:3]
        row[0:3] = base
        row[9] = float(get_in(sh, "roughness", 0.5))
        row[10] = float(get_in(sh, "metallic", 0.0))
        e = np.array(get_in(sh, "emissiveColor", [0, 0, 0]), np.float32)[:3]
        row[12:15] = e
        return row
    t = tex.get(get_in(sh, "diffuse_texture"))
    if t < 0 and "diffuse_reflection_color_image" in sh["inputs"]:  # OmniSurface
        t = tex.get(get_in(sh, "diffuse_reflection_color_image"))
    if t >= 0:
        row[0:3] = tint
        row[3] = t
    else:
        c = get_in(sh, "diffuse_color_constant", None)
        if c is None:
            c = get_in(sh, "diffuse_reflection_color", [0.2, 0.2, 0.2])
        row[0:3] = np.array(c, np.float32)[:3] * tint
    sc = get_in(sh, "texture_scale", [1, 1])
    tr = get_in(sh, "texture_translate", [0, 0])
    row[4:6] = np.array(sc, np.float32)[:2]
    row[6:8] = np.array(tr, np.float32)[:2]
    row[8] = math.radians(float(get_in(sh, "texture_rotate", 0.0)))
    row[9] = float(get_in(sh, "reflection_roughness_constant", get_in(sh, "specular_reflection_roughness", 0.5)))
    row[10] = float(get_in(sh, "metallic_constant", get_in(sh, "metalness", 0.0)))
    if get_in(sh, "enable_opacity", False):
        row[11] = float(get_in(sh, "opacity_constant", 1.0))
        row[15] = 1
    if get_in(sh, "enable_emission", False):
        e = np.array(get_in(sh, "emissive_color", [1, 1, 1]), np.float32)[:3]
        row[12:15] = e * float(get_in(sh, "emissive_intensity", 40.0))
    row[16] = float(get_in(sh, "albedo_add", 0.0))
    row[17] = float(get_in(sh, "albedo_brightness", 1.0))
    return row


def light_row(L, anchor_idx, tex=None):
    row = np.zeros(32, np.float32)
    a = L["attrs"]

    def g(*names, default=0.0):
        for n in names:
            for p in ("inputs:", ""):
                if p + n in a and a[p + n] is not None:
                    return a[p + n]
        return default

    row[0] = LIGHT_TYPES.get(L["type"], -1)
    row[1] = anchor_idx
    row[2] = 1.0 if L.get("visible0", True) else 0.0
    inten = float(g("intensity", default=1.0))
    expo = float(g("exposure", default=0.0))
    col = np.array(g("color", default=[1, 1, 1]), np.float64)[:3]
    if g("enableColorTemperature", default=False):
        col = col * blackbody_rgb(float(g("colorTemperature", default=6500.0)))
    row[3:6] = (inten * (2.0 ** expo) * col).astype(np.float32)
    row[6] = float(g("radius", default=0.5))
    row[7] = float(g("width", default=1.0))
    row[8] = float(g("height", default=1.0))
    row[9] = float(g("length", default=1.0))
    row[10] = float(g("angle", default=0.53))
    row[11] = float(a.get("inputs:shaping:cone:angle", a.get("shaping:cone:angle", 90.0)) or 90.0)
    row[12] = float(a.get("inputs:shaping:cone:softness", a.get("shaping:cone:softness", 0.0)) or 0.0)
    row[13:25] = usd_to_aff(np.eye(4))
    row[25] = 1.0 if g("normalize", default=False) else 0.0  # USD Lux normalize: 휘도를 조명 표면적으로 나눔(렌더러가 한다)
    if tex is not None and L["type"] == "DomeLight":  # 돔 위경도 텍스처 번호 + 1 (0 = 없음) (render B, rsc_io.h light_from)
        tf = g("texture:file", default=None)
        row[26] = float(tex.get(tf) + 1) if tf else 0.0
    return row


def main():
    argv = sys.argv[1:]
    d = argv[0]
    tex_max = int(argv[argv.index("--tex-max") + 1]) if "--tex-max" in argv else 1024
    out = argv[argv.index("--out") + 1] if "--out" in argv else d
    os.makedirs(out, exist_ok=True)
    G = np.load(os.path.join(d, "geom.npz"))
    with open(os.path.join(d, "scene.json"), encoding="utf-8") as f:
        sc = json.load(f)
    npts, ntri_g, has_n, has_uv = G["n_points"], G["n_tri"], G["has_normals"], G["has_uv"]
    p_off = np.concatenate([[0], np.cumsum(npts)])
    t_off = np.concatenate([[0], np.cumsum(ntri_g)])
    n_off = np.concatenate([[0], np.cumsum(np.where(has_n, ntri_g, 0))])
    u_off = np.concatenate([[0], np.cumsum(np.where(has_uv, ntri_g, 0))])
    points, tri, fot, normals, uv = G["points"], G["tri"], G["face_of_tri"], G["normals"], G["uv"]

    meshes, anchors, mats, lights = sc["meshes"], sc["anchors"], sc["materials"], sc["lights"]
    tex = TexCache(tex_max)
    stats = {"no_shader": 0}
    mat_rows = [material_row(m, tex, stats) for m in mats]
    default_mat = len(mat_rows)
    mat_rows.append(material_row(None, tex, stats))

    # 기하 × 칸 배치
    conv_key, conv = {}, []  # key -> 새 기하 번호, conv = (원 기하, tri_slot)
    inst_rows, slot_mat = [], []
    for m in meshes:
        g = m["geom"]
        nt = int(ntri_g[g])
        slot = np.zeros(nt, np.int32)
        fo = fot[t_off[g]:t_off[g + 1]]
        for k, s in enumerate(m["subsets"]):
            faces = np.asarray(s["faces"], np.int64)
            if faces.size:
                slot[np.isin(fo, faces)] = k + 1
        key = (g, hashlib.blake2b(slot.tobytes(), digest_size=8).hexdigest())
        if key not in conv_key:
            conv_key[key] = len(conv)
            conv.append((g, slot))
        base = len(slot_mat)
        slot_mat.append(m["material"] if m["material"] >= 0 else default_mat)
        for s in m["subsets"]:
            slot_mat.append(s["material"] if s["material"] >= 0 else slot_mat[base])
        flags = 1 if m.get("double_sided") else 0
        inst_rows.append((conv_key[key], m["anchor"], base, flags, usd_to_aff(m["rel"])))
    n_mesh_anchor = len(anchors)
    light_rows = [light_row(L, n_mesh_anchor + i, tex) for i, L in enumerate(lights)]
    n_anchor = n_mesh_anchor + len(lights)

    # 쓰기
    tv_all, tn_all, tu_all, ts_all, gtab = [], [], [], [], []
    off = 0
    for g, slot in conv:
        nt = int(ntri_g[g])
        P = points[p_off[g]:p_off[g + 1]]
        T = tri[t_off[g]:t_off[g + 1]]
        tv_all.append(P[T].reshape(nt, 9).astype(np.float32))
        tn_all.append(normals[n_off[g]:n_off[g + 1]].reshape(nt, 9).astype(np.float32) if has_n[g] else np.zeros((nt, 9), np.float32))
        tu_all.append(uv[u_off[g]:u_off[g + 1]].reshape(nt, 6).astype(np.float32) if has_uv[g] else np.zeros((nt, 6), np.float32))
        ts_all.append(slot)
        gtab.append((off, nt, (1 if has_n[g] else 0) | (2 if has_uv[g] else 0)))
        off += nt
    n_tri = off
    texels = np.concatenate(tex.chunks) if tex.chunks else np.zeros(0, np.uint32)
    # 음영 기본값 (render B 가 radio 기준 자료 스텝 1·2·101 × 카메라 3 과 SSIM·EMD 로 고름, docs 14.6.3). core/render/rsc_io.h load_scene 순서
    sp = np.zeros(16, np.float32)
    sp[0:3] = 1.0      # 주변광 (가림 없음, 노출 곱하면 거의 0)
    sp[3] = 0.00283    # 노출 (Hable 에서 히스토그램 EMD 최소; 물리 추정 ISO100·1/50 s·f5 = 0.0008 과 같은 자릿수)
    sp[4] = 2          # spp
    sp[5] = 1          # 그림자 광선 (기여 비례 조명 고르기)
    sp[6] = 5          # Hable + sRGB (공식 op 6 ACES 는 우리 ACES 두 판보다 Hable 이 더 가깝다)
    sp[7] = 1          # 튕김
    sp[10] = 4         # à-trous 잡음 제거 4 번 (간격 1,2,4,8)
    sp[11] = 0.01      # 잡음 제거 평면 거리 허용
    sp[12] = 1         # 1차 면 GGX 반사 광선
    sp[13] = 2.0       # 간접광 표본 자르기 (× 1/노출)
    sp[14] = 2         # 텍스처 밉 = 짧은 축 발자국 (RTX 비등방 필터 흉내)
    sp[15] = -0.5      # 밉 단계 더하기
    with open(os.path.join(out, "scene.rsc"), "wb") as f:
        f.write(b"RSCENE01")
        cnt = np.zeros(16, np.int64)
        cnt[:9] = [len(conv), n_tri, len(inst_rows), n_anchor, len(mat_rows), len(tex.infos), len(slot_mat), len(light_rows), texels.size]
        f.write(cnt.tobytes())
        f.write(np.array(gtab, np.int64).reshape(-1, 3).tobytes())
        for arrs in (tv_all, tn_all, tu_all, ts_all):
            for a in arrs:
                f.write(np.ascontiguousarray(a).tobytes())
        for g, anc, base, flags, rel in inst_rows:
            f.write(struct.pack("<4i", g, anc, base, flags))
            f.write(rel.tobytes())
        f.write(np.array(slot_mat, np.int32).tobytes())
        f.write(np.stack(mat_rows).astype(np.float32).tobytes())
        f.write((np.stack(light_rows) if light_rows else np.zeros((0, 32), np.float32)).astype(np.float32).tobytes())
        f.write(np.array(tex.infos, np.int64).reshape(-1, 3).tobytes())
        f.write(texels.tobytes())
        f.write(sp.tobytes())
    print(f"scene.rsc: 기하 {len(conv)} (원 {len(ntri_g)}), 삼각형 {n_tri:,}, 인스턴스 {len(inst_rows)}, 기준 prim {n_anchor} "
          f"(메시 {n_mesh_anchor} + 조명 {len(lights)}), 재질 {len(mat_rows)}, 텍스처 {len(tex.infos)} ({texels.size * 4 / 2**20:.0f} MiB), "
          f"조명 {len(light_rows)}, 재질 종류 {stats.get('kinds')}, 셰이더 없음 {stats['no_shader']}, 못 찾은 텍스처 {len(tex.missing)}")

    # 프레임
    roles = ["head", "left_wrist", "right_wrist"]
    # 투영 행렬(cameraProjection)은 첫 기록 스텝에만 뜬다 -> 모든 프레임에 같은 값을 쓴다(프레임마다 tan 이 달라지지 않게)
    proj = {}
    for fn in sorted(x for x in os.listdir(d) if x.startswith("frame_") and x.endswith(".npz")):
        for r, c in json.loads(str(np.load(os.path.join(d, fn), allow_pickle=False)["cams_json"])).items():
            cp = c.get("camera_parameters")
            if r not in proj and cp and "cameraProjection" in cp:
                proj[r] = cp
    for fn in sorted(x for x in os.listdir(d) if x.startswith("frame_") and x.endswith(".npz")):
        Z = np.load(os.path.join(d, fn), allow_pickle=False)
        cams = json.loads(str(Z["cams_json"]))
        k = int(fn[6:10])
        aw = [usd_to_aff(M) for M in Z["anchor_world"]]
        aw += [usd_to_aff(M) for M in Z["light_world"]]
        vis = Z["mesh_visible"].astype(bool)
        words = np.zeros((len(vis) + 31) // 32, np.uint32)
        for i in np.nonzero(vis)[0]:
            words[i >> 5] |= np.uint32(1) << np.uint32(i & 31)
        cam_rows = []
        for r in roles:
            c = cams[r]
            M = np.asarray(c["world"], np.float64)
            ax = [M[i, :3] / np.linalg.norm(M[i, :3]) for i in range(3)]
            row = np.zeros(20, np.float32)
            for rr in range(3):
                for cc in range(3):
                    row[rr * 4 + cc] = ax[cc][rr]
                row[rr * 4 + 3] = M[3][rr]
            p = c["params"]
            W, H = p["image_width"], p["image_height"]
            fl = float(p["focalLength"])
            ha = float(p["horizontalAperture"])
            va = float(p.get("verticalAperture") or ha * H / W)
            tanx, tany = ha / 2 / fl, ha * H / W / 2 / fl  # 가로 맞춤(세로 = 가로 × H/W) — 투영 행렬로 확인
            cp = proj.get(r)
            if cp and "cameraProjection" in cp:
                P = np.asarray(cp["cameraProjection"], np.float64).reshape(4, 4)
                tanx, tany = 1.0 / P[0, 0], 1.0 / P[1, 1]
            row[12], row[13] = tanx, tany
            cr = p.get("clippingRange") or [0.01, 1e6]
            row[14], row[15] = cr[0], cr[1]
            row[16], row[17] = W, H
            cam_rows.append(row)
        imgs = []
        for key in Z.files:
            if key.startswith("img::") or key.startswith("noise"):
                a = Z[key]
                if a.dtype == np.uint8:
                    a3 = a.reshape(a.shape[0], a.shape[1], -1)
                    imgs.append((key, a3, 0))
                else:
                    a3 = a.astype(np.float32).reshape(a.shape[0], a.shape[1], -1)
                    imgs.append((key, a3, 1))
        with open(os.path.join(out, f"frame_{k:04d}.rfr"), "wb") as f:
            f.write(b"RFRAME01")
            f.write(np.array([k, len(aw), len(vis), len(cam_rows), len(imgs), 0, 0, 0], np.int64).tobytes())
            f.write(np.stack(aw).astype(np.float32).tobytes())
            f.write(words.tobytes())
            f.write(np.stack(cam_rows).tobytes())
            pos = f.tell() + len(imgs) * (48 + 5 * 8)
            for name, a, dt in imgs:
                nb = name.encode()[:48]
                f.write(nb + b"\0" * (48 - len(nb)))
                f.write(np.array([a.shape[0], a.shape[1], a.shape[2], dt, pos], np.int64).tobytes())
                pos += a.nbytes
            for name, a, dt in imgs:
                f.write(np.ascontiguousarray(a).tobytes())
        print(f"frame_{k:04d}.rfr: 영상 {len(imgs)}, 보이는 메시 {int(vis.sum())}/{len(vis)}")


if __name__ == "__main__":
    main()
