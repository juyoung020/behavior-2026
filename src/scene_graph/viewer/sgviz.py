#!/usr/bin/env python3
"""Live Spark-DSG viewer for the robot's object memory.

    ~/sdsg_venv/bin/python src/scene_graph/viewer/sgviz.py <memory_dir> [--port 8080]

<memory_dir> is the directory the memory runtime rewrites every ~1 s:
scene.json (Spark-DSG DynamicSceneGraph), map.pgm + map.yaml, objects/*.png.

Built on spark_dsg.viser.ViserRenderer (server ownership / clearing) and the
spark_dsg python API (graph loading, bounding-box corners and edge indices).
The stock GraphHandle redraws the whole graph per call (and uses viser APIs
that are gone in viser 1.x), so the per-node incremental drawing, map, robot
and metadata panel are done here on top of the renderer's server.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# `python sgviz.py` puts this folder on sys.path; nothing here may shadow
# `viser` or `spark_dsg` (Spark-DSG's own python dir contains a viser.py).
import spark_dsg as dsg
from spark_dsg.viser import BOUNDING_BOX_EDGE_INDICES, ViserRenderer

STATE_COLORS = {
    "seen": (46, 160, 67),
    "moved": (245, 140, 20),
    "held": (40, 110, 230),
    "gone": (140, 140, 140),
}
DEFAULT_COLOR = (200, 60, 200)
RELATION_COLORS = {"on": (90, 60, 200), "in": (200, 40, 120)}


# --------------------------------------------------------------------------
# Parsing (headless, testable)
# --------------------------------------------------------------------------


@dataclass
class ObjInfo:
    id: int
    sym: str
    name: str
    pos: Tuple[float, float, float]
    state: str = "seen"
    n_obs: int = 0
    score: float = 0.0
    first_pos: Optional[Tuple[float, float, float]] = None
    last_seen: float = 0.0  # seconds (from last_update_time_ns)
    is_active: bool = True
    structural: bool = False
    handled: bool = False
    bbox_dims: Optional[Tuple[float, float, float]] = None
    bbox_center: Optional[Tuple[float, float, float]] = None
    bbox_corners: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    rgbd: Optional[Dict[str, Any]] = None
    points: Optional[Dict[str, Any]] = None  # {"path", "n", "voxel", "stamp"}
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        return f"{self.name}#{self.id}"

    def draw_sig(self):
        """Fields that affect the 3D drawing."""
        return (self.name, self.pos, self.state, self.structural, self.is_active,
                self.first_pos, self.bbox_dims, self.bbox_center, repr(self.points))

    def full_sig(self):
        return (self.draw_sig(), self.n_obs, self.score, self.last_seen,
                self.handled, repr(self.rgbd), repr(sorted(self.metadata.items())))


@dataclass
class EdgeInfo:
    source: int  # object
    target: int  # support
    relation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[int, int]:
        return (self.source, self.target)

    def sig(self):
        return (self.relation, repr(sorted(self.metadata.items())))


@dataclass
class Scene:
    stamp: float = 0.0
    robot_pose: Optional[Tuple[float, float, float]] = None
    grid: Optional[str] = None
    objects: Dict[int, ObjInfo] = field(default_factory=dict)
    edges: Dict[Tuple[int, int], EdgeInfo] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _r(v, nd=3):
    return tuple(round(float(x), nd) for x in v)


def _plain(v):
    """spark_dsg metadata comes back as (nested) mappingproxy; make it plain."""
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def _meta(obj) -> Dict[str, Any]:
    try:
        m = _plain(obj.metadata.get())
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def scene_from_graph(G: dsg.DynamicSceneGraph) -> Scene:
    gm = _meta(G)
    sc = Scene(metadata=gm)
    sc.stamp = float(gm.get("stamp", 0.0) or 0.0)
    rp = gm.get("robot_pose")
    if isinstance(rp, (list, tuple)) and len(rp) >= 3:
        sc.robot_pose = _r(rp[:3], 4)
    sc.grid = gm.get("grid")

    if not G.has_layer(dsg.DsgLayers.OBJECTS):
        return sc
    layer = G.get_layer(dsg.DsgLayers.OBJECTS)
    for node in layer.nodes:
        a = node.attributes
        md = _meta(a)
        sym = node.id
        oid = int(sym.category_id)
        fp = md.get("first_pos")
        info = ObjInfo(
            id=oid,
            sym=sym.str(),
            name=str(getattr(a, "name", "") or sym.str()),
            pos=_r(a.position),
            state=str(md.get("state", "seen")),
            n_obs=int(md.get("n_obs", 0) or 0),
            score=float(md.get("score", 0.0) or 0.0),
            first_pos=_r(fp) if isinstance(fp, (list, tuple)) and len(fp) == 3 else None,
            last_seen=float(getattr(a, "last_update_time_ns", 0)) * 1e-9,
            is_active=bool(getattr(a, "is_active", True)),
            structural=bool(md.get("structural", False)),
            handled=bool(md.get("handled", False)),
            rgbd=md.get("rgbd") if isinstance(md.get("rgbd"), dict) else None,
            points=md.get("points") if isinstance(md.get("points"), dict) else None,
            metadata=md,
        )
        bb = getattr(a, "bounding_box", None)
        try:
            if bb is not None and bb.is_valid():
                info.bbox_dims = _r(bb.dimensions)
                info.bbox_center = _r(bb.world_P_center)
                info.bbox_corners = np.asarray(bb.corners(), dtype=np.float64)
        except Exception:
            pass
        sc.objects[oid] = info

    for e in layer.edges:
        s, t = dsg.NodeSymbol(e.source), dsg.NodeSymbol(e.target)
        md = _meta(e.info)
        ei = EdgeInfo(int(s.category_id), int(t.category_id),
                      str(md.get("relation", "")), md)
        sc.edges[ei.key] = ei
    return sc


def load_scene(dirpath: str) -> Scene:
    G = dsg.DynamicSceneGraph.load(os.path.join(dirpath, "scene.json"))
    return scene_from_graph(G)


@dataclass
class Diff:
    added: List[int] = field(default_factory=list)
    redraw: List[int] = field(default_factory=list)   # geometry changed
    changed: List[int] = field(default_factory=list)  # any field changed (superset of redraw)
    removed: List[int] = field(default_factory=list)
    edges_added: List[Tuple[int, int]] = field(default_factory=list)
    edges_changed: List[Tuple[int, int]] = field(default_factory=list)
    edges_removed: List[Tuple[int, int]] = field(default_factory=list)

    def empty(self) -> bool:
        return not (self.added or self.changed or self.removed or self.edges_added
                    or self.edges_changed or self.edges_removed)


def diff_scenes(old: Optional[Scene], new: Scene) -> Diff:
    d = Diff()
    oo = old.objects if old else {}
    for oid, o in new.objects.items():
        p = oo.get(oid)
        if p is None:
            d.added.append(oid)
        else:
            if p.draw_sig() != o.draw_sig():
                d.redraw.append(oid)
            if p.full_sig() != o.full_sig():
                d.changed.append(oid)
    d.removed = [oid for oid in oo if oid not in new.objects]
    oe = old.edges if old else {}
    for k, e in new.edges.items():
        if k not in oe:
            d.edges_added.append(k)
        elif oe[k].sig() != e.sig():
            d.edges_changed.append(k)
    d.edges_removed = [k for k in oe if k not in new.edges]
    return d


def relations_of(scene: Scene, oid: int) -> List[str]:
    out = []
    for (s, t), e in scene.edges.items():
        rel = e.relation or "-"
        if s == oid:
            tn = scene.objects[t].label if t in scene.objects else f"#{t}"
            out.append(f"{rel} {tn}")
        elif t == oid:
            sn = scene.objects[s].label if s in scene.objects else f"#{s}"
            out.append(f"{sn} {rel} this")
    return out


# --------------------------------------------------------------------------
# Map / images
# --------------------------------------------------------------------------


def read_map_yaml(path: str) -> Dict[str, Any]:
    """Tiny parser for ROS map_server yaml (no pyyaml dependency)."""
    out: Dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            k, v = (x.strip() for x in line.split(":", 1))
            if v.startswith("["):
                out[k] = [float(x) for x in v.strip("[]").split(",") if x.strip()]
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def map_texture(pgm: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Occupancy PGM -> RGB texture.

    Cells keep their grey level (0 = occupied ... 254 = free; the runtime writes
    graded values), except the map_saver "unknown" value 205, drawn blue-grey.
    """
    g = pgm.astype(np.uint8)
    if int(meta.get("negate", 0) or 0):
        g = 255 - g
    rgb = np.repeat(g[..., None], 3, axis=2)
    rgb[pgm == 205] = (150, 150, 165)
    return rgb


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Minimal PLY vertex reader (binary little/big endian or ascii; numpy only).

    Returns (xyz float32 [N,3], rgb uint8 [N,3] or None). Elements after
    "vertex" (faces etc.) are ignored.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt, n, props, cur, before = None, 0, [], None, 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header without end_header")
            tok = line.decode("ascii", "replace").split()
            if not tok or tok[0] in ("comment", "obj_info"):
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                cur = tok[1]
                if cur == "vertex":
                    n = int(tok[2])
                elif n == 0:
                    before += 1  # an element before vertex: unsupported layout
            elif tok[0] == "property" and cur == "vertex":
                if tok[1] == "list":
                    raise ValueError("list property in vertex element")
                props.append((tok[2], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        if before:
            raise ValueError("elements before vertex are not supported")
        names = [p[0] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, ndmin=2)
            col = {nm: data[:, i] for i, nm in enumerate(names)}
        else:
            end = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(nm, end + t) for nm, t in props])
            buf = f.read(dt.itemsize * n)
            if len(buf) < dt.itemsize * n:
                raise ValueError("truncated PLY")
            arr = np.frombuffer(buf, dtype=dt, count=n)
            col = {nm: arr[nm] for nm in names}
    xyz = np.stack([col["x"], col["y"], col["z"]], axis=1).astype(np.float32)
    rgb = None
    for keys in (("red", "green", "blue"), ("r", "g", "b")):
        if all(k in col for k in keys):
            rgb = np.stack([col[k] for k in keys], axis=1)
            if rgb.dtype.kind == "f":
                rgb = rgb * (255.0 if rgb.max() <= 1.0 else 1.0)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            break
    ok = np.all(np.isfinite(xyz), axis=1)
    return xyz[ok], (rgb[ok] if rgb is not None else None)


class PointCache:
    """(path, mtime_ns, size)-keyed cache of object point clouds."""

    def __init__(self, maxlen: int = 512):
        self._d: Dict[str, Tuple[Tuple[int, int], Tuple[np.ndarray, Optional[np.ndarray]]]] = {}
        self._maxlen = maxlen

    @staticmethod
    def file_key(path: str) -> Optional[Tuple[int, int]]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def get(self, path: str):
        """-> (key, (xyz, rgb)) or (None, None) if missing/unreadable."""
        key = self.file_key(path)
        if key is None:
            return None, None
        hit = self._d.get(path)
        if hit is not None and hit[0] == key:
            return key, hit[1]
        try:
            pts = read_ply(path)
        except Exception as ex:  # half-written -> try again on a later poll
            print(f"[sgviz] ply read failed {path}: {ex}", file=sys.stderr)
            return None, None
        if len(self._d) >= self._maxlen and path not in self._d:
            self._d.pop(next(iter(self._d)))
        self._d[path] = (key, pts)
        return key, pts


def _erode(m: np.ndarray) -> np.ndarray:
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def rgb_overlay(rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """RGB crop with the segment shown: outside dimmed, inside kept, yellow outline."""
    if mask is None:
        return rgb
    out = rgb.astype(np.float32)
    out[~mask] = out[~mask] * 0.45 + 30.0
    edge = mask & ~_erode(mask)
    edge |= np.roll(edge, 1, axis=1)  # 2 px wide
    out[edge] = (255, 210, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def depth_gray(depth: np.ndarray, mask: Optional[np.ndarray] = None):
    """Depth crop -> grayscale RGB (near bright, far dark) + stats in metres.

    Range is a robust 2-98 percentile of valid depth inside the mask (whole crop
    when no mask). Pixels outside the mask are dimmed; invalid (0) is black.
    Input: uint16 mm, or float metres. Returns (rgb uint8, (min, median, max) m or None).
    """
    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if depth.dtype.kind in "ui":
        d = d * 1e-3
    valid = np.isfinite(d) & (d > 0)
    sel = valid & mask if mask is not None else valid
    if not sel.any():
        sel = valid
    rgb = np.zeros(d.shape + (3,), np.uint8)
    if not sel.any():
        return rgb, None
    vals = d[sel]
    lo, hi = np.percentile(vals, [2, 98])
    if hi - lo < 1e-3:
        lo, hi = lo - 0.05, hi + 0.05
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    g = 255.0 - 200.0 * t  # near 255, far 55
    if mask is not None:
        g = np.where(mask, g, g * 0.35)
    g = np.where(valid, g, 0.0)
    rgb[:] = g.astype(np.uint8)[..., None]
    stats = (float(vals.min()), float(np.median(vals)), float(vals.max()))
    return rgb, stats


def maybe_crop(img: np.ndarray, box_px, margin: int = 8) -> np.ndarray:
    """Crop to box_px if the stored image is a full frame (box fits well inside)."""
    if not box_px or len(box_px) != 4:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box_px)
    if (x1 - x0, y1 - y0) == (w, h):
        return img  # stored image is exactly the box crop
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        return img  # already a crop (box is in full-frame coords)
    if (x1 - x0) * (y1 - y0) > 0.8 * w * h:
        return img
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(w, x1 + margin), min(h, y1 + margin)
    return img[y0:y1, x0:x1]


def _read_image(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            return np.array(im, dtype=np.uint16 if im.mode.startswith("I;16") else np.int32)
        return np.array(im.convert("RGB"))


def _fit(img: np.ndarray, max_side: int = 320, min_side: int = 240) -> np.ndarray:
    """Downsample big crops; upscale small ones (nearest, integer) so the panel shows them."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s < min_side:
        k = max(1, min(8, max_side // s))
        return np.repeat(np.repeat(img, k, axis=0), k, axis=1) if k > 1 else img
    if s <= max_side:
        return img
    step = int(math.ceil(s / max_side))
    return img[::step, ::step]


class CropCache:
    """Lazy cache of the panel images for one rgbd record (keyed by file mtimes)."""

    def __init__(self, maxlen: int = 64):
        self._d: Dict[Any, Dict[str, Any]] = {}
        self._maxlen = maxlen

    def get(self, root: str, rgbd: Dict[str, Any]) -> Dict[str, Any]:
        """-> {"rgb": img|None, "depth": img|None, "stats": (min,med,max)|None, "mask": bool}"""
        paths = {k: os.path.join(root, rgbd[k]) for k in ("rgb", "depth", "mask")
                 if isinstance(rgbd.get(k), str)}
        key = [repr(rgbd.get("box_px"))]
        for k in sorted(paths):
            fk = PointCache.file_key(paths[k])
            key.append((k, fk))
            if fk is None:
                paths.pop(k)
        key = tuple(key)
        if key in self._d:
            return self._d[key]
        box = rgbd.get("box_px")
        out: Dict[str, Any] = {"rgb": None, "depth": None, "stats": None, "mask": False}
        rgb = dep = None
        try:
            if "rgb" in paths:
                rgb = maybe_crop(_read_image(paths["rgb"]), box)
                if rgb.ndim == 2:
                    rgb = np.repeat(rgb[..., None], 3, axis=2).astype(np.uint8)
            if "depth" in paths:
                dep = maybe_crop(_read_image(paths["depth"]), box)
        except Exception as ex:
            print(f"[sgviz] crop read failed: {ex}", file=sys.stderr)
            return out  # not cached: retry on next selection/update
        mask_rgb = mask_dep = None
        if "mask" in paths:
            ref = rgb if rgb is not None else dep
            try:
                if ref is not None:
                    full = _read_image(paths["mask"])
                    full = full if full.ndim == 2 else full[..., 0]
                    full = maybe_crop(full, box)
                    from PIL import Image as _I

                    def fit_mask(shape):
                        im = _I.fromarray(full.astype(np.uint8))
                        if im.size != (shape[1], shape[0]):
                            im = im.resize((shape[1], shape[0]), _I.NEAREST)
                        return np.array(im) >= 128

                    mask_rgb = fit_mask(rgb.shape) if rgb is not None else None
                    mask_dep = fit_mask(dep.shape) if dep is not None else None
                    out["mask"] = True
            except Exception as ex:
                print(f"[sgviz] mask read failed: {ex}", file=sys.stderr)
        if rgb is not None:
            # scale first so the outline stays thin on upscaled small crops
            m = _fit(mask_rgb) if mask_rgb is not None else None
            out["rgb"] = np.ascontiguousarray(rgb_overlay(_fit(rgb), m))
        if dep is not None:
            img, stats = depth_gray(dep, mask_dep)
            out["depth"], out["stats"] = _fit(np.ascontiguousarray(img)), stats
        if len(self._d) >= self._maxlen:
            self._d.pop(next(iter(self._d)))
        self._d[key] = out
        return out


# --------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------


class PickIndex:
    """Ray picking over object point clouds (and fallback spheres), numpy only.

    Each entry keeps its points (float32 [N,3]), a hit radius and an AABB
    grown by that radius. pick(origin, direction) returns the id of the object
    whose nearest hit point (perpendicular distance <= radius, in front of the
    origin) is closest along the ray, or None.
    """

    def __init__(self):
        self._e: Dict[int, Tuple[np.ndarray, float, np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return len(self._e)

    def __contains__(self, oid):
        return oid in self._e

    def set_points(self, oid: int, xyz: np.ndarray, radius: float):
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        self._e[oid] = (xyz, float(radius), xyz.min(axis=0) - radius, xyz.max(axis=0) + radius)

    def set_sphere(self, oid: int, center, radius: float):
        c = np.asarray(center, np.float32).reshape(1, 3)
        self._e[oid] = (c, float(radius), c[0] - radius, c[0] + radius)

    def remove(self, oid: int):
        self._e.pop(oid, None)

    def pick(self, origin, direction) -> Optional[int]:
        o = np.asarray(origin, np.float64)
        d = np.asarray(direction, np.float64)
        n = np.linalg.norm(d)
        if not np.isfinite(n) or n == 0:
            return None
        d = d / n
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
        o32, d32 = o.astype(np.float32), d.astype(np.float32)
        best, best_t = None, np.inf
        for oid, (pts, r, lo, hi) in self._e.items():
            # slab test against the radius-grown AABB (cheap reject)
            with np.errstate(invalid="ignore"):
                t1, t2 = (lo - o) * inv, (hi - o) * inv
            t1 = np.where(np.isnan(t1), -np.inf, t1)
            t2 = np.where(np.isnan(t2), np.inf, t2)
            tmin = float(np.max(np.minimum(t1, t2)))
            tmax = float(np.min(np.maximum(t1, t2)))
            if tmax < max(tmin, 0.0) or tmin > best_t:
                continue
            v = pts - o32
            t = v @ d32
            perp2 = np.einsum("ij,ij->i", v, v) - t * t
            ok = (t > 0) & (perp2 <= r * r)
            if not ok.any():
                continue
            th = float(t[ok].min())
            if th < best_t:
                best, best_t = oid, th
        return best


def _yaw_wxyz(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _box_segments(o: ObjInfo, max_dim: float) -> Optional[np.ndarray]:
    if o.bbox_dims is None or o.bbox_center is None:
        return None
    dims = np.asarray(o.bbox_dims, float)
    if not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        return None
    if np.all(dims <= max_dim) and o.bbox_corners is not None:
        corners = o.bbox_corners
    else:
        # clamp absurd extents (merged floor blobs etc.) around the center
        dims = np.minimum(dims, max_dim)
        c = np.asarray(o.bbox_center, float)
        bb = dsg.BoundingBox(dims.astype(np.float32), c.astype(np.float32))
        corners = np.asarray(bb.corners(), float)
    return corners[BOUNDING_BOX_EDGE_INDICES]


class SgViewer(ViserRenderer):
    def __init__(self, dirpath: str, ip: str = "0.0.0.0", port: int = 8080, poll: float = 0.25):
        super().__init__(ip, port=port, clear_at_exit=False)
        if self._server is None:
            raise RuntimeError("viser is not installed in this environment")
        self.server = self._server
        self.dir = os.path.abspath(dirpath)
        self.poll = poll
        self.lock = threading.RLock()
        self.scene: Optional[Scene] = None
        self.selected: Optional[int] = None
        self.crops = CropCache()
        self.points = PointCache()
        self._points_key: Dict[int, Any] = {}  # oid -> file key of the drawn cloud
        self._scene_mtime = None
        self._map_hash = None
        self._map_handle = None
        self._robot = None
        self._node_handles: Dict[int, List[Any]] = {}
        self._edge_handles: Dict[Tuple[int, int], List[Any]] = {}
        self._sel_handle = None
        self._camera_set = False
        self.picker = PickIndex()
        self._suppress_dropdown = False
        self._build_gui()
        from sglayers import LayerView  # Hydra-style stacked graph layers (view.json "graph")
        self.layers = LayerView(self.server, self.dir)

    # ---------------- GUI ----------------
    def _build_gui(self):
        gui = self.server.gui
        self.g_status = gui.add_markdown("waiting for scene.json ...")
        with gui.add_folder("Display", expand_by_default=False):
            self.g_labels = gui.add_checkbox("Labels", True)
            self.g_points = gui.add_checkbox("Segment points", True)
            self.g_pcolor = gui.add_dropdown("Point colour", ["true colour", "state colour"],
                                             initial_value="true colour")
            self.g_psize = gui.add_number("Point size x voxel", 1.0, min=0.2, max=5.0, step=0.1)
            self.g_markers = gui.add_checkbox("Centre markers", False)
            self.g_boxes = gui.add_checkbox("Boxes (objects w/o points)", False)
            self.g_edges = gui.add_checkbox("Support edges", True)
            self.g_trails = gui.add_checkbox("Moved trails", True)
            self.g_struct = gui.add_checkbox("Structural objects", True)
            self.g_gone = gui.add_checkbox("Gone objects", True)
            self.g_map = gui.add_checkbox("Occupancy map", True)
            self.g_size = gui.add_number("Node radius (no points)", 0.06, min=0.01, max=0.5, step=0.01)
            self.g_maxbox = gui.add_number("Max box side [m]", 2.0, min=0.1, max=20.0, step=0.1)
        for h in (self.g_labels, self.g_points, self.g_pcolor, self.g_psize, self.g_markers,
                  self.g_boxes, self.g_edges, self.g_trails,
                  self.g_struct, self.g_gone, self.g_size, self.g_maxbox):
            h.on_update(lambda _: self._redraw_all())
        self.g_map.on_update(lambda _: self._map_visibility())

        with gui.add_folder("Object", expand_by_default=True):
            self.g_select = gui.add_dropdown("Select", ["(none)"], initial_value="(none)")
            self.g_info = gui.add_markdown("Click a node or pick one above.")
            blank = np.zeros((2, 2, 3), np.uint8)
            self.g_rgb = gui.add_image(blank, label="RGB crop", visible=False)
            self.g_depth = gui.add_image(blank, label="Depth crop (near bright)", visible=False)
            self.g_dstats = gui.add_markdown("", visible=False)
        self.g_select.on_update(lambda _: self._on_dropdown())

        # click anywhere in the 3D view -> ray pick over the object clouds/spheres
        @self.server.scene.on_click()
        def _(ev):
            self.on_scene_click(ev.ray_origin, ev.ray_direction)

    def on_scene_click(self, origin, direction) -> Optional[int]:
        """Select the object hit by the click ray (a miss keeps the current selection)."""
        if origin is None or direction is None:
            return None
        with self.lock:
            oid = self.picker.pick(origin, direction)
        if oid is not None and oid != self.selected:
            self.select(oid)
        return oid

    def _options(self) -> List[str]:
        objs = self.scene.objects if self.scene else {}
        return ["(none)"] + [objs[k].label for k in sorted(objs)]

    def _on_dropdown(self):
        if self._suppress_dropdown:
            return
        v = self.g_select.value
        oid = None
        if v and "#" in v:
            try:
                oid = int(v.rsplit("#", 1)[1])
            except ValueError:
                oid = None
        self.select(oid, from_dropdown=True)

    def select(self, oid: Optional[int], from_dropdown: bool = False):
        with self.lock:
            prev, self.selected = self.selected, oid
            if not from_dropdown:
                self._set_dropdown()
            if self.scene is not None:  # re-tint old/new selection
                with self.server.atomic():
                    for k in {prev, oid}:
                        if k is not None and k in self.scene.objects and k in self._node_handles:
                            self._draw_node(self.scene.objects[k])
            self._draw_selection()
            self._render_panel()

    def _set_dropdown(self):
        opts = self._options()
        cur = "(none)"
        if self.scene and self.selected in self.scene.objects:
            cur = self.scene.objects[self.selected].label
        self._suppress_dropdown = True
        try:
            if tuple(self.g_select.options) != tuple(opts):
                self.g_select.options = opts
            if self.g_select.value != cur:
                self.g_select.value = cur
        finally:
            self._suppress_dropdown = False

    def _render_panel(self):
        sc = self.scene
        o = sc.objects.get(self.selected) if (sc and self.selected is not None) else None
        if o is None:
            self.g_info.content = "Click a node or pick one above."
            self.g_rgb.visible = False
            self.g_depth.visible = False
            self.g_dstats.visible = False
            return
        f3 = lambda v: "-" if v is None else "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
        rels = relations_of(sc, o.id)
        lines = [
            f"**{o.name}** #{o.id} (`{o.sym}`)",
            "",
            "| | |", "|---|---|",
            f"| state | **{o.state}**{' (structural)' if o.structural else ''}{' handled' if o.handled else ''} |",
            f"| pos | {f3(o.pos)} |",
            f"| first_pos | {f3(o.first_pos)} |",
            f"| n_obs | {o.n_obs} |",
            f"| score | {o.score:.3f} |",
            f"| last seen | {o.last_seen:.2f} s (now {sc.stamp:.2f}) |",
            f"| bbox | {f3(o.bbox_dims)} |",
            f"| relations | {'; '.join(rels) if rels else '-'} |",
        ]
        if o.points:
            n = o.points.get("n", "?")
            vx = o.points.get("voxel")
            lines.append(f"| points | {n} pts, voxel {vx} m, stamp {o.points.get('stamp', '-')} |")
        rgbd = o.rgbd or {}
        if rgbd:
            lines.append(f"| rgbd stamp | {rgbd.get('stamp', '-')} |")
            lines.append(f"| box_px | {rgbd.get('box_px', '-')} |")
            dm = rgbd.get("depth_m")
            lines.append(f"| depth_m | {dm:.3f} |" if isinstance(dm, (int, float)) else f"| depth_m | {dm} |")
            if "mask_area" in rgbd:
                lines.append(f"| mask_area | {rgbd.get('mask_area')} |")
        self.g_info.content = "\n".join(lines)

        crop = self.crops.get(self.dir, rgbd) if rgbd else {}
        for kind, handle in (("rgb", self.g_rgb), ("depth", self.g_depth)):
            img = crop.get(kind)
            if img is None:
                handle.visible = False
            else:
                handle.image = img
                handle.visible = True
        st = crop.get("stats")
        if st is not None and self.g_depth.visible:
            where = "inside mask" if crop.get("mask") else "whole crop"
            self.g_dstats.content = (f"depth min / median / max = **{st[0]:.3f} / {st[1]:.3f} / "
                                     f"{st[2]:.3f} m** ({where})")
            self.g_dstats.visible = True
        else:
            self.g_dstats.visible = False

    # ---------------- drawing ----------------
    def _visible(self, o: ObjInfo) -> bool:
        if o.structural and not self.g_struct.value:
            return False
        if o.state == "gone" and not self.g_gone.value:
            return False
        return True

    def _remove_node(self, oid: int):
        self.picker.remove(oid)
        for h in self._node_handles.pop(oid, []):
            try:
                h.remove()
            except Exception:
                pass

    def _points_path(self, o: ObjInfo) -> Optional[str]:
        p = (o.points or {}).get("path")
        if not isinstance(p, str) or not p:
            return None
        return p if os.path.isabs(p) else os.path.join(self.dir, p)

    def _draw_node(self, o: ObjInfo):
        self._remove_node(o.id)
        self._points_key.pop(o.id, None)
        if not self._visible(o):
            return
        sc = self.server.scene
        base = f"/objects/O{o.id}"
        col = STATE_COLORS.get(o.state, DEFAULT_COLOR)
        opacity = 0.35 if o.state == "gone" else None
        r = float(self.g_size.value)
        hs: List[Any] = []

        # segment shape (true-colour points) is the object's representation
        top_z = o.pos[2]
        has_pts = False
        path = self._points_path(o)
        if path is not None:
            key, pts = self.points.get(path)
            self._points_key[o.id] = key  # None = missing/unreadable -> retried in poll
            if pts is not None and len(pts[0]) > 0 and self.g_points.value:
                xyz, rgb = pts
                c = xyz.mean(axis=0)
                if rgb is None or self.g_pcolor.value == "state colour":
                    colors = np.tile(np.asarray(col, np.uint8), (len(xyz), 1))
                else:
                    colors = rgb
                if o.state == "gone":
                    colors = (colors.astype(np.float32) * 0.4 + 140 * 0.6).astype(np.uint8)
                vox = float((o.points or {}).get("voxel") or 0.02)
                psize = max(1e-3, vox * float(self.g_psize.value))
                if o.id == self.selected:  # highlight: tint towards yellow, slightly bigger
                    colors = (colors.astype(np.float32) * 0.45
                              + np.array([255, 225, 40], np.float32) * 0.55).astype(np.uint8)
                    psize *= 1.25
                hs.append(sc.add_point_cloud(
                    f"{base}/points", (xyz - c).astype(np.float32), colors,
                    point_size=psize,
                    point_shape="square", position=tuple(float(v) for v in c)))
                self.picker.set_points(o.id, xyz, max(2.0 * vox, 0.02))
                top_z = float(xyz[:, 2].max())
                has_pts = True
            elif pts is not None and len(pts[0]) > 0:
                has_pts = True  # points hidden by toggle: still no box

        # fallback sphere (objects without points) / optional centre marker.
        # Clicks are handled by the scene-level ray pick, not per-mesh on_click.
        if not has_pts or self.g_markers.value:
            rad = 0.02 if has_pts else r
            hs.append(sc.add_icosphere(f"{base}/node", radius=rad, color=col, position=o.pos,
                                       subdivisions=2, opacity=opacity))
            if not has_pts:
                self.picker.set_sphere(o.id, o.pos, 1.5 * r)
        if self.g_labels.value:
            z = max(top_z, o.pos[2]) + (0.04 if has_pts else 1.8 * r)
            hs.append(sc.add_label(f"{base}/label", o.label, position=(o.pos[0], o.pos[1], z),
                                   anchor="bottom-center", font_screen_scale=0.8))
        if self.g_boxes.value and not has_pts:
            seg = _box_segments(o, float(self.g_maxbox.value))
            if seg is not None:
                hs.append(sc.add_line_segments(f"{base}/bbox", seg, col, thickness=1.5,
                                               thickness_units="screen"))
        if self.g_trails.value and o.first_pos is not None and o.state == "moved":
            fp = np.asarray(o.first_pos, float)
            if np.linalg.norm(fp - np.asarray(o.pos)) > 0.02:
                hs.append(sc.add_line_segments(
                    f"{base}/trail", np.array([[fp, o.pos]], float), STATE_COLORS["moved"],
                    thickness=3.0, thickness_units="screen"))
                hs.append(sc.add_icosphere(f"{base}/first", radius=0.03, color=(180, 180, 180),
                                           position=tuple(fp), subdivisions=1, opacity=0.6))
        self._node_handles[o.id] = hs

    def _refresh_points(self) -> List[int]:
        """Redraw objects whose PLY file appeared/changed after the node was drawn."""
        if self.scene is None:
            return []
        todo = []
        for oid, o in self.scene.objects.items():
            path = self._points_path(o)
            if path is None or oid not in self._node_handles:
                continue
            if PointCache.file_key(path) != self._points_key.get(oid):
                todo.append(oid)
        if todo:
            with self.server.atomic():
                for oid in todo:
                    self._draw_node(self.scene.objects[oid])
        return todo

    def _remove_edge(self, k):
        for h in self._edge_handles.pop(k, []):
            try:
                h.remove()
            except Exception:
                pass

    def _draw_edge(self, e: EdgeInfo):
        self._remove_edge(e.key)
        objs = self.scene.objects
        if not self.g_edges.value or e.source not in objs or e.target not in objs:
            return
        a, b = objs[e.source], objs[e.target]
        if not (self._visible(a) and self._visible(b)):
            return
        col = RELATION_COLORS.get(e.relation, (60, 60, 60))
        sc = self.server.scene
        name = f"/edges/O{e.source}_O{e.target}"
        hs = [sc.add_line_segments(f"{name}/line", np.array([[a.pos, b.pos]], float), col,
                                   thickness=2.0, thickness_units="screen")]
        if e.relation:
            mid = tuple((np.asarray(a.pos) + np.asarray(b.pos)) / 2)
            hs.append(sc.add_label(f"{name}/label", e.relation, position=mid,
                                   anchor="center-center", font_screen_scale=0.7))
        self._edge_handles[e.key] = hs

    def _draw_selection(self):
        if self._sel_handle is not None:
            self._sel_handle.remove()
            self._sel_handle = None
        sc = self.scene
        if sc is None or self.selected not in sc.objects:
            return
        if any(h.name.endswith("/points") for h in self._node_handles.get(self.selected, [])):
            return  # point-cloud objects are highlighted by tinting their points
        o = sc.objects[self.selected]
        self._sel_handle = self.server.scene.add_icosphere(
            "/selection", radius=2.2 * float(self.g_size.value), color=(255, 220, 0),
            position=o.pos, subdivisions=2, wireframe=True)

    def _redraw_all(self):
        with self.lock:
            if self.scene is None:
                return
            with self.server.atomic():
                for o in self.scene.objects.values():
                    self._draw_node(o)
                for e in self.scene.edges.values():
                    self._draw_edge(e)
                self._draw_selection()

    def _draw_robot(self, pose):
        if pose is None:
            return
        x, y, yaw = pose
        if self._robot is None:
            sc = self.server.scene
            root = sc.add_frame("/robot", show_axes=False, position=(x, y, 0.0), wxyz=_yaw_wxyz(yaw))
            body = sc.add_icosphere("/robot/body", radius=0.25, color=(30, 30, 30),
                                    scale=(1.0, 1.0, 0.15), position=(0, 0, 0.05), subdivisions=2)
            arrow = sc.add_arrows("/robot/heading", np.array([[[0, 0, 0.1], [0.6, 0, 0.1]]], float),
                                  (220, 30, 30), shaft_radius=0.03, head_radius=0.08, head_length=0.15)
            self._robot = (root, body, arrow)
        else:
            self._robot[0].position = (x, y, 0.0)
            self._robot[0].wxyz = _yaw_wxyz(yaw)

    def _map_visibility(self):
        if self._map_handle is not None:
            self._map_handle.visible = bool(self.g_map.value)

    def _update_map(self, grid_name: Optional[str]):
        pgm_path = os.path.join(self.dir, grid_name or "map.pgm")
        yaml_path = os.path.splitext(pgm_path)[0] + ".yaml"
        try:
            with open(pgm_path, "rb") as f:
                raw = f.read()
            with open(yaml_path, "rb") as f:
                ytxt = f.read()
        except OSError:
            return
        h = hashlib.md5(raw + ytxt).hexdigest()
        if h == self._map_hash:
            return
        try:
            import io
            from PIL import Image

            meta = read_map_yaml(yaml_path)
            pgm = np.array(Image.open(io.BytesIO(raw)).convert("L"))
        except Exception as ex:  # partially written file -> retry next poll
            print(f"[sgviz] map read failed: {ex}", file=sys.stderr)
            return
        self._map_hash = h
        res = float(meta.get("resolution", 0.05))
        ox, oy = (meta.get("origin") or [0.0, 0.0])[:2]
        H, W = pgm.shape
        tex = map_texture(pgm, meta)
        # viser image planes have row 0 at local -Y; PGM row 0 is max-y.
        tex = np.ascontiguousarray(np.flipud(tex))
        if self._map_handle is not None:
            self._map_handle.remove()
        self._map_handle = self.server.scene.add_image(
            "/map", tex, render_width=W * res, render_height=H * res, format="png",
            position=(ox + W * res / 2, oy + H * res / 2, -0.01),
            cast_shadow=False, receive_shadow=False, visible=bool(self.g_map.value))

    def apply(self, new: Scene) -> Diff:
        with self.lock:
            d = diff_scenes(self.scene, new)
            self.scene = new
            if not d.empty():
                with self.server.atomic():
                    for oid in d.removed:
                        self._remove_node(oid)
                    for oid in d.added + d.redraw:
                        self._draw_node(new.objects[oid])
                    moved = set(d.redraw) | set(d.removed) | set(d.added)
                    for k in d.edges_removed:
                        self._remove_edge(k)
                    redo = set(d.edges_added) | set(d.edges_changed)
                    redo |= {k for k in new.edges if k[0] in moved or k[1] in moved}
                    for k in redo:
                        self._draw_edge(new.edges[k])
                    if self.selected is not None and self.selected in moved:
                        self._draw_selection()
            if d.added or d.removed:
                self._set_dropdown()
            if self.selected is not None and (self.selected in d.changed or d.removed
                                              or d.edges_added or d.edges_removed
                                              or d.edges_changed):
                if self.selected not in new.objects:
                    self.selected = None
                    self._set_dropdown()
                    self._draw_selection()
                self._render_panel()
            self._draw_robot(new.robot_pose)
            self._update_map(new.grid)
            n_moved = sum(o.state == "moved" for o in new.objects.values())
            self.g_status.content = (
                f"`{self.dir}`  \nstamp **{new.stamp:.1f} s** | objects {len(new.objects)} "
                f"(moved {n_moved}) | edges {len(new.edges)}")
            if not self._camera_set and new.robot_pose is not None:
                x, y, _ = new.robot_pose
                self.server.initial_camera.position = (x - 3.0, y - 3.0, 4.0)
                self.server.initial_camera.look_at = (x + 1.0, y, 0.5)
                self._camera_set = True
            return d

    def poll_once(self) -> Optional[Diff]:
        path = os.path.join(self.dir, "scene.json")
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size, st.st_ino)
        if key == self._scene_mtime:
            return None
        try:
            new = load_scene(self.dir)
        except Exception as ex:  # mid-write / malformed: retry next poll
            print(f"[sgviz] scene.json load failed: {ex}", file=sys.stderr)
            return None
        self._scene_mtime = key
        return self.apply(new)

    def run(self):
        print(f"[sgviz] watching {self.dir}; open http://localhost:{self.server.get_port()}", flush=True)
        while True:
            try:
                self.poll_once()
                with self.lock:
                    self._refresh_points()
                    if self.layers.poll() and self.scene is not None:
                        self.g_status.content = self.g_status.content.split("  \ngraph:")[0] + "  \n" + self.layers.status()
            except Exception as ex:
                print(f"[sgviz] update error: {ex!r}", file=sys.stderr)
            time.sleep(self.poll)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", help="memory output dir (contains scene.json)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--poll", type=float, default=0.25, help="scene.json poll period [s]")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.dir):
        ap.error(f"not a directory: {args.dir}")
    viewer = SgViewer(args.dir, ip=args.host, port=args.port, poll=min(args.poll, 0.5))
    try:
        viewer.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
