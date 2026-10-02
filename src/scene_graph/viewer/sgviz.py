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
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        return f"{self.name}#{self.id}"

    def draw_sig(self):
        """Fields that affect the 3D drawing."""
        return (self.name, self.pos, self.state, self.structural, self.is_active,
                self.first_pos, self.bbox_dims, self.bbox_center)

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


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """uint16 mm (or float m) depth -> RGB uint8 (turbo-like ramp, 0 = black)."""
    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    valid = d > 0
    rgb = np.zeros(d.shape + (3,), np.uint8)
    if not valid.any():
        return rgb
    lo, hi = np.percentile(d[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    # piecewise "jet": near = red, far = blue
    t = 1.0 - t
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    rgb[..., 0] = (r * 255).astype(np.uint8)
    rgb[..., 1] = (g * 255).astype(np.uint8)
    rgb[..., 2] = (b * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def maybe_crop(img: np.ndarray, box_px, margin: int = 8) -> np.ndarray:
    """Crop to box_px if the stored image is a full frame (box fits well inside)."""
    if not box_px or len(box_px) != 4:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box_px)
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


def _fit(img: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    step = int(math.ceil(s / max_side))
    return img[::step, ::step]


class ImageCache:
    """Lazy (path, mtime)-keyed cache of display-ready crops."""

    def __init__(self, maxlen: int = 64):
        self._d: Dict[Tuple[str, float, str, str], np.ndarray] = {}
        self._maxlen = maxlen

    def get(self, path: str, kind: str, box_px) -> Optional[np.ndarray]:
        try:
            mt = os.stat(path).st_mtime
        except OSError:
            return None
        key = (path, mt, kind, repr(box_px))
        if key in self._d:
            return self._d[key]
        try:
            img = _read_image(path)
        except Exception:
            return None
        img = maybe_crop(img, box_px)
        if kind == "depth":
            img = colorize_depth(img)
        elif img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2).astype(np.uint8)
        img = _fit(np.ascontiguousarray(img))
        if len(self._d) >= self._maxlen:
            self._d.pop(next(iter(self._d)))
        self._d[key] = img
        return img


# --------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------


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
        self.images = ImageCache()
        self._scene_mtime = None
        self._map_hash = None
        self._map_handle = None
        self._robot = None
        self._node_handles: Dict[int, List[Any]] = {}
        self._edge_handles: Dict[Tuple[int, int], List[Any]] = {}
        self._sel_handle = None
        self._camera_set = False
        self._suppress_dropdown = False
        self._build_gui()

    # ---------------- GUI ----------------
    def _build_gui(self):
        gui = self.server.gui
        self.g_status = gui.add_markdown("waiting for scene.json ...")
        with gui.add_folder("Display", expand_by_default=False):
            self.g_labels = gui.add_checkbox("Labels", True)
            self.g_boxes = gui.add_checkbox("Bounding boxes", True)
            self.g_edges = gui.add_checkbox("Support edges", True)
            self.g_trails = gui.add_checkbox("Moved trails", True)
            self.g_struct = gui.add_checkbox("Structural objects", True)
            self.g_gone = gui.add_checkbox("Gone objects", True)
            self.g_map = gui.add_checkbox("Occupancy map", True)
            self.g_size = gui.add_number("Node radius", 0.06, min=0.01, max=0.5, step=0.01)
            self.g_maxbox = gui.add_number("Max box side [m]", 2.0, min=0.1, max=20.0, step=0.1)
        for h in (self.g_labels, self.g_boxes, self.g_edges, self.g_trails,
                  self.g_struct, self.g_gone, self.g_size, self.g_maxbox):
            h.on_update(lambda _: self._redraw_all())
        self.g_map.on_update(lambda _: self._map_visibility())

        with gui.add_folder("Object", expand_by_default=True):
            self.g_select = gui.add_dropdown("Select", ["(none)"], initial_value="(none)")
            self.g_info = gui.add_markdown("Click a node or pick one above.")
            blank = np.zeros((2, 2, 3), np.uint8)
            self.g_rgb = gui.add_image(blank, label="RGB crop", visible=False)
            self.g_depth = gui.add_image(blank, label="Depth crop", visible=False)
        self.g_select.on_update(lambda _: self._on_dropdown())

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
            self.selected = oid
            if not from_dropdown:
                self._set_dropdown()
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
        rgbd = o.rgbd or {}
        if rgbd:
            lines.append(f"| rgbd stamp | {rgbd.get('stamp', '-')} |")
            lines.append(f"| box_px | {rgbd.get('box_px', '-')} |")
            dm = rgbd.get("depth_m")
            lines.append(f"| depth_m | {dm:.3f} |" if isinstance(dm, (int, float)) else f"| depth_m | {dm} |")
            if "mask_area" in rgbd:
                lines.append(f"| mask_area | {rgbd.get('mask_area')} |")
        self.g_info.content = "\n".join(lines)

        box = rgbd.get("box_px")
        for kind, handle in (("rgb", self.g_rgb), ("depth", self.g_depth)):
            rel = rgbd.get(kind)
            img = self.images.get(os.path.join(self.dir, rel), kind, box) if rel else None
            if img is None:
                handle.visible = False
            else:
                handle.image = img
                handle.visible = True

    # ---------------- drawing ----------------
    def _visible(self, o: ObjInfo) -> bool:
        if o.structural and not self.g_struct.value:
            return False
        if o.state == "gone" and not self.g_gone.value:
            return False
        return True

    def _remove_node(self, oid: int):
        for h in self._node_handles.pop(oid, []):
            try:
                h.remove()
            except Exception:
                pass

    def _draw_node(self, o: ObjInfo):
        self._remove_node(o.id)
        if not self._visible(o):
            return
        sc = self.server.scene
        base = f"/objects/O{o.id}"
        col = STATE_COLORS.get(o.state, DEFAULT_COLOR)
        opacity = 0.35 if o.state == "gone" else None
        r = float(self.g_size.value)
        hs: List[Any] = []
        sph = sc.add_icosphere(f"{base}/node", radius=r, color=col, position=o.pos,
                               subdivisions=2, opacity=opacity)
        sph.on_click(lambda _ev, oid=o.id: self.select(oid))
        hs.append(sph)
        if self.g_labels.value:
            hs.append(sc.add_label(f"{base}/label", o.label,
                                   position=(o.pos[0], o.pos[1], o.pos[2] + 1.8 * r),
                                   anchor="bottom-center", font_screen_scale=0.8))
        if self.g_boxes.value:
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
                hs.append(sc.add_icosphere(f"{base}/first", radius=0.5 * r, color=(180, 180, 180),
                                           position=tuple(fp), subdivisions=1, opacity=0.6))
        self._node_handles[o.id] = hs

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
