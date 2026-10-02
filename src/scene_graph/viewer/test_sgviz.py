#!/usr/bin/env python3
"""Headless test of sgviz loading/diff code (no browser, no viser server).

    ~/sdsg_venv/bin/python src/scene_graph/viewer/test_sgviz.py
"""

import os
import sys
import tempfile

import numpy as np
import spark_dsg as dsg
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgviz  # noqa: E402


def _obj(name, pos, dims, md):
    a = dsg.ObjectNodeAttributes()
    a.name = name
    a.position = np.array(pos, float)
    a.bounding_box = dsg.BoundingBox(np.array(dims, np.float32), np.array(pos, np.float32))
    a.last_update_time_ns = int(12.5e9)
    a.metadata.set(md)
    return a


def write_dir(d, cup_pos=(1.0, 0.5, 0.8), with_edge=True):
    os.makedirs(os.path.join(d, "objects"), exist_ok=True)
    G = dsg.DynamicSceneGraph()
    G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 1),
               _obj("table", (1.0, 0.5, 0.4), (1.2, 0.8, 0.8),
                    {"state": "seen", "n_obs": 40, "score": 0.8, "first_pos": [1.0, 0.5, 0.4],
                     "structural": True, "handled": False}))
    G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 2),
               _obj("cup", cup_pos, (0.08, 0.08, 0.1),
                    {"state": "moved", "n_obs": 7, "score": 0.6, "first_pos": [0.2, 0.1, 0.8],
                     "structural": False, "handled": False,
                     "rgbd": {"rgb": "objects/O2_rgb.png", "depth": "objects/O2_depth.png",
                              "stamp": 12.4, "box_px": [10, 12, 30, 40], "mask_area": 410,
                              "depth_m": 1.234, "cam_T": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}}))
    G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 3),
               _obj("floor", (0.0, 0.0, 0.0), (30.0, 20.0, 0.5),
                    {"state": "gone", "n_obs": 3, "score": 0.3, "first_pos": [0, 0, 0],
                     "structural": True}))
    if with_edge:
        e = dsg.EdgeAttributes()
        e.metadata.set({"relation": "on"})
        G.insert_edge(dsg.NodeSymbol("O", 2), dsg.NodeSymbol("O", 1), e)
    G.metadata.set({"stamp": 12.5, "robot_pose": [0.1, -0.2, 0.5], "grid": "map.pgm"})
    G.save(os.path.join(d, "scene.json"))

    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[..., 0] = 200
    Image.fromarray(rgb).save(os.path.join(d, "objects", "O2_rgb.png"))
    depth = (np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) % 3000) + 500
    depth[0, :] = 0
    Image.fromarray(depth).save(os.path.join(d, "objects", "O2_depth.png"))

    pgm = np.full((20, 30), 205, np.uint8)
    pgm[5:15, 5:25] = 254
    pgm[10, :] = 0
    Image.fromarray(pgm).save(os.path.join(d, "map.pgm"))
    with open(os.path.join(d, "map.yaml"), "w") as f:
        f.write("image: map.pgm\nresolution: 0.05\norigin: [-0.5, -0.5, 0.0]\nnegate: 0\n"
                "occupied_thresh: 0.65\nfree_thresh: 0.25\n")


def main():
    with tempfile.TemporaryDirectory() as d:
        write_dir(d)
        sc = sgviz.load_scene(d)
        assert set(sc.objects) == {1, 2, 3}, sc.objects.keys()
        assert abs(sc.stamp - 12.5) < 1e-9 and sc.robot_pose == (0.1, -0.2, 0.5)
        assert sc.grid == "map.pgm"
        cup = sc.objects[2]
        assert cup.name == "cup" and cup.label == "cup#2" and cup.sym == "O2"
        assert cup.state == "moved" and cup.n_obs == 7 and abs(cup.score - 0.6) < 1e-9
        assert cup.first_pos == (0.2, 0.1, 0.8) and abs(cup.last_seen - 12.5) < 1e-6
        assert cup.rgbd["box_px"] == [10, 12, 30, 40] and cup.rgbd["depth_m"] == 1.234
        assert sc.objects[1].rgbd is None and sc.objects[1].structural
        assert list(sc.edges) == [(2, 1)] and sc.edges[(2, 1)].relation == "on"
        assert sgviz.relations_of(sc, 2) == ["on table#1"]
        assert sgviz.relations_of(sc, 1) == ["cup#2 on this"]

        # boxes: real one kept, absurd floor clamped
        seg = sgviz._box_segments(sc.objects[2], 2.0)
        assert seg.shape == (12, 2, 3)
        fl = sgviz._box_segments(sc.objects[3], 2.0)
        assert np.ptp(fl[..., 0]) <= 2.0 + 1e-5 and np.ptp(fl[..., 1]) <= 2.0 + 1e-5

        # images (lazy cache, crop + depth colorize)
        cache = sgviz.ImageCache()
        rgb = cache.get(os.path.join(d, cup.rgbd["rgb"]), "rgb", cup.rgbd["box_px"])
        dep = cache.get(os.path.join(d, cup.rgbd["depth"]), "depth", cup.rgbd["box_px"])
        assert rgb.shape == (44, 36, 3) and rgb.dtype == np.uint8, rgb.shape  # 20x28 + 8 margin
        assert dep.shape == (44, 36, 3) and dep.dtype == np.uint8 and dep.max() > 0
        assert cache.get(os.path.join(d, "objects/missing.png"), "rgb", None) is None

        # map
        meta = sgviz.read_map_yaml(os.path.join(d, "map.yaml"))
        assert meta["resolution"] == 0.05 and meta["origin"] == [-0.5, -0.5, 0.0]
        tex = sgviz.map_texture(np.array(Image.open(os.path.join(d, "map.pgm"))), meta)
        assert tuple(tex[10, 0]) == (0, 0, 0) and tuple(tex[6, 6]) == (254, 254, 254)
        assert tuple(tex[0, 0]) == (150, 150, 165)

        # diff: identical -> empty; move cup + drop edge -> redraw cup, edge removed
        assert sgviz.diff_scenes(sc, sgviz.load_scene(d)).empty()
        d0 = sgviz.diff_scenes(None, sc)
        assert sorted(d0.added) == [1, 2, 3] and d0.edges_added == [(2, 1)]
        write_dir(d, cup_pos=(1.5, 0.5, 0.8), with_edge=False)
        sc2 = sgviz.load_scene(d)
        df = sgviz.diff_scenes(sc, sc2)
        assert df.redraw == [2] and df.changed == [2] and not df.added and not df.removed, df
        assert df.edges_removed == [(2, 1)]

        # no rgbd / no edges / no metadata at all must not crash
        G = dsg.DynamicSceneGraph()
        a = dsg.ObjectNodeAttributes()
        a.position = np.zeros(3)
        G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 9), a)
        s3 = sgviz.scene_from_graph(G)
        assert s3.objects[9].state == "seen" and s3.objects[9].bbox_dims is None
        assert s3.robot_pose is None and not s3.edges

        server_check(d)

    print("test_sgviz: OK")


def server_check(d):
    """Run the real viewer (viser server, no browser) on the synthetic dir."""
    import socket
    import urllib.request

    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    v = sgviz.SgViewer(d, ip="127.0.0.1", port=port)
    try:
        write_dir(d)  # back to the version with the edge
        df = v.poll_once()
        assert sorted(df.added) == [1, 2, 3] and df.edges_added == [(2, 1)]
        assert v.poll_once() is None  # unchanged mtime -> no work
        assert set(v._node_handles) == {1, 2, 3} and set(v._edge_handles) == {(2, 1)}
        assert "cup#2" in v.g_select.options
        v.select(2)
        assert v.g_select.value == "cup#2"
        txt = v.g_info.content
        for frag in ("cup", "moved", "n_obs | 7", "first_pos", "on table#1", "box_px", "1.234"):
            assert frag in txt, (frag, txt)
        assert v.g_rgb.visible and v.g_depth.visible
        v.select(1)
        assert not v.g_rgb.visible  # table has no rgbd
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert r.status == 200
    finally:
        v.server.stop()


if __name__ == "__main__":
    main()
