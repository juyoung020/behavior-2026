"""렌더 장면의 기준 prim 경로 목록 (G3, native_loop 가 물리 행위자와 이름으로 짝지음).

    python export_render_anchors.py <렌더 덤프 폴더>   # 예: src/sim/engine/dumps/render_radio_rgbd

덤프 폴더의 scene.json(render_capture.py) 기준 prim 차례 그대로 <폴더>/rsc/anchors.txt 에 한 줄에 경로 하나를 쓴다.
결과는 덤프 폴더(git 밖)에만 둔다.
"""
import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    with open(os.path.join(d, "scene.json"), encoding="utf-8") as f:
        sc = json.load(f)
    paths = [a["path"] for a in sc["anchors"]]
    out = os.path.join(d, "rsc", "anchors.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"{out}: 기준 prim {len(paths)} (강체 {sum(1 for a in sc['anchors'] if a['rigid'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
