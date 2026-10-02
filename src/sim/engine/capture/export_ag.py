"""보조 잡기(AG) 입력 내보내기: scope.json(physx_capture 의 robot.ag·ag_scene_objects) + s1_setup.txt -> ag_setup.txt
(ovd_replay --ag <폴더>, engine_capi 가 읽음). 원본 robot.py:835 _handle_assisted_grasping 이 쓰는 값들.

  python3 export_ag.py <기록 폴더>

ag_setup.txt (줄 단위)
  robot <로봇 prim 경로>                         자기 몸 걸러내기 (결과 경로에 이 문자열이 들어 있으면 버림)
  robotlink <링크 경로>                          robot.link_prim_paths (접촉 상대에서 뺌)
  arm <이름> <손끝 링크> <손가락1> <손가락2>
  grip <이름> <dof> <위쪽 한계 (s1_setup dof 줄의 pos_hi 문자열)>  (grasping_direction = lower: 제어값 < 위쪽 한계면 잡는 중)
  start <이름> <링크 경로> <x y z float32 비트>   광선 시작 점 (링크 틀)
  end <이름> <링크 경로> <x y z 비트>
  obj <물체 prim> <고정 바닥 0/1> <뿌리 링크 이름>
  link <물체 prim> <링크 이름> <링크 경로> <질량 비트|-> <동적 0/1> <고정 아닌 조상 관절 있음 0/1>
"""
import json
import os
import sys


def main():
    d = sys.argv[1]
    sc = json.load(open(os.path.join(d, "scope.json")))[0]
    r = sc["robot"]
    if "ag" not in r:
        raise SystemExit(f"scope.json 에 AG 입력이 없다 ({r.get('ag_error')}) — 새 기록 도구로 다시 뜰 것")
    links = r["links"] if isinstance(r["links"], dict) else {}
    s1 = {}
    for line in open(os.path.join(d, "s1_setup.txt")):
        k = line.split()
        if k and k[0] == "group":
            s1[k[1]] = [int(x) for x in k[2:]]
        if k and k[0] == "dof":
            s1.setdefault("dof", {})[int(k[1])] = k
    out = [f"robot {r['prim_path']}"]
    for lp in r.get("link_prim_paths", []):
        out.append(f"robotlink {lp}")

    def lpath(name):
        return links.get(name, f"{r['prim_path']}/{name}")

    for arm, a in r["ag"].items():
        out.append(f"arm {arm} {a['eef']} {a['fingers'][0]} {a['fingers'][1]}")
        for dof in s1.get(f"gripper_{arm}", []):
            hi = s1["dof"][dof][5]  # dof d link sign pos_lo pos_hi ... (문자열 그대로)
            out.append(f"grip {arm} {dof} {hi}")
        for kind in ("start", "end"):
            for ln, bits in a[kind]:
                out.append(f"{kind} {arm} {lpath(ln)} " + " ".join("%08x" % b for b in bits))
    for op, o in r.get("ag_scene_objects", {}).items():
        out.append(f"obj {op} {int(o['fixed_base'])} {o['root_link_name']}")
        # 조상 관절: nx.edge_dfs(tree, link, orientation='reverse') = 뿌리 쪽으로 올라가며 만나는 모든 관절
        parent = {}
        for u, v, jt in o.get("tree", []):
            parent[v] = (u, jt)
        for ln, l in o["links"].items():
            nonfixed, cur, seen = 0, ln, set()
            while cur in parent and cur not in seen:
                seen.add(cur)
                u, jt = parent[cur]
                if "FIXED" not in jt.upper():
                    nonfixed = 1
                cur = u
            m = "%08x" % l["mass_bits"] if l.get("mass_bits") is not None else "-"
            out.append(f"link {op} {ln} {l['prim_path']} {m} {int(l['dynamic'])} {nonfixed}")
    open(os.path.join(d, "ag_setup.txt"), "w").write("\n".join(out) + "\n")
    print(f"AG 입력: 팔 {len(r['ag'])}, 물체 {len(r.get('ag_scene_objects', {}))} -> {d}/ag_setup.txt")


if __name__ == "__main__":
    main()
