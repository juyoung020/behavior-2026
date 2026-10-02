# -*- coding: utf-8 -*-
"""behavior 환경에 Isaac Sim 5.1 휠을 설치하고 setup.ps1 의 나머지 단계를 마친다.

왜 따로 만들었나: setup.ps1 은 휠을 임시 폴더에 받고 pip 설치가 실패해도 이유를 남기지 않은 채
임시 폴더를 지운다(PowerShell 5.1 기록에는 pip 출력이 안 남는다). 여기서는
- 휠을 `C:\\behavior-2026\\cache\\isaac_wheels` 에 받아 두고(다시 돌려도 이미 받은 건 건너뜀),
- pip 출력을 그대로 보여 주며 설치한 뒤,
- setup.ps1 과 같은 후처리(websockets 프리번들 제거, cffi 1.17.1)와 JoyLo, 평가 의존성까지 한다.

반드시 behavior 환경의 python 으로 실행한다:
  C:\\Users\\user one\\anaconda3\\envs\\behavior\\python.exe install_isaacsim.py
"""
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\behavior-2026\BEHAVIOR-1K")
CACHE = Path(r"C:\behavior-2026\cache\isaac_wheels")
PY = sys.executable


def run(args, **kw):
    print(">", " ".join(map(str, args)), flush=True)
    r = subprocess.run(args, **kw)
    if r.returncode != 0:
        raise SystemExit(f"실패 (exit {r.returncode}): {' '.join(map(str, args))}")


def wheel_list():
    text = (ROOT / "setup.ps1").read_text(encoding="utf-8")
    block = re.search(r"\$packages = @\((.*?)\)", text, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def download(pkg):
    parts = pkg.split("-")
    name = "-".join(parts[:-1]).replace("_", "-")
    fn = f"{pkg}-cp311-none-win_amd64.whl"
    url = f"https://pypi.nvidia.com/{name}/{fn}"
    out = CACHE / fn
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        size = int(r.headers["Content-Length"])
    if out.exists() and out.stat().st_size == size:
        print(f"  있음 {size/1e9:6.2f} GB {fn}", flush=True)
        return out
    print(f"  받는 중 {size/1e9:6.2f} GB {fn}", flush=True)
    tmp = out.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size != size:
        raise SystemExit(f"크기 불일치: {fn}")
    tmp.replace(out)
    return out


def main():
    assert sys.version_info[:2] == (3, 11), "behavior 환경(파이썬 3.11)에서 실행해야 한다"
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    CACHE.mkdir(parents=True, exist_ok=True)

    pkgs = wheel_list()
    print(f"Isaac Sim 휠 {len(pkgs)}개", flush=True)
    files = [download(p) for p in pkgs]

    run([PY, "-m", "pip", "install", *map(str, files)])
    run([PY, "-c", "import isaacsim; print('isaacsim import OK')"])

    # setup.ps1 과 같은 후처리: extscache 안의 websockets 프리번들 제거
    isaac_path = subprocess.run([PY, "-c", "import isaacsim, os; print(os.environ.get('ISAAC_PATH', ''))"],
                                capture_output=True, text=True).stdout.strip().splitlines()
    isaac_path = isaac_path[-1] if isaac_path else ""
    if isaac_path and (Path(isaac_path) / "extscache").exists():
        for p in (Path(isaac_path) / "extscache").rglob("websockets"):
            if p.is_dir() and p.parent.name == "pip_prebundle":
                print("  websockets 프리번들 제거:", p, flush=True)
                import shutil
                shutil.rmtree(p, ignore_errors=True)
    run([PY, "-m", "pip", "install", "--force-reinstall", "cffi==1.17.1"])

    # JoyLo (-Eval 이 요구) 와 평가 의존성
    run([PY, "-m", "pip", "install", "-e", str(ROOT / "joylo")])
    tv = subprocess.run([PY, "-c", "import torch; print(torch.__version__)"],   # 예: 2.7.0+cu128 (setup.ps1 과 같다)
                        capture_output=True, text=True).stdout.strip()
    run([PY, "-m", "pip", "install", "torch-cluster", "-f", f"https://data.pyg.org/whl/torch-{tv}.html"])
    print("\n다음 단계(conda): conda install -n behavior av \"numpy<2\" -c conda-forge -y", flush=True)
    print("=== install_isaacsim 끝 ===", flush=True)


if __name__ == "__main__":
    main()
