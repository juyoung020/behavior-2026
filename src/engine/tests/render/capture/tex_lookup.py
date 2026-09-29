"""재질 텍스처 경로 찾기 (변환기 공용). OmniGibson 은 암호화 USD 를 임시 폴더에 풀어서 읽기 때문에 재질의 텍스처 경로
(`../../material/<모델>__<링크>__diffuse.png` 같은 상대 경로)가 덤프 때 풀린 경로(resolvedPath)로 남지 않는 경우가 많다.
텍스처 PNG 는 암호화돼 있지 않고 에셋 폴더 `objects/<분류>/<모델 id>/material/` 에 있으므로, 파일 이름 앞부분(모델 id)으로 찾는다.
못 찾으면 로봇 에셋 폴더를 한 번 훑은 이름표에서 찾는다.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "BEHAVIOR-1K", "datasets"))
_model_dir = None
_name_index = None


def _build():
    global _model_dir, _name_index
    _model_dir, _name_index = {}, {}
    objs = os.path.join(DATASETS, "behavior-1k-assets", "objects")
    if os.path.isdir(objs):
        for cat in os.scandir(objs):
            if cat.is_dir():
                for mdl in os.scandir(cat.path):
                    if mdl.is_dir():
                        _model_dir[mdl.name] = mdl.path
    for root, _, files in os.walk(os.path.join(DATASETS, "omnigibson-robot-assets")):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".exr")):
                _name_index.setdefault(fn, os.path.join(root, fn))


def find_texture(asset):
    """asset: {"asset": 원래 경로, "resolved": 풀린 경로} 또는 문자열 -> 있는 파일 경로 또는 None"""
    if not asset:
        return None
    res, path = (asset.get("resolved") or "", asset.get("asset") or "") if isinstance(asset, dict) else ("", str(asset))
    if res and os.path.isfile(res):
        return res
    if path and os.path.isfile(path):
        return path
    base = os.path.basename(path.replace("\\", "/"))
    if not base:
        return None
    if _model_dir is None:
        _build()
    d = _model_dir.get(base.split("__")[0])
    if d:
        for sub in ("material", ""):
            p = os.path.join(d, sub, base)
            if os.path.isfile(p):
                return p
    return _name_index.get(base)
