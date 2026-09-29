# -*- coding: utf-8 -*-
"""로봇 에셋 · 2026 과제 인스턴스 · 에셋 복호화 키를 받는다 (공식 함수 그대로).

큰 에셋(behavior-1k-assets)은 stream_assets.py 가 따로 받는다. 키는 그 에셋(암호화된 USD)을 여는 데 필요하다.
키를 받는 것은 BEHAVIOR 데이터 라이선스에 동의하는 것과 같다 (asset_utils.print_user_agreement).
behavior 환경에서 실행: C:\\Users\\user one\\anaconda3\\envs\\behavior\\python.exe fetch_small_datasets.py
"""
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from omnigibson.utils import asset_utils as au  # noqa: E402

print("DATA_PATH:", au.gm.DATA_PATH, flush=True)
au.download_omnigibson_robot_assets()
print("robot assets OK", flush=True)
if not os.path.exists(au.get_key_path()):
    au.download_key()
print("key:", os.path.exists(au.get_key_path()), flush=True)
au.download_2026_challenge_task_instances()
print("=== fetch_small_datasets done ===", flush=True)
