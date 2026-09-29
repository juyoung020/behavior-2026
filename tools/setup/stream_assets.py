# -*- coding: utf-8 -*-
"""BEHAVIOR-1K 에셋(31.5 GB zip)을 **zip 파일을 통째로 받지 않고** 풀면서 받는다.

왜: 공식 `download_behavior_1k_assets()` 는 zip 전체를 임시 폴더에 받은 뒤 풀고 지운다.
그래서 푸는 동안 zip(31.5 GB) + 푼 파일(34.7 GB) = 약 66 GB 가 동시에 필요하다.
이 스크립트는 HuggingFace 의 범위 요청(HTTP Range)으로 zip 안 파일을 하나씩 읽어 바로 쓴다.
최대 디스크 사용량 = 푼 크기(34.7 GB)뿐이다.

- `<대상>.partial` 에 풀고, 다 끝나면 `<대상>` 으로 이름을 바꾼다.
  (공식 함수는 폴더가 있기만 하면 '이미 설치됨' 으로 넘어가므로 반쯤 푼 폴더를 대상 이름으로 두면 안 된다.)
- 끊겨도 다시 실행하면 이미 다 쓴 파일(크기 일치)은 건너뛰고 이어서 받는다.
- 네트워크 오류는 zip 을 다시 열어 같은 파일부터 재시도한다.

사용: python stream_assets.py [--dest <datasets 폴더>] [--file behavior-1k-assets-3.9.0.zip]
"""
import argparse
import os
import shutil
import sys
import time
import zipfile

from huggingface_hub import HfFileSystem

REPO = "datasets/behavior-1k/zipped-datasets"
DEFAULT_DEST = r"C:\behavior-2026\BEHAVIOR-1K\datasets"


def open_zip(path):
    f = HfFileSystem().open(f"{REPO}/{path}", "rb", block_size=64 << 20, cache_type="readahead")
    return f, zipfile.ZipFile(f)


def safe_join(root, name):
    out = os.path.normpath(os.path.join(root, name))
    if not out.startswith(os.path.normpath(root) + os.sep):
        raise ValueError(f"zip 밖을 가리키는 경로: {name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--file", default="behavior-1k-assets-3.9.0.zip")
    ap.add_argument("--name", default="behavior-1k-assets", help="풀 폴더 이름")
    a = ap.parse_args()

    target = os.path.join(a.dest, a.name)
    part = target + ".partial"
    if os.path.isdir(target):
        print(f"이미 있다: {target}")
        return
    os.makedirs(part, exist_ok=True)

    f, z = open_zip(a.file)
    infos = sorted(z.infolist(), key=lambda i: i.header_offset)   # zip 안 순서대로 읽어야 미리읽기가 먹힌다
    total = sum(i.file_size for i in infos)
    print(f"{a.file}: 파일 {len(infos):,}개, 풀면 {total/1e9:.2f} GB -> {part}", flush=True)

    done = skipped = 0
    t0 = time.time()
    last = 0
    idx = 0
    tries = 0
    while idx < len(infos):
        i = infos[idx]
        try:
            out = safe_join(part, i.filename)
            if i.is_dir():
                os.makedirs(out, exist_ok=True)
            elif os.path.exists(out) and os.path.getsize(out) == i.file_size:
                skipped += i.file_size
            else:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(i) as src, open(out + ".tmp", "wb") as dst:
                    shutil.copyfileobj(src, dst, 8 << 20)
                os.replace(out + ".tmp", out)
            done += i.file_size
            idx += 1
            tries = 0
        except (OSError, zipfile.BadZipFile, EOFError, ValueError) as e:
            if isinstance(e, ValueError) and "zip 밖" in str(e):
                raise
            tries += 1
            if tries > 20:
                raise
            print(f"\n[재시도 {tries}] {i.filename}: {e}", flush=True)
            time.sleep(min(60, 5 * tries))
            try:
                f.close()
            except Exception:
                pass
            f, z = open_zip(a.file)
            continue
        if done - last >= 500e6 or idx == len(infos):
            last = done
            el = time.time() - t0
            got = done - skipped
            rate = got / el / 1e6 if el > 0 else 0
            eta = (total - done) / (got / el) / 60 if got > 0 else 0
            print(f"\r{done/1e9:6.2f} / {total/1e9:.2f} GB  ({idx:,}/{len(infos):,})  "
                  f"{rate:5.1f} MB/s  남은 약 {eta:4.0f}분   ", end="", flush=True)

    f.close()
    os.replace(part, target)
    print(f"\n끝: {target}  ({(time.time()-t0)/60:.1f}분)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
