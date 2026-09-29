"""π0.5 학습 데이터 파이프라인 측정 (docs/학습환경_가속.md 1절).

WSL ~/openpi 환경에서 돌린다 (tools/ft_run.sh 가 PYTHONPATH·환경변수를 맞춘다):
    bash tools/ft_run.sh tools/ft_bench.py stages  --n 64            # 샘플 하나를 구간별로 (한 프로세스)
    bash tools/ft_run.sh tools/ft_bench.py loader  --workers 0,1,2,4 --batches 12   # 원래 로더 처리량
    bash tools/ft_run.sh tools/ft_bench.py fast    --batches 30      # 가속 로더 처리량
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= bash tools/ft_run.sh tools/ft_bench.py cpu   # 가속판 워커 쪽(CPU)만

구간 정의 (원래 코드 위치):
  행 읽기       hf_dataset[idx] + 행동 32개 + 영상 시각 (lerobot dataset_reader.py:292-306)
  파일 읽기     영상 디코딩에 필요한 압축 바이트(직전 키프레임~목표)를 디스크 캐시에서 읽는 시간만 따로 잰 것
  RGB/깊이 디코딩  카메라별로 하나씩 순서대로 잰 값 (원래는 6개를 스레드로 동시에, dataset_reader.py:281-283)
  변환         openpi 변환 하나하나 (data_loader.py:150-153, 214-222)
  묶기         np.stack 32개 (data_loader.py:527-531)
  GPU 전송     jax.make_array_from_process_local_data + Observation.from_dict (data_loader.py:522, 594-596)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def now():
    return time.perf_counter()


class GpuMem:
    """nvidia-smi 로 GPU 전체 사용량을 0.3 초마다 읽어 최대값을 기록 (다른 프로세스 몫 포함 → 시작값을 뺀다)."""

    def __init__(self):
        self.base = self._read()
        self.peak = self.base
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    @staticmethod
    def _read():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return -1

    def _loop(self):
        while not self._stop:
            self.peak = max(self.peak, self._read())
            time.sleep(0.3)

    def stop(self):
        self._stop = True
        self._t.join()
        return self.peak - self.base


def packet_bytes(path, target_ts):
    """목표 프레임을 풀려면 읽어야 하는 압축 바이트 범위 (직전 키프레임부터 목표까지) — 파일 읽기 시간 측정용."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(target_ts * av.time_base), backward=True)
        rng = []
        for p in c.demux(s):
            if p.pts is None:
                continue
            rng.append((p.pos, p.size))
            if float(p.pts * s.time_base) >= target_ts - 1e-4:
                break
    return rng


def cmd_stages(a):
    import torch
    import jax
    import openpi.models.model as _model
    from openpi.training import data_loader as DL
    import lerobot.datasets.dataset_reader as R
    from lerobot.datasets.depth_utils import dequantize_depth
    from fasttrain import orig

    cfg = orig.train_config()
    ds, dc = orig.dataset(cfg)
    lds = orig.lerobot_of(ds)
    reader = lds.reader
    meta = lds.meta
    tfs = orig.transform_list(dc)
    print(f"샘플 수 {len(ds)}, 영상 키 {len(meta.video_keys)}개 (깊이 {len(meta.depth_keys)}개), 백엔드 {reader._video_backend}")
    rng = random.Random(a.seed)
    idxs = [rng.randrange(len(ds)) for _ in range(a.n + a.warm)]
    T = defaultdict(float)
    nbytes = defaultdict(int)
    samples = []
    for it, idx in enumerate(idxs):
        rec = it >= a.warm
        t0 = now()
        item = reader.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()
        q, pad = reader._get_query_indices(abs_idx, ep_idx)
        qr = reader._query_hf_dataset(q)
        item = {**item, **pad}
        item.update(qr)
        cur = item["timestamp"].item()
        qts = reader._get_query_timestamps(cur, q)
        t1 = now()
        if rec:
            T["행 읽기(parquet→torch, 행동 32개)"] += t1 - t0
        ep = meta.episodes[ep_idx]
        frames = {}
        for key, ts in qts.items():
            is_depth = key in meta.depth_keys
            path = reader.root / meta.get_video_file_path(ep_idx, key)
            shifted = [ep[f"videos/{key}/from_timestamp"] + t for t in ts]
            if rec and a.read_bytes:
                rngs = packet_bytes(str(path), shifted[0])
                fd = os.open(path, os.O_RDONLY)
                tr = now()
                for pos, size in rngs:
                    os.pread(fd, size, pos)
                T["파일 읽기(압축 바이트, 캐시)"] += now() - tr
                os.close(fd)
                nbytes["깊이" if is_depth else "RGB"] += sum(s for _, s in rngs)
            td = now()
            fr = R.decode_video_frames(path, shifted, reader._tolerance_s, reader._video_backend,
                                       return_uint8=reader._return_uint8, is_depth=is_depth)
            if is_depth:
                enc = reader._depth_encoder_configs[key]
                fr = dequantize_depth(fr, depth_min=enc.depth_min, depth_max=enc.depth_max, shift=enc.shift,
                                      use_log=enc.use_log, output_unit=enc.output_unit, output_tensor=True)
            frames[key] = fr.squeeze(0)
            if rec:
                cam = key.split(".")[-1].replace("_link_camera_0", "").replace("_realsense", "")
                T[f"{'깊이' if is_depth else 'RGB'} 디코딩 {cam}"] += now() - td
        t2 = now()
        vw = reader._query_videos(qts, ep_idx)  # 원래 방식 (6개 스레드 동시) 벽시계
        if rec:
            T["(참고) 영상 6개 원래 방식 동시 디코딩 벽시계"] += now() - t2
        item = {**frames, **item}
        item["task"] = meta.tasks.iloc[item["task_index"].item()].name
        x = item
        for tf in tfs:
            tt = now()
            x = tf(x)
            if rec:
                T[f"변환 {type(tf).__name__}"] += now() - tt
        # 기준 경로와 같은지 (계측이 원래와 다른 일을 하지 않았는지)
        if it == a.warm:
            ref = ds[idx]
            same = all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in
                       zip(jax.tree.leaves(ref), jax.tree.leaves(x)))
            print(f"계측 경로 == ds[idx]: {same}")
        if rec:
            samples.append(x)
    n = a.n
    print(f"\n샘플 {n}개 평균 (한 프로세스, 무작위 인덱스 seed={a.seed})")
    tot = 0.0
    for k, v in T.items():
        ms = 1e3 * v / n
        if not k.startswith("(참고)"):
            tot += ms
        print(f"  {k:<44}{ms:9.2f} ms")
    print(f"  {'합 (디코딩은 순차 합)':<44}{tot:9.2f} ms")
    if nbytes:
        print(f"  읽은 압축 바이트/샘플: RGB {nbytes['RGB']/n/1e3:.0f} kB, 깊이 {nbytes['깊이']/n/1e3:.0f} kB")
    # 묶기 + GPU 전송 (배치 32)
    B = 32
    batch_items = (samples * ((B + len(samples) - 1) // len(samples)))[:B]
    tc = now()
    for _ in range(5):
        batch = DL._collate_fn(batch_items)
    tcol = (now() - tc) / 5
    sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("B",)), jax.sharding.PartitionSpec("B"))

    def to_dev(b):
        arr = jax.tree.map(lambda x: jax.make_array_from_process_local_data(sharding, x), b)
        obs = _model.Observation.from_dict(arr)
        jax.block_until_ready((obs, arr["actions"]))
        return obs

    to_dev(batch)
    tg = now()
    for _ in range(5):
        to_dev(batch)
    tgpu = (now() - tg) / 5
    mb = sum(np.asarray(v).nbytes for v in jax.tree.leaves(batch)) / 1e6
    print(f"\n배치 {B}: 묶기 {1e3*tcol:.2f} ms, GPU 전송+uint8→float {1e3*tgpu:.2f} ms ({mb:.1f} MB)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"per_sample_ms": {k: 1e3 * v / n for k, v in T.items()}, "collate_ms": 1e3 * tcol,
                       "to_gpu_ms": 1e3 * tgpu, "batch_mb": mb,
                       "bytes": {k: v / n for k, v in nbytes.items()}}, f, ensure_ascii=False, indent=1)


def run_loader(it_factory, batches, warm):
    import jax

    it = iter(it_factory())
    t0 = now()
    for _ in range(warm):
        jax.block_until_ready(next(it))
    t1 = now()
    for _ in range(batches):
        jax.block_until_ready(next(it))
    t2 = now()
    return t1 - t0, (t2 - t1) / batches


def cmd_loader(a):
    import dataclasses
    from fasttrain import orig

    res = []
    for w in [int(x) for x in a.workers.split(",")]:
        cfg = orig.train_config(num_workers=w, batch_size=a.batch)
        mem = GpuMem()
        # 워커마다 2배치씩 미리 만들어 두므로(prefetch_factor 2) 그만큼 먼저 빼고 잰다 — 안 그러면 버퍼를 재게 된다.
        warm = a.warm + 2 * w
        batches = max(a.batches, 2 * w)
        startup, per = run_loader(lambda: orig.loader(cfg, shuffle=True), batches, warm)
        dmem = mem.stop()
        r = dict(workers=w, batch=a.batch, sec_per_batch=per, samples_per_s=a.batch / per,
                 batches_per_s=1 / per, startup_s=startup, gpu_mem_mb=dmem,
                 load=os.getloadavg()[0])
        res.append(r)
        print(f"워커 {w:2d}: {r['samples_per_s']:7.1f} 샘플/s, {r['batches_per_s']:5.2f} 배치/s "
              f"(배치 {a.batch}, 첫 {warm}배치 {startup:.1f}s 제외, {batches}배치 측정), GPU 메모리 +{dmem} MB, load {r['load']:.1f}",
              flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


def cmd_fast(a):
    """가속 로더 처리량. --step-ms 를 주면 학습 스텝을 흉내내(GPU 행렬곱을 그 시간만큼) 소비 쪽이 데이터를 기다린
    시간(= GPU 가 데이터를 기다린 시간)을 잰다."""
    import jax
    import jax.numpy as jnp
    from fasttrain import fast

    res = []
    for w in [int(x) for x in a.workers.split(",")]:
        for th in [int(x) for x in str(a.threads).split(",")]:
            mem = GpuMem()
            t0 = now()
            ld = fast.loader(num_workers=w, batch_size=a.batch, shuffle=True, decode_threads=th)
            it = iter(ld)
            warm, nb = a.warm + 2 * w, max(a.batches, 2 * w)
            for _ in range(warm):
                jax.block_until_ready(next(it))
            startup = now() - t0
            eng0 = ld.engine.stats()
            x = jnp.ones((2048, 2048), jnp.float32)
            step = jax.jit(lambda x: (x @ x) * 1e-3)
            jax.block_until_ready(step(x))
            wait = 0.0
            t1 = now()
            for _ in range(nb):
                s = now()
                b = next(it)
                jax.block_until_ready(b)
                wait += now() - s
                if a.step_ms:  # 학습 스텝 흉내: GPU 를 step_ms 동안 바쁘게
                    e = now() + a.step_ms / 1e3
                    while now() < e:
                        x = jax.block_until_ready(step(x))
            per = (now() - t1) / nb
            eng1 = ld.engine.stats()
            dmem = mem.stop()
            frames = (eng1[1] - eng0[1]) / max(1.0, eng1[0] - eng0[0])
            r = dict(workers=w, decode_threads=th, batch=a.batch, step_ms=a.step_ms, sec_per_batch=per,
                     samples_per_s=a.batch / per, batches_per_s=1 / per, wait_ms_per_batch=1e3 * wait / nb,
                     startup_s=startup, gpu_mem_mb=dmem, frames_per_request=frames, load=os.getloadavg()[0])
            res.append(r)
            print(f"[가속] 워커 {w:2d} 디코딩 스레드 {th:2d}: {r['samples_per_s']:7.1f} 샘플/s, "
                  f"{r['batches_per_s']:5.2f} 배치/s, 스텝 {a.step_ms} ms 일 때 데이터 대기 {r['wait_ms_per_batch']:.1f} ms/배치, "
                  f"요청당 디코딩 {frames:.2f} 프레임, GPU 메모리 +{dmem} MB, load {r['load']:.1f}", flush=True)
            del it, ld
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


def cmd_cpu(a):
    """가속판의 워커 쪽(CPU)만: 샘플 구간 시간과 워커 수별 CPU 배치 처리량. GPU 는 전혀 안 쓴다."""
    import torch
    from openpi.training import data_loader as DL
    from fasttrain import fast, orig

    cfg = orig.train_config()
    fds = fast.FastDataset(cfg)
    reader, meta = fds.lds.reader, fds.lds.meta
    rng = random.Random(a.seed)
    idxs = [rng.randrange(len(fds)) for _ in range(a.n + 4)]
    T = defaultdict(float)
    for it, idx in enumerate(idxs):
        rec = it >= 4
        t0 = now()
        item = reader.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        q, pad = reader._get_query_indices(item["index"].item(), ep_idx)
        qr = reader._query_hf_dataset(q)
        t1 = now()
        x = fds[idx]
        t2 = now()
        if rec:
            T["행 읽기(원래와 같음)"] += t1 - t0
            T["샘플 전체(행 읽기+프레임 번호+변환)"] += t2 - t1
    for k, v in T.items():
        print(f"  {k:<40}{1e3 * v / a.n:8.2f} ms/샘플")
    # 변환 하나하나
    tt = defaultdict(float)
    for idx in idxs[4:]:
        item = reader.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        q, pad = reader._get_query_indices(item["index"].item(), ep_idx)
        item = {**item, **pad, **reader._query_hf_dataset(q)}
        for key in fds.cams:
            item[key] = fast._placeholder
        item["task"] = meta.tasks.iloc[item["task_index"].item()].name
        x = item
        for tf in fds.tfs:
            s = now()
            x = tf(x)
            tt[type(tf).__name__] += now() - s
    for k, v in tt.items():
        print(f"    변환 {k:<34}{1e3 * v / a.n:8.2f} ms/샘플")
    import multiprocessing

    res = []
    for w in [int(x) for x in a.workers.split(",")]:
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        dl = torch.utils.data.DataLoader(fds, batch_size=a.batch, shuffle=True, num_workers=w,
                                         multiprocessing_context=multiprocessing.get_context("spawn") if w else None,
                                         persistent_workers=w > 0, collate_fn=DL._collate_fn, drop_last=True,
                                         generator=g)
        it = iter(dl)
        warm, nb = 2 + 2 * w, max(a.batches, 2 * w)
        for _ in range(warm):
            next(it)
        t = now()
        for _ in range(nb):
            next(it)
        per = (now() - t) / nb
        r = dict(workers=w, samples_per_s=a.batch / per, batches_per_s=1 / per, load=os.getloadavg()[0])
        res.append(r)
        print(f"[가속·CPU 쪽] 워커 {w:2d}: {r['samples_per_s']:8.1f} 샘플/s ({r['batches_per_s']:.2f} 배치/s), "
              f"load {r['load']:.1f}", flush=True)
        del it, dl
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"per_sample_ms": {k: 1e3 * v / a.n for k, v in T.items()},
                       "transforms_ms": {k: 1e3 * v / a.n for k, v in tt.items()}, "loader": res}, f,
                      ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stages")
    s.add_argument("--n", type=int, default=64)
    s.add_argument("--warm", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--read-bytes", type=int, default=1)
    s.add_argument("--json", default="")
    s = sp.add_parser("loader")
    s.add_argument("--workers", default="0,1,2,4")
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--batches", type=int, default=10)
    s.add_argument("--warm", type=int, default=2)
    s.add_argument("--json", default="")
    s = sp.add_parser("cpu")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", default="1,2,4,8")
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--batches", type=int, default=10)
    s.add_argument("--json", default="")
    s = sp.add_parser("fast")
    s.add_argument("--workers", default="2")
    s.add_argument("--threads", default="6", help="디코딩 스레드 수(쉼표로 여러 개)")
    s.add_argument("--step-ms", type=float, default=0.0, help="학습 스텝 흉내 시간")
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--batches", type=int, default=30)
    s.add_argument("--warm", type=int, default=3)
    s.add_argument("--json", default="")
    a = ap.parse_args()
    {"stages": cmd_stages, "loader": cmd_loader, "fast": cmd_fast, "cpu": cmd_cpu}[a.cmd](a)


if __name__ == "__main__":
    main()
