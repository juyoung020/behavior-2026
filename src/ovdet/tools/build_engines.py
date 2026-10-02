#!/usr/bin/env python3
"""YOLOE ONNX (export_yoloe.py) -> TensorRT FP16 plan (4 GiB workspace), <onnx>.names.txt copied next to it.
GPU: run under the shared GPU lock (src/engine/scripts/gpu_lock.sh).

  ~/ovdet_venv/bin/python build_engines.py ~/ovdet_models/onnx/yoloe-11l-all.onnx ... --out ~/ovdet_models/x86_sm120
"""
import argparse
import shutil
import time
from pathlib import Path

import tensorrt as trt


def build(onnx_path, out_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise SystemExit('ONNX parse failed: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    for t in [network.get_input(i) for i in range(network.num_inputs)] +              [network.get_output(i) for i in range(network.num_outputs)]:
        print(f'  {t.name} {tuple(t.shape)}')
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit(f'engine build failed: {onnx_path}')
    out_path.write_bytes(bytes(blob))
    print(f'  -> {out_path} ({out_path.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx', nargs='+')
    ap.add_argument('--out', default=str(Path.home() / 'ovdet_models' / 'x86_sm120'))
    a = ap.parse_args()
    for o in map(Path, a.onnx):
        dst = Path(a.out) / (o.stem + '.plan')
        print(f'{o} -> {dst}', flush=True)
        build(o, dst)
        names = Path(str(o) + '.names.txt')
        if names.exists():
            shutil.copy(names, str(dst) + '.names.txt')


if __name__ == '__main__':
    main()
