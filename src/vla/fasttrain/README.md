# fasttrain — native, torch-free data loader for π0.5 (openpi `pi05_b1k`) training

Drop-in replacement for openpi's `create_b1k_data_loader` that produces **bit-identical** batches
(same tensors, same shuffle order) while moving all hot-path work into C++/CUDA and Rust.
Full write-up (Korean): [`docs/학습환경_가속.md`](../../../docs/학습환경_가속.md).

## What replaces what

| original (openpi + LeRobot) | here |
|---|---|
| torch `DataLoader`, 8 spawn workers | one C++ producer thread + N NVDEC engine threads (`csrc/loader.cpp`, `csrc/nvdec.cpp`) |
| CPU HEVC decode of 6 videos (3 depth videos decoded and then dropped) | NVDEC via the cuvid driver API, RGB only, reading just the needed packets (`.ftidx` index from `ftprep index`) |
| swscale YUV→RGB, /255, ×255 | 2^24-entry colour LUT built by decoding a synthetic video with the *original* decoder (`lut.py`) |
| `openpi_client.image_tools.resize_with_pad` (PIL BILINEAR on CPU) | hand-written CUDA kernels doing Pillow's fixed-point (22-bit) two-pass resample exactly (`csrc/pil_resize.h`, `csrc/kernels.cu`); checked against PIL at start-up |
| per-sample Python transforms (state extraction, normalisation, tokenisation, delta actions) | per-frame table built once in Rust (`ftprep table`), per-sample action window + delta + normalisation in C++ |
| torch `randperm` shuffle (seed 42) | the same MT19937 / randperm / seed-consumption sequence in C++ |
| `np.stack` + host→GPU copy (14.8 MB/batch) | images written straight into GPU slots; 0.3 MB of non-image data via pinned memory |
| — | batch handed to JAX with zero-copy DLPack; a slot is recycled when JAX releases all 10 tensors |

Python (`fast.py`) is only ctypes glue. The verification tools stay in Python because their *reference*
side is the original openpi/JAX/torchcodec code.

## Build (Linux)

```bash
bash src/vla/fasttrain/build.sh [all|rust|native]   # syncs sources to $FT_WORK/src, builds ftprep (Rust) and libftcore.so + ftbench (C++/CUDA)
# env: FT_WORK (~/fasttrain_work), CUDA_HOME (/usr/local/cuda-12.8), FT_CUDA_ARCH (120; A100 80, H100 90),
#      FT_NVHDR (nv-codec-headers include dir; cloned into $FT_WORK/third_party if missing)
```

Needs: NVIDIA driver with `libnvcuvid.so.1`, CUDA toolkit (nvcc), g++, cmake (for the sentencepiece crate), Rust,
system ffmpeg with libx265 (only to build the colour LUT).

## Use

```bash
export PYTHONPATH=<repo>/src FT_WORK=~/fasttrain_work
python src/vla/fasttrain/lut.py build && python src/vla/fasttrain/lut.py check   # once per machine
python tools/ft_verify.py table && python tools/ft_verify.py order      # CPU-only checks
python tools/ft_verify.py stage && python tools/ft_verify.py ref --tag a && python tools/ft_verify.py check --tag a
FT_FAST_DATA=1 uv run scripts/b1k/train_b1k.py pi05_b1k ...             # openpi train script, loader switched by env var
```

The per-frame table is built automatically on first use (`fast.ensure_table`, cached by a hash of the spec).

## Layout

| path | language | role |
|---|---|---|
| `csrc/nvdec.cpp`, `engine.h` | C++ | NVDEC engine: job queue, per-thread decoders and CUDA streams |
| `csrc/kernels.cu`, `kernels.h`, `pil_resize.h` | CUDA | colour-LUT kernel, Pillow-exact integer resize kernels (plan built on the host) |
| `csrc/loader.cpp`, `loader.h` | C++ | table (mmap), torch-identical sampler, GPU slot ring, DLPack tensors |
| `csrc/capi.cpp`, `dlpack_min.h` | C++ | C ABI (`ft_loader_*`, `ft_table_*`, `ft_engine_*`, `ft_sampler_order`) |
| `csrc/ftbench.cpp` | C++ | loader throughput without Python |
| `ftprep/` | Rust | `index` (mp4 → packet index), `table` (per-frame table), `yuvgrid` (LUT source) |
| `fast.py` | Python | ctypes glue, table spec from the openpi config, DLPack → JAX |
| `lut.py` | Python | colour-LUT build — it must call the original torchcodec decoder |
| `orig.py` | Python | the original pipeline on local paths (measurement and verification reference) |
| `openpi-fast-data.patch` | — | the 20-line `train_b1k.py` switch (`FT_FAST_DATA=1`), already merged into `~/openpi` `behavior` |
