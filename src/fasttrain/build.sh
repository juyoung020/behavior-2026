#!/bin/bash
# Build the native data-loader pieces (WSL / Linux). Sources are synced out of the repo first,
# so nothing is built inside /mnt/c.
#   ftprep       Rust  — mp4 packet index, per-frame tables, LUT grid
#   libftcore.so C++/CUDA — NVDEC engine, colour-LUT and resize kernels, native loader, C ABI (no torch)
#   ftbench      C++ — loader throughput without Python
# Env: FT_WORK (default ~/fasttrain_work), CUDA_HOME (default /usr/local/cuda-12.8),
#      FT_CUDA_ARCH (default 120 = RTX 50xx; A100 80, H100 90), FT_NVHDR (nv-codec-headers include dir)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FT_WORK=${FT_WORK:-$HOME/fasttrain_work}
CUDA=${CUDA_HOME:-/usr/local/cuda-12.8}
ARCH=${FT_CUDA_ARCH:-120}
NVHDR=${FT_NVHDR:-$FT_WORK/third_party/nv-codec-headers/include}
SRC=$FT_WORK/src/fasttrain
OUT=$FT_WORK/build/native
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}

mkdir -p "$SRC" "$OUT"
if [ ! -d "$NVHDR/ffnvcodec" ]; then
  git clone -q --depth 1 https://github.com/FFmpeg/nv-codec-headers "$FT_WORK/third_party/nv-codec-headers"
fi
rsync -a --delete --exclude target --exclude __pycache__ "$HERE/" "$SRC/"

what=${1:-all}
if [ "$what" = all ] || [ "$what" = rust ]; then
  CARGO_TARGET_DIR=$FT_WORK/target "$CARGO" build --release -q --manifest-path "$SRC/ftprep/Cargo.toml"
  echo "ftprep: $FT_WORK/target/release/ftprep"
fi
if [ "$what" = all ] || [ "$what" = native ]; then
  # -ffp-contract=off / --fmad=false: no fused multiply-add anywhere except the explicit __fmaf_rn in the resize
  # kernels, so the float arithmetic is exactly what the source says (bit-identity with the original pipeline).
  CXXF="-O3 -std=c++17 -fPIC -ffp-contract=off -Wall -Wno-unused-function"
  "$CUDA/bin/nvcc" -O3 -std=c++17 --fmad=false -Xcompiler -fPIC -gencode arch=compute_$ARCH,code=sm_$ARCH \
      -c "$SRC/csrc/kernels.cu" -o "$OUT/kernels.o"
  for f in nvdec loader capi; do
    g++ $CXXF -I"$SRC/csrc" -I"$NVHDR" -I"$CUDA/include" -c "$SRC/csrc/$f.cpp" -o "$OUT/$f.o"
  done
  g++ -shared -o "$OUT/libftcore.so" "$OUT"/kernels.o "$OUT"/nvdec.o "$OUT"/loader.o "$OUT"/capi.o \
      -L"$CUDA/lib64" -Wl,-Bstatic -lcudart_static -Wl,-Bdynamic -ldl -lpthread -lrt
  g++ $CXXF -o "$OUT/ftbench" "$SRC/csrc/ftbench.cpp" -L"$OUT" -lftcore -Wl,-rpath,"$OUT" -lpthread
  echo "libftcore: $OUT/libftcore.so, ftbench: $OUT/ftbench"
fi
