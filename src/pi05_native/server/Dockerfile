# Submission image: the native pi0.5 policy server (openpi / BEHAVIOR websocket protocol, /healthz, msgpack).
# No Python, PyTorch or JAX inside: one C++ binary (CUDA runtime linked statically) + the weight files.
#
#   docker build -f src/pi05_native/server/Dockerfile -t behavior-policy:pb2025 \
#       --build-arg MODEL=pb2025 <context dir holding src/pi05_native and weights/>
#   docker run --gpus all -p 8000:8000 behavior-policy:pb2025
#
# Build context layout (see README "Submission image"):
#   src/pi05_native/...                          engine sources
#   weights/pb2025_ckpt{1,2,3,4}.pi05w           MODEL=pb2025 (2025 1st place, one checkpoint per task group)
#   weights/task_checkpoint_mapping.json
#   weights/pi05_radio.pi05w                     MODEL=radio
# GPU architectures: Turing (sm_75 = TitanRTX: fp16 tensor-core path), Ampere (sm_80, sm_86 = RTX 3090 / A5000),
# Ada (sm_89), Hopper (sm_90), Blackwell (sm_120). The driver picks the matching code at run time.

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build
ARG ARCHS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120"
COPY src/pi05_native /src/pi05_native
RUN CUDA_HOME=/usr/local/cuda PI05_ARCH="${ARCHS}" bash /src/pi05_native/build_linux.sh /opt/pi05 \
    && strip /opt/pi05/pi05_server

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04
ARG MODEL=pb2025
ENV MODEL=${MODEL} PORT=8000
COPY --from=build /opt/pi05/pi05_server /usr/local/bin/pi05_server
COPY weights/ /models/
EXPOSE 8000
# pb2025: the task id in each observation selects the checkpoint (up to 3 resident on a 24 GB GPU, LRU);
# radio: the openpi radio checkpoint with its task prompt.
CMD if [ "$MODEL" = "pb2025" ]; then \
      exec pi05_server --task-map /models/task_checkpoint_mapping.json --weights-dir /models --port "$PORT" --max-engines 3; \
    else \
      exec pi05_server --weights /models/pi05_radio.pi05w --port "$PORT"; \
    fi
