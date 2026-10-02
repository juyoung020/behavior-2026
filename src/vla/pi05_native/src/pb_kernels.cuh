// PiBehavior (2025 BEHAVIOR 1st place) specific kernels, see pb_kernels.cu.
#pragma once
#include "common.cuh"

namespace pi05 {

struct PbTaskArgs {
  const bf16* task_emb;   // [tasks][2048]
  const bf16* stage_emb;  // [sum of stages][1024]
  const int* ints;        // device: [0] task id, [1] stage, [2] inpaint on
  const int* num_stages;  // device [tasks]
  const int* offsets;     // device [tasks]
};

void launch_pb_task_prep(PbTaskArgs a, float* all, bf16* x_rows, cudaStream_t st);
void launch_pb_stage_feat(const float* all, const float* g_sc, const float* g_ts, float* sf, cudaStream_t st);
void launch_pb_task_rows(const float* all, const float* g_t, const float* bal, const float* sd, bf16* x_rows,
                         cudaStream_t st);
void launch_kv_transform(const bf16* kc, const bf16* vt, bf16* kc2, bf16* vt2, const bf16* kcoef, const bf16* vcoef,
                         const bf16* kbias, const bf16* vbias, int depth, int sc, const int* d_T, cudaStream_t st);
void launch_linear_bf16in(const bf16* x, const bf16* w, const bf16* b, bf16* y, int in_dim, int out_dim,
                          cudaStream_t st);
void launch_inpaint(float* x, const float* x0O, const float* zO, const float* C, int nO, int nU, float t_new,
                    const int* ints, cudaStream_t st);

}  // namespace pi05
