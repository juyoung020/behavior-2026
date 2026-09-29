// Non-GEMM kernels: stem, norms, RoPE/KV-cache writes, attention softmax, suffix helpers.
// Every kernel documents the JAX expression it reproduces and where JAX rounds to bf16.
#pragma once
#include "common.cuh"

namespace pi05 {

// Device-side sizes shared by all kernels of one inference (so one CUDA graph serves every prompt).
struct Dims {
  int T;    // prefix tokens actually used (768 image + valid text tokens)
  int T8;   // T rounded up to 8 (K extent of the P.V GEMM)
  int S;    // suffix attention keys = T + action_horizon
  int S8;   // S rounded up to 8
  int L;    // valid text tokens
  int TH;   // T * heads (rows of the prefix attention logits)
};

void launch_stem(const uint8_t* img, const bf16* w, const bf16* b, const bf16* pos, float* stem_out, bf16* x,
                 int n_img, cudaStream_t st);
void launch_layernorm(const bf16* x, const bf16* scale, const bf16* bias, bf16* y, int rows, int dim, cudaStream_t st);
void launch_siglip_attn_prep(const bf16* qkv, bf16* qs, bf16* vt, int n_img, cudaStream_t st);
void launch_softmax_bf16_rows(bf16* s, int rows, int cols, cudaStream_t st);
void launch_rmsnorm(const bf16* x, const bf16* scale, bf16* y, int rows, int dim, const int* d_rows, cudaStream_t st);
void launch_adarms(const bf16* x, const bf16* mod, bf16* y, int rows, int dim, cudaStream_t st);
void launch_rope_tables(float2* table, int max_pos, int half, cudaStream_t st);
// qkv [rows, (H+2)*256] -> q_out [rows*H, 256] (roped, * 1/16), k -> kc[pos0 + r], v -> vt[:, pos0 + r]
void launch_rope_split(const bf16* qkv, const float2* rope, bf16* q_out, bf16* kc, bf16* vt, int rows,
                       const int* d_rows, int heads, int vt_ld, int pos0, const int* d_pos0, cudaStream_t st);
void launch_softmax_f32_rows(const float* logits, int ld_in, bf16* p, int ld_out, int rows, const int* d_rows,
                             const int* d_cols, const int* d_cols8, cudaStream_t st);
void launch_action_in(const float* x, const bf16* w, const bf16* b, bf16* h, int rows, int in_dim, int out_dim,
                      cudaStream_t st);
void launch_flow_update(float* x, const bf16* v, int n, cudaStream_t st);
void launch_time_embed(const float* times, int n_steps, int dim, float* out, cudaStream_t st);
void launch_linear_f32(const float* x, const bf16* w, const bf16* b, float* y, int rows, int in_dim, int out_dim,
                       int swish, cudaStream_t st);
void launch_linear_bf16_vec(const float* x, const bf16* w, const bf16* b, bf16* y, int rows, int in_dim,
                            int out_dim, cudaStream_t st);

}  // namespace pi05
