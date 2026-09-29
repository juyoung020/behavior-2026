// pi0.5 (openpi pi05, PaliGemma 3B + 300M action expert) inference engine: weights, buffers, forward.
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "common.cuh"
#include "kernels.cuh"
#include "weights.h"

namespace pi05 {

struct ModelCfg {
  int img_w = 1152, img_depth = 27, img_mlp = 4352, img_heads = 16, img_hd = 72, n_img = 3;
  int w = 2048, depth = 18, mlp = 16384, heads = 8, hd = 256;
  int ae_w = 1024, ae_mlp = 4096;
  int ah = 32, ad = 32, steps = 10, max_tok = 200;
  // PiBehavior (2025 1st place): 5 task/stage tokens computed on the GPU after the images, then max_tok = 32 state
  // tokens gathered on the host; two attention groups (tokens < g0 only see tokens < g0).
  bool pb = false;
  int extra_dev = 0, g0 = 0, num_tasks = 0;
  float dtb = -0.10009765625f;  // bf16(-1 / steps)
  std::vector<int> stages;      // per task (PiBehavior)
  int img_tokens() const { return n_img * 256; }
  int t_cap() const { return img_tokens() + extra_dev + max_tok; }  // pi05 968, PiBehavior 805
  int s_cap() const { return 1024; }                    // key capacity >= t_cap + ah, multiple of 64
};

// Debug tap: called with device pointers at named points when tapping is enabled (never during graph capture).
using Tap = std::function<void(const std::string& name, const void* dptr, int dtype /*0 bf16,1 f32*/,
                               std::vector<int64_t> shape)>;

class Model {
 public:
  bool load(const WeightFile& wf, std::string* err);
  ~Model();

  // Host inputs for one inference: images [n_img][224][224][3] uint8, text tokens, noise [ah][ad] f32.
  void upload_inputs(const uint8_t* img, const std::vector<int>& tokens, const float* noise, cudaStream_t st);
  // PiBehavior per-call inputs (call before upload_inputs/forward): task, stage, inpainting (x0O/zO: nO floats or null)
  void set_pb_inputs(int task, int stage, const float* x0O, const float* zO, int nO, cudaStream_t st);
  void download_stage_logits(float* out, cudaStream_t st);  // [15] as float (bf16 values)
  // Full forward on `st`. With a graph already captured this just launches it.
  void forward(cudaStream_t st);
  void forward_eager(cudaStream_t st);
  bool capture_graph(cudaStream_t st, std::string* err);
  void download_actions(float* out, cudaStream_t st);  // [ah][ad]

  // Stages (used by forward and by the layer-by-layer verifier)
  void siglip(cudaStream_t st);
  void siglip_layer(int l, cudaStream_t st);
  void siglip_head(cudaStream_t st);
  void text_embed(cudaStream_t st);
  void prefix_layer(int l, cudaStream_t st);
  void prefix_tail(cudaStream_t st);  // PiBehavior: stage head + KV layer mixing
  void denoise_step(int s, cudaStream_t st);
  void action_in(int s, cudaStream_t st);
  void suffix_layer(int s, int l, cudaStream_t st);
  void suffix_head(int s, cudaStream_t st);

  Tap tap;  // optional
  ModelCfg cfg;
  size_t weight_bytes = 0, act_bytes = 0;

  // ---- buffers (public for the verifier) ----
  uint8_t* d_img = nullptr;
  int* d_tokens = nullptr;
  Dims* d_dims = nullptr;
  Dims h_dims{};
  bf16 *xi = nullptr, *ni = nullptr, *qkvi = nullptr, *qsi = nullptr, *vti = nullptr, *si = nullptr, *ai = nullptr,
       *hi = nullptr;
  float* stem = nullptr;
  bf16 *x = nullptr, *n = nullptr, *qkv = nullptr, *q = nullptr, *pr = nullptr, *att = nullptr, *hid = nullptr;
  float* lg = nullptr;
  bf16 *kc = nullptr, *vt = nullptr;  // [depth][s_cap][hd], [depth][hd][s_cap]
  float* xt = nullptr;                // [ah][ad] f32 flow state
  bf16 *h = nullptr, *hn = nullptr, *sqkv = nullptr, *sq = nullptr, *spr = nullptr, *satt = nullptr, *shid = nullptr,
       *v = nullptr;
  float* slg = nullptr;
  float* ws = nullptr;
  int* counters = nullptr;
  size_t ws_floats = 0;
  // PiBehavior
  int* pb_ints = nullptr;  // [task, stage, inpaint_on, pad]
  int h_pb_ints[4] = {0, 0, 0, 0};
  int *pb_nstages = nullptr, *pb_offsets = nullptr;
  float *pb_all = nullptr, *pb_gsc = nullptr, *pb_gts = nullptr, *pb_gt = nullptr, *pb_h1 = nullptr, *pb_bal = nullptr,
        *pb_sf = nullptr, *pb_sd = nullptr, *pb_x0O = nullptr, *pb_zO = nullptr, *pb_C = nullptr;
  bf16 *pb_stage_n = nullptr, *pb_logits = nullptr, *kc2 = nullptr, *vt2 = nullptr;
  int pb_nO = 128, pb_nU = 832;
  std::vector<float> t_new;          // flow time after each step (inpainting threshold)
  float inpaint_threshold = 0.3f;
  std::vector<double> corr_L;        // PiBehavior correlated noise factor [ah*ad]^2 (row major)
  bf16* mods = nullptr;  // adaRMS scale|shift|gate per (layer, norm, step), see mod()
  float2* rope = nullptr;
  bf16* embed_host = nullptr;  // token embedding table, pinned host memory (gathered on the CPU)
  std::vector<float> times;

  // mods layout: [(l*2 + which)][step][3*ae_w], final norm at l = depth
  bf16* mod(int s, int l, int which) const { return mods + ((size_t)(l * 2 + which) * cfg.steps + s) * 3 * cfg.ae_w; }
  bf16* final_mod(int s) const { return mods + ((size_t)(cfg.depth * 2) * cfg.steps + s) * 3 * cfg.ae_w; }
  std::vector<float> cond_dbg;    // time conditioning [steps][ae_w] (f32), kept for the verifier
  std::vector<uint16_t> h_emb;    // host staging for gathered text embeddings

 private:
  struct ImgLayer { const bf16 *ln1_s, *ln1_b, *qkv_w, *qkv_b, *o_w, *o_b, *ln2_s, *ln2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b; };
  struct LlmLayer { const bf16 *attn_norm, *qkv_w, *o_w, *ffn_norm, *gu_w, *down_w; };
  struct AeLayer { const bf16 *qkv_w, *o_w, *gu_w, *down_w; };
  std::vector<ImgLayer> il_;
  std::vector<LlmLayer> ll_;
  std::vector<AeLayer> al_;
  const bf16 *patch_w_, *patch_b_, *pos_, *post_s_, *post_b_, *head_w_, *head_b_;
  const bf16 *ain_w_, *ain_b_, *aout_w_, *aout_b_;
  const bf16 *pb_task_emb_ = nullptr, *pb_stage_emb_ = nullptr, *pb_w_[7] = {}, *pb_b_[7] = {}, *pb_kv_[4] = {},
             *final_norm_ = nullptr;
  uint8_t* arena_ = nullptr;
  uint8_t* act_arena_ = nullptr;
  cudaGraphExec_t graph_ = nullptr;
  void do_tap(const std::string& name, const void* p, int dt, std::vector<int64_t> shape, cudaStream_t st);
  bool precompute_modulation(const WeightFile& wf, std::string* err);
};

}  // namespace pi05
