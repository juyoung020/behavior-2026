// Native pi0.5 trainer (openpi pi05 training step without JAX/PyTorch).
//
// Mode "expert": the action expert (Gemma 300M, every llm *_1 parameter), action_in/out_proj and the time MLP are
// trained; SigLIP and Gemma 2B are frozen (bf16) and run through the inference engine (pi05::Model), whose prefix KV
// cache is exactly what the suffix attends to in openpi's joint forward (prefix tokens never attend to the suffix).
// One sample at a time (micro-batch 1): forward with saved activations, backward, gradients summed in f32; at the end
// of the batch every gradient that JAX produces as a bf16 einsum result is rounded once (finalize), then
// clip_by_global_norm -> AdamW -> EMA exactly as optax / scripts/train.py.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../pi05_native/src/model.h"
#include "tparams.h"

namespace pi05t {

struct Sample {
  const uint8_t* img = nullptr;  // [3][224][224][3] host, or:
  const float* img_f32_dev = nullptr;  // device f32 images in [-1, 1] (augmented)
  std::vector<int> tokens;       // valid prompt tokens
  const float* actions = nullptr;  // [ah][ad] normalized
  const float* noise = nullptr;    // [ah][ad]
  float time = 0.5f;
};

class Trainer {
 public:
  ~Trainer();
  // state: params container (p.<name> f32 trainable, f.<name> bf16 frozen; cfg lr/adam/clip/ema), model: the same
  // (cut) model in the inference format for the frozen prefix.
  // offload: 0 = optimizer state on the GPU, 1 = EMA in pinned host memory, 2 = EMA and Adam moments on the host
  bool init(const std::string& state_path, const std::string& model_path, std::string* err, int offload = 0);
  void zero_grads();
  // forward + backward of one sample; returns its loss (mean over horizon and dims); `scale` = 1 / batch
  float accumulate(const Sample& s, float scale);
  void finalize_grads();
  double grad_norm();
  void opt_step();  // one optimizer step from the current gradients (after finalize)
  int step_count() const { return count_; }
  std::vector<TParam>& params() { return ps_.all(); }
  TParam* param(const std::string& name) { return ps_.get(name); }
  OptCfg opt;
  cudaStream_t st = nullptr;

 private:
  void forward_suffix(const Sample& s);
  void backward_suffix(const Sample& s);
  void alloc_work();
  pi05::WeightFile wf_;
  pi05::Model prefix_;
  DevMem mem_;
  ParamSet ps_;
  int L_ = 0, count_ = 0;
  int Tp_ = 0, S_ = 0, Sp_ = 0;  // valid prefix tokens, keys, keys padded to 8
  static constexpr int AH = 32, AD = 32, W = 1024, F = 4096, NH = 8, HD = 256;
  float2* rope_ = nullptr;
  // per-sample tensors (device)
  float *xt_ = nullptr, *u_ = nullptr, *temb_ = nullptr, *z1_ = nullptr, *c1_ = nullptr, *z2_ = nullptr,
        *cond_ = nullptr, *h0f_ = nullptr, *v_ = nullptr, *dv_ = nullptr, *lossr_ = nullptr, *dcond_ = nullptr,
        *dc1_ = nullptr, *dtmp_ = nullptr, *dyf_ = nullptr, *dss_ = nullptr, *dk32_ = nullptr, *acc32_ = nullptr,
        *dh0f_ = nullptr;
  struct Lay {
    bf16 *h_in, *mod_a, *n1, *q, *kv, *pb, *enc, *o, *h_mid, *mod_f, *n2, *g, *u, *a, *d;
    float *r1, *r2, *p32;
  };
  std::vector<Lay> lay_;
  bf16 *h_fin_ = nullptr, *mod_fin_ = nullptr, *y_ = nullptr;
  float* r_fin_ = nullptr;
  bf16 *Kall_ = nullptr, *Vall_ = nullptr, *K3_ = nullptr;  // [Sp][256], [Sp][256], [3*Sp][256]
  float* logits_ = nullptr;
  // backward scratch
  bf16 *dh_ = nullptr, *dh2_ = nullptr, *dx_ = nullptr, *dd_ = nullptr, *da_ = nullptr, *dg_ = nullptr, *du_ = nullptr,
       *dn_ = nullptr, *dn_b_ = nullptr, *do_ = nullptr, *denc_ = nullptr, *dp_ = nullptr, *d3_ = nullptr,
       *dq_ = nullptr, *dkv_ = nullptr, *dvall_ = nullptr, *dmod_ = nullptr, *dy_ = nullptr;
  template <class T>
  T* dalloc(size_t n) { return mem_.dev<T>(n); }
};

}  // namespace pi05t
