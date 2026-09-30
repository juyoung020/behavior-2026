// Native pi0.5 trainer, LoRA mode = openpi Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora",
// action_expert_variant="gemma_300m_lora") with its get_freeze_filter (models/pi0_config.py:88-117):
//   trainable: SigLIP (all of PaliGemma/img, f32), every LoRA factor of Gemma 2B (rank 16) and of the action expert
//              (rank 32), action_in/out_proj, time_mlp_in/out
//   frozen (bf16): Gemma 2B and action expert base weights, norms, adaRMS modulation Dense, token embedding
// Everything runs in this trainer (no inference engine): SigLIP -> joint Gemma pass (prefix + suffix, one attention per
// layer) -> flow loss, and the full backward down to the SigLIP stem. Like openpi (nn.remat(..., nothing_saveable) on
// every Gemma / SigLIP layer) only each layer's input is kept; the layer is recomputed during backward.
#pragma once
#include <string>
#include <vector>

#include "tparams.h"

namespace pi05t {

struct LoraSample {
  const uint8_t* img_u8 = nullptr;  // [3][224][224][3] host (or img_f32 on the device: augmented, [-1, 1])
  const float* img_f32_dev = nullptr;
  std::vector<int> tokens;          // valid prompt tokens
  const float* actions = nullptr;   // [32][32] host, normalized
  const float* noise = nullptr;     // [32][32] host
  float time = 0.5f;
};

class LoraTrainer {
 public:
  ~LoraTrainer();
  // state: p.<name> f32 trainable, f.<name> bf16 frozen; model: inference-format file (token embedding table only)
  bool init(const std::string& state_path, const std::string& model_path, std::string* err, int offload = 0);
  void zero_grads() { ps_.zero_grads(st); }
  float accumulate(const LoraSample& s, float scale);
  void finalize_grads() { ps_.finalize(st); }
  double grad_norm() { return ps_.grad_norm(st); }
  void opt_step() { ps_.step(opt, count_, st); ++count_; }
  std::vector<TParam>& params() { return ps_.all(); }
  TParam* param(const std::string& n) { return ps_.get(n); }
  OptCfg opt;
  cudaStream_t st = nullptr;
  size_t frozen_bytes() const { return fz_.bytes; }

 private:
  static constexpr int NI = 3, TI = 768, DI = 1152, HI = 16, DHI = 72, FI = 4304;
  static constexpr int DP = 2048, FP = 16384, NH = 8, HD = 256, RP = 16;
  static constexpr int AH = 32, AD = 32, DS = 1024, FS = 4096, RS = 32;
  static constexpr int TCAP = 968, SCAP = 1008;  // prefix / key capacity (multiples of 8)

  struct Sig {  // one SigLIP layer's recomputed activations
    bf16 *ln1, *qkv, *qs, *s, *p, *att, *xmid, *ln2, *h, *a;
    float *mu1, *rs1, *mu2, *rs2;
  };
  struct Joint {  // one joint Gemma layer's recomputed activations (p = prefix / expert 0, s = suffix / expert 1)
    bf16 *n1p, *qp, *kvp, *mqp, *mkvp, *pbp, *encp, *mop, *op, *xmidp, *n2p, *gup, *mgup, *ap, *mlp, *dp;
    float *rp1, *rp2, *p32p;
    bf16 *moda, *n1s, *qs, *kvs, *mqs, *mkvs, *pbs, *encs, *mos, *os, *xmids, *modf, *n2s, *gus, *mgus, *as, *mls, *ds;
    float *rs1, *rs2, *p32s;
  };

  void siglip_layer_fwd(int l, const bf16* x, bf16* out);
  void siglip_layer_bwd(int l, const bf16* x, const bf16* dout, bf16* dx);
  void joint_layer_fwd(int l, const bf16* xp, const bf16* xs, bf16* xp_out, bf16* xs_out);
  void joint_layer_bwd(int l, const bf16* xp, const bf16* xs, const bf16* dxp_out, const bf16* dxs_out, bf16* dxp,
                       bf16* dxs);
  void attention_fwd(int l);
  void attention_bwd(int l, const bf16* dencp, const bf16* dencs, bf16* dqp, bf16* dkvp, bf16* dqs, bf16* dkvs);
  const bf16* F(const std::string& n) const;  // frozen weight
  TParam* P(const std::string& n);

  pi05::WeightFile wf_;
  DevMem mem_;
  ParamSet ps_;
  FrozenSet fz_;
  bf16* embed_host_ = nullptr;
  int Li_ = 0, Lg_ = 0, count_ = 0;
  int T_ = 0, Tp8_ = 0, S_ = 0, Sp_ = 0;
  float2* rope_ = nullptr;
  // saved layer inputs
  std::vector<bf16*> sig_in_, xp_in_, xs_in_;
  bf16 *sig_last_ = nullptr, *enc_ln_ = nullptr, *xp_ = nullptr, *xs_fin_ = nullptr, *xp_fin_ = nullptr;
  float *enc_mu_ = nullptr, *enc_rs_ = nullptr;
  float *patches_ = nullptr, *stem_ = nullptr;
  Sig sg_{};
  Joint jt_{};
  bf16 *Kall_ = nullptr, *Vall_ = nullptr, *K3_ = nullptr;
  float* logits_ = nullptr;
  // suffix / heads
  float *xt_ = nullptr, *u_ = nullptr, *temb_ = nullptr, *z1_ = nullptr, *c1_ = nullptr, *z2_ = nullptr, *cond_ = nullptr,
        *h0f_ = nullptr, *v_ = nullptr, *dv_ = nullptr, *lossr_ = nullptr, *dcond_ = nullptr, *dc1_ = nullptr,
        *tmpf_ = nullptr, *dss_ = nullptr, *dyf_ = nullptr;
  bf16 *modfin_ = nullptr, *y_ = nullptr, *pseudo_p_ = nullptr;
  float* rfin_ = nullptr;
  // backward scratch
  bf16 *gA_ = nullptr, *gB_ = nullptr, *gC_ = nullptr, *gD_ = nullptr, *gE_ = nullptr;  // SigLIP pools
  bf16 *dxp_a_ = nullptr, *dxp_b_ = nullptr, *dxs_a_ = nullptr, *dxs_b_ = nullptr;
  bf16 *dmid_ = nullptr, *d3_ = nullptr, *dP_ = nullptr, *dmod_ = nullptr, *dy_ = nullptr, *bsum_ = nullptr;
  float *acc_ = nullptr, *acc2_ = nullptr, *dk32_ = nullptr, *dv32_ = nullptr, *dbs_ = nullptr, *accb_ = nullptr;
  uint8_t* img_u8_ = nullptr;
  bf16 *junkp_ = nullptr, *junks_ = nullptr, *sb_[7] = {}, *pA_ = nullptr, *pB_ = nullptr, *pC_ = nullptr,
       *pD_ = nullptr, *pF_ = nullptr, *pG_ = nullptr, *pH_ = nullptr, *sencs_ = nullptr, *sq_ = nullptr,
       *skv_ = nullptr, *sgd_[2] = {}, *kvtmp_ = nullptr;
};

}  // namespace pi05t
