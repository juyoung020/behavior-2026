#include "lora.h"

#include <cmath>
#include <cstring>

#include "../../pi05_native/src/kernels.cuh"
#include "tgemm.cuh"

namespace pi05t {
namespace {
const std::string IMG = "PaliGemma/img/", EB = "PaliGemma/img/Transformer/encoderblock/", LY = "PaliGemma/llm/layers/";
inline int r8(int x) { return (x + 7) / 8 * 8; }
TGemm G(const bf16* A, long long lda, const bf16* B, long long ldb, int M, int N, int K) {
  TGemm p;
  p.A = A; p.lda = lda; p.B = B; p.ldb = ldb; p.M = M; p.N = N; p.K = K;
  return p;
}
TGemm GB(const bf16* A, long long lda, long long sA, const bf16* B, long long ldb, long long sB, long long sC, int M,
         int N, int K) {
  TGemm p = G(A, lda, B, ldb, M, N, K);
  p.sA1 = sA; p.sB1 = sB; p.sC1 = sC;
  return p;
}
}  // namespace

LoraTrainer::~LoraTrainer() {
  if (embed_host_) cudaFreeHost(embed_host_);
  if (st) cudaStreamDestroy(st);
}

const bf16* LoraTrainer::F(const std::string& n) const {
  const bf16* p = fz_.get(n);
  if (!p) { fprintf(stderr, "missing frozen weight %s\n", n.c_str()); abort(); }
  return p;
}
TParam* LoraTrainer::P(const std::string& n) {
  TParam* p = ps_.get(n);
  if (!p) { fprintf(stderr, "missing trainable %s\n", n.c_str()); abort(); }
  return p;
}

bool LoraTrainer::init(const std::string& state_path, const std::string& model_path, std::string* err, int offload) {
  PI05_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  if (!wf_.open(state_path, err)) return false;
  if (wf_.cfg_str("mode", "") != "lora") { *err = "state is not a lora-mode state"; return false; }
  opt.read(wf_);
  Li_ = wf_.cfg_int("img.depth", 27);
  Lg_ = wf_.cfg_int("llm.depth", 18);
  // bf16 working copies for everything used through a bf16 matmul; gradients of those are bf16 results
  auto f32_use = [](const std::string& n) {
    return n.find("LayerNorm") != std::string::npos || n.find("encoder_norm") != std::string::npos ||
           n.rfind(IMG + "embedding/", 0) == 0 || n == IMG + "pos_embedding" || n.rfind("PaliGemma/", 0) != 0;
  };
  auto bf = [&](const std::string& n) { return !f32_use(n); };
  if (!ps_.load(wf_, offload, opt, mem_, bf, bf, st, err)) return false;
  if (!fz_.load(wf_, mem_, err)) return false;
  if (const std::string ff = wf_.cfg_str("frozen_from", ""); !ff.empty()) {  // checkpoint saved by pi05_train
    pi05::WeightFile orig;
    if (!orig.open(ff, err) || !fz_.load(orig, mem_, err)) return false;
  }
  {  // token embedding table (frozen) from the inference-format model file, pinned host memory for row gathers
    pi05::WeightFile mf;
    if (!mf.open(model_path, err)) return false;
    const pi05::TensorInfo* t = mf.find("llm.embed");
    if (!t) { *err = "model file has no llm.embed"; return false; }
    PI05_CUDA(cudaMallocHost(&embed_host_, t->nbytes));
    if (!mf.read(*t, embed_host_, err)) return false;
  }
  rope_ = mem_.dev<float2>(1024 * 128);
  pi05::launch_rope_tables(rope_, 1024, 128, st);
  auto B = [&](size_t n) { return mem_.dev<bf16>(n); };
  auto Fl = [&](size_t n) { return mem_.dev<float>(n); };
  for (int l = 0; l < Li_; ++l) sig_in_.push_back(B((size_t)TI * DI));
  for (int l = 0; l < Lg_; ++l) { xp_in_.push_back(B((size_t)TCAP * DP)); xs_in_.push_back(B((size_t)AH * DS)); }
  sig_last_ = B((size_t)TI * DI); enc_ln_ = B((size_t)TI * DI); enc_mu_ = Fl(TI); enc_rs_ = Fl(TI);
  xp_fin_ = B((size_t)TCAP * DP); xs_fin_ = B((size_t)AH * DS);
  patches_ = Fl((size_t)TI * 588); stem_ = Fl((size_t)TI * DI);
  sg_.ln1 = B((size_t)TI * DI); sg_.qkv = B((size_t)TI * 3 * DI); sg_.qs = B((size_t)TI * DI);
  sg_.s = B((size_t)NI * HI * 256 * 256); sg_.p = B((size_t)NI * HI * 256 * 256); sg_.att = B((size_t)TI * DI);
  sg_.xmid = B((size_t)TI * DI); sg_.ln2 = B((size_t)TI * DI); sg_.h = B((size_t)TI * FI); sg_.a = B((size_t)TI * FI);
  sg_.mu1 = Fl(TI); sg_.rs1 = Fl(TI); sg_.mu2 = Fl(TI); sg_.rs2 = Fl(TI);
  Joint& j = jt_;
  j.n1p = B((size_t)TCAP * DP); j.qp = B((size_t)TCAP * NH * HD); j.kvp = B((size_t)TCAP * 2 * HD);
  j.mqp = B((size_t)TCAP * NH * RP); j.mkvp = B((size_t)TCAP * 2 * RP); j.pbp = B((size_t)TCAP * NH * TCAP);
  j.p32p = Fl((size_t)TCAP * NH * TCAP); j.encp = B((size_t)TCAP * NH * HD); j.mop = B((size_t)TCAP * RP);
  j.op = B((size_t)TCAP * DP); j.xmidp = B((size_t)TCAP * DP); j.n2p = B((size_t)TCAP * DP);
  j.gup = B((size_t)2 * TCAP * FP);
  gup_scratch_ = j.gup;
  sig_h_scratch_ = sg_.h;
  if (const char* k = getenv("PI05_LORA_KEEP_MLP"); !k || k[0] != '0') {  // +1.3 GB: skips most of the recompute
    for (int l = 0; l < Lg_; ++l) gup_l_.push_back(B((size_t)2 * TCAP * FP));
    for (int l = 0; l < Li_; ++l) sig_h_.push_back(B((size_t)TI * FI));
  } j.mgup = B((size_t)2 * TCAP * RP); j.ap = B((size_t)TCAP * FP);
  j.mlp = B((size_t)TCAP * RP); j.dp = B((size_t)TCAP * DP); j.rp1 = Fl(TCAP); j.rp2 = Fl(TCAP);
  j.moda = B(3 * DS); j.n1s = B(AH * DS); j.qs = B(AH * NH * HD); j.kvs = B(AH * 2 * HD); j.mqs = B(AH * NH * RS);
  j.mkvs = B(AH * 2 * RS); j.pbs = B((size_t)AH * NH * SCAP); j.p32s = Fl((size_t)AH * NH * SCAP);
  j.encs = B(AH * NH * HD); j.mos = B(AH * RS); j.os = B(AH * DS); j.xmids = B(AH * DS); j.modf = B(3 * DS);
  j.n2s = B(AH * DS); j.gus = B(2 * AH * FS); j.mgus = B(2 * AH * RS); j.as = B(AH * FS); j.mls = B(AH * RS);
  j.ds = B(AH * DS); j.rs1 = Fl(AH); j.rs2 = Fl(AH);
  Kall_ = B((size_t)SCAP * HD); Vall_ = B((size_t)SCAP * HD); K3_ = B((size_t)3 * SCAP * HD);
  logits_ = Fl((size_t)TCAP * NH * TCAP);
  xt_ = Fl(AH * AD); u_ = Fl(AH * AD); temb_ = Fl(DS); z1_ = Fl(DS); c1_ = Fl(DS); z2_ = Fl(DS); cond_ = Fl(DS);
  h0f_ = Fl(AH * DS); v_ = Fl(AH * AD); dv_ = Fl(AH * AD); lossr_ = Fl(AH); dcond_ = Fl(DS); dc1_ = Fl(DS);
  tmpf_ = Fl((size_t)AH * DS * 4); dss_ = Fl(2 * DP); dyf_ = Fl(AH * DS);
  modfin_ = B(3 * DS); y_ = B(AH * DS); rfin_ = Fl(AH); pseudo_p_ = B(2 * DP);
  gA_ = B((size_t)2 * TCAP * FP); gB_ = B((size_t)TCAP * FP); gC_ = B((size_t)TCAP * DP); gD_ = B((size_t)TCAP * DP);
  gE_ = B((size_t)TCAP * DP);
  dxp_a_ = B((size_t)TCAP * DP); dxp_b_ = B((size_t)TCAP * DP); dxs_a_ = B(AH * DS); dxs_b_ = B(AH * DS);
  dmid_ = B((size_t)2 * TCAP * NH * RS); d3_ = B((size_t)3 * TCAP * NH * TCAP); dP_ = B((size_t)TCAP * NH * TCAP);
  dmod_ = B(3 * DS); dy_ = B(AH * DS); bsum_ = B((size_t)RS * DP);
  acc_ = Fl((size_t)TCAP * DP); acc2_ = Fl((size_t)TCAP * DP); dk32_ = Fl((size_t)SCAP * HD);
  dv32_ = Fl((size_t)SCAP * HD); dbs_ = Fl((size_t)RS * DP); accb_ = Fl((size_t)TCAP * FP);
  img_u8_ = mem_.dev<uint8_t>((size_t)NI * 224 * 224 * 3);
  junkp_ = B((size_t)TCAP * DP); junks_ = B(AH * DS);
  for (auto& b : sb_) b = B((size_t)AH * FS);
  pA_ = B((size_t)TCAP * DP); pB_ = B((size_t)TCAP * DP); pC_ = B((size_t)TCAP * DP); pD_ = B((size_t)TCAP * DP);
  pF_ = B((size_t)TCAP * FP); pG_ = B((size_t)TCAP * FP); pH_ = B((size_t)TCAP * FP);
  sencs_ = B(AH * NH * HD); sq_ = B(AH * NH * HD); skv_ = B(AH * 2 * HD);
  sgd_[0] = B((size_t)TI * DI); sgd_[1] = B((size_t)TI * DI); kvtmp_ = B((size_t)SCAP * HD);
  PI05_CUDA(cudaStreamSynchronize(st));
  return true;
}

// ======================================================================================================================
// SigLIP layer (siglip.py Encoder1DBlock, dtype_mm bf16)
// ======================================================================================================================
void LoraTrainer::siglip_layer_fwd(int l, const bf16* x, bf16* out) {
  Sig& s = sg_;
  const size_t LD = (size_t)DI * DI;
  layernorm_fwd(x, P(EB + "LayerNorm_0/scale")->p + l * DI, P(EB + "LayerNorm_0/bias")->p + l * DI, s.ln1, s.mu1, s.rs1,
                TI, DI, st);
  const char* qkvn[3] = {"query", "key", "value"};
  for (int k = 0; k < 3; ++k) {  // DenseGeneral(dtype=bf16): bf16(bf16(y W) + b)
    const std::string n = EB + "MultiHeadDotProductAttention_0/" + qkvn[k];
    tgemm<true, false>(G(s.ln1, DI, P(n + "/kernel")->pb + l * LD, DI, TI, DI, DI),
                       EBiasBf16{s.qkv + k * DI, 3 * DI, P(n + "/bias")->pb + l * DI}, 1, st);
  }
  copy_cols(s.qkv, 3 * DI, s.qs, DI, TI, DI, st);
  div_bf16(s.qs, (long long)TI * DI, 8.5f, st);  // query / bf16(sqrt(72))
  {  // logits per (image, head) bf16
    TGemm p = G(s.qs, DI, s.qkv + DI, 3 * DI, 256, 256, DHI);
    p.nb2 = HI; p.sA1 = 256LL * DI; p.sA2 = DHI; p.sB1 = 256LL * 3 * DI; p.sB2 = DHI;
    p.sC1 = (long long)HI * 65536; p.sC2 = 65536;
    tgemm<true, true>(p, EStoreBf16{s.s, 256}, NI * HI, st);
  }
  softmax_bf16_fwd(s.s, s.p, NI * HI * 256, 256, st);
  {  // att[img, tok, h, d] = P . V
    TGemm p = G(s.p, 256, s.qkv + 2 * DI, 3 * DI, 256, DHI, 256);
    p.nb2 = HI; p.sA1 = (long long)HI * 65536; p.sA2 = 65536; p.sB1 = 256LL * 3 * DI; p.sB2 = DHI;
    p.sC1 = 256LL * DI; p.sC2 = DHI;
    tgemm<true, false>(p, EStoreBf16{s.att, DI}, NI * HI, st);
  }
  const std::string on = EB + "MultiHeadDotProductAttention_0/out";
  tgemm<true, false>(G(s.att, DI, P(on + "/kernel")->pb + l * LD, DI, TI, DI, DI),
                     EBiasResidBf16{s.xmid, DI, P(on + "/bias")->pb + l * DI, x}, 1, st);
  layernorm_fwd(s.xmid, P(EB + "LayerNorm_1/scale")->p + l * DI, P(EB + "LayerNorm_1/bias")->p + l * DI, s.ln2, s.mu2,
                s.rs2, TI, DI, st);
  const std::string m = EB + "MlpBlock_0/";
  s.h = sig_h_.empty() ? sig_h_scratch_ : sig_h_[l];  // fc1 output kept per layer when there is room
  const bool reuse = remat_ && !sig_h_.empty();
  if (!reuse)
    tgemm<true, false>(G(s.ln2, DI, P(m + "Dense_0/kernel")->pb + (size_t)l * DI * FI, FI, TI, FI, DI),
                       EBiasBf16{s.h, FI, P(m + "Dense_0/bias")->pb + (size_t)l * FI}, 1, st);
  gelu_fwd(s.h, s.a, (long long)TI * FI, st);
  if (!reuse)  // in a recompute the layer output is not needed
    tgemm<true, false>(G(s.a, FI, P(m + "Dense_1/kernel")->pb + (size_t)l * FI * DI, DI, TI, DI, FI),
                       EBiasResidBf16{out, DI, P(m + "Dense_1/bias")->pb + (size_t)l * DI, s.xmid}, 1, st);
}

void LoraTrainer::siglip_layer_bwd(int l, const bf16* x, const bf16* dout, bf16* dx) {
  Sig& s = sg_;
  remat_ = true;
  siglip_layer_fwd(l, x, gC_);  // recompute (remat); output unused
  remat_ = false;
  const size_t LD = (size_t)DI * DI;
  const std::string m = EB + "MlpBlock_0/";
  TParam *W1 = P(m + "Dense_0/kernel"), *B1 = P(m + "Dense_0/bias"), *W2 = P(m + "Dense_1/kernel"),
         *B2 = P(m + "Dense_1/bias");
  // out = xmid + Dense_1(a)
  tgemm<false, false>(G(s.a, FI, dout, DI, FI, DI, TI), EAddF32{W2->g + (size_t)l * FI * DI, DI}, 1, st);
  colsum_bf16(dout, B2->g + (size_t)l * DI, TI, DI, DI, st);
  bf16* da = gB_;
  tgemm<true, true>(G(dout, DI, W2->pb + (size_t)l * FI * DI, DI, TI, FI, DI), EStoreBf16{da, FI}, 1, st);
  bf16* dh = gA_;
  gelu_bwd(s.h, da, dh, (long long)TI * FI, st);
  tgemm<false, false>(G(s.ln2, DI, dh, FI, DI, FI, TI), EAddF32{W1->g + (size_t)l * DI * FI, FI}, 1, st);
  colsum_bf16(dh, B1->g + (size_t)l * FI, TI, FI, FI, st);
  bf16* dln = gD_;
  tgemm<true, true>(G(dh, FI, W1->pb + (size_t)l * DI * FI, FI, TI, DI, FI), EStoreBf16{dln, DI}, 1, st);
  bf16* dxm = gE_;
  layernorm_bwd(s.xmid, P(EB + "LayerNorm_1/scale")->p + l * DI, s.mu2, s.rs2, dln, dxm,
                P(EB + "LayerNorm_1/scale")->g + l * DI, P(EB + "LayerNorm_1/bias")->g + l * DI, TI, DI, st);
  add_bf16(dout, dxm, dxm, TI * DI, st);  // cotangent of xmid (residual first, then the norm path)
  // attention: xmid = x + out(att)
  const std::string an = EB + "MultiHeadDotProductAttention_0/";
  TParam* Wo = P(an + "out/kernel");
  tgemm<false, false>(G(s.att, DI, dxm, DI, DI, DI, TI), EAddF32{Wo->g + l * LD, DI}, 1, st);
  colsum_bf16(dxm, P(an + "out/bias")->g + l * DI, TI, DI, DI, st);
  bf16* datt = gD_;
  tgemm<true, true>(G(dxm, DI, Wo->pb + l * LD, DI, TI, DI, DI), EStoreBf16{datt, DI}, 1, st);
  bf16* dqkv = gA_;  // [3][TI][DI]: dq (scaled), dk, dv
  bf16 *dq = dqkv, *dk = dqkv + (size_t)TI * DI, *dv = dqkv + 2 * (size_t)TI * DI;
  {  // dP = dAtt . V^T ; dV = P^T . dAtt
    TGemm p = G(datt, DI, s.qkv + 2 * DI, 3 * DI, 256, 256, DHI);
    p.nb2 = HI; p.sA1 = 256LL * DI; p.sA2 = DHI; p.sB1 = 256LL * 3 * DI; p.sB2 = DHI;
    p.sC1 = (long long)HI * 65536; p.sC2 = 65536;
    tgemm<true, true>(p, EStoreBf16{s.s, 256}, NI * HI, st);
    TGemm q = G(s.p, 256, datt, DI, 256, DHI, 256);
    q.nb2 = HI; q.sA1 = (long long)HI * 65536; q.sA2 = 65536; q.sB1 = 256LL * DI; q.sB2 = DHI;
    q.sC1 = 256LL * DI; q.sC2 = DHI;
    tgemm<false, false>(q, EStoreBf16{dv, DI}, NI * HI, st);
  }
  softmax_bf16_bwd(s.p, s.s, s.s, NI * HI * 256, 256, st);  // dS in place of dP
  {  // dQs = dS . K ; dK = dS^T . Qs
    TGemm p = G(s.s, 256, s.qkv + DI, 3 * DI, 256, DHI, 256);
    p.nb2 = HI; p.sA1 = (long long)HI * 65536; p.sA2 = 65536; p.sB1 = 256LL * 3 * DI; p.sB2 = DHI;
    p.sC1 = 256LL * DI; p.sC2 = DHI;
    tgemm<true, false>(p, EStoreBf16{dq, DI}, NI * HI, st);
    TGemm q = G(s.s, 256, s.qs, DI, 256, DHI, 256);
    q.nb2 = HI; q.sA1 = (long long)HI * 65536; q.sA2 = 65536; q.sB1 = 256LL * DI; q.sB2 = DHI;
    q.sC1 = 256LL * DI; q.sC2 = DHI;
    tgemm<false, false>(q, EStoreBf16{dk, DI}, NI * HI, st);
  }
  div_bf16(dq, (long long)TI * DI, 8.5f, st);
  // q/k/v Dense: weight/bias grads; input grads folded in reverse call order v, k, q (bf16 adds)
  bf16* dln1 = gD_;
  const char* qkvn[3] = {"query", "key", "value"};
  for (int k = 2; k >= 0; --k) {
    const std::string n = an + qkvn[k];
    TParam *Wk = P(n + "/kernel"), *Bk = P(n + "/bias");
    const bf16* g = dqkv + (size_t)k * TI * DI;
    tgemm<false, false>(G(s.ln1, DI, g, DI, DI, DI, TI), EAddF32{Wk->g + l * LD, DI}, 1, st);
    colsum_bf16(g, Bk->g + l * DI, TI, DI, DI, st);
    if (k == 2) tgemm<true, true>(G(g, DI, Wk->pb + l * LD, DI, TI, DI, DI), EStoreBf16{dln1, DI}, 1, st);
    else tgemm<true, true>(G(g, DI, Wk->pb + l * LD, DI, TI, DI, DI), EAddToBf16{dln1, DI}, 1, st);
  }
  layernorm_bwd(x, P(EB + "LayerNorm_0/scale")->p + l * DI, s.mu1, s.rs1, dln1, dx,
                P(EB + "LayerNorm_0/scale")->g + l * DI, P(EB + "LayerNorm_0/bias")->g + l * DI, TI, DI, st);
  add_bf16(dxm, dx, dx, TI * DI, st);
}


// ======================================================================================================================
// LoRA-augmented projections (lora.py Einsum / FeedForward._dot): out = bf16(bf16(x W) + bf16(bf16(x A) B))
// n matrices batched with strides (heads of q / kv, gate and up of the MLP); openpi [in, out] layouts used as stored.
// ======================================================================================================================
namespace {
struct LoraMat {
  const bf16 *W, *A, *B;   // [n][din][dout], [n][din][r], [n][r][dout]
  float *gA, *gB;          // gradients of A, B (f32), same layout
  int din, dout, r, n;
};
// out(t, k, :) at out + k*sO + t*ldo ; mid(t, k, :) at mid + k*sM + t*ldm
void lora_fwd(const LoraMat& L, const bf16* x, int rows, bf16* out, long long ldo, long long sO, bf16* mid,
              long long ldm, long long sM, cudaStream_t st) {
  tgemm<true, false>(GB(x, L.din, 0, L.W, L.dout, (long long)L.din * L.dout, sO, rows, L.dout, L.din),
                     EStoreBf16{out, ldo}, L.n, st);
  tgemm<true, false>(GB(x, L.din, 0, L.A, L.r, (long long)L.din * L.r, sM, rows, L.r, L.din), EStoreBf16{mid, ldm}, L.n,
                     st);
  tgemm<true, false>(GB(mid, ldm, sM, L.B, L.dout, (long long)L.r * L.dout, sO, rows, L.dout, L.r),
                     EAddToBf16{out, ldo}, L.n, st);
}
// gradients of A, B; dx_lora = bf16(sum_k dmid_k A_k^T), dx_base = bf16(sum_k dout_k W_k^T) (one rounding per einsum)
void lora_bwd(const LoraMat& L, const bf16* x, int rows, const bf16* mid, long long ldm, long long sM, const bf16* dout,
              long long ldd, long long sD, bf16* dmid, float* acc, bf16* dx_lora, bf16* dx_base, cudaStream_t st) {
  tgemm<false, false>(GB(mid, ldm, sM, dout, ldd, sD, (long long)L.r * L.dout, L.r, L.dout, rows),
                      EAddF32{L.gB, L.dout}, L.n, st);
  tgemm<true, true>(GB(dout, ldd, sD, L.B, L.dout, (long long)L.r * L.dout, sM, rows, L.r, L.dout),
                    EStoreBf16{dmid, ldm}, L.n, st);
  tgemm<false, false>(GB(x, L.din, 0, dmid, ldm, sM, (long long)L.din * L.r, L.din, L.r, rows), EAddF32{L.gA, L.r}, L.n,
                      st);
  PI05_CUDA(cudaMemsetAsync(acc, 0, (size_t)rows * L.din * 4, st));
  for (int k = 0; k < L.n; ++k)
    tgemm<true, true>(G(dmid + k * sM, ldm, L.A + (size_t)k * L.din * L.r, L.r, rows, L.din, L.r), EAddF32{acc, L.din}, 1,
                      st);
  f32_to_bf16(acc, dx_lora, (long long)rows * L.din, st);
  PI05_CUDA(cudaMemsetAsync(acc, 0, (size_t)rows * L.din * 4, st));
  for (int k = 0; k < L.n; ++k)
    tgemm<true, true>(G(dout + k * sD, ldd, L.W + (size_t)k * L.din * L.dout, L.dout, rows, L.din, L.dout),
                      EAddF32{acc, L.din}, 1, st);
  f32_to_bf16(acc, dx_base, (long long)rows * L.din, st);
}
// attn_vec_einsum ("BTNH,NHD->BTD"; LoRA "BTNH,NHL->BTL", "BTL,NLD->BTD" sums B over heads)
struct LoraOut {
  const bf16 *W, *A, *B;  // [NH*HD][dout], [NH*HD][r], [NH][r][dout]
  float *gA, *gB;
  int dout, r;
};
void lout_fwd(const LoraOut& L, const bf16* enc, int rows, bf16* out, bf16* mid, bf16* bsum, cudaStream_t st) {
  const int K = 8 * 256;
  tgemm<true, false>(G(enc, K, L.W, L.dout, rows, L.dout, K), EStoreBf16{out, L.dout}, 1, st);
  tgemm<true, false>(G(enc, K, L.A, L.r, rows, L.r, K), EStoreBf16{mid, L.r}, 1, st);
  sum_heads_bf16(L.B, bsum, 8, (long long)L.r * L.dout, st);
  tgemm<true, false>(G(mid, L.r, bsum, L.dout, rows, L.dout, L.r), EAddToBf16{out, L.dout}, 1, st);
}
// denc = bf16(bf16(dmid A^T) + bf16(dout W^T))
void lout_bwd(const LoraOut& L, const bf16* enc, int rows, const bf16* mid, const bf16* bsum, const bf16* dout,
              bf16* dmid, float* dbs, bf16* denc, bf16* tmp, cudaStream_t st) {
  const int K = 8 * 256;
  tgemm<false, false>(G(mid, L.r, dout, L.dout, L.r, L.dout, rows), EStoreF32{dbs, L.dout}, 1, st);
  bcast_add_f32(dbs, L.gB, 8, (long long)L.r * L.dout, st);
  tgemm<true, true>(G(dout, L.dout, bsum, L.dout, rows, L.r, L.dout), EStoreBf16{dmid, L.r}, 1, st);
  tgemm<false, false>(G(enc, K, dmid, L.r, K, L.r, rows), EAddF32{L.gA, L.r}, 1, st);
  tgemm<true, true>(G(dmid, L.r, L.A, L.r, rows, K, L.r), EStoreBf16{tmp, K}, 1, st);
  tgemm<true, true>(G(dout, L.dout, L.W, L.dout, rows, K, L.dout), EStoreBf16{denc, K}, 1, st);
  add_bf16(tmp, denc, denc, rows * K, st);
}
}  // namespace

// weights of joint layer l for expert e (0 = PaliGemma prefix, 1 = action expert)
struct LoraTrainer_W {
  LoraMat q, kv, g, lin;
  LoraOut o;
  const bf16 *s1, *s2;              // prefix RMSNorm scales
  const bf16 *mw_a, *mb_a, *mw_f, *mb_f;  // expert adaRMS modulation Dense (frozen)
};

static LoraTrainer_W layer_w(LoraTrainer* T, int l, int e, const std::function<const bf16*(const std::string&)>& Fz,
                             const std::function<TParam*(const std::string&)>& Pp) {
  const std::string sfx = e ? "_1" : "";
  const int D = e ? 1024 : 2048, Fd = e ? 4096 : 16384, r = e ? 32 : 16;
  LoraTrainer_W w{};
  auto lm = [&](const std::string& base, const std::string& wname, int n, int din, int dout) {
    TParam *A = Pp(base + "lora_a"), *B = Pp(base + "lora_b");
    LoraMat m{Fz(wname) + (size_t)l * n * din * dout, A->pb + (size_t)l * n * din * r, B->pb + (size_t)l * n * r * dout,
              A->g + (size_t)l * n * din * r, B->g + (size_t)l * n * r * dout, din, dout, r, n};
    return m;
  };
  const std::string at = LY + "attn/";
  w.q = lm(at + "q_einsum" + sfx + "/", at + "q_einsum" + sfx + "/w", 8, D, 256);
  w.kv = lm(at + "kv_einsum" + sfx + "/", at + "kv_einsum" + sfx + "/w", 2, D, 256);
  {
    TParam *A = Pp(at + "attn_vec_einsum" + sfx + "/lora_a"), *B = Pp(at + "attn_vec_einsum" + sfx + "/lora_b");
    w.o = LoraOut{Fz(at + "attn_vec_einsum" + sfx + "/w") + (size_t)l * 2048 * D, A->pb + (size_t)l * 2048 * r,
                  B->pb + (size_t)l * 8 * r * D, A->g + (size_t)l * 2048 * r, B->g + (size_t)l * 8 * r * D, D, r};
  }
  const std::string ml = LY + "mlp" + sfx + "/";
  {
    TParam *A = Pp(ml + "gating_einsum_lora_a"), *B = Pp(ml + "gating_einsum_lora_b");
    w.g = LoraMat{Fz(ml + "gating_einsum") + (size_t)l * 2 * D * Fd, A->pb + (size_t)l * 2 * D * r,
                  B->pb + (size_t)l * 2 * r * Fd, A->g + (size_t)l * 2 * D * r, B->g + (size_t)l * 2 * r * Fd, D, Fd, r, 2};
    TParam *A2 = Pp(ml + "linear_lora_a"), *B2 = Pp(ml + "linear_lora_b");
    w.lin = LoraMat{Fz(ml + "linear") + (size_t)l * Fd * D, A2->pb + (size_t)l * Fd * r, B2->pb + (size_t)l * r * D,
                    A2->g + (size_t)l * Fd * r, B2->g + (size_t)l * r * D, Fd, D, r, 1};
  }
  if (e == 0) {
    w.s1 = Fz(LY + "pre_attention_norm/scale") + (size_t)l * D;
    w.s2 = Fz(LY + "pre_ffw_norm/scale") + (size_t)l * D;
  } else {
    w.mw_a = Fz(LY + "pre_attention_norm_1/Dense_0/kernel") + (size_t)l * D * 3 * D;
    w.mb_a = Fz(LY + "pre_attention_norm_1/Dense_0/bias") + (size_t)l * 3 * D;
    w.mw_f = Fz(LY + "pre_ffw_norm_1/Dense_0/kernel") + (size_t)l * D * 3 * D;
    w.mb_f = Fz(LY + "pre_ffw_norm_1/Dense_0/bias") + (size_t)l * 3 * D;
  }
  (void)T;
  return w;
}

// ---- joint layer forward ---------------------------------------------------------------------------------------------
void LoraTrainer::joint_layer_fwd(int l, const bf16* xp, const bf16* xs, bf16* xp_out, bf16* xs_out) {
  Joint& j = jt_;
  auto Fz = [&](const std::string& n) { return F(n); };
  auto Pp = [&](const std::string& n) { return P(n); };
  const LoraTrainer_W w0 = layer_w(this, l, 0, Fz, Pp), w1 = layer_w(this, l, 1, Fz, Pp);
  const int T = T_;
  // prefix: RMSNorm (frozen scale; shift 0 through the adaRMS kernel), q / kv with LoRA
  PI05_CUDA(cudaMemcpyAsync(pseudo_p_, w0.s1, DP * 2, cudaMemcpyDeviceToDevice, st));
  adarms_fwd(xp, pseudo_p_, j.n1p, j.rp1, T, DP, st);
  lora_fwd(w0.q, j.n1p, T, j.qp, NH * HD, HD, j.mqp, NH * RP, RP, st);
  lora_fwd(w0.kv, j.n1p, T, j.kvp, 2 * HD, HD, j.mkvp, 2 * RP, RP, st);
  // suffix: adaRMS (frozen modulation Dense on the time condition)
  mod_fwd(cond_, w1.mw_a, w1.mb_a, j.moda, DS, 3 * DS, st);
  adarms_fwd(xs, j.moda, j.n1s, j.rs1, AH, DS, st);
  lora_fwd(w1.q, j.n1s, AH, j.qs, NH * HD, HD, j.mqs, NH * RS, RS, st);
  lora_fwd(w1.kv, j.n1s, AH, j.kvs, 2 * HD, HD, j.mkvs, 2 * RS, RS, st);
  rope_fwd(j.qp, rope_, T, NH, NH * HD, 0, true, st);
  rope_fwd(j.kvp, rope_, T, 1, 2 * HD, 0, false, st);
  rope_fwd(j.qs, rope_, AH, NH, NH * HD, T, true, st);
  rope_fwd(j.kvs, rope_, AH, 1, 2 * HD, T, false, st);
  attention_fwd(l);
  // prefix: out projection, residual (x + y), MLP
  lout_fwd(w0.o, j.encp, T, j.op, j.mop, bsum_, st);
  add_bf16(xp, j.op, j.xmidp, T * DP, st);
  PI05_CUDA(cudaMemcpyAsync(pseudo_p_, w0.s2, DP * 2, cudaMemcpyDeviceToDevice, st));
  adarms_fwd(j.xmidp, pseudo_p_, j.n2p, j.rp2, T, DP, st);
  // gate/up output kept per layer when there is room: the recompute then skips the two big MLP products (the
  // values are the ones the first forward produced, so the backward is unchanged)
  j.gup = gup_l_.empty() ? gup_scratch_ : gup_l_[l];
  if (!remat_ || gup_l_.empty()) {
    lora_fwd(w0.g, j.n2p, T, j.gup, FP, (long long)TCAP * FP, j.mgup, RP, (long long)TCAP * RP, st);
  } else {
    tgemm<true, false>(GB(j.n2p, DP, 0, w0.g.A, RP, (long long)DP * RP, (long long)TCAP * RP, T, RP, DP),
                       EStoreBf16{j.mgup, RP}, 2, st);
  }
  gelu_mul_fwd(j.gup, j.gup + (size_t)TCAP * FP, j.ap, T * FP, st);
  if (!remat_ || gup_l_.empty()) {
    lora_fwd(w0.lin, j.ap, T, j.dp, DP, 0, j.mlp, RP, 0, st);
    add_bf16(j.xmidp, j.dp, xp_out, T * DP, st);
  } else {  // the layer output is not needed in the backward, only the LoRA mid of the down projection
    tgemm<true, false>(G(j.ap, FP, w0.lin.A, RP, T, RP, FP), EStoreBf16{j.mlp, RP}, 1, st);
  }
  // suffix: out projection, gated residual, MLP
  lout_fwd(w1.o, j.encs, AH, j.os, j.mos, bsum_, st);
  gated_res_fwd(xs, j.os, j.moda + 2 * DS, j.xmids, AH, DS, st);
  mod_fwd(cond_, w1.mw_f, w1.mb_f, j.modf, DS, 3 * DS, st);
  adarms_fwd(j.xmids, j.modf, j.n2s, j.rs2, AH, DS, st);
  lora_fwd(w1.g, j.n2s, AH, j.gus, FS, (long long)AH * FS, j.mgus, RS, (long long)AH * RS, st);
  gelu_mul_fwd(j.gus, j.gus + (size_t)AH * FS, j.as, AH * FS, st);
  lora_fwd(w1.lin, j.as, AH, j.ds, DS, 0, j.mls, RS, 0, st);
  gated_res_fwd(j.xmids, j.ds, j.modf + 2 * DS, xs_out, AH, DS, st);
}

// one attention op over [prefix; suffix] keys: prefix queries see the prefix, suffix queries see everything
void LoraTrainer::attention_fwd(int) {
  Joint& j = jt_;
  const int T = T_;
  copy_cols(j.kvp, 2 * HD, Kall_, HD, T, HD, st);
  copy_cols(j.kvp + HD, 2 * HD, Vall_, HD, T, HD, st);
  copy_cols(j.kvs, 2 * HD, Kall_ + (size_t)T * HD, HD, AH, HD, st);
  copy_cols(j.kvs + HD, 2 * HD, Vall_ + (size_t)T * HD, HD, AH, HD, st);
  if (Sp_ > S_) {
    PI05_CUDA(cudaMemsetAsync(Kall_ + (size_t)S_ * HD, 0, (size_t)(Sp_ - S_) * HD * 2, st));
    PI05_CUDA(cudaMemsetAsync(Vall_ + (size_t)S_ * HD, 0, (size_t)(Sp_ - S_) * HD * 2, st));
  }
  tgemm<true, true>(G(j.qp, HD, Kall_, HD, T * NH, Tp8_, HD), EStoreF32{logits_, Tp8_}, 1, st);
  softmax_fwd(logits_, j.p32p, j.pbp, T * NH, T, Tp8_, st);
  tgemm<true, false>(G(j.pbp, Tp8_, Vall_, HD, T * NH, HD, Tp8_), EStoreBf16{j.encp, HD}, 1, st);
  tgemm<true, true>(G(j.qs, HD, Kall_, HD, AH * NH, Sp_, HD), EStoreF32{logits_, Sp_}, 1, st);
  softmax_fwd(logits_, j.p32s, j.pbs, AH * NH, S_, Sp_, st);
  tgemm<true, false>(G(j.pbs, Sp_, Vall_, HD, AH * NH, HD, Sp_), EStoreBf16{j.encs, HD}, 1, st);
}

// cotangents of the attention outputs -> cotangents of pre-RoPE q and of k / v. dK, dV: all queries summed in f32 and
// rounded once (one dot per einsum in JAX); dq = bf16(f32 dlogits . K) via the exact 3-way bf16 split of dlogits.
void LoraTrainer::attention_bwd(int, const bf16* dencp, const bf16* dencs, bf16* dqp, bf16* dkvp, bf16* dqs,
                                bf16* dkvs) {
  Joint& j = jt_;
  const int T = T_;
  PI05_CUDA(cudaMemsetAsync(dk32_, 0, (size_t)Sp_ * HD * 4, st));
  PI05_CUDA(cudaMemsetAsync(dv32_, 0, (size_t)Sp_ * HD * 4, st));
  tgemm<true, true>(G(dencs, HD, Vall_, HD, AH * NH, Sp_, HD), EStoreBf16{dP_, Sp_}, 1, st);
  tgemm<false, false>(G(j.pbs, Sp_, dencs, HD, Sp_, HD, AH * NH), EAddF32{dv32_, HD}, 1, st);
  softmax_bwd_split(j.p32s, dP_, d3_, AH * NH, S_, Sp_, st);
  for (int k = 0; k < 3; ++k)
    PI05_CUDA(cudaMemcpyAsync(K3_ + (size_t)k * Sp_ * HD, Kall_, (size_t)Sp_ * HD * 2, cudaMemcpyDeviceToDevice, st));
  tgemm<true, false>(G(d3_, 3LL * Sp_, K3_, HD, AH * NH, HD, 3 * Sp_), EStoreBf16{dqs, HD}, 1, st);
  for (int k = 0; k < 3; ++k)
    tgemm<false, false>(G(d3_ + (size_t)k * Sp_, 3LL * Sp_, j.qs, HD, Sp_, HD, AH * NH), EAddF32{dk32_, HD}, 1, st);
  tgemm<true, true>(G(dencp, HD, Vall_, HD, T * NH, Tp8_, HD), EStoreBf16{dP_, Tp8_}, 1, st);
  tgemm<false, false>(G(j.pbp, Tp8_, dencp, HD, Tp8_, HD, T * NH), EAddF32{dv32_, HD}, 1, st);
  softmax_bwd_split(j.p32p, dP_, d3_, T * NH, T, Tp8_, st);
  for (int k = 0; k < 3; ++k)
    PI05_CUDA(cudaMemcpyAsync(K3_ + (size_t)k * Tp8_ * HD, Kall_, (size_t)Tp8_ * HD * 2, cudaMemcpyDeviceToDevice, st));
  tgemm<true, false>(G(d3_, 3LL * Tp8_, K3_, HD, T * NH, HD, 3 * Tp8_), EStoreBf16{dqp, HD}, 1, st);
  for (int k = 0; k < 3; ++k)
    tgemm<false, false>(G(d3_ + (size_t)k * Tp8_, 3LL * Tp8_, j.qp, HD, Tp8_, HD, T * NH), EAddF32{dk32_, HD}, 1, st);
  bf16* tk = kvtmp_;
  f32_to_bf16(dk32_, tk, (long long)S_ * HD, st);
  copy_cols(tk, HD, dkvp, 2 * HD, T, HD, st);
  copy_cols(tk + (size_t)T * HD, HD, dkvs, 2 * HD, AH, HD, st);
  f32_to_bf16(dv32_, tk, (long long)S_ * HD, st);
  copy_cols(tk, HD, dkvp + HD, 2 * HD, T, HD, st);
  copy_cols(tk + (size_t)T * HD, HD, dkvs + HD, 2 * HD, AH, HD, st);
  rope_bwd(dqp, rope_, T, NH, NH * HD, 0, true, st);
  rope_bwd(dkvp, rope_, T, 1, 2 * HD, 0, false, st);
  rope_bwd(dqs, rope_, AH, NH, NH * HD, T, true, st);
  rope_bwd(dkvs, rope_, AH, 1, 2 * HD, T, false, st);
}

// ---- joint layer backward: recompute (remat), then reverse ----------------------------------------------------------
// Cotangent sums follow JAX's reverse order (bf16 adds): for an input feeding several einsums the later einsum's
// contributions come first, and inside one LoRA einsum the LoRA path comes before the base product.
void LoraTrainer::joint_layer_bwd(int l, const bf16* xp, const bf16* xs, const bf16* dxp_out, const bf16* dxs_out,
                                  bf16* dxp, bf16* dxs) {
  Joint& j = jt_;
  const int T = T_;
  remat_ = true;
  joint_layer_fwd(l, xp, xs, junkp_, junks_);
  remat_ = false;
  auto Fz = [&](const std::string& n) { return F(n); };
  auto Pp = [&](const std::string& n) { return P(n); };
  const LoraTrainer_W w0 = layer_w(this, l, 0, Fz, Pp), w1 = layer_w(this, l, 1, Fz, Pp);
  float* dgate = tmpf_;  // [DS] f32
  bf16 *s0 = sb_[0], *s1 = sb_[1], *s2 = sb_[2], *s3 = sb_[3], *s4 = sb_[4], *s5 = sb_[5], *s6 = sb_[6];
  bf16 *tA = pA_, *tB = pB_, *tC = pC_, *tD = pD_;  // [TCAP][DP]
  auto split = [](const LoraMat& m, int k, long long sMid) {
    LoraMat o = m;
    o.n = 1;
    o.W += (size_t)k * m.din * m.dout; o.A += (size_t)k * m.din * m.r; o.B += (size_t)k * m.r * m.dout;
    o.gA += (size_t)k * m.din * m.r; o.gB += (size_t)k * m.r * m.dout;
    (void)sMid;
    return o;
  };
  // ================= suffix MLP: xs_out = xmids + ds * gate_f
  gated_res_bwd(dxs_out, j.ds, j.modf + 2 * DS, s0, dgate, AH, DS, st);  // s0 = d(ds)
  lora_bwd(w1.lin, j.as, AH, j.mls, RS, 0, s0, DS, 0, dmid_, acc_, s1, s2, st);  // s1 lora, s2 base: [AH][FS]
  add_bf16(s1, s2, s1, AH * FS, st);                                            // d(as)
  gelu_mul_bwd(j.gus, j.gus + (size_t)AH * FS, s1, s2, s3, AH * FS, st);         // s2 = dg, s3 = du
  {
    const LoraMat up = split(w1.g, 1, 0), gt = split(w1.g, 0, 0);
    lora_bwd(up, j.n2s, AH, j.mgus + AH * RS, RS, 0, s3, FS, 0, dmid_, acc_, s4, s5, st);
    add_bf16(s4, s5, s4, AH * DS, st);
    lora_bwd(gt, j.n2s, AH, j.mgus, RS, 0, s2, FS, 0, dmid_, acc_, s5, s6, st);
    add_bf16(s4, s5, s4, AH * DS, st);
    add_bf16(s4, s6, s4, AH * DS, st);  // s4 = d(n2s)
  }
  adarms_bwd(j.xmids, j.modf, j.rs2, s4, s5, dss_, AH, DS, st);
  pack_dmod(dss_, dgate, dmod_, DS, st);
  mod_bwd(cond_, w1.mw_f, dmod_, nullptr, nullptr, dcond_, DS, 3 * DS, st);
  bf16* dxmids = s6;
  add_bf16(dxs_out, s5, dxmids, AH * DS, st);
  // ================= prefix MLP: xp_out = xmidp + dp
  lora_bwd(w0.lin, j.ap, T, j.mlp, RP, 0, dxp_out, DP, 0, dmid_, accb_, pF_, pG_, st);  // [T][FP] lora, base
  add_bf16(pF_, pG_, pF_, T * FP, st);                                                 // d(ap)
  gelu_mul_bwd(j.gup, j.gup + (size_t)TCAP * FP, pF_, pG_, pH_, T * FP, st);           // pG_ = dg, pH_ = du
  {
    const LoraMat up = split(w0.g, 1, 0), gt = split(w0.g, 0, 0);
    lora_bwd(up, j.n2p, T, j.mgup + (size_t)TCAP * RP, RP, 0, pH_, FP, 0, dmid_, acc_, tA, tB, st);
    add_bf16(tA, tB, tA, T * DP, st);
    lora_bwd(gt, j.n2p, T, j.mgup, RP, 0, pG_, FP, 0, dmid_, acc_, tB, tC, st);
    add_bf16(tA, tB, tA, T * DP, st);
    add_bf16(tA, tC, tA, T * DP, st);  // tA = d(n2p)
  }
  PI05_CUDA(cudaMemcpyAsync(pseudo_p_, w0.s2, DP * 2, cudaMemcpyDeviceToDevice, st));
  adarms_bwd(j.xmidp, pseudo_p_, j.rp2, tA, tB, dss_, T, DP, st);
  add_bf16(dxp_out, tB, dxp, T * DP, st);  // dxp holds d(xmidp) until the end
  // ================= attention output projections
  gated_res_bwd(dxmids, j.os, j.moda + 2 * DS, s0, dgate, AH, DS, st);  // s0 = d(os)
  sum_heads_bf16(w1.o.B, bsum_, 8, (long long)RS * DS, st);
  bf16* dencs = sencs_;
  lout_bwd(w1.o, j.encs, AH, j.mos, bsum_, s0, dmid_, dbs_, dencs, sq_, st);
  sum_heads_bf16(w0.o.B, bsum_, 8, (long long)RP * DP, st);
  bf16* dencp = pF_;
  lout_bwd(w0.o, j.encp, T, j.mop, bsum_, dxp, dmid_, dbs_, dencp, tA, st);
  // ================= attention
  bf16 *dqp = pG_, *dkvp = pH_, *dqs = sq_, *dkvs = skv_;
  attention_bwd(l, dencp, dencs, dqp, dkvp, dqs, dkvs);
  // ================= suffix q / kv projections -> n1s -> adaRMS -> xs
  lora_bwd(w1.kv, j.n1s, AH, j.mkvs, 2 * RS, RS, dkvs, 2 * HD, HD, dmid_, acc_, s1, s2, st);
  add_bf16(s1, s2, s1, AH * DS, st);
  lora_bwd(w1.q, j.n1s, AH, j.mqs, NH * RS, RS, dqs, NH * HD, HD, dmid_, acc_, s2, s3, st);
  add_bf16(s1, s2, s1, AH * DS, st);
  add_bf16(s1, s3, s1, AH * DS, st);  // d(n1s)
  adarms_bwd(xs, j.moda, j.rs1, s1, s2, dss_, AH, DS, st);
  pack_dmod(dss_, dgate, dmod_, DS, st);
  mod_bwd(cond_, w1.mw_a, dmod_, nullptr, nullptr, dcond_, DS, 3 * DS, st);
  add_bf16(dxmids, s2, dxs, AH * DS, st);
  // ================= prefix q / kv projections -> n1p -> RMSNorm -> xp
  lora_bwd(w0.kv, j.n1p, T, j.mkvp, 2 * RP, RP, dkvp, 2 * HD, HD, dmid_, acc_, tA, tB, st);
  add_bf16(tA, tB, tA, T * DP, st);
  lora_bwd(w0.q, j.n1p, T, j.mqp, NH * RP, RP, dqp, NH * HD, HD, dmid_, acc_, tB, tC, st);
  add_bf16(tA, tB, tA, T * DP, st);
  add_bf16(tA, tC, tA, T * DP, st);  // d(n1p)
  PI05_CUDA(cudaMemcpyAsync(pseudo_p_, w0.s1, DP * 2, cudaMemcpyDeviceToDevice, st));
  adarms_bwd(xp, pseudo_p_, j.rp1, tA, tD, dss_, T, DP, st);
  add_bf16(dxp, tD, dxp, T * DP, st);
}

// ======================================================================================================================
// one sample: forward (layer inputs kept), loss, backward
// ======================================================================================================================
float LoraTrainer::accumulate(const LoraSample& s, float scale) {
  const size_t LD = (size_t)DI * DI;
  // ---- SigLIP stem (f32 conv with f32 params, + bias + pos_embedding in f32, then bf16; siglip.py:211-239)
  if (s.img_u8) {
    PI05_CUDA(cudaMemcpyAsync(img_u8_, s.img_u8, (size_t)NI * 224 * 224 * 3, cudaMemcpyHostToDevice, st));
    im2col_patches(img_u8_, nullptr, patches_, NI, st);
  } else {
    im2col_patches(nullptr, s.img_f32_dev, patches_, NI, st);
  }
  TParam *Pk = P(IMG + "embedding/kernel"), *Pb = P(IMG + "embedding/bias"), *Ppos = P(IMG + "pos_embedding");
  sgemm(true, false, TI, DI, 588, patches_, 588, Pk->p, DI, stem_, DI, false, st);
  stem_finish(stem_, Pb->p, Ppos->p, sig_in_[0], TI, st);
  prof_mark(0);
  for (int l = 0; l < Li_; ++l) siglip_layer_fwd(l, sig_in_[l], l + 1 < Li_ ? sig_in_[l + 1] : sig_last_);
  prof_mark(1);
  TParam *Ens = P(IMG + "Transformer/encoder_norm/scale"), *Enb = P(IMG + "Transformer/encoder_norm/bias");
  layernorm_fwd(sig_last_, Ens->p, Enb->p, enc_ln_, enc_mu_, enc_rs_, TI, DI, st);
  TParam *Hw = P(IMG + "head/kernel"), *Hb = P(IMG + "head/bias");
  tgemm<true, false>(G(enc_ln_, DI, Hw->pb, DP, TI, DP, DI), EBiasBf16{xp_in_[0], DP, Hb->pb}, 1, st);
  // ---- text tokens: frozen embedding rows * bf16(sqrt(2048)) (gemma.py:143-146)
  T_ = TI + (int)s.tokens.size();
  Tp8_ = r8(T_);
  S_ = T_ + AH;
  Sp_ = r8(S_);
  {
    std::vector<uint16_t> rows(s.tokens.size() * DP);
    const uint16_t* tab = reinterpret_cast<const uint16_t*>(embed_host_);
    for (size_t r = 0; r < s.tokens.size(); ++r)
      for (int i = 0; i < DP; ++i) {
        uint32_t u = (uint32_t)tab[(size_t)s.tokens[r] * DP + i] << 16;
        float f;
        memcpy(&f, &u, 4);
        f *= 45.25f;
        memcpy(&u, &f, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        rows[r * DP + i] = (uint16_t)(u >> 16);
      }
    PI05_CUDA(cudaMemcpyAsync(xp_in_[0] + (size_t)TI * DP, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice, st));
    PI05_CUDA(cudaStreamSynchronize(st));  // host staging buffer goes out of scope
  }
  // ---- suffix inputs (pi0.py:195-200, 161-167)
  PI05_CUDA(cudaMemcpyAsync(tmpf_, s.noise, AH * AD * 4, cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaMemcpyAsync(tmpf_ + AH * AD, s.actions, AH * AD * 4, cudaMemcpyHostToDevice, st));
  flow_inputs(tmpf_, tmpf_ + AH * AD, s.time, xt_, u_, AH * AD, st);
  PI05_CUDA(cudaMemcpyAsync(dc1_, &s.time, 4, cudaMemcpyHostToDevice, st));
  pi05::launch_time_embed(dc1_, 1, DS, temb_, st);
  lin_f32_fwd(temb_, nullptr, P("time_mlp_in/kernel")->p, P("time_mlp_in/bias")->p, z1_, c1_, 1, DS, DS, 1, st);
  lin_f32_fwd(c1_, nullptr, P("time_mlp_out/kernel")->p, P("time_mlp_out/bias")->p, z2_, cond_, 1, DS, DS, 1, st);
  lin_f32_fwd(xt_, nullptr, P("action_in_proj/kernel")->p, P("action_in_proj/bias")->p, nullptr, h0f_, AH, AD, DS, 0, st);
  f32_to_bf16(h0f_, xs_in_[0], AH * DS, st);
  prof_mark(2);
  // ---- joint layers
  for (int l = 0; l < Lg_; ++l)
    joint_layer_fwd(l, xp_in_[l], xs_in_[l], l + 1 < Lg_ ? xp_in_[l + 1] : xp_fin_, l + 1 < Lg_ ? xs_in_[l + 1] : xs_fin_);
  mod_fwd(cond_, F("PaliGemma/llm/final_norm_1/Dense_0/kernel"), F("PaliGemma/llm/final_norm_1/Dense_0/bias"), modfin_,
          DS, 3 * DS, st);
  adarms_fwd(xs_fin_, modfin_, y_, rfin_, AH, DS, st);
  lin_f32_fwd(nullptr, y_, P("action_out_proj/kernel")->p, P("action_out_proj/bias")->p, nullptr, v_, AH, DS, AD, 0, st);
  flow_loss(v_, u_, lossr_, dv_, AH, AD, scale / AH, st);

  prof_mark(3);
  // ================= backward
  PI05_CUDA(cudaMemsetAsync(dcond_, 0, DS * 4, st));
  lin_f32_bwd(nullptr, y_, P("action_out_proj/kernel")->p, nullptr, dv_, 0, dyf_, P("action_out_proj/kernel")->g,
              P("action_out_proj/bias")->g, AH, DS, AD, nullptr, st);
  f32_to_bf16(dyf_, dy_, AH * DS, st);
  bf16 *dxs = dxs_a_, *dxs2 = dxs_b_, *dxp = dxp_a_, *dxp2 = dxp_b_;
  adarms_bwd(xs_fin_, modfin_, rfin_, dy_, dxs, dss_, AH, DS, st);
  pack_dmod(dss_, nullptr, dmod_, DS, st);
  mod_bwd(cond_, F("PaliGemma/llm/final_norm_1/Dense_0/kernel"), dmod_, nullptr, nullptr, dcond_, DS, 3 * DS, st);
  PI05_CUDA(cudaMemsetAsync(dxp, 0, (size_t)T_ * DP * 2, st));  // the prefix output is not used by the loss
  for (int l = Lg_ - 1; l >= 0; --l) {
    joint_layer_bwd(l, xp_in_[l], xs_in_[l], dxp, dxs, dxp2, dxs2);
    std::swap(dxp, dxp2);
    std::swap(dxs, dxs2);
  }
  prof_mark(4);
  // heads
  bf16_to_f32(dxs, h0f_, AH * DS, st);
  lin_f32_bwd(xt_, nullptr, P("action_in_proj/kernel")->p, nullptr, h0f_, 0, nullptr, P("action_in_proj/kernel")->g,
              P("action_in_proj/bias")->g, AH, AD, DS, nullptr, st);
  lin_f32_bwd(c1_, nullptr, P("time_mlp_out/kernel")->p, z2_, dcond_, 1, dc1_, P("time_mlp_out/kernel")->g,
              P("time_mlp_out/bias")->g, 1, DS, DS, dyf_, st);
  lin_f32_bwd(temb_, nullptr, P("time_mlp_in/kernel")->p, z1_, dc1_, 1, nullptr, P("time_mlp_in/kernel")->g,
              P("time_mlp_in/bias")->g, 1, DS, DS, dyf_, st);
  prof_mark(5);
  // image tokens: head Dense, encoder_norm, SigLIP layers, stem
  const bf16* dimg = dxp;  // rows [0, 768)
  tgemm<false, false>(G(enc_ln_, DI, dimg, DP, DI, DP, TI), EAddF32{Hw->g, DP}, 1, st);
  colsum_bf16(dimg, Hb->g, TI, DP, DP, st);
  bf16 *dsa = sgd_[0], *dsb = sgd_[1];
  tgemm<true, true>(G(dimg, DP, Hw->pb, DP, TI, DI, DP), EStoreBf16{dsb, DI}, 1, st);
  layernorm_bwd(sig_last_, Ens->p, enc_mu_, enc_rs_, dsb, dsa, Ens->g, Enb->g, TI, DI, st);
  for (int l = Li_ - 1; l >= 0; --l) {
    siglip_layer_bwd(l, sig_in_[l], dsa, dsb);
    std::swap(dsa, dsb);
  }
  // stem: x0 = bf16(conv + b + pos) -> f32 cotangent
  bf16_to_f32(dsa, stem_, (long long)TI * DI, st);
  sgemm(false, false, 588, DI, TI, patches_, 588, stem_, DI, Pk->g, DI, true, st);
  stem_bias_pos_grads(stem_, Pb->g, Ppos->g, NI, st);
  (void)LD;
  prof_mark(6);
  prof_report();
  float lr[AH];
  PI05_CUDA(cudaMemcpyAsync(lr, lossr_, AH * 4, cudaMemcpyDeviceToHost, st));
  PI05_CUDA(cudaStreamSynchronize(st));
  double sum = 0;
  for (float x : lr) sum += x;
  return (float)(sum / AH);
}

}  // namespace pi05t

namespace pi05t {
// PI05_TRAIN_PROFILE=1: per-phase GPU time of accumulate() (events), averaged and printed every 32 samples
void LoraTrainer::prof_mark(int i) {
  static const bool on = getenv("PI05_TRAIN_PROFILE") != nullptr;
  if (!on) return;
  if (!prof_ev_[0]) for (auto& ev : prof_ev_) cudaEventCreate(&ev);
  cudaEventRecord(prof_ev_[i], st);
}
void LoraTrainer::prof_report() {
  static const bool on = getenv("PI05_TRAIN_PROFILE") != nullptr;
  if (!on) return;
  cudaEventSynchronize(prof_ev_[6]);
  for (int i = 0; i < 6; ++i) {
    float ms = 0;
    cudaEventElapsedTime(&ms, prof_ev_[i], prof_ev_[i + 1]);
    prof_ms_[i] += ms;
  }
  if (++prof_n_ % 32 == 0) {
    const char* nm[6] = {"siglip fwd", "prefix+text+suffix inputs", "joint fwd + loss", "joint bwd (remat)", "heads bwd",
                         "siglip bwd (remat) + stem"};
    printf("profile per sample:");
    for (int i = 0; i < 6; ++i) printf("  %s %.1f ms", nm[i], prof_ms_[i] / prof_n_);
    printf("\n");
  }
}
}  // namespace pi05t
