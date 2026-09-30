#include "trainer.h"

#include <cmath>
#include <cstring>

#include "tgemm.cuh"

namespace pi05t {
namespace {
const std::string LY = "PaliGemma/llm/layers/";
const std::string N_Q = LY + "attn/q_einsum_1/w", N_KV = LY + "attn/kv_einsum_1/w", N_O = LY + "attn/attn_vec_einsum_1/w",
                  N_G = LY + "mlp_1/gating_einsum", N_LIN = LY + "mlp_1/linear",
                  N_AMW = LY + "pre_attention_norm_1/Dense_0/kernel", N_AMB = LY + "pre_attention_norm_1/Dense_0/bias",
                  N_FMW = LY + "pre_ffw_norm_1/Dense_0/kernel", N_FMB = LY + "pre_ffw_norm_1/Dense_0/bias",
                  N_FINW = "PaliGemma/llm/final_norm_1/Dense_0/kernel", N_FINB = "PaliGemma/llm/final_norm_1/Dense_0/bias",
                  N_INW = "action_in_proj/kernel", N_INB = "action_in_proj/bias", N_OUTW = "action_out_proj/kernel",
                  N_OUTB = "action_out_proj/bias", N_T1W = "time_mlp_in/kernel", N_T1B = "time_mlp_in/bias",
                  N_T2W = "time_mlp_out/kernel", N_T2B = "time_mlp_out/bias";
inline int r8(int x) { return (x + 7) / 8 * 8; }

TGemm G(const bf16* A, long long lda, const bf16* B, long long ldb, int M, int N, int K) {
  TGemm p;
  p.A = A; p.lda = lda; p.B = B; p.ldb = ldb; p.M = M; p.N = N; p.K = K;
  return p;
}
}  // namespace

Trainer::~Trainer() {
  if (st) cudaStreamDestroy(st);
}

bool Trainer::init(const std::string& state_path, const std::string& model_path, std::string* err, int offload) {
  PI05_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  pi05::WeightFile mf;
  if (!mf.open(model_path, err)) return false;
  if (!prefix_.load(mf, err)) return false;
  if (!wf_.open(state_path, err)) return false;
  opt.read(wf_);
  if (wf_.cfg_str("mode", "expert") != "expert") { *err = "only mode expert is implemented"; return false; }
  L_ = wf_.cfg_int("llm.depth", 18);
  if (L_ != prefix_.cfg.depth) { *err = "state/model depth mismatch"; return false; }

  auto is_llm = [](const std::string& n) { return n.rfind("PaliGemma/llm/", 0) == 0; };
  if (!ps_.load(wf_, offload, opt, mem_, is_llm, is_llm, st, err)) return false;  // expert einsums + bf16 Dense
  for (const std::string& n : {N_Q, N_KV, N_O, N_G, N_LIN, N_AMW, N_AMB, N_FMW, N_FMB, N_FINW, N_FINB, N_INW, N_INB,
                               N_OUTW, N_OUTB, N_T1W, N_T1B, N_T2W, N_T2B})
    if (!param(n)) { *err = "missing trainable " + n; return false; }
  rope_ = dalloc<float2>(1024 * 128);
  pi05::launch_rope_tables(rope_, 1024, 128, st);
  alloc_work();
  PI05_CUDA(cudaStreamSynchronize(st));
  return true;
}

void Trainer::alloc_work() {
  const int SC = 1024;  // key capacity (968 prefix + 32 suffix, padded)
  xt_ = dalloc<float>(AH * AD); u_ = dalloc<float>(AH * AD);
  temb_ = dalloc<float>(W); z1_ = dalloc<float>(W); c1_ = dalloc<float>(W); z2_ = dalloc<float>(W);
  cond_ = dalloc<float>(W); h0f_ = dalloc<float>(AH * W); v_ = dalloc<float>(AH * AD); dv_ = dalloc<float>(AH * AD);
  lossr_ = dalloc<float>(AH); dcond_ = dalloc<float>(W); dc1_ = dalloc<float>(W); dtmp_ = dalloc<float>(AH * W);
  dyf_ = dalloc<float>(AH * W); dss_ = dalloc<float>(2 * W); dk32_ = dalloc<float>((size_t)SC * HD);
  acc32_ = dalloc<float>(AH * W); dh0f_ = dalloc<float>(AH * W);
  lay_.resize(L_);
  for (auto& l : lay_) {
    l.h_in = dalloc<bf16>(AH * W); l.mod_a = dalloc<bf16>(3 * W); l.n1 = dalloc<bf16>(AH * W);
    l.q = dalloc<bf16>(AH * NH * HD); l.kv = dalloc<bf16>(AH * 2 * HD);
    l.pb = dalloc<bf16>((size_t)AH * NH * SC); l.p32 = dalloc<float>((size_t)AH * NH * SC);
    l.enc = dalloc<bf16>(AH * NH * HD); l.o = dalloc<bf16>(AH * W); l.h_mid = dalloc<bf16>(AH * W);
    l.mod_f = dalloc<bf16>(3 * W); l.n2 = dalloc<bf16>(AH * W);
    l.g = dalloc<bf16>(2 * AH * F); l.u = l.g + AH * F; l.a = dalloc<bf16>(AH * F); l.d = dalloc<bf16>(AH * W);
    l.r1 = dalloc<float>(AH); l.r2 = dalloc<float>(AH);
  }
  h_fin_ = dalloc<bf16>(AH * W); mod_fin_ = dalloc<bf16>(3 * W); y_ = dalloc<bf16>(AH * W); r_fin_ = dalloc<float>(AH);
  Kall_ = dalloc<bf16>((size_t)SC * HD); Vall_ = dalloc<bf16>((size_t)SC * HD); K3_ = dalloc<bf16>((size_t)3 * SC * HD);
  logits_ = dalloc<float>((size_t)AH * NH * SC);
  dh_ = dalloc<bf16>(AH * W); dh2_ = dalloc<bf16>(AH * W); dx_ = dalloc<bf16>(AH * W); dd_ = dalloc<bf16>(AH * W);
  da_ = dalloc<bf16>(AH * F); dg_ = dalloc<bf16>(2 * AH * F); du_ = dg_ + AH * F;
  dn_ = dalloc<bf16>(2 * AH * W); dn_b_ = dn_ + AH * W; do_ = dalloc<bf16>(AH * W);
  denc_ = dalloc<bf16>(AH * NH * HD); dp_ = dalloc<bf16>((size_t)AH * NH * SC); d3_ = dalloc<bf16>((size_t)3 * AH * NH * SC);
  dq_ = dalloc<bf16>(AH * NH * HD); dkv_ = dalloc<bf16>(AH * 2 * HD); dvall_ = dalloc<bf16>((size_t)SC * HD);
  dmod_ = dalloc<bf16>(3 * W); dy_ = dalloc<bf16>(AH * W);
}

void Trainer::zero_grads() { ps_.zero_grads(st); }

// K_all / V_all for layer l: rows [0, Tp) = prefix cache of the frozen inference engine, [Tp, Tp+32) = suffix k / v
// of this layer (after RoPE), rows up to Sp zero.
static void build_kv(const pi05::Model& m, int l, const bf16* kv, int Tp, int Sp, bf16* K, bf16* V, cudaStream_t st) {
  const int SC = m.cfg.s_cap();
  PI05_CUDA(cudaMemcpyAsync(K, m.kc + (size_t)l * SC * 256, (size_t)Tp * 256 * 2, cudaMemcpyDeviceToDevice, st));
  transpose_bf16(m.vt + (size_t)l * 256 * SC, V, 256, Tp, SC, 256, st);
  copy_cols(kv, 512, K + (size_t)Tp * 256, 256, 32, 256, st);
  copy_cols(kv + 256, 512, V + (size_t)Tp * 256, 256, 32, 256, st);
  const int S = Tp + 32;
  if (Sp > S) {
    PI05_CUDA(cudaMemsetAsync(K + (size_t)S * 256, 0, (size_t)(Sp - S) * 256 * 2, st));
    PI05_CUDA(cudaMemsetAsync(V + (size_t)S * 256, 0, (size_t)(Sp - S) * 256 * 2, st));
  }
}

void Trainer::forward_suffix(const Sample& s) {
  // frozen prefix through the inference engine (bf16 frozen params = the inference weights)
  prefix_.ext_imgf = s.img_f32_dev;
  prefix_.upload_inputs(s.img_f32_dev ? nullptr : s.img, s.tokens, s.noise, st);
  prefix_.siglip(st);
  prefix_.text_embed(st);
  for (int l = 0; l < L_; ++l) prefix_.prefix_layer(l, st);
  Tp_ = 768 + (int)s.tokens.size();
  S_ = Tp_ + AH;
  Sp_ = r8(S_);
  // flow matching inputs (pi0.py:196-200) and time conditioning (pi0.py:161-167, f32 params)
  float* tn = dtmp_;  // noise, actions staged in dtmp_ [0, 2048)
  PI05_CUDA(cudaMemcpyAsync(tn, s.noise, AH * AD * 4, cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaMemcpyAsync(tn + AH * AD, s.actions, AH * AD * 4, cudaMemcpyHostToDevice, st));
  flow_inputs(tn, tn + AH * AD, s.time, xt_, u_, AH * AD, st);
  PI05_CUDA(cudaMemcpyAsync(dc1_, &s.time, 4, cudaMemcpyHostToDevice, st));
  pi05::launch_time_embed(dc1_, 1, W, temb_, st);
  lin_f32_fwd(temb_, nullptr, param(N_T1W)->p, param(N_T1B)->p, z1_, c1_, 1, W, W, 1, st);
  lin_f32_fwd(c1_, nullptr, param(N_T2W)->p, param(N_T2B)->p, z2_, cond_, 1, W, W, 1, st);
  lin_f32_fwd(xt_, nullptr, param(N_INW)->p, param(N_INB)->p, nullptr, h0f_, AH, AD, W, 0, st);
  f32_to_bf16(h0f_, lay_[0].h_in, AH * W, st);  // embedded.astype(bf16) (gemma.py:400)

  const bf16 *wq = param(N_Q)->pb, *wkv = param(N_KV)->pb, *wo = param(N_O)->pb, *wg = param(N_G)->pb,
             *wl = param(N_LIN)->pb, *amw = param(N_AMW)->pb, *amb = param(N_AMB)->pb, *fmw = param(N_FMW)->pb,
             *fmb = param(N_FMB)->pb;
  for (int l = 0; l < L_; ++l) {
    Lay& y = lay_[l];
    mod_fwd(cond_, amw + (size_t)l * W * 3 * W, amb + (size_t)l * 3 * W, y.mod_a, W, 3 * W, st);
    adarms_fwd(y.h_in, y.mod_a, y.n1, y.r1, AH, W, st);
    {  // q = n1 . Wq[n] per head -> [t][n][h]; k, v -> kv [t][2][h]  (gemma.py:185-201)
      TGemm p = G(y.n1, W, wq + (size_t)l * NH * W * HD, HD, AH, HD, W);
      p.sB1 = (long long)W * HD; p.sC1 = HD;
      tgemm<true, false>(p, EStoreBf16{y.q, NH * HD}, NH, st);
      TGemm p2 = G(y.n1, W, wkv + (size_t)l * 2 * W * HD, HD, AH, HD, W);
      p2.sB1 = (long long)W * HD; p2.sC1 = HD;
      tgemm<true, false>(p2, EStoreBf16{y.kv, 2 * HD}, 2, st);
    }
    rope_fwd(y.q, rope_, AH, NH, NH * HD, Tp_, true, st);
    rope_fwd(y.kv, rope_, AH, 1, 2 * HD, Tp_, false, st);
    build_kv(prefix_, l, y.kv, Tp_, Sp_, Kall_, Vall_, st);
    // logits [(t,n)][s] f32 = q . K^T ; probs ; enc = probs . V   (gemma.py:212-230)
    tgemm<true, true>(G(y.q, HD, Kall_, HD, AH * NH, Sp_, HD), EStoreF32{logits_, Sp_}, 1, st);
    softmax_fwd(logits_, y.p32, y.pb, AH * NH, S_, Sp_, st);
    tgemm<true, false>(G(y.pb, Sp_, Vall_, HD, AH * NH, HD, Sp_), EStoreBf16{y.enc, HD}, 1, st);
    tgemm<true, false>(G(y.enc, NH * HD, wo + (size_t)l * NH * HD * W, W, AH, W, NH * HD), EStoreBf16{y.o, W}, 1, st);
    gated_res_fwd(y.h_in, y.o, y.mod_a + 2 * W, y.h_mid, AH, W, st);
    mod_fwd(cond_, fmw + (size_t)l * W * 3 * W, fmb + (size_t)l * 3 * W, y.mod_f, W, 3 * W, st);
    adarms_fwd(y.h_mid, y.mod_f, y.n2, y.r2, AH, W, st);
    {  // gate, up (gemma.py:344-352)
      TGemm p = G(y.n2, W, wg + (size_t)l * 2 * W * F, F, AH, F, W);
      p.sB1 = (long long)W * F; p.sC1 = (long long)AH * F;
      tgemm<true, false>(p, EStoreBf16{y.g, F}, 2, st);
    }
    gelu_mul_fwd(y.g, y.u, y.a, AH * F, st);
    tgemm<true, false>(G(y.a, F, wl + (size_t)l * F * W, W, AH, W, F), EStoreBf16{y.d, W}, 1, st);
    gated_res_fwd(y.h_mid, y.d, y.mod_f + 2 * W, l + 1 < L_ ? lay_[l + 1].h_in : h_fin_, AH, W, st);
  }
  mod_fwd(cond_, param(N_FINW)->pb, param(N_FINB)->pb, mod_fin_, W, 3 * W, st);
  adarms_fwd(h_fin_, mod_fin_, y_, r_fin_, AH, W, st);
  lin_f32_fwd(nullptr, y_, param(N_OUTW)->p, param(N_OUTB)->p, nullptr, v_, AH, W, AD, 0, st);
}

void Trainer::backward_suffix(const Sample&) {
  PI05_CUDA(cudaMemsetAsync(dcond_, 0, W * 4, st));
  // action_out_proj (f32): dy = dv . W^T -> bf16 (cotangent of the bf16 input), dW += y^T dv
  lin_f32_bwd(nullptr, y_, param(N_OUTW)->p, nullptr, dv_, 0, dyf_, param(N_OUTW)->g, param(N_OUTB)->g, AH, W, AD,
              nullptr, st);
  f32_to_bf16(dyf_, dy_, AH * W, st);
  adarms_bwd(h_fin_, mod_fin_, r_fin_, dy_, dh_, dss_, AH, W, st);
  pack_dmod(dss_, nullptr, dmod_, W, st);
  mod_bwd(cond_, param(N_FINW)->pb, dmod_, param(N_FINW)->g, param(N_FINB)->g, dcond_, W, 3 * W, st);

  TParam *Pq = param(N_Q), *Pkv = param(N_KV), *Po = param(N_O), *Pg = param(N_G), *Pl = param(N_LIN),
         *Pamw = param(N_AMW), *Pamb = param(N_AMB), *Pfmw = param(N_FMW), *Pfmb = param(N_FMB);
  float* dgate = acc32_;             // [W]
  float* acc = acc32_ + W;           // [AH][W] scratch accumulator (acc32_ has AH*W floats; use dtmp_ instead)
  acc = dtmp_;
  for (int l = L_ - 1; l >= 0; --l) {
    Lay& y = lay_[l];
    // ---- MLP block: h_out = h_mid + d * gate_f
    gated_res_bwd(dh_, y.d, y.mod_f + 2 * W, dd_, dgate, AH, W, st);
    tgemm<true, true>(G(dd_, W, Pl->pb + (size_t)l * F * W, W, AH, F, W), EStoreBf16{da_, F}, 1, st);
    tgemm<false, false>(G(y.a, F, dd_, W, F, W, AH), EAddF32{Pl->g + (size_t)l * F * W, W}, 1, st);
    gelu_mul_bwd(y.g, y.u, da_, dg_, du_, AH * F, st);
    {
      TGemm p = G(y.n2, W, dg_, F, W, F, AH);
      p.sB1 = (long long)AH * F; p.sC1 = (long long)W * F;
      tgemm<false, false>(p, EAddF32{Pg->g + (size_t)l * 2 * W * F, F}, 2, st);
      TGemm q = G(dg_, F, Pg->pb + (size_t)l * 2 * W * F, F, AH, W, F);
      q.sA1 = (long long)AH * F; q.sB1 = (long long)W * F; q.sC1 = (long long)AH * W;
      tgemm<true, true>(q, EStoreBf16{dn_, W}, 2, st);  // dn_ = dg.W0^T, dn_b_ = du.W1^T
      add_bf16(dn_b_, dn_, dn_, AH * W, st);
    }
    adarms_bwd(y.h_mid, y.mod_f, y.r2, dn_, dx_, dss_, AH, W, st);
    pack_dmod(dss_, dgate, dmod_, W, st);
    mod_bwd(cond_, Pfmw->pb + (size_t)l * W * 3 * W, dmod_, Pfmw->g + (size_t)l * W * 3 * W, Pfmb->g + (size_t)l * 3 * W,
            dcond_, W, 3 * W, st);
    add_bf16(dh_, dx_, dh2_, AH * W, st);  // cotangent of h_mid
    // ---- attention block: h_mid = h_in + o * gate_a
    gated_res_bwd(dh2_, y.o, y.mod_a + 2 * W, do_, dgate, AH, W, st);
    tgemm<true, true>(G(do_, W, Po->pb + (size_t)l * NH * HD * W, W, AH, NH * HD, W), EStoreBf16{denc_, NH * HD}, 1, st);
    tgemm<false, false>(G(y.enc, NH * HD, do_, W, NH * HD, W, AH), EAddF32{Po->g + (size_t)l * NH * HD * W, W}, 1, st);
    build_kv(prefix_, l, y.kv, Tp_, Sp_, Kall_, Vall_, st);
    tgemm<true, true>(G(denc_, HD, Vall_, HD, AH * NH, Sp_, HD), EStoreBf16{dp_, Sp_}, 1, st);
    tgemm<false, false>(G(y.pb, Sp_, denc_, HD, Sp_, HD, AH * NH), EStoreBf16{dvall_, HD}, 1, st);
    softmax_bwd_split(y.p32, dp_, d3_, AH * NH, S_, Sp_, st);
    for (int k = 0; k < 3; ++k)
      PI05_CUDA(cudaMemcpyAsync(K3_ + (size_t)k * Sp_ * HD, Kall_, (size_t)Sp_ * HD * 2, cudaMemcpyDeviceToDevice, st));
    tgemm<true, false>(G(d3_, 3LL * Sp_, K3_, HD, AH * NH, HD, 3 * Sp_), EStoreBf16{dq_, HD}, 1, st);
    PI05_CUDA(cudaMemsetAsync(dk32_, 0, (size_t)Sp_ * HD * 4, st));
    for (int k = 0; k < 3; ++k)
      tgemm<false, false>(G(d3_ + (size_t)k * Sp_, 3LL * Sp_, y.q, HD, Sp_, HD, AH * NH), EAddF32{dk32_, HD}, 1, st);
    f32_to_bf16(dk32_ + (size_t)Tp_ * HD, dx_, AH * HD, st);  // suffix rows of dK (dx_ as scratch)
    copy_cols(dx_, HD, dkv_, 2 * HD, AH, HD, st);
    copy_cols(dvall_ + (size_t)Tp_ * HD, HD, dkv_ + HD, 2 * HD, AH, HD, st);
    rope_bwd(dq_, rope_, AH, NH, NH * HD, Tp_, true, st);
    rope_bwd(dkv_, rope_, AH, 1, 2 * HD, Tp_, false, st);
    {  // weight grads of q / kv, input grads summed in f32 and rounded once per einsum
      TGemm p = G(y.n1, W, dq_, NH * HD, W, HD, AH);
      p.sB1 = HD; p.sC1 = (long long)W * HD;
      tgemm<false, false>(p, EAddF32{Pq->g + (size_t)l * NH * W * HD, HD}, NH, st);
      TGemm p2 = G(y.n1, W, dkv_, 2 * HD, W, HD, AH);
      p2.sB1 = HD; p2.sC1 = (long long)W * HD;
      tgemm<false, false>(p2, EAddF32{Pkv->g + (size_t)l * 2 * W * HD, HD}, 2, st);
      PI05_CUDA(cudaMemsetAsync(acc, 0, AH * W * 4, st));
      for (int k = 0; k < 2; ++k)
        tgemm<true, true>(G(dkv_ + k * HD, 2 * HD, Pkv->pb + ((size_t)l * 2 + k) * W * HD, HD, AH, W, HD), EAddF32{acc, W},
                          1, st);
      f32_to_bf16(acc, dn_b_, AH * W, st);
      PI05_CUDA(cudaMemsetAsync(acc, 0, AH * W * 4, st));
      for (int n = 0; n < NH; ++n)
        tgemm<true, true>(G(dq_ + n * HD, NH * HD, Pq->pb + ((size_t)l * NH + n) * W * HD, HD, AH, W, HD),
                          EAddF32{acc, W}, 1, st);
      f32_to_bf16(acc, dn_, AH * W, st);
      add_bf16(dn_b_, dn_, dn_, AH * W, st);
    }
    adarms_bwd(y.h_in, y.mod_a, y.r1, dn_, dx_, dss_, AH, W, st);
    pack_dmod(dss_, dgate, dmod_, W, st);
    mod_bwd(cond_, Pamw->pb + (size_t)l * W * 3 * W, dmod_, Pamw->g + (size_t)l * W * 3 * W, Pamb->g + (size_t)l * 3 * W,
            dcond_, W, 3 * W, st);
    add_bf16(dh2_, dx_, dh_, AH * W, st);  // cotangent of h_in
  }
  // action_in_proj (f32 in, f32 out, then .astype(bf16)): dW += x_t^T f32(dh0)
  bf16_to_f32(dh_, dh0f_, AH * W, st);
  lin_f32_bwd(xt_, nullptr, param(N_INW)->p, nullptr, dh0f_, 0, nullptr, param(N_INW)->g, param(N_INB)->g, AH, AD, W,
              nullptr, st);
  // time MLP: cond = swish(z2), z2 = c1 W2 + b2, c1 = swish(z1), z1 = temb W1 + b1
  lin_f32_bwd(c1_, nullptr, param(N_T2W)->p, z2_, dcond_, 1, dc1_, param(N_T2W)->g, param(N_T2B)->g, 1, W, W, dyf_, st);
  lin_f32_bwd(temb_, nullptr, param(N_T1W)->p, z1_, dc1_, 1, nullptr, param(N_T1W)->g, param(N_T1B)->g, 1, W, W, dyf_,
              st);
}

float Trainer::accumulate(const Sample& s, float scale) {
  forward_suffix(s);
  flow_loss(v_, u_, lossr_, dv_, AH, AD, scale / AH, st);
  backward_suffix(s);
  float lr[AH];
  PI05_CUDA(cudaMemcpyAsync(lr, lossr_, AH * 4, cudaMemcpyDeviceToHost, st));
  PI05_CUDA(cudaStreamSynchronize(st));
  double sum = 0;
  for (float x : lr) sum += x;
  return (float)(sum / AH);
}

void Trainer::finalize_grads() { ps_.finalize(st); }
double Trainer::grad_norm() { return ps_.grad_norm(st); }
void Trainer::opt_step() {
  ps_.step(opt, count_, st);
  ++count_;
}

}  // namespace pi05t
