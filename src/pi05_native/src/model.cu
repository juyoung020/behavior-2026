#include "model.h"

#include <cstring>
#include <map>

#include "gemm.cuh"

namespace pi05 {

// ---------------------------------------------------------------------------------------------------
// GEMM dispatch: a handful of tile shapes, picked by estimated waves on this GPU.
// ---------------------------------------------------------------------------------------------------
namespace {

int g_sms = 70;

enum class Tile { L, M, S, K };

template <int BM, int BN, int WM, int WN, int ST, class Epi>
void launch_tile(const GemmParams& p, const Epi& epi, int batch, cudaStream_t st) {
  using C = GemmCfg<BM, BN, WM, WN, ST>;
  dim3 grid(cdiv(p.N, BN), cdiv(p.M, BM), batch * p.splits);
  gemm_nt_kernel<BM, BN, WM, WN, ST, Epi><<<grid, C::kThreads, C::kSmem, st>>>(p, epi);
}

Tile choose_tile(int M, int N, int batch) {
  if (M <= 32) return Tile::K;
  struct Opt { Tile t; int bm, bn, occ; };
  const Opt opts[] = {{Tile::L, 128, 128, 2}, {Tile::M, 128, 64, 2}, {Tile::S, 64, 64, 3}};
  Tile best = Tile::L;
  double best_cost = 1e30;
  for (const Opt& o : opts) {
    long ctas = (long)cdiv(M, o.bm) * cdiv(N, o.bn) * batch;
    long slots = (long)g_sms * o.occ;
    long waves = (ctas + slots - 1) / slots;
    // cost ~ waves * work per CTA / relative efficiency of the tile (bigger tiles reuse more)
    double eff = o.t == Tile::L ? 1.0 : o.t == Tile::M ? 0.9 : 0.75;
    double cost = waves * (double)o.bm * o.bn / o.occ / eff;
    if (cost < best_cost - 1e-9) { best_cost = cost; best = o.t; }
  }
  return best;
}

// C[M,N] = A[M,K] . B[N,K]^T with epilogue. `ws` is needed when the skinny tile splits K.
template <class Epi>
void gemm(GemmParams p, const Epi& epi, cudaStream_t st, float* ws = nullptr, size_t ws_floats = 0, int batch = 1) {
  Tile t = choose_tile(p.M, p.N, batch);
  if (t == Tile::K) {
    const int ctas = cdiv(p.N, 64) * batch;
    const int ktiles = cdiv(p.K, GEMM_BK);
    int splits = 1;
    while (ctas * splits < 2 * g_sms && ktiles / (splits * 2) >= 4 && batch == 1) splits *= 2;
    if (splits > 1 && (size_t)splits * p.M * p.N > ws_floats) splits = 1;
    p.splits = splits;
    p.ws = ws;
    launch_tile<32, 64, 1, 4, 6>(p, epi, batch, st);
    if (splits > 1) {
      const int pairs = p.M * p.N / 2;
      splitk_reduce_kernel<<<cdiv(pairs, 256), 256, 0, st>>>(ws, splits, p.M, p.N, p.dM, epi);
    }
    return;
  }
  p.splits = 1;
  if (t == Tile::L) launch_tile<128, 128, 2, 4, 3>(p, epi, batch, st);
  else if (t == Tile::M) launch_tile<128, 64, 2, 2, 4>(p, epi, batch, st);
  else launch_tile<64, 64, 2, 2, 4>(p, epi, batch, st);
}

GemmParams gp(const bf16* A, int lda, const bf16* B, int ldb, int M, int N, int K) {
  GemmParams p;
  p.A = A; p.lda = lda; p.B = B; p.ldb = ldb; p.M = M; p.N = N; p.K = K;
  return p;
}

uint16_t host_f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);  // NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
float host_bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------------------------------
Model::~Model() {
  if (graph_) cudaGraphExecDestroy(graph_);
  if (arena_) cudaFree(arena_);
  if (act_arena_) cudaFree(act_arena_);
  if (embed_host) cudaFreeHost(embed_host);
}

bool Model::load(const WeightFile& wf, std::string* err) {
  err->clear();
  cudaDeviceProp prop;
  PI05_CUDA(cudaGetDeviceProperties(&prop, 0));
  g_sms = prop.multiProcessorCount;
  cfg.ah = wf.cfg_int("action_horizon", 32);
  cfg.ad = wf.cfg_int("action_dim", 32);
  cfg.steps = wf.cfg_int("num_steps", 10);
  cfg.max_tok = wf.cfg_int("max_token_len", 200);
  cfg.depth = wf.cfg_int("llm.depth", 18);
  cfg.img_depth = wf.cfg_int("img.depth", 27);
  if (wf.cfg_int("llm.width", 0) != 2048 || wf.cfg_int("ae.width", 0) != 1024 || wf.cfg_int("img.width", 0) != 1152 ||
      wf.cfg_int("llm.head_dim", 0) != 256 || wf.cfg_int("llm.heads", 0) != 8 || wf.cfg_int("llm.kv_heads", 0) != 1) {
    *err = "unsupported model dimensions";
    return false;
  }

  // ---- persistent device weights: one arena ----
  std::vector<const TensorInfo*> dev;
  for (const TensorInfo& t : wf.tensors()) {
    const std::string& nm = t.name;
    if (t.dtype != DType::BF16) continue;
    if (nm == "llm.embed" || nm.find("_mod_") != std::string::npos || nm == "llm.final_norm") continue;
    if (nm.rfind("time_mlp", 0) == 0) continue;
    dev.push_back(&t);
  }
  std::map<std::string, size_t> off;
  size_t total = 0;
  for (auto* t : dev) {
    off[t->name] = total;
    total += (t->nbytes + 255) / 256 * 256;
  }
  PI05_CUDA(cudaMalloc(&arena_, total));
  weight_bytes = total;
  const size_t chunk = 64 << 20;
  uint8_t* stage = nullptr;
  PI05_CUDA(cudaMallocHost(&stage, chunk));
  for (auto* t : dev) {
    for (uint64_t o = 0; o < t->nbytes; o += chunk) {
      uint64_t n = std::min<uint64_t>(chunk, t->nbytes - o);
      if (!wf.read_range(*t, o, n, stage, err)) { cudaFreeHost(stage); return false; }
      PI05_CUDA(cudaMemcpy(arena_ + off[t->name] + o, stage, n, cudaMemcpyHostToDevice));
    }
  }
  cudaFreeHost(stage);
  auto W = [&](const std::string& nm) -> const bf16* {
    auto it = off.find(nm);
    if (it == off.end()) { *err = "missing weight " + nm; return nullptr; }
    return reinterpret_cast<const bf16*>(arena_ + it->second);
  };
  patch_w_ = W("img.patch_w"); patch_b_ = W("img.patch_b"); pos_ = W("img.pos");
  post_s_ = W("img.post_s"); post_b_ = W("img.post_b"); head_w_ = W("img.head_w"); head_b_ = W("img.head_b");
  ain_w_ = W("action_in_proj.w"); ain_b_ = W("action_in_proj.b");
  aout_w_ = W("action_out_proj.w"); aout_b_ = W("action_out_proj.b");
  il_.resize(cfg.img_depth);
  for (int l = 0; l < cfg.img_depth; ++l) {
    std::string p = "img.l" + std::to_string(l) + ".";
    il_[l] = {W(p + "ln1_s"), W(p + "ln1_b"), W(p + "qkv_w"), W(p + "qkv_b"), W(p + "o_w"), W(p + "o_b"),
              W(p + "ln2_s"), W(p + "ln2_b"), W(p + "fc1_w"), W(p + "fc1_b"), W(p + "fc2_w"), W(p + "fc2_b")};
  }
  ll_.resize(cfg.depth);
  al_.resize(cfg.depth);
  for (int l = 0; l < cfg.depth; ++l) {
    std::string p = "llm.l" + std::to_string(l) + ".", a = "ae.l" + std::to_string(l) + ".";
    ll_[l] = {W(p + "attn_norm"), W(p + "qkv_w"), W(p + "o_w"), W(p + "ffn_norm"), W(p + "gu_w"), W(p + "down_w")};
    al_[l] = {W(a + "qkv_w"), W(a + "o_w"), W(a + "gu_w"), W(a + "down_w")};
  }
  if (!err->empty()) return false;

  // ---- token embedding table stays in host memory (only <= 200 rows are used per call) ----
  const TensorInfo* emb = wf.find("llm.embed");
  if (!emb) { *err = "missing llm.embed"; return false; }
  PI05_CUDA(cudaMallocHost(&embed_host, emb->nbytes));
  if (!wf.read(*emb, embed_host, err)) return false;

  // ---- activations: one arena ----
  const int IT = cfg.img_tokens(), TC = cfg.t_cap(), SC = cfg.s_cap();
  struct Buf { void** p; size_t bytes; };
  ws_floats = (size_t)4 * cfg.ah * 8192;
  std::vector<Buf> bufs = {
      {(void**)&d_img, (size_t)cfg.n_img * 224 * 224 * 3},
      {(void**)&d_tokens, (size_t)cfg.max_tok * 4},
      {(void**)&d_dims, sizeof(Dims)},
      {(void**)&xi, (size_t)IT * cfg.img_w * 2},
      {(void**)&ni, (size_t)IT * cfg.img_w * 2},
      {(void**)&qkvi, (size_t)IT * 3 * cfg.img_w * 2},
      {(void**)&qsi, (size_t)IT * cfg.img_w * 2},
      {(void**)&vti, (size_t)IT * cfg.img_w * 2},
      {(void**)&si, (size_t)cfg.n_img * cfg.img_heads * 256 * 256 * 2},
      {(void**)&ai, (size_t)IT * cfg.img_w * 2},
      {(void**)&hi, (size_t)IT * cfg.img_mlp * 2},
      {(void**)&stem, (size_t)IT * cfg.img_w * 4},
      {(void**)&x, (size_t)TC * cfg.w * 2},
      {(void**)&n, (size_t)TC * cfg.w * 2},
      {(void**)&qkv, (size_t)TC * (cfg.heads + 2) * cfg.hd * 2},
      {(void**)&q, (size_t)TC * cfg.heads * cfg.hd * 2},
      {(void**)&lg, (size_t)TC * cfg.heads * SC * 4},
      {(void**)&pr, (size_t)TC * cfg.heads * SC * 2},
      {(void**)&att, (size_t)TC * cfg.w * 2},
      {(void**)&hid, (size_t)TC * cfg.mlp * 2},
      {(void**)&kc, (size_t)cfg.depth * SC * cfg.hd * 2},
      {(void**)&vt, (size_t)cfg.depth * cfg.hd * SC * 2},
      {(void**)&xt, (size_t)cfg.ah * cfg.ad * 4},
      {(void**)&h, (size_t)cfg.ah * cfg.ae_w * 2},
      {(void**)&hn, (size_t)cfg.ah * cfg.ae_w * 2},
      {(void**)&sqkv, (size_t)cfg.ah * (cfg.heads + 2) * cfg.hd * 2},
      {(void**)&sq, (size_t)cfg.ah * cfg.heads * cfg.hd * 2},
      {(void**)&slg, (size_t)cfg.ah * cfg.heads * SC * 4},
      {(void**)&spr, (size_t)cfg.ah * cfg.heads * SC * 2},
      {(void**)&satt, (size_t)cfg.ah * cfg.heads * cfg.hd * 2},
      {(void**)&shid, (size_t)cfg.ah * cfg.ae_mlp * 2},
      {(void**)&v, (size_t)cfg.ah * cfg.ad * 2},
      {(void**)&ws, ws_floats * 4},
      {(void**)&mods, ((size_t)cfg.steps * cfg.depth * 2 + cfg.steps) * 3 * cfg.ae_w * 2},
      {(void**)&rope, (size_t)(SC + 64) * 128 * sizeof(float2)},
  };
  size_t atot = 0;
  for (auto& b : bufs) atot += (b.bytes + 255) / 256 * 256;
  PI05_CUDA(cudaMalloc(&act_arena_, atot));
  PI05_CUDA(cudaMemset(act_arena_, 0, atot));  // K/V pad rows must be finite
  act_bytes = atot;
  size_t o = 0;
  for (auto& b : bufs) {
    *b.p = act_arena_ + o;
    o += (b.bytes + 255) / 256 * 256;
  }
  launch_rope_tables(rope, SC + 64, 128, 0);
  if (!precompute_modulation(wf, err)) return false;
  PI05_CUDA(cudaDeviceSynchronize());
  return true;
}

// adaRMS modulation depends only on the flow time, which takes the same 10 values every call
// (t = 1, 1-0.1, ... in f32, pi0.py:228-278), so scale/shift/gate for every layer are computed once here with
// the same arithmetic the model uses per step (pi0.py:161-167, gemma.py:128-130).
bool Model::precompute_modulation(const WeightFile& wf, std::string* err) {
  times.resize(cfg.steps);
  float t = 1.0f;
  const float dt = (float)(-1.0 / cfg.steps);
  for (int s = 0; s < cfg.steps; ++s) {
    times[s] = t;
    t = t + dt;
  }
  auto load_tmp = [&](const std::string& nm, bf16** dst) -> bool {
    const TensorInfo* ti = wf.find(nm);
    if (!ti) { *err = "missing " + nm; return false; }
    std::vector<uint8_t> hbuf(ti->nbytes);
    if (!wf.read(*ti, hbuf.data(), err)) return false;
    PI05_CUDA(cudaMalloc(dst, ti->nbytes));
    PI05_CUDA(cudaMemcpy(*dst, hbuf.data(), ti->nbytes, cudaMemcpyHostToDevice));
    return true;
  };
  const int S = cfg.steps, D = cfg.ae_w;
  float *d_t, *temb, *t1, *cond;
  PI05_CUDA(cudaMalloc(&d_t, S * 4));
  PI05_CUDA(cudaMalloc(&temb, (size_t)S * D * 4));
  PI05_CUDA(cudaMalloc(&t1, (size_t)S * D * 4));
  PI05_CUDA(cudaMalloc(&cond, (size_t)S * D * 4));
  PI05_CUDA(cudaMemcpy(d_t, times.data(), S * 4, cudaMemcpyHostToDevice));
  launch_time_embed(d_t, S, D, temb, 0);
  bf16 *wi, *bi, *wo, *bo;
  if (!load_tmp("time_mlp_in.w", &wi) || !load_tmp("time_mlp_in.b", &bi) || !load_tmp("time_mlp_out.w", &wo) ||
      !load_tmp("time_mlp_out.b", &bo))
    return false;
  launch_linear_f32(temb, wi, bi, t1, S, D, D, 1, 0);
  launch_linear_f32(t1, wo, bo, cond, S, D, D, 1, 0);
  cond_dbg.resize((size_t)S * D);
  PI05_CUDA(cudaMemcpy(cond_dbg.data(), cond, cond_dbg.size() * 4, cudaMemcpyDeviceToHost));
  for (bf16* p : {wi, bi, wo, bo}) cudaFree(p);
  // mods layout: [(l*2 + which)][s][3D], then final [s][3D]
  for (int l = 0; l <= cfg.depth; ++l) {
    for (int which = 0; which < 2; ++which) {
      if (l == cfg.depth && which == 1) break;
      std::string base = l == cfg.depth ? "ae.final_mod_" : "ae.l" + std::to_string(l) + (which ? ".ffn_mod_" : ".attn_mod_");
      bf16 *mw, *mb;
      if (!load_tmp(base + "w", &mw) || !load_tmp(base + "b", &mb)) return false;
      bf16* dst = mods + ((size_t)(l * 2 + which) * S) * 3 * D;
      launch_linear_bf16_vec(cond, mw, mb, dst, S, D, 3 * D, 0);
      PI05_CUDA(cudaDeviceSynchronize());
      cudaFree(mw);
      cudaFree(mb);
    }
  }
  for (float* p : {d_t, temb, t1, cond}) cudaFree(p);
  return true;
}

// ---------------------------------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------------------------------
void Model::do_tap(const std::string& name, const void* p, int dt, std::vector<int64_t> shape, cudaStream_t st) {
  if (!tap) return;
  PI05_CUDA(cudaStreamSynchronize(st));
  tap(name, p, dt, std::move(shape));
}

void Model::upload_inputs(const uint8_t* img, const std::vector<int>& tokens, const float* noise, cudaStream_t st) {
  const int L = (int)tokens.size();
  h_dims.L = L;
  h_dims.T = cfg.img_tokens() + L;
  h_dims.T8 = (h_dims.T + 7) / 8 * 8;
  h_dims.S = h_dims.T + cfg.ah;
  h_dims.S8 = (h_dims.S + 7) / 8 * 8;
  h_dims.TH = h_dims.T * cfg.heads;
  // text embedding rows gathered on the host: x = table[tok] * bf16(sqrt(2048)) (bf16 multiply)
  const size_t row = cfg.w;
  if (h_emb.size() < (size_t)cfg.max_tok * row) h_emb.resize((size_t)cfg.max_tok * row);
  const uint16_t* tab = reinterpret_cast<const uint16_t*>(embed_host);
  for (int r = 0; r < L; ++r) {
    const uint16_t* src = tab + (size_t)tokens[r] * row;
    uint16_t* dst = h_emb.data() + (size_t)r * row;
    for (size_t i = 0; i < row; ++i) dst[i] = host_f2bf(host_bf2f(src[i]) * 45.25f);
  }
  if (img) PI05_CUDA(cudaMemcpyAsync(d_img, img, (size_t)cfg.n_img * 224 * 224 * 3, cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaMemcpyAsync(d_tokens, tokens.data(), (size_t)L * 4, cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaMemcpyAsync(d_dims, &h_dims, sizeof(Dims), cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaMemcpyAsync(x + (size_t)cfg.img_tokens() * row, h_emb.data(), (size_t)L * row * 2,
                            cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaMemcpyAsync(xt, noise, (size_t)cfg.ah * cfg.ad * 4, cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaStreamSynchronize(st));  // host buffers may be reused by the caller
}

void Model::siglip_layer(int l, cudaStream_t st) {
  const ImgLayer& w = il_[l];
  const int M = cfg.img_tokens(), D = cfg.img_w;
  launch_layernorm(xi, w.ln1_s, w.ln1_b, ni, M, D, st);
  gemm(gp(ni, D, w.qkv_w, D, M, 3 * D, D), EpiBias{qkvi, 3 * D, w.qkv_b}, st);
  launch_siglip_attn_prep(qkvi, qsi, vti, cfg.n_img, st);
  {  // logits per (image, head): [256, 72] . [256, 72]^T -> bf16
    GemmParams p = gp(qsi, D, qkvi + D, 3 * D, 256, 256, cfg.img_hd);
    p.nb2 = cfg.img_heads;
    p.sA1 = 256LL * D; p.sA2 = cfg.img_hd;
    p.sB1 = 256LL * 3 * D; p.sB2 = cfg.img_hd;
    p.sC1 = (long long)cfg.img_heads * 65536; p.sC2 = 65536;
    gemm(p, EpiBf16{si, 256}, st, nullptr, 0, cfg.n_img * cfg.img_heads);
  }
  launch_softmax_bf16_rows(si, cfg.n_img * cfg.img_heads * 256, 256, st);
  {  // out per (image, head): P[256,256] . Vt[72,256]^T -> ai[img*256 + r, h*72 + d]
    GemmParams p = gp(si, 256, vti, 256, 256, cfg.img_hd, 256);
    p.nb2 = cfg.img_heads;
    p.sA1 = (long long)cfg.img_heads * 65536; p.sA2 = 65536;
    p.sB1 = (long long)cfg.img_heads * cfg.img_hd * 256; p.sB2 = (long long)cfg.img_hd * 256;
    p.sC1 = 256LL * D; p.sC2 = cfg.img_hd;
    gemm(p, EpiBf16{ai, D}, st, nullptr, 0, cfg.n_img * cfg.img_heads);
  }
  gemm(gp(ai, D, w.o_w, D, M, D, D), EpiBiasResid{xi, D, w.o_b, xi}, st);
  if (tap) for (int c = 0; c < cfg.n_img; ++c)
    do_tap("img.c" + std::to_string(c) + ".l" + std::to_string(l) + ".res1", xi + (size_t)c * 256 * D, 0, {256, D}, st);
  launch_layernorm(xi, w.ln2_s, w.ln2_b, ni, M, D, st);
  gemm(gp(ni, D, w.fc1_w, D, M, cfg.img_mlp, D), EpiBiasGelu{hi, cfg.img_mlp, w.fc1_b}, st);
  gemm(gp(hi, cfg.img_mlp, w.fc2_w, cfg.img_mlp, M, D, cfg.img_mlp), EpiBiasResid{xi, D, w.fc2_b, xi}, st);
  if (tap) for (int c = 0; c < cfg.n_img; ++c)
    do_tap("img.c" + std::to_string(c) + ".l" + std::to_string(l) + ".out", xi + (size_t)c * 256 * D, 0, {256, D}, st);
}

void Model::siglip_head(cudaStream_t st) {
  const int M = cfg.img_tokens(), D = cfg.img_w;
  launch_layernorm(xi, post_s_, post_b_, ni, M, D, st);
  if (tap) for (int c = 0; c < cfg.n_img; ++c)
    do_tap("img.c" + std::to_string(c) + ".encoded", ni + (size_t)c * 256 * D, 0, {256, D}, st);
  gemm(gp(ni, D, head_w_, D, M, cfg.w, D), EpiBias{x, cfg.w, head_b_}, st);
  if (tap) for (int c = 0; c < cfg.n_img; ++c)
    do_tap("img.c" + std::to_string(c) + ".tokens", x + (size_t)c * 256 * cfg.w, 0, {256, cfg.w}, st);
}

void Model::siglip(cudaStream_t st) {
  launch_stem(d_img, patch_w_, patch_b_, pos_, tap ? stem : nullptr, xi, cfg.n_img, st);
  if (tap) for (int c = 0; c < cfg.n_img; ++c) {
    do_tap("img.c" + std::to_string(c) + ".stem", stem + (size_t)c * 256 * cfg.img_w, 1, {256, cfg.img_w}, st);
    do_tap("img.c" + std::to_string(c) + ".x0", xi + (size_t)c * 256 * cfg.img_w, 0, {256, cfg.img_w}, st);
  }
  for (int l = 0; l < cfg.img_depth; ++l) siglip_layer(l, st);
  siglip_head(st);
}

void Model::text_embed(cudaStream_t st) {
  if (tap) do_tap("txt.emb", x + (size_t)cfg.img_tokens() * cfg.w, 0, {h_dims.L, cfg.w}, st);
}

void Model::prefix_layer(int l, cudaStream_t st) {
  const LlmLayer& w = ll_[l];
  const int TC = cfg.t_cap(), D = cfg.w, H = cfg.heads, HD = cfg.hd, SC = cfg.s_cap();
  const int* dT = &d_dims->T;
  const std::string p = "pre.l" + std::to_string(l);
  launch_rmsnorm(x, w.attn_norm, n, TC, D, dT, st);
  if (tap) do_tap(p + ".norm1", n, 0, {h_dims.T, D}, st);
  {
    GemmParams g = gp(n, D, w.qkv_w, D, TC, (H + 2) * HD, D);
    g.dM = dT;
    gemm(g, EpiBf16{qkv, (H + 2) * HD}, st);
  }
  if (tap) do_tap(p + ".qkv", qkv, 0, {h_dims.T, (H + 2) * HD}, st);
  bf16* kcl = kc + (size_t)l * SC * HD;
  bf16* vtl = vt + (size_t)l * HD * SC;
  launch_rope_split(qkv, rope, q, kcl, vtl, TC, dT, H, SC, 0, nullptr, st);
  if (tap) {
    do_tap(p + ".k", kcl, 0, {h_dims.T, HD}, st);
    do_tap(p + ".vt", vtl, 0, {HD, SC}, st);
  }
  {  // logits [T*H, T] f32 (query rows are (token, head))
    GemmParams g = gp(q, HD, kcl, HD, TC * H, TC, HD);
    g.dM = &d_dims->TH;
    g.dN = dT;
    gemm(g, EpiF32{lg, SC}, st);
  }
  launch_softmax_f32_rows(lg, SC, pr, SC, TC * H, &d_dims->TH, dT, &d_dims->T8, st);
  {
    GemmParams g = gp(pr, SC, vtl, SC, TC * H, HD, SC);
    g.dM = &d_dims->TH;
    g.dK = &d_dims->T8;
    gemm(g, EpiBf16{att, HD}, st);
  }
  {
    GemmParams g = gp(att, D, w.o_w, D, TC, D, D);
    g.dM = dT;
    gemm(g, EpiResid{x, D, x}, st);
  }
  if (tap) do_tap(p + ".res1", x, 0, {h_dims.T, D}, st);
  launch_rmsnorm(x, w.ffn_norm, n, TC, D, dT, st);
  if (tap) do_tap(p + ".norm2", n, 0, {h_dims.T, D}, st);
  {
    GemmParams g = gp(n, D, w.gu_w, D, TC, 2 * cfg.mlp, D);
    g.dM = dT;
    gemm(g, EpiGeluGate{hid, cfg.mlp}, st);
  }
  {
    GemmParams g = gp(hid, cfg.mlp, w.down_w, cfg.mlp, TC, D, cfg.mlp);
    g.dM = dT;
    gemm(g, EpiResid{x, D, x}, st);
  }
  if (tap) do_tap(p + ".out", x, 0, {h_dims.T, D}, st);
}

void Model::suffix_layer(int s, int l, cudaStream_t st) {
  const AeLayer& w = al_[l];
  const int A = cfg.ah, D = cfg.ae_w, H = cfg.heads, HD = cfg.hd, SC = cfg.s_cap();
  const std::string p = "suf.s" + std::to_string(s) + ".l" + std::to_string(l);
  bf16* m1 = mod(s, l, 0);
  bf16* m2 = mod(s, l, 1);
  launch_adarms(h, m1, hn, A, D, st);
  if (tap) do_tap(p + ".norm1", hn, 0, {A, D}, st);
  gemm(gp(hn, D, w.qkv_w, D, A, (H + 2) * HD, D), EpiBf16{sqkv, (H + 2) * HD}, st, ws, ws_floats);
  bf16* kcl = kc + (size_t)l * SC * HD;
  bf16* vtl = vt + (size_t)l * HD * SC;
  launch_rope_split(sqkv, rope, sq, kcl, vtl, A, nullptr, H, SC, 0, &d_dims->T, st);
  {
    GemmParams g = gp(sq, HD, kcl, HD, A * H, SC, HD);
    g.dN = &d_dims->S;
    gemm(g, EpiF32{slg, SC}, st);
  }
  launch_softmax_f32_rows(slg, SC, spr, SC, A * H, nullptr, &d_dims->S, &d_dims->S8, st);
  {
    GemmParams g = gp(spr, SC, vtl, SC, A * H, HD, SC);
    g.dK = &d_dims->S8;
    gemm(g, EpiBf16{satt, HD}, st);
  }
  gemm(gp(satt, H * HD, w.o_w, H * HD, A, D, H * HD), EpiGatedResid{h, D, h, m1 + 2 * D}, st, ws, ws_floats);
  if (tap) do_tap(p + ".res1", h, 0, {A, D}, st);
  launch_adarms(h, m2, hn, A, D, st);
  if (tap) do_tap(p + ".norm2", hn, 0, {A, D}, st);
  gemm(gp(hn, D, w.gu_w, D, A, 2 * cfg.ae_mlp, D), EpiGeluGate{shid, cfg.ae_mlp}, st, ws, ws_floats);
  gemm(gp(shid, cfg.ae_mlp, w.down_w, cfg.ae_mlp, A, D, cfg.ae_mlp), EpiGatedResid{h, D, h, m2 + 2 * D}, st, ws,
       ws_floats);
  if (tap) do_tap(p + ".out", h, 0, {A, D}, st);
}

void Model::suffix_head(int s, cudaStream_t st) {
  const int A = cfg.ah, D = cfg.ae_w;
  launch_adarms(h, final_mod(s), hn, A, D, st);
  if (tap) do_tap("suf.s" + std::to_string(s) + ".final", hn, 0, {A, D}, st);
  gemm(gp(hn, D, aout_w_, D, A, cfg.ad, D), EpiBias{v, cfg.ad, aout_b_}, st, ws, ws_floats);
  if (tap) do_tap("suf.s" + std::to_string(s) + ".v", v, 0, {A, cfg.ad}, st);
  launch_flow_update(xt, v, A * cfg.ad, st);
  if (tap) do_tap("suf.s" + std::to_string(s) + ".x", xt, 1, {A, cfg.ad}, st);
}

void Model::action_in(int s, cudaStream_t st) {
  launch_action_in(xt, ain_w_, ain_b_, h, cfg.ah, cfg.ad, cfg.ae_w, st);
  if (tap) do_tap("suf.s" + std::to_string(s) + ".in", h, 0, {cfg.ah, cfg.ae_w}, st);
}

void Model::denoise_step(int s, cudaStream_t st) {
  action_in(s, st);
  for (int l = 0; l < cfg.depth; ++l) suffix_layer(s, l, st);
  suffix_head(s, st);
}

void Model::forward_eager(cudaStream_t st) {
  siglip(st);
  text_embed(st);
  for (int l = 0; l < cfg.depth; ++l) prefix_layer(l, st);
  for (int s = 0; s < cfg.steps; ++s) denoise_step(s, st);
}

bool Model::capture_graph(cudaStream_t st, std::string* err) {
  Tap saved = tap;
  tap = nullptr;
  cudaGraph_t g;
  if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { *err = "begin capture"; return false; }
  forward_eager(st);
  if (cudaStreamEndCapture(st, &g) != cudaSuccess) { *err = "end capture"; return false; }
  if (cudaGraphInstantiate(&graph_, g, 0) != cudaSuccess) { *err = "instantiate"; return false; }
  cudaGraphDestroy(g);
  tap = saved;
  return true;
}

void Model::forward(cudaStream_t st) {
  if (graph_ && !tap) PI05_CUDA(cudaGraphLaunch(graph_, st));
  else forward_eager(st);
}

void Model::download_actions(float* out, cudaStream_t st) {
  PI05_CUDA(cudaMemcpyAsync(out, xt, (size_t)cfg.ah * cfg.ad * 4, cudaMemcpyDeviceToHost, st));
  PI05_CUDA(cudaStreamSynchronize(st));
}

}  // namespace pi05
