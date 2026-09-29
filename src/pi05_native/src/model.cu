#include "model.h"

#include <cstring>
#include <map>

#include "gemm.cuh"
#include "pb_kernels.cuh"

namespace pi05 {

// ---------------------------------------------------------------------------------------------------
// GEMM dispatch: a handful of tile shapes, picked by estimated waves on this GPU.
// ---------------------------------------------------------------------------------------------------
namespace {

int g_sms = 70;

enum class Tile { L, M, S, K };

template <int BM, int BN, int WM, int WN, int ST, int BK, class Epi>
void launch_tile(const GemmParams& p, const Epi& epi, int batch, cudaStream_t st) {
  using C = GemmCfg<BM, BN, WM, WN, ST, BK>;
  auto kern = gemm_nt_kernel<BM, BN, WM, WN, ST, BK, Epi>;
  static bool attr = false;  // one static per template instance
  if (!attr && C::kSmem > 48 * 1024) {
    PI05_CUDA(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem));
    attr = true;
  }
  dim3 grid(cdiv(p.M, BM), cdiv(p.N, BN), batch * p.splits);
  launch_k(kern, grid, C::kThreads, C::kSmem, st, p, epi);
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
int* g_counters = nullptr;  // split-K tile tickets, zeroed once; every reducing CTA re-arms its own

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
    p.counters = g_counters;
    launch_tile<32, 64, 1, 4, 6, 32>(p, epi, batch, st);
    if (splits > 1 && !p.counters) {
      const int pairs = p.M * p.N / 2;
      launch_k(splitk_reduce_kernel<Epi>, cdiv(pairs, 256), 256, 0, st, (const float*)ws, splits, p.M, p.N, p.dM, epi);
    }
    return;
  }
  p.splits = 1;
  if (t == Tile::L) launch_tile<128, 128, 2, 4, 3, 32>(p, epi, batch, st);
  else if (t == Tile::M) launch_tile<128, 64, 2, 2, 4, 32>(p, epi, batch, st);
  else launch_tile<64, 64, 2, 2, 4, 32>(p, epi, batch, st);
}

GemmParams gp(const bf16* A, int lda, const bf16* B, int ldb, int M, int N, int K) {
  GemmParams p;
  p.A = A; p.lda = lda; p.B = B; p.ldb = ldb; p.M = M; p.N = N; p.K = K;
  return p;
}
// B is a model weight: its tiles may be prefetched before the PDL wait
GemmParams gw(const bf16* A, int lda, const bf16* B, int ldb, int M, int N, int K) {
  GemmParams p = gp(A, lda, B, ldb, M, N, K);
  p.b_static = true;
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
  cfg.pb = wf.cfg_str("model", "pi05") == "pi_behavior";
  if (cfg.pb) {
    cfg.extra_dev = 5;   // base task + 4 fused task/stage tokens (pi_behavior.py:577-590)
    cfg.max_tok = 32;    // one token per padded state dim (pi_behavior.py:592-609)
    cfg.g0 = cfg.img_tokens() + 1;  // images + base task token attend among themselves only
    cfg.num_tasks = wf.cfg_int("pb.num_tasks", 50);
    cfg.stages.clear();
    if (auto v = wf.cfg("pb.stages"))
      for (auto& s : *v) cfg.stages.push_back(std::stoi(s));
    if ((int)cfg.stages.size() != cfg.num_tasks) { *err = "pb.stages size"; return false; }
    inpaint_threshold = std::stof(wf.cfg_str("pb.inpaint_threshold", "0.3"));
  }
  cfg.dtb = host_bf2f(host_f2bf((float)(-1.0 / cfg.steps)));
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
    if (nm == "llm.embed" || nm.find("_mod_") != std::string::npos) continue;
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
  std::vector<uint8_t> perm;
  for (auto* t : dev) {
    const std::string& nm = t->name;
    const bool rope_rows = nm.size() > 6 && nm.compare(nm.size() - 6, 6, ".qkv_w") == 0 && nm.rfind("img.", 0) != 0;
    if (rope_rows) {
      // q and k rows of every 256-wide head reordered to (0,128,1,129,...) so the GEMM epilogue sees RoPE pairs
      // side by side (EpiQKVRope); v rows unchanged.
      const int64_t rows = t->shape[0], K = t->shape[1];
      std::vector<uint8_t> src(t->nbytes);
      if (!wf.read(*t, src.data(), err)) { cudaFreeHost(stage); return false; }
      perm.assign(t->nbytes, 0);
      const int64_t rb = K * 2, rope_blocks = rows / 256 - 1;  // all heads + k; the last block is v
      for (int64_t r = 0; r < rows; ++r) {
        int64_t srow = r;
        if (r / 256 < rope_blocks) {
          const int64_t c = r % 256;
          srow = (r / 256) * 256 + (c >> 1) + (c & 1) * 128;
        }
        memcpy(&perm[r * rb], &src[srow * rb], rb);
      }
      PI05_CUDA(cudaMemcpy(arena_ + off[nm], perm.data(), t->nbytes, cudaMemcpyHostToDevice));
      continue;
    }
    for (uint64_t o = 0; o < t->nbytes; o += chunk) {
      uint64_t n = std::min<uint64_t>(chunk, t->nbytes - o);
      if (!wf.read_range(*t, o, n, stage, err)) { cudaFreeHost(stage); return false; }
      PI05_CUDA(cudaMemcpy(arena_ + off[nm] + o, stage, n, cudaMemcpyHostToDevice));
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
  final_norm_ = W("llm.final_norm");
  if (cfg.pb) {
    pb_task_emb_ = W("pb.task_emb");
    pb_stage_emb_ = W("pb.task_stage_emb");
    const char* lin[7] = {"gate_sincos", "gate_task_stage", "gate_task", "fusion_layer1", "fusion_layer2",
                          "stage_projection", "stage_pred_from_vlm"};
    for (int i = 0; i < 7; ++i) {
      pb_w_[i] = W(std::string("pb.") + lin[i] + ".w");
      pb_b_[i] = W(std::string("pb.") + lin[i] + ".b");
    }
    const char* kv[4] = {"k_coeffs", "v_coeffs", "k_bias", "v_bias"};
    for (int i = 0; i < 4; ++i) pb_kv_[i] = W(std::string("pb.kv.") + kv[i]);
  }
  if (!err->empty()) return false;

  // ---- token embedding table stays in host memory (only <= 200 rows are used per call) ----
  // PiBehavior only embeds state bins 0..255, so only those rows are read.
  const TensorInfo* emb = wf.find("llm.embed");
  if (!emb) { *err = "missing llm.embed"; return false; }
  const uint64_t emb_bytes = cfg.pb ? (uint64_t)256 * cfg.w * 2 : emb->nbytes;
  PI05_CUDA(cudaMallocHost(&embed_host, emb_bytes));
  if (!wf.read_range(*emb, 0, emb_bytes, embed_host, err)) return false;

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
      {(void**)&counters, 4096 * 4},
      {(void**)&mods, ((size_t)cfg.steps * cfg.depth * 2 + cfg.steps) * 3 * cfg.ae_w * 2},
      {(void**)&rope, (size_t)(SC + 64) * 128 * sizeof(float2)},
  };
  if (cfg.pb) {
    const std::vector<Buf> pbb = {
        {(void**)&pb_ints, 16},
        {(void**)&pb_nstages, (size_t)cfg.num_tasks * 4},
        {(void**)&pb_offsets, (size_t)cfg.num_tasks * 4},
        {(void**)&pb_all, 4096 * 4},
        {(void**)&pb_gsc, 1024 * 4},
        {(void**)&pb_gts, 1024 * 4},
        {(void**)&pb_gt, 2048 * 4},
        {(void**)&pb_h1, 4096 * 4},
        {(void**)&pb_bal, 2048 * 4},
        {(void**)&pb_sf, 2048 * 4},
        {(void**)&pb_sd, 2048 * 4},
        {(void**)&pb_x0O, 512 * 4},
        {(void**)&pb_zO, 512 * 4},
        {(void**)&pb_C, (size_t)pb_nU * pb_nO * 4},
        {(void**)&pb_stage_n, 2048 * 2},
        {(void**)&pb_logits, 16 * 2},
        {(void**)&kc2, (size_t)cfg.depth * SC * cfg.hd * 2},
        {(void**)&vt2, (size_t)cfg.depth * cfg.hd * SC * 2},
    };
    bufs.insert(bufs.end(), pbb.begin(), pbb.end());
  }
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
  g_counters = counters;
  if (cfg.pb) {
    std::vector<int> offs(cfg.num_tasks, 0);
    for (int i = 1; i < cfg.num_tasks; ++i) offs[i] = offs[i - 1] + cfg.stages[i - 1];
    PI05_CUDA(cudaMemcpy(pb_nstages, cfg.stages.data(), cfg.num_tasks * 4, cudaMemcpyHostToDevice));
    PI05_CUDA(cudaMemcpy(pb_offsets, offs.data(), cfg.num_tasks * 4, cudaMemcpyHostToDevice));
    const TensorInfo* tc = wf.find("pb.inpaint_C4");
    const TensorInfo* tl = wf.find("pb.corr_L");
    if (!tc || !tl || tc->numel() != (int64_t)pb_nU * pb_nO) { *err = "missing pb.inpaint_C4 / pb.corr_L"; return false; }
    std::vector<float> Ch(tc->numel());
    if (!wf.read(*tc, Ch.data(), err)) return false;
    PI05_CUDA(cudaMemcpy(pb_C, Ch.data(), Ch.size() * 4, cudaMemcpyHostToDevice));
    corr_L.resize(tl->numel());
    if (!wf.read(*tl, corr_L.data(), err)) return false;
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
  t_new.resize(cfg.steps);
  for (int s = 0; s < cfg.steps; ++s) {
    times[s] = t;
    t = t + dt;
    t_new[s] = t;  // time after this Euler step (pi_behavior.py:1064 time_new = time + dt)
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
  h_dims.L = cfg.extra_dev + L;
  h_dims.T = cfg.img_tokens() + cfg.extra_dev + L;
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
  PI05_CUDA(cudaMemcpyAsync(x + (size_t)(cfg.img_tokens() + cfg.extra_dev) * row, h_emb.data(), (size_t)L * row * 2,
                            cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaMemcpyAsync(xt, noise, (size_t)cfg.ah * cfg.ad * 4, cudaMemcpyHostToDevice, st));
  PI05_CUDA(cudaStreamSynchronize(st));  // host buffers may be reused by the caller
}

void Model::set_pb_inputs(int task, int stage, const float* x0O, const float* zO, int nO, cudaStream_t st) {
  if (!cfg.pb) return;
  h_pb_ints[0] = task;
  h_pb_ints[1] = stage;
  h_pb_ints[2] = (x0O && zO && nO == pb_nO) ? 1 : 0;
  PI05_CUDA(cudaMemcpyAsync(pb_ints, h_pb_ints, sizeof h_pb_ints, cudaMemcpyHostToDevice, st));
  if (h_pb_ints[2]) {
    PI05_CUDA(cudaMemcpyAsync(pb_x0O, x0O, (size_t)nO * 4, cudaMemcpyHostToDevice, st));
    PI05_CUDA(cudaMemcpyAsync(pb_zO, zO, (size_t)nO * 4, cudaMemcpyHostToDevice, st));
  }
  PI05_CUDA(cudaStreamSynchronize(st));
}

void Model::download_stage_logits(float* out, cudaStream_t st) {
  uint16_t hb[16];
  PI05_CUDA(cudaMemcpyAsync(hb, pb_logits, 15 * 2, cudaMemcpyDeviceToHost, st));
  PI05_CUDA(cudaStreamSynchronize(st));
  for (int i = 0; i < 15; ++i) out[i] = host_bf2f(hb[i]);
}

void Model::siglip_layer(int l, cudaStream_t st) {
  const ImgLayer& w = il_[l];
  const int M = cfg.img_tokens(), D = cfg.img_w;
  launch_layernorm(xi, w.ln1_s, w.ln1_b, ni, M, D, st);
  gemm(gw(ni, D, w.qkv_w, D, M, 3 * D, D), EpiSiglipQKV{qkvi, qsi, vti, w.qkv_b}, st);
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
  gemm(gw(ai, D, w.o_w, D, M, D, D), EpiBiasResid{xi, D, w.o_b, xi}, st);
  if (tap) for (int c = 0; c < cfg.n_img; ++c)
    do_tap("img.c" + std::to_string(c) + ".l" + std::to_string(l) + ".res1", xi + (size_t)c * 256 * D, 0, {256, D}, st);
  launch_layernorm(xi, w.ln2_s, w.ln2_b, ni, M, D, st);
  gemm(gw(ni, D, w.fc1_w, D, M, cfg.img_mlp, D), EpiBiasGelu{hi, cfg.img_mlp, w.fc1_b}, st);
  gemm(gw(hi, cfg.img_mlp, w.fc2_w, cfg.img_mlp, M, D, cfg.img_mlp), EpiBiasResid{xi, D, w.fc2_b, xi}, st);
  if (tap) for (int c = 0; c < cfg.n_img; ++c)
    do_tap("img.c" + std::to_string(c) + ".l" + std::to_string(l) + ".out", xi + (size_t)c * 256 * D, 0, {256, D}, st);
}

void Model::siglip_head(cudaStream_t st) {
  const int M = cfg.img_tokens(), D = cfg.img_w;
  launch_layernorm(xi, post_s_, post_b_, ni, M, D, st);
  if (tap) for (int c = 0; c < cfg.n_img; ++c)
    do_tap("img.c" + std::to_string(c) + ".encoded", ni + (size_t)c * 256 * D, 0, {256, D}, st);
  gemm(gw(ni, D, head_w_, D, M, cfg.w, D), EpiBias{x, cfg.w, head_b_}, st);
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
  if (cfg.pb) {  // task + stage tokens (pi_behavior.py:452-517), rows [img_tokens, img_tokens + 5)
    bf16* rows = x + (size_t)cfg.img_tokens() * cfg.w;
    launch_pb_task_prep(PbTaskArgs{pb_task_emb_, pb_stage_emb_, pb_ints, pb_nstages, pb_offsets}, pb_all, rows, st);
    launch_linear_f32(pb_all, pb_w_[0], pb_b_[0], pb_gsc, 1, 4096, 1024, 2, st);  // sigmoid(gate_sincos)
    launch_linear_f32(pb_all, pb_w_[1], pb_b_[1], pb_gts, 1, 4096, 1024, 2, st);  // sigmoid(gate_task_stage)
    launch_linear_f32(pb_all, pb_w_[2], pb_b_[2], pb_gt, 1, 4096, 2048, 2, st);   // sigmoid(gate_task)
    launch_linear_f32(pb_all, pb_w_[3], pb_b_[3], pb_h1, 1, 4096, 4096, 3, st);   // relu(fusion_layer1)
    launch_linear_f32(pb_h1, pb_w_[4], pb_b_[4], pb_bal, 1, 4096, 2048, 0, st);   // fusion_layer2
    launch_pb_stage_feat(pb_all, pb_gsc, pb_gts, pb_sf, st);
    launch_linear_f32(pb_sf, pb_w_[5], pb_b_[5], pb_sd, 1, 2048, 2048, 0, st);    // stage_projection
    launch_pb_task_rows(pb_all, pb_gt, pb_bal, pb_sd, rows, st);
  }
  if (tap && cfg.pb) do_tap("pre.x0", x, 0, {h_dims.T, cfg.w}, st);
  if (tap) do_tap("txt.emb", x + (size_t)cfg.img_tokens() * cfg.w, 0, {h_dims.L, cfg.w}, st);
}

void Model::prefix_tail(cudaStream_t st) {
  if (!cfg.pb) return;
  // stage head: final_norm(prefix_out[base task token]) -> stage_pred_from_vlm (pi_behavior.py:1009-1022)
  launch_rmsnorm(x + (size_t)cfg.img_tokens() * cfg.w, final_norm_, pb_stage_n, 1, cfg.w, nullptr, st);
  launch_linear_bf16in(pb_stage_n, pb_w_[6], pb_b_[6], pb_logits, cfg.w, 15, st);
  // each action-expert layer attends to a learned mix of all VLM layers' K/V (pi_behavior.py:1024-1026)
  launch_kv_transform(kc, vt, kc2, vt2, pb_kv_[0], pb_kv_[1], pb_kv_[2], pb_kv_[3], cfg.depth, cfg.s_cap(), &d_dims->T,
                      st);
  if (tap) {
    do_tap("pre.stage_logits", pb_logits, 0, {15}, st);
    do_tap("pre.base_final", pb_stage_n, 0, {cfg.w}, st);
    for (int l : {0, cfg.depth - 1}) {
      do_tap("pre.kv2.l" + std::to_string(l) + ".k", kc2 + (size_t)l * cfg.s_cap() * cfg.hd, 0, {h_dims.T, cfg.hd}, st);
      do_tap("pre.kv2.l" + std::to_string(l) + ".vt", vt2 + (size_t)l * cfg.hd * cfg.s_cap(), 0, {cfg.hd, cfg.s_cap()}, st);
    }
  }
}

void Model::prefix_layer(int l, cudaStream_t st) {
  const LlmLayer& w = ll_[l];
  const int TC = cfg.t_cap(), D = cfg.w, H = cfg.heads, HD = cfg.hd, SC = cfg.s_cap();
  const int* dT = &d_dims->T;
  const std::string p = "pre.l" + std::to_string(l);
  launch_rmsnorm(x, w.attn_norm, n, TC, D, dT, st);
  if (tap) do_tap(p + ".norm1", n, 0, {h_dims.T, D}, st);
  {
    GemmParams g = gw(n, D, w.qkv_w, D, TC, (H + 2) * HD, D);
    g.dM = dT;
    gemm(g, EpiQKVRope{q, kc + (size_t)l * SC * HD, vt + (size_t)l * HD * SC, rope, nullptr, H, SC}, st);
  }
  bf16* kcl = kc + (size_t)l * SC * HD;
  bf16* vtl = vt + (size_t)l * HD * SC;
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
  launch_softmax_f32_rows(lg, SC, pr, SC, TC * H, &d_dims->TH, dT, &d_dims->T8, st, cfg.g0, H);
  {
    GemmParams g = gp(pr, SC, vtl, SC, TC * H, HD, SC);
    g.dM = &d_dims->TH;
    g.dK = &d_dims->T8;
    gemm(g, EpiBf16{att, HD}, st);
  }
  {
    GemmParams g = gw(att, D, w.o_w, D, TC, D, D);
    g.dM = dT;
    gemm(g, EpiResid{x, D, x}, st);
  }
  if (tap) do_tap(p + ".res1", x, 0, {h_dims.T, D}, st);
  launch_rmsnorm(x, w.ffn_norm, n, TC, D, dT, st);
  if (tap) do_tap(p + ".norm2", n, 0, {h_dims.T, D}, st);
  {
    GemmParams g = gw(n, D, w.gu_w, D, TC, 2 * cfg.mlp, D);
    g.dM = dT;
    gemm(g, EpiGeluGate{hid, cfg.mlp}, st);
  }
  {
    GemmParams g = gw(hid, cfg.mlp, w.down_w, cfg.mlp, TC, D, cfg.mlp);
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
  bf16* kcl = (cfg.pb ? kc2 : kc) + (size_t)l * SC * HD;  // PiBehavior: layer-mixed prefix cache
  bf16* vtl = (cfg.pb ? vt2 : vt) + (size_t)l * HD * SC;
  gemm(gw(hn, D, w.qkv_w, D, A, (H + 2) * HD, D), EpiQKVRope{sq, kcl, vtl, rope, &d_dims->T, H, SC}, st, ws, ws_floats);
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
  gemm(gw(satt, H * HD, w.o_w, H * HD, A, D, H * HD), EpiGatedResid{h, D, h, m1 + 2 * D}, st, ws, ws_floats);
  if (tap) do_tap(p + ".res1", h, 0, {A, D}, st);
  launch_adarms(h, m2, hn, A, D, st);
  if (tap) do_tap(p + ".norm2", hn, 0, {A, D}, st);
  gemm(gw(hn, D, w.gu_w, D, A, 2 * cfg.ae_mlp, D), EpiGeluGate{shid, cfg.ae_mlp}, st, ws, ws_floats);
  gemm(gw(shid, cfg.ae_mlp, w.down_w, cfg.ae_mlp, A, D, cfg.ae_mlp), EpiGatedResid{h, D, h, m2 + 2 * D}, st, ws,
       ws_floats);
  if (tap) do_tap(p + ".out", h, 0, {A, D}, st);
}

void Model::suffix_head(int s, cudaStream_t st) {
  const int A = cfg.ah, D = cfg.ae_w;
  launch_adarms(h, final_mod(s), hn, A, D, st);
  if (tap) do_tap("suf.s" + std::to_string(s) + ".final", hn, 0, {A, D}, st);
  gemm(gw(hn, D, aout_w_, D, A, cfg.ad, D), EpiBias{v, cfg.ad, aout_b_}, st, ws, ws_floats);
  if (tap) do_tap("suf.s" + std::to_string(s) + ".v", v, 0, {A, cfg.ad}, st);
  launch_flow_update(xt, v, A * cfg.ad, cfg.dtb, st);
  if (cfg.pb && t_new[s] > inpaint_threshold)  // soft inpainting only while time_new > 0.3 (pi_behavior.py:1101-1107)
    launch_inpaint(xt, pb_x0O, pb_zO, pb_C, pb_nO, pb_nU, t_new[s], pb_ints, st);
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
  prefix_tail(st);
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
