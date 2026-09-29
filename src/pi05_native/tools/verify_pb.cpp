// pi05_verify_pb: PiBehavior (2025 1st place) engine vs JAX reference dumps (tools/dump_reference_pb.py).
//   pi05_verify_pb --weights pb.pi05w --ref DIR [--tag gpu] [--floor cpu] [--samples 0-3] [--check] [--time]
// Points: prefix embeddings, every prefix layer, stage head, layer-mixed KV cache, every denoising step, final raw and
// robot-unit actions, and an end-to-end rolling-inpainting call driven by the same JAX key stream as the policy.
// Floor: another JAX run (--floor tag) and the JAX per-layer chain vs JAX fused (XLA fusion noise).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../src/jax_rng.h"
#include "../src/model.h"
#include "../src/pb_host.h"
#include "../src/weights.h"

using namespace pi05;

namespace {
float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

struct Arr { std::vector<float> v; std::vector<int64_t> shape; bool ok = false; };
Arr read_arr(const WeightFile& f, const std::string& name) {
  Arr a;
  const TensorInfo* t = f.find(name);
  if (!t) return a;
  std::string err;
  std::vector<uint8_t> raw(t->nbytes);
  if (!f.read(*t, raw.data(), &err)) return a;
  a.shape = t->shape;
  a.v.resize(t->numel());
  for (int64_t i = 0; i < t->numel(); ++i) {
    switch (t->dtype) {
      case DType::BF16: a.v[i] = bf2f(reinterpret_cast<uint16_t*>(raw.data())[i]); break;
      case DType::F32: a.v[i] = reinterpret_cast<float*>(raw.data())[i]; break;
      case DType::F64: a.v[i] = (float)reinterpret_cast<double*>(raw.data())[i]; break;
      case DType::I32: a.v[i] = (float)reinterpret_cast<int32_t*>(raw.data())[i]; break;
      case DType::U8: a.v[i] = raw[i]; break;
    }
  }
  a.ok = true;
  return a;
}
std::vector<double> read_f64(const WeightFile& f, const std::string& name) {
  const TensorInfo* t = f.find(name);
  std::vector<double> v(t ? t->numel() : 0);
  std::string err;
  if (t) f.read(*t, v.data(), &err);
  return v;
}

struct Stat { double rel = 0, maxabs = 0, eq = 0; };
Stat compare(const float* a, const float* b, int64_t n) {
  Stat s;
  double num = 0, den = 0;
  int64_t eq = 0, cnt = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (!std::isfinite(b[i]) && !std::isfinite(a[i])) { ++eq; ++cnt; continue; }
    const double d = (double)a[i] - b[i];
    num += d * d;
    den += (double)b[i] * b[i];
    s.maxabs = std::max(s.maxabs, std::fabs(d));
    eq += a[i] == b[i];
    ++cnt;
  }
  s.rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  s.eq = cnt ? (double)eq / cnt : 1;
  return s;
}

struct Row { std::string name; Stat ours, floor; bool hf = false; };
void add(std::vector<Row>& rows, const std::string& name, Stat o, const Stat* f) {
  for (Row& r : rows)
    if (r.name == name) {
      r.ours.rel = std::max(r.ours.rel, o.rel); r.ours.maxabs = std::max(r.ours.maxabs, o.maxabs);
      r.ours.eq = std::min(r.ours.eq, o.eq);
      if (f) { r.floor.rel = std::max(r.floor.rel, f->rel); r.floor.maxabs = std::max(r.floor.maxabs, f->maxabs); r.hf = true; }
      return;
    }
  rows.push_back({name, o, f ? *f : Stat{}, f != nullptr});
}

std::map<std::string, Arr> g_taps;
Tap make_tap() {
  return [](const std::string& name, const void* p, int dt, std::vector<int64_t> shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    Arr a;
    a.shape = shape;
    a.v.resize(n);
    if (dt == 1) PI05_CUDA(cudaMemcpy(a.v.data(), p, n * 4, cudaMemcpyDeviceToHost));
    else {
      std::vector<uint16_t> t(n);
      PI05_CUDA(cudaMemcpy(t.data(), p, n * 2, cudaMemcpyDeviceToHost));
      for (int64_t i = 0; i < n; ++i) a.v[i] = bf2f(t[i]);
    }
    a.ok = true;
    g_taps[name] = a;
  };
}
}  // namespace

int main(int argc, char** argv) {
  std::string weights, ref, tag = "gpu", floor_tag, samples = "0-3";
  bool check = false, time_it = false;
  double factor = 2.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--weights") weights = next();
    else if (a == "--ref") ref = next();
    else if (a == "--tag") tag = next();
    else if (a == "--floor") floor_tag = next();
    else if (a == "--samples") samples = next();
    else if (a == "--check") check = true;
    else if (a == "--time") time_it = true;
    else if (a == "--factor") factor = std::stod(next());
  }
  std::string err;
  WeightFile wf;
  if (!wf.open(weights, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  PbSpec spec;
  if (!load_pb_spec(wf, &spec, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  Model m;
  if (!m.load(wf, &err)) { fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
  printf("PiBehavior weights on GPU %.2f GiB, activations %.2f GiB, steps %d, horizon %d\n",
         m.weight_bytes / 1073741824.0, m.act_bytes / 1073741824.0, m.cfg.steps, m.cfg.ah);
  cudaStream_t st;
  PI05_CUDA(cudaStreamCreate(&st));
  const int ah = spec.ah, ad = spec.ad, n = ah * ad;
  std::vector<Row> rows;
  int fails = 0;
  auto dump = [&](const std::string& t, int i) {
    char b[64];
    snprintf(b, sizeof b, "%s_%03d.pi05d", t.c_str(), i);
    return ref + "/" + b;
  };
  const int s0 = std::stoi(samples.substr(0, samples.find('-'))), s1 = std::stoi(samples.substr(samples.find('-') + 1));
  std::vector<uint8_t> img(3 * 224 * 224 * 3);
  for (int si = s0; si <= s1; ++si) {
    WeightFile R;
    if (!R.open(dump(tag, si), &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    WeightFile F;
    const bool hf = !floor_tag.empty() && F.open(dump(floor_tag, si), &err);
    Arr ai = read_arr(R, "in.img"), ap = read_arr(R, "in.proprio"), an = read_arr(R, "in.noise"), ats = read_arr(R, "in.task_stage");
    for (size_t k = 0; k < img.size(); ++k) img[k] = (uint8_t)ai.v[k];
    const int task = (int)ats.v[0], stage = (int)ats.v[1];
    float st23[23], norm32[32];
    std::vector<int> tok(32);
    pb_extract_state(ap.v.data(), (int)ap.v.size(), st23);
    pb_state_tokens(spec, st23, norm32, tok.data());
    Arr hs = read_arr(R, "host.state");
    Stat shs = compare(norm32, hs.v.data(), std::min<int64_t>(32, hs.v.size()));
    add(rows, "host.state (normalized)", shs, nullptr);
    // ---- chain with taps, explicit noise, no inpainting
    g_taps.clear();
    m.tap = make_tap();
    m.set_pb_inputs(task, stage, nullptr, nullptr, 0, st);
    m.upload_inputs(img.data(), tok, an.v.data(), st);
    m.forward_eager(st);
    m.tap = nullptr;
    std::vector<float> raw(n);
    m.download_actions(raw.data(), st);
    float logits[15];
    m.download_stage_logits(logits, st);
    auto cmp_tap = [&](const std::string& tapname, const std::string& refname, const std::string& rowname, int64_t count,
                       bool transpose_vt = false) {
      Arr r = read_arr(R, refname);
      auto it = g_taps.find(tapname);
      if (!r.ok || it == g_taps.end()) return;
      std::vector<float> o = it->second.v;
      if (transpose_vt) {  // ours [256][SC] vs ref [T][256]
        const int64_t T = r.v.size() / 256, SC = it->second.shape[1];
        std::vector<float> tr(T * 256);
        for (int64_t t = 0; t < T; ++t)
          for (int d = 0; d < 256; ++d) tr[t * 256 + d] = o[d * SC + t];
        o = tr;
      }
      count = std::min<int64_t>(count, std::min<int64_t>(o.size(), r.v.size()));
      Stat so = compare(o.data(), r.v.data(), count);
      Stat sf;
      Arr fr = hf ? read_arr(F, refname) : Arr{};
      if (fr.ok) sf = compare(fr.v.data(), r.v.data(), count);
      add(rows, rowname, so, fr.ok ? &sf : nullptr);
    };
    const int64_t T = m.h_dims.T;
    cmp_tap("pre.x0", "pre.x0", "pre.x0 (images+task+state)", T * 2048);
    for (int l = 0; l < 18; ++l) cmp_tap("pre.l" + std::to_string(l) + ".out", "pre.l" + std::to_string(l) + ".out", "pre.l" + std::to_string(l) + ".out", T * 2048);
    cmp_tap("pre.base_final", "pre.base_final", "pre.base_final", 2048);
    for (int l : {0, 17}) {
      cmp_tap("pre.kv2.l" + std::to_string(l) + ".k", "pre.kv2.l" + std::to_string(l) + ".k", "pre.kv2.l" + std::to_string(l) + ".k", T * 256);
      cmp_tap("pre.kv2.l" + std::to_string(l) + ".vt", "pre.kv2.l" + std::to_string(l) + ".v", "pre.kv2.l" + std::to_string(l) + ".v", T * 256, true);
    }
    for (int s = 0; s < m.cfg.steps; ++s) {
      cmp_tap("suf.s" + std::to_string(s) + ".v", "suf.s" + std::to_string(s) + ".v", "suf.s" + std::to_string(s) + ".v", n);
      cmp_tap("suf.s" + std::to_string(s) + ".x", "suf.s" + std::to_string(s) + ".x", "suf.s" + std::to_string(s) + ".x", n);
    }
    Arr rr = read_arr(R, "out.actions_raw"), rl = read_arr(R, "out.stage_logits"), rc = read_arr(R, "out.actions_chain");
    Stat s_raw = compare(raw.data(), rr.v.data(), n), s_chain = compare(rc.v.data(), rr.v.data(), n);
    Arr fraw = hf ? read_arr(F, "out.actions_raw") : Arr{};
    Stat s_f;
    if (fraw.ok) s_f = compare(fraw.v.data(), rr.v.data(), n);
    add(rows, "actions_raw (normalized)", s_raw, fraw.ok ? &s_f : &s_chain);
    std::vector<float> lg(logits, logits + 15);
    for (int i = spec.stages[task]; i < 15; ++i) lg[i] = -INFINITY;
    add(rows, "stage_logits", compare(lg.data(), rl.v.data(), 15), nullptr);
    std::vector<double> act((size_t)ah * 23);
    pb_postprocess(spec, raw.data(), norm32, act.data());
    std::vector<float> actf(act.begin(), act.end());
    Arr ra = read_arr(R, "out.actions");
    Stat s_act = compare(actf.data(), ra.v.data(), (int64_t)ah * 23);
    Arr fa = hf ? read_arr(F, "out.actions") : Arr{};
    Stat s_fa;
    if (fa.ok) s_fa = compare(fa.v.data(), ra.v.data(), (int64_t)ah * 23);
    add(rows, "actions (robot units)", s_act, fa.ok ? &s_fa : nullptr);
    int pred = 0, rpred = 0;
    for (int i = 1; i < 15; ++i) { if (lg[i] > lg[pred]) pred = i; if (rl.v[i] > rl.v[rpred]) rpred = i; }
    printf("sample %d: task %d stage %d  predicted stage ours %d jax %d\n", si, task, stage, pred, rpred);
    // ---- end to end: rolling inpainting + correlated noise through the policy key stream
    Arr seed = read_arr(R, "inp.seed");
    std::vector<double> kept = read_f64(R, "inp.kept");
    if (seed.ok && !kept.empty()) {
      JaxKey key = jax_key((uint64_t)seed.v[0]), sample, r2, nk;
      jax_split(key, &key, &sample);
      jax_split(sample, &r2, &nk);
      std::vector<float> z(n), noise(n), x0O(4 * ad);
      jax_normal(nk, z.data(), n);
      pb_correlate(spec, z.data(), noise.data());
      pb_initial_actions(spec, kept.data(), 4, st23, x0O.data());
      m.set_pb_inputs(task, stage, x0O.data(), noise.data(), 4 * ad, st);
      m.upload_inputs(img.data(), tok, noise.data(), st);
      m.forward_eager(st);
      m.download_actions(raw.data(), st);
      pb_postprocess(spec, raw.data(), norm32, act.data());
      actf.assign(act.begin(), act.end());
      Arr ria = read_arr(R, "inp.actions");
      Stat s_inp = compare(actf.data(), ria.v.data(), (int64_t)ah * 23);
      Arr fia = hf ? read_arr(F, "inp.actions") : Arr{};
      Stat s_fi;
      if (fia.ok) s_fi = compare(fia.v.data(), ria.v.data(), (int64_t)ah * 23);
      add(rows, "inpainting call, actions (robot units)", s_inp, fia.ok ? &s_fi : nullptr);
      // kept steps must follow the pinned actions closely (inpainted in 70% of the steps)
      double kd = 0;
      for (int t = 0; t < 4; ++t)
        for (int d = 0; d < 23; ++d) kd = std::max(kd, std::fabs(act[t * 23 + d] - kept[t * 23 + d]));
      printf("  inpainting: max |first 4 actions - kept| = %.3e\n", kd);
    }
  }
  printf("\n%-40s %11s %11s %11s %11s %8s %5s\n", "point", "rel_ours", "rel_floor", "max_ours", "max_floor", "eq_ours", "ok");
  for (const Row& r : rows) {
    const bool ok = !r.hf || r.ours.rel <= factor * r.floor.rel + 1e-6;
    if (!ok) ++fails;
    printf("%-40s %11.3e %11.3e %11.3e %11.3e %8.4f %5s\n", r.name.c_str(), r.ours.rel, r.hf ? r.floor.rel : NAN,
           r.ours.maxabs, r.hf ? r.floor.maxabs : NAN, r.ours.eq, ok ? "ok" : "FAIL");
  }
  if (time_it) {
    std::string e2;
    if (!m.capture_graph(st, &e2)) { fprintf(stderr, "graph: %s\n", e2.c_str()); return 1; }
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int w = 0; w < 3; ++w) m.forward(st);
    std::vector<float> ms;
    for (int it = 0; it < 20; ++it) {
      cudaEventRecord(a, st);
      m.forward(st);
      cudaEventRecord(b, st);
      cudaEventSynchronize(b);
      float t;
      cudaEventElapsedTime(&t, a, b);
      ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    printf("graph forward (20 steps, 805 prefix tokens): median %.2f ms, min %.2f ms\n", ms[ms.size() / 2], ms[0]);
  }
  printf("%s: %d failing points (floor = other JAX run, else JAX per-layer vs fused)\n", fails ? "FAIL" : "PASS", fails);
  return (check && fails) ? 1 : 0;
}
