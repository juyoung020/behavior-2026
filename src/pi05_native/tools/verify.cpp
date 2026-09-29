// pi05_verify: layer-by-layer comparison of the native engine against JAX reference dumps
// (the JSBSim fdm_verify pattern: plant the reference state, advance one layer, compare everything).
//
//   pi05_verify --weights W.pi05w --ref DIR --tag gpu [--floor cpu --floor-single cpu_planted]
//               [--samples 0-3] [--mode chain|single|both] [--check] [--factor 2]
//
// chain : engine runs end to end from the raw inputs; every tap is compared with the reference.
// single: teacher forcing. Before each layer the engine buffers are overwritten with the reference input of that
//         layer (previous layer's reference output, reference K/V cache), so the error of one layer is isolated.
// floor : the same comparison between two JAX runs (GPU vs CPU) = the noise floor. With --check, every point must
//         satisfy ours <= factor * floor (relative L2), otherwise exit 1.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../src/host_io.h"
#include "../src/model.h"
#include "../src/tokenizer.h"
#include "../src/weights.h"

using namespace pi05;

namespace {

float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

struct Arr {
  std::vector<float> v;
  std::vector<int64_t> shape;
  bool ok = false;
};

// Reads a dump tensor as float.
Arr read_arr(const WeightFile& f, const std::string& name) {
  Arr a;
  const TensorInfo* t = f.find(name);
  if (!t) return a;
  std::string err;
  std::vector<uint8_t> raw(t->nbytes);
  if (!f.read(*t, raw.data(), &err)) return a;
  a.shape = t->shape;
  const int64_t n = t->numel();
  a.v.resize(n);
  for (int64_t i = 0; i < n; ++i) {
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

struct Stat {
  double rel = 0, maxabs = 0, eq = 0;
  int64_t n = 0;
};

// Compares a (ours) with b (reference) over the first `rows` rows of width `cols`; strides allow sub-blocks.
Stat compare(const float* a, int64_t lda, const float* b, int64_t ldb, int64_t rows, int64_t cols) {
  Stat s;
  double num = 0, den = 0;
  int64_t eq = 0;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const double x = a[r * lda + c], y = b[r * ldb + c];
      const double d = x - y;
      num += d * d;
      den += y * y;
      s.maxabs = std::max(s.maxabs, std::fabs(d));
      eq += (x == y);
    }
  s.n = rows * cols;
  s.rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  s.eq = s.n ? (double)eq / s.n : 1;
  return s;
}

void round_bf16(std::vector<float>& v) {
  for (float& x : v) x = bf2f(f2bf(x));
}

struct Row {
  std::string name;
  Stat ours, floor;
  bool has_floor = false;
};

struct Report {
  std::vector<Row> rows;
  void add(const std::string& name, Stat ours, const Stat* floor) {
    for (Row& r : rows)
      if (r.name == name) {  // aggregate (e.g. several samples): keep the worst
        r.ours.rel = std::max(r.ours.rel, ours.rel);
        r.ours.maxabs = std::max(r.ours.maxabs, ours.maxabs);
        r.ours.eq = std::min(r.ours.eq, ours.eq);
        if (floor) {
          r.floor.rel = std::max(r.floor.rel, floor->rel);
          r.floor.maxabs = std::max(r.floor.maxabs, floor->maxabs);
          r.floor.eq = std::min(r.floor.eq, floor->eq);
        }
        return;
      }
    Row r{name, ours, floor ? *floor : Stat{}, floor != nullptr};
    rows.push_back(r);
  }
};

// Collected engine taps: name -> host floats
struct Taps {
  std::map<std::string, Arr> m;
  Tap fn() {
    return [this](const std::string& name, const void* p, int dt, std::vector<int64_t> shape) {
      int64_t n = 1;
      for (auto d : shape) n *= d;
      Arr a;
      a.shape = shape;
      a.v.resize(n);
      if (dt == 1) {
        PI05_CUDA(cudaMemcpy(a.v.data(), p, n * 4, cudaMemcpyDeviceToHost));
      } else {
        std::vector<uint16_t> tmp(n);
        PI05_CUDA(cudaMemcpy(tmp.data(), p, n * 2, cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < n; ++i) a.v[i] = bf2f(tmp[i]);
      }
      a.ok = true;
      m[name] = std::move(a);
    };
  }
};

void upload_bf16(void* dst, const std::vector<float>& v, size_t n, size_t src_off = 0) {
  std::vector<uint16_t> t(n);
  for (size_t i = 0; i < n; ++i) t[i] = f2bf(v[src_off + i]);
  PI05_CUDA(cudaMemcpy(dst, t.data(), n * 2, cudaMemcpyHostToDevice));
}

std::vector<int> parse_range(const std::string& s) {
  std::vector<int> r;
  auto d = s.find('-');
  int a = std::stoi(s.substr(0, d)), b = d == std::string::npos ? a : std::stoi(s.substr(d + 1));
  for (int i = a; i <= b; ++i) r.push_back(i);
  return r;
}

std::string dump_path(const std::string& dir, const std::string& tag, int i) {
  char b[64];
  snprintf(b, sizeof b, "%s_%03d.pi05d", tag.c_str(), i);
  return dir + "/" + b;
}

// How one tap maps to reference arrays. kind: plain / ref_bf16 (round ref) / qkv / vt
struct Point {
  std::string tap, ref;
  int kind = 0;  // 0 plain, 1 round ref to bf16, 2 qkv assembled from q/kv, 3 vt vs v^T
};

std::vector<Point> points_for(const ModelCfg& c, bool single) {
  std::vector<Point> p;
  for (int cam = 0; cam < c.n_img; ++cam) {
    std::string b = "img.c" + std::to_string(cam);
    if (!single) {
      p.push_back({b + ".stem", b + ".stem", 0});
      p.push_back({b + ".x0", b + ".posemb", 1});
    }
    for (int l = 0; l < c.img_depth; ++l) {
      std::string L = b + ".l" + std::to_string(l);
      p.push_back({L + ".res1", L + ".res1", 0});
      p.push_back({L + ".out", L + ".out", 0});
    }
    p.push_back({b + ".encoded", b + ".encoded", 0});
    p.push_back({b + ".tokens", b + ".tokens", 0});
  }
  if (!single) p.push_back({"txt.emb", "txt.emb", 0});
  for (int l = 0; l < c.depth; ++l) {
    std::string L = "pre.l" + std::to_string(l);
    p.push_back({L + ".norm1", L + ".norm1", 0});
    p.push_back({L + ".qkv", L, 2});
    p.push_back({L + ".k", L + ".k", 0});
    p.push_back({L + ".vt", L + ".v", 3});
    p.push_back({L + ".norm2", L + ".norm2", 0});
    p.push_back({L + ".out", L + ".out", 0});
  }
  for (int s = 0; s < c.steps; ++s) {
    std::string S = "suf.s" + std::to_string(s);
    p.push_back({S + ".in", S + ".in", 1});
    for (int l = 0; l < c.depth; ++l) {
      std::string L = S + ".l" + std::to_string(l);
      p.push_back({L + ".norm1", L + ".norm1", 0});
      p.push_back({L + ".norm2", L + ".norm2", 0});
      p.push_back({L + ".out", L + ".out", 0});
    }
    p.push_back({S + ".final", S + ".final", 0});
    p.push_back({S + ".v", S + ".v", 0});
    p.push_back({S + ".x", S + ".x", 0});
  }
  return p;
}

// Compares a tap (ours, or another reference in the same naming) to ref for one point. rows = valid prefix rows.
bool compare_point(const Point& pt, const std::map<std::string, Arr>& ours_taps, const WeightFile* ours_dump,
                   const WeightFile& ref, int T, int L, Stat* out) {
  // fetch "ours": from engine taps, or (floor) from another dump under the reference naming
  auto get_ours = [&](const std::string& tapname, const std::string& refname) -> Arr {
    if (ours_dump) return read_arr(*ours_dump, refname);
    auto it = ours_taps.find(tapname);
    return it == ours_taps.end() ? Arr{} : it->second;
  };
  if (pt.kind == 2) {  // qkv: ours [T, 2560] vs ref q [968,8,256] + kv [2,968,1,256]
    Arr rq = read_arr(ref, pt.ref + ".q"), rkv = read_arr(ref, pt.ref + ".kv");
    if (!rq.ok || !rkv.ok) return false;
    std::vector<float> full((size_t)T * 2560);
    for (int t = 0; t < T; ++t) {
      memcpy(&full[(size_t)t * 2560], &rq.v[(size_t)t * 2048], 2048 * 4);
      memcpy(&full[(size_t)t * 2560 + 2048], &rkv.v[(size_t)t * 256], 256 * 4);
      memcpy(&full[(size_t)t * 2560 + 2304], &rkv.v[(size_t)(rkv.shape[1] + t) * 256], 256 * 4);
    }
    if (ours_dump) {
      Arr oq = read_arr(*ours_dump, pt.ref + ".q"), okv = read_arr(*ours_dump, pt.ref + ".kv");
      if (!oq.ok || !okv.ok) return false;
      std::vector<float> of((size_t)T * 2560);
      for (int t = 0; t < T; ++t) {
        memcpy(&of[(size_t)t * 2560], &oq.v[(size_t)t * 2048], 2048 * 4);
        memcpy(&of[(size_t)t * 2560 + 2048], &okv.v[(size_t)t * 256], 256 * 4);
        memcpy(&of[(size_t)t * 2560 + 2304], &okv.v[(size_t)(okv.shape[1] + t) * 256], 256 * 4);
      }
      *out = compare(of.data(), 2560, full.data(), 2560, T, 2560);
      return true;
    }
    auto it = ours_taps.find(pt.tap);
    if (it == ours_taps.end()) return false;
    *out = compare(it->second.v.data(), 2560, full.data(), 2560, T, 2560);
    return true;
  }
  Arr r = read_arr(ref, pt.ref);
  if (!r.ok) return false;
  if (pt.kind == 1) round_bf16(r.v);
  if (pt.kind == 3) {  // ours vt [256, SC] vs ref v [968, 256]
    if (ours_dump) {
      Arr o = read_arr(*ours_dump, pt.ref);
      if (!o.ok) return false;
      *out = compare(o.v.data(), 256, r.v.data(), 256, T, 256);
      return true;
    }
    auto it = ours_taps.find(pt.tap);
    if (it == ours_taps.end()) return false;
    const Arr& o = it->second;
    std::vector<float> tr((size_t)T * 256);
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < 256; ++d) tr[(size_t)t * 256 + d] = o.v[(size_t)d * o.shape[1] + t];
    *out = compare(tr.data(), 256, r.v.data(), 256, T, 256);
    return true;
  }
  Arr o = get_ours(pt.tap, pt.ref);
  if (!o.ok) return false;
  if (pt.kind == 1 && ours_dump) round_bf16(o.v);
  int64_t cols = r.shape.back();
  int64_t rows = (int64_t)r.v.size() / cols;
  const bool prefix = pt.ref.rfind("pre.", 0) == 0;
  if (prefix) rows = std::min<int64_t>(rows, T);
  if (pt.ref == "txt.emb") rows = L;
  int64_t orows = (int64_t)o.v.size() / cols;
  rows = std::min(rows, orows);
  *out = compare(o.v.data(), cols, r.v.data(), cols, rows, cols);
  return true;
}

void print_report(const char* title, const Report& rep, double factor, int* fails) {
  printf("\n== %s ==\n", title);
  printf("%-28s %12s %12s %12s %12s %8s %6s\n", "point", "rel_ours", "rel_floor", "max_ours", "max_floor", "eq_ours", "ok");
  for (const Row& r : rep.rows) {
    bool ok = !r.has_floor || r.ours.rel <= factor * r.floor.rel + 1e-6;
    if (!ok) ++*fails;
    printf("%-28s %12.3e %12.3e %12.3e %12.3e %8.4f %6s\n", r.name.c_str(), r.ours.rel, r.has_floor ? r.floor.rel : NAN,
           r.ours.maxabs, r.has_floor ? r.floor.maxabs : NAN, r.ours.eq, ok ? "ok" : "FAIL");
  }
}

// Short names for the table: collapse camera index (worst over cameras) and keep layer granularity.
std::string row_name(const std::string& ref) {
  std::string s = ref;
  auto p = s.find(".c");
  if (s.rfind("img.", 0) == 0 && p != std::string::npos) s = "img" + s.substr(s.find('.', p + 1));
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights, ref_dir, tag = "gpu", floor_tag, floor_single_tag, mode = "both", samples = "0-3";
  double factor = 2.0;
  bool check = false, time_it = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--weights") weights = next();
    else if (a == "--ref") ref_dir = next();
    else if (a == "--tag") tag = next();
    else if (a == "--floor") floor_tag = next();
    else if (a == "--floor-single") floor_single_tag = next();
    else if (a == "--mode") mode = next();
    else if (a == "--samples") samples = next();
    else if (a == "--factor") factor = std::stod(next());
    else if (a == "--check") check = true;
    else if (a == "--time") time_it = true;
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (weights.empty() || ref_dir.empty()) {
    fprintf(stderr, "usage: pi05_verify --weights W --ref DIR [--tag gpu] [--floor cpu] [--floor-single cpu_planted] "
                    "[--samples 0-3] [--mode chain|single|both] [--check] [--factor 2]\n");
    return 2;
  }
  std::string err;
  WeightFile wf;
  if (!wf.open(weights, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  Tokenizer tok;
  {
    const TensorInfo* t = wf.find("tokenizer.model");
    std::vector<uint8_t> m(t->nbytes);
    wf.read(*t, m.data(), &err);
    if (!tok.load(m.data(), m.size(), &err)) { fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
  }
  RobotSpec rs;
  if (!load_robot_spec(wf, &rs, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  Model model;
  if (!model.load(wf, &err)) { fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
  printf("weights on GPU %.2f GiB, activations %.2f GiB\n", model.weight_bytes / 1073741824.0, model.act_bytes / 1073741824.0);
  const ModelCfg& c = model.cfg;
  cudaStream_t st;
  PI05_CUDA(cudaStreamCreate(&st));

  Report chain, single, host, finals;
  int fails = 0;
  double worst_final_raw = 0, worst_final = 0, floor_final_raw = 0, floor_final = 0;
  for (int si : parse_range(samples)) {
    WeightFile ref;
    if (!ref.open(dump_path(ref_dir, tag, si), &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::unique_ptr<WeightFile> fl, fls;
    if (!floor_tag.empty()) {
      fl = std::make_unique<WeightFile>();
      if (!fl->open(dump_path(ref_dir, floor_tag, si), &err)) fl.reset();
    }
    if (!floor_single_tag.empty()) {
      fls = std::make_unique<WeightFile>();
      if (!fls->open(dump_path(ref_dir, floor_single_tag, si), &err)) fls.reset();
    }
    // ---- host transforms ----
    Arr img = read_arr(ref, "in.img"), prop = read_arr(ref, "in.proprio"), noise = read_arr(ref, "in.noise");
    const TensorInfo* pt = ref.find("in.prompt");
    std::string prompt(pt->nbytes, '\0');
    ref.read(*pt, &prompt[0], &err);
    PreparedInput in;
    prepare_input(rs, tok, prop.v.data(), (int)prop.v.size(), prompt, c.max_tok, c.ad, &in);
    Arr rtok = read_arr(ref, "host.tokens"), rmask = read_arr(ref, "host.mask"), rstate = read_arr(ref, "host.state");
    int L = 0;
    for (float m : rmask.v) L += m != 0;
    bool tok_ok = (int)in.tokens.size() == L;
    for (int i = 0; tok_ok && i < L; ++i) tok_ok = in.tokens[i] == (int)rtok.v[i];
    Stat sst = compare(in.state_f32.data(), 1, rstate.v.data(), 1, (int64_t)rstate.v.size(), 1);
    printf("sample %d: tokens %d (%s), state rel %.2e eq %.3f\n", si, L, tok_ok ? "match" : "MISMATCH", sst.rel, sst.eq);
    if (!tok_ok || sst.eq < 1.0) ++fails;
    const int T = c.img_tokens() + L;
    std::vector<uint8_t> img_u8(img.v.size());
    for (size_t i = 0; i < img.v.size(); ++i) img_u8[i] = (uint8_t)img.v[i];

    // ---- chain ----
    if (mode == "chain" || mode == "both") {
      Taps taps;
      model.tap = taps.fn();
      model.upload_inputs(img_u8.data(), in.tokens, noise.v.data(), st);
      model.forward_eager(st);
      model.tap = nullptr;
      PI05_CUDA(cudaStreamSynchronize(st));
      for (const Point& p : points_for(c, false)) {
        Stat o, f;
        if (!compare_point(p, taps.m, nullptr, ref, T, L, &o)) continue;
        bool hf = fl && compare_point(p, taps.m, fl.get(), ref, T, L, &f);
        chain.add(row_name(p.ref), o, hf ? &f : nullptr);
      }
      // time conditioning
      Arr rc = read_arr(ref, "suf.s0.cond");
      if (rc.ok) {
        Stat o = compare(model.cond_dbg.data(), 1, rc.v.data(), 1, (int64_t)rc.v.size(), 1);
        Stat f;
        Arr fc = fl ? read_arr(*fl, "suf.s0.cond") : Arr{};
        if (fc.ok) f = compare(fc.v.data(), 1, rc.v.data(), 1, (int64_t)rc.v.size(), 1);
        chain.add("suf.s0.cond", o, fc.ok ? &f : nullptr);
      }
      // final actions: raw (normalized) and after output transforms
      std::vector<float> raw((size_t)c.ah * c.ad);
      model.download_actions(raw.data(), st);
      Arr rraw = read_arr(ref, "out.actions_raw"), ract = read_arr(ref, "out.actions");
      std::vector<double> act((size_t)c.ah * rs.action_dim);
      postprocess_actions(rs, raw.data(), c.ah, c.ad, in.state_f32, act.data());
      std::vector<float> actf(act.begin(), act.end());
      Stat o1 = compare(raw.data(), c.ad, rraw.v.data(), c.ad, c.ah, c.ad);
      Stat o2 = compare(actf.data(), rs.action_dim, ract.v.data(), rs.action_dim, c.ah, rs.action_dim);
      Stat f1, f2;
      bool hf = false;
      if (fl) {
        Arr a1 = read_arr(*fl, "out.actions_raw"), a2 = read_arr(*fl, "out.actions");
        if (a1.ok && a2.ok) {
          f1 = compare(a1.v.data(), c.ad, rraw.v.data(), c.ad, c.ah, c.ad);
          f2 = compare(a2.v.data(), rs.action_dim, ract.v.data(), rs.action_dim, c.ah, rs.action_dim);
          hf = true;
          floor_final_raw = std::max(floor_final_raw, f1.maxabs);
          floor_final = std::max(floor_final, f2.maxabs);
        }
      }
      worst_final_raw = std::max(worst_final_raw, o1.maxabs);
      worst_final = std::max(worst_final, o2.maxabs);
      finals.add("actions_raw (normalized)", o1, hf ? &f1 : nullptr);
      finals.add("actions (robot units)", o2, hf ? &f2 : nullptr);
      Arr chainref = read_arr(ref, "out.actions_chain");
      if (chainref.ok) {
        Stat cs = compare(chainref.v.data(), c.ad, rraw.v.data(), c.ad, c.ah, c.ad);
        finals.add("jax per-layer vs jax fused", cs, nullptr);
      }
    }

    // ---- single (teacher forcing) ----
    if ((mode == "single" || mode == "both") && ref.find("img.c0.l0.out")) {
      Taps taps;
      model.tap = taps.fn();
      model.upload_inputs(img_u8.data(), in.tokens, noise.v.data(), st);
      const size_t ID = c.img_w;
      for (int l = 0; l < c.img_depth; ++l) {
        for (int cam = 0; cam < c.n_img; ++cam) {
          std::string b = "img.c" + std::to_string(cam);
          Arr src = l == 0 ? read_arr(ref, b + ".posemb") : read_arr(ref, b + ".l" + std::to_string(l - 1) + ".out");
          upload_bf16(model.xi + (size_t)cam * 256 * ID, src.v, 256 * ID);
        }
        model.siglip_layer(l, st);
      }
      for (int cam = 0; cam < c.n_img; ++cam) {
        Arr src = read_arr(ref, "img.c" + std::to_string(cam) + ".l" + std::to_string(c.img_depth - 1) + ".out");
        upload_bf16(model.xi + (size_t)cam * 256 * ID, src.v, 256 * ID);
      }
      model.siglip_head(st);
      for (int l = 0; l < c.depth; ++l) {
        if (l == 0) {
          for (int cam = 0; cam < c.n_img; ++cam) {
            Arr src = read_arr(ref, "img.c" + std::to_string(cam) + ".tokens");
            upload_bf16(model.x + (size_t)cam * 256 * c.w, src.v, 256 * c.w);
          }
          Arr te = read_arr(ref, "txt.emb");
          upload_bf16(model.x + (size_t)c.img_tokens() * c.w, te.v, (size_t)L * c.w);
        } else {
          Arr src = read_arr(ref, "pre.l" + std::to_string(l - 1) + ".out");
          upload_bf16(model.x, src.v, (size_t)T * c.w);
        }
        model.prefix_layer(l, st);
      }
      // plant the reference prefix KV cache for every layer
      for (int l = 0; l < c.depth; ++l) {
        Arr k = read_arr(ref, "pre.l" + std::to_string(l) + ".k"), vv = read_arr(ref, "pre.l" + std::to_string(l) + ".v");
        upload_bf16(model.kc + (size_t)l * c.s_cap() * c.hd, k.v, (size_t)T * c.hd);
        std::vector<float> tr((size_t)c.hd * c.s_cap(), 0.f);
        for (int t = 0; t < T; ++t)
          for (int d = 0; d < c.hd; ++d) tr[(size_t)d * c.s_cap() + t] = vv.v[(size_t)t * c.hd + d];
        upload_bf16(model.vt + (size_t)l * c.hd * c.s_cap(), tr, tr.size());
      }
      for (int s = 0; s < c.steps; ++s) {
        std::string S = "suf.s" + std::to_string(s);
        if (s > 0) {
          Arr xp = read_arr(ref, "suf.s" + std::to_string(s - 1) + ".x");
          PI05_CUDA(cudaMemcpy(model.xt, xp.v.data(), xp.v.size() * 4, cudaMemcpyHostToDevice));
        }
        model.action_in(s, st);
        for (int l = 0; l < c.depth; ++l) {
          Arr src = l == 0 ? read_arr(ref, S + ".in") : read_arr(ref, S + ".l" + std::to_string(l - 1) + ".out");
          upload_bf16(model.h, src.v, (size_t)c.ah * c.ae_w);
          model.suffix_layer(s, l, st);
        }
        Arr src = read_arr(ref, S + ".l" + std::to_string(c.depth - 1) + ".out");
        upload_bf16(model.h, src.v, (size_t)c.ah * c.ae_w);
        if (s > 0) {
          Arr xp = read_arr(ref, "suf.s" + std::to_string(s - 1) + ".x");
          PI05_CUDA(cudaMemcpy(model.xt, xp.v.data(), xp.v.size() * 4, cudaMemcpyHostToDevice));
        } else {
          PI05_CUDA(cudaMemcpy(model.xt, noise.v.data(), noise.v.size() * 4, cudaMemcpyHostToDevice));
        }
        model.suffix_head(s, st);
      }
      model.tap = nullptr;
      for (const Point& p : points_for(c, true)) {
        Stat o, f;
        if (!compare_point(p, taps.m, nullptr, ref, T, L, &o)) continue;
        bool hf = fls && compare_point(p, taps.m, fls.get(), ref, T, L, &f);
        single.add(row_name(p.ref), o, hf ? &f : nullptr);
      }
    }
  }
  print_report("final actions (worst over samples)", finals, factor, &fails);
  if (!chain.rows.empty()) print_report("chain: engine end-to-end vs JAX (floor: JAX other platform)", chain, factor, &fails);
  if (!single.rows.empty()) print_report("single layer (teacher forced) vs JAX (floor: JAX other platform, same inputs)", single, factor, &fails);
  printf("\nfinal max|diff| raw %.3e (floor %.3e)   robot units %.3e (floor %.3e)\n", worst_final_raw, floor_final_raw,
         worst_final, floor_final);

  if (time_it) {
    // speed: graph vs eager on sample inputs
    WeightFile ref;
    ref.open(dump_path(ref_dir, tag, parse_range(samples)[0]), &err);
    Arr img = read_arr(ref, "in.img"), prop = read_arr(ref, "in.proprio"), noise = read_arr(ref, "in.noise");
    std::vector<uint8_t> img_u8(img.v.begin(), img.v.end());
    PreparedInput in;
    const TensorInfo* pt = ref.find("in.prompt");
    std::string prompt(pt->nbytes, '\0');
    ref.read(*pt, &prompt[0], &err);
    prepare_input(rs, tok, prop.v.data(), (int)prop.v.size(), prompt, c.max_tok, c.ad, &in);
    model.upload_inputs(img_u8.data(), in.tokens, noise.v.data(), st);
    if (!model.capture_graph(st, &err)) { fprintf(stderr, "graph: %s\n", err.c_str()); return 1; }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int w = 0; w < 3; ++w) model.forward(st);
    std::vector<float> ms;
    for (int it = 0; it < 20; ++it) {
      model.upload_inputs(img_u8.data(), in.tokens, noise.v.data(), st);
      cudaEventRecord(e0, st);
      model.forward(st);
      cudaEventRecord(e1, st);
      cudaEventSynchronize(e1);
      float t;
      cudaEventElapsedTime(&t, e0, e1);
      ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    printf("graph forward: median %.2f ms, min %.2f ms (T=%d)\n", ms[ms.size() / 2], ms[0], model.h_dims.T);
    size_t fr, tot;
    cudaMemGetInfo(&fr, &tot);
    printf("device memory used by process (approx): %.2f GiB\n", (tot - fr) / 1073741824.0);
  }
  printf("\n%s: %d failing points\n", fails ? "FAIL" : "PASS", fails);
  return (check && fails) ? 1 : 0;
}
