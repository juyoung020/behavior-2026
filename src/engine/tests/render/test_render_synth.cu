// 렌더 층 1(C++) = 층 2(CUDA) 비트 시험 — 합성 장면 (에셋 없이, 공식 기록 없이 도구 자체를 검증).
//   test_render_synth [--envs E] [--inst N] [--spp S] [--bounces B] [--check K] [--reps R] [--res 720,480,480] [--ao 거리] [--tonemap 0..5] [--gpu 0 --dump 폴더]
// 장면: 상자·판·구 삼각형 묶음 인스턴스 N 개, 기준 prim(강체 흉내) 16 개, 조명(구·사각·원판·먼 조명), 체크무늬 텍스처.
// 판 e 는 기준 prim 을 e 에 따라 조금씩 옮겨(움직이는 물체 흉내) 판마다 다른 TLAS 가 된다.
// 비교: 판 0..K-1 을 층 1 로 그려 층 2 의 depth(f32 비트)·RGB(u8) 와 전부 비교. 처리량: GPU 커널 시간.
#include <cuda_runtime.h>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "cuda/render/render_cuda.cuh"

using namespace eng;
using namespace eng::rnd;

static void push_tri(std::vector<float>& tv, std::vector<float>& tn, std::vector<float>& tu, std::vector<int32_t>& ts,
                     const float* a, const float* b, const float* c, const float* n, int slot) {
  const float* p[3] = {a, b, c};
  const float uvs[6] = {0, 0, 1, 0, 1, 1};
  for (int k = 0; k < 3; ++k)
    for (int q = 0; q < 3; ++q) tv.push_back(p[k][q]);
  for (int k = 0; k < 3; ++k)
    for (int q = 0; q < 3; ++q) tn.push_back(n[q]);
  for (int k = 0; k < 6; ++k) tu.push_back(uvs[k] * 3.0f);
  ts.push_back(slot);
}

static int32_t make_box(HostScene& S) {
  std::vector<float> tv, tn, tu;
  std::vector<int32_t> ts;
  const float P[8][3] = {{-.5f, -.5f, -.5f}, {.5f, -.5f, -.5f}, {.5f, .5f, -.5f}, {-.5f, .5f, -.5f},
                         {-.5f, -.5f, .5f},  {.5f, -.5f, .5f},  {.5f, .5f, .5f},  {-.5f, .5f, .5f}};
  const int F[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {1, 2, 6, 5}, {0, 4, 7, 3}};
  const float N[6][3] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {1, 0, 0}, {-1, 0, 0}};
  for (int f = 0; f < 6; ++f) {
    push_tri(tv, tn, tu, ts, P[F[f][0]], P[F[f][1]], P[F[f][2]], N[f], f % 2);
    push_tri(tv, tn, tu, ts, P[F[f][0]], P[F[f][2]], P[F[f][3]], N[f], f % 2);
  }
  return add_geometry(S, tv.data(), tn.data(), tu.data(), ts.data(), uint32_t(ts.size()), 3);
}
static int32_t make_sphere(HostScene& S, int seg) {
  std::vector<float> tv, tn, tu;
  std::vector<int32_t> ts;
  auto pt = [&](int i, int j, float* o) {
    const double th = M_PI * i / seg, ph = 2 * M_PI * j / seg;
    o[0] = float(std::sin(th) * std::cos(ph) * 0.5);
    o[1] = float(std::sin(th) * std::sin(ph) * 0.5);
    o[2] = float(std::cos(th) * 0.5);
  };
  for (int i = 0; i < seg; ++i)
    for (int j = 0; j < seg; ++j) {
      float a[3], b[3], c[3], d[3];
      pt(i, j, a); pt(i + 1, j, b); pt(i + 1, j + 1, c); pt(i, j + 1, d);
      float n[3] = {a[0] * 2, a[1] * 2, a[2] * 2};
      push_tri(tv, tn, tu, ts, a, b, c, n, 0);
      push_tri(tv, tn, tu, ts, a, c, d, n, 0);
    }
  return add_geometry(S, tv.data(), tn.data(), tu.data(), ts.data(), uint32_t(ts.size()), 1);
}
static int32_t make_plane(HostScene& S) {
  std::vector<float> tv, tn, tu;
  std::vector<int32_t> ts;
  const float a[3] = {-10, -10, 0}, b[3] = {10, -10, 0}, c[3] = {10, 10, 0}, d[3] = {-10, 10, 0}, n[3] = {0, 0, 1};
  push_tri(tv, tn, tu, ts, a, b, c, n, 0);
  push_tri(tv, tn, tu, ts, a, c, d, n, 0);
  return add_geometry(S, tv.data(), tn.data(), tu.data(), ts.data(), 2, 3);
}

static Aff aff_trs(float tx, float ty, float tz, float yaw, float s) {
  const float c = std::cos(yaw), sn = std::sin(yaw);
  return Aff{{c * s, -sn * s, 0, tx, sn * s, c * s, 0, ty, 0, 0, s, tz}};
}
// 카메라: 위치 e 에서 목표 t 를 본다 (열 = 오른쪽, 위, 뒤)
static Camera look_at(float ex, float ey, float ez, float tx, float ty, float tz, int w, int h, float fov_deg) {
  double f[3] = {tx - ex, ty - ey, tz - ez};
  double fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (double& x : f) x /= fl;
  double r[3] = {f[1] * 1 - f[2] * 0, f[2] * 0 - f[0] * 1, 0};  // f × z
  double rl = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  for (double& x : r) x /= rl;
  double u[3] = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
  Camera c{};
  const double col[3][3] = {{r[0], r[1], r[2]}, {u[0], u[1], u[2]}, {-f[0], -f[1], -f[2]}};
  for (int row = 0; row < 3; ++row) {
    for (int k = 0; k < 3; ++k) c.world.m[row * 4 + k] = float(col[k][row]);
  }
  c.world.m[3] = ex; c.world.m[7] = ey; c.world.m[11] = ez;
  c.tanx = c.tany = float(std::tan(fov_deg * M_PI / 360.0));
  c.znear = 0.01f; c.zfar = 1000.0f;
  c.w = w; c.h = h;
  return c;
}

// 단계별 탐침: 두 층이 어디서 갈라지는지 (첫 다름 픽셀에서)
constexpr int kProbeN = 24;
static const char* kProbeName[kProbeN] = {"depth", "p.x", "p.y", "p.z", "ns.x", "ns.y", "ns.z", "alb.r", "alb.g", "alb.b",
                                          "L0.r", "L1.r", "L2.r", "L3.r", "rad.b0.r", "rad.b0.g", "rad.r", "rad.g", "rad.b",
                                          "rnd0", "cos.x", "srgb(.5)", "exp2(.3)", "log2(3)"};
EHD void probe(const SceneView& S0, const EnvView& E, const Camera& cm, int e, int c, int px, int py, float* o) {
  const Ray r0 = camera_ray(cm, float(px) + 0.5f, float(py) + 0.5f);
  const Hit h = trace(S0, E, r0, cm.znear, cm.zfar);
  o[0] = h.t;
  for (int k = 1; k < kProbeN; ++k) o[k] = 0.0f;
  if (h.inst >= 0) {
    const Surf s = surface(S0, E, r0, h, 2.0f * cm.tanx / float(cm.w) * h.t * mag(r0.d));
    o[1] = s.p.x; o[2] = s.p.y; o[3] = s.p.z;
    o[4] = s.ns.x; o[5] = s.ns.y; o[6] = s.ns.z;
    o[7] = s.albedo.x; o[8] = s.albedo.y; o[9] = s.albedo.z;
    const V3 po = s.p + s.ng * 1e-4f;
    for (int li = 0; li < 4 && li < S0.n_lights; ++li) o[10 + li] = light_direct(S0, E, li, po, s.ns, 0.37f, 0.61f).x;
  }
  SceneView S1 = S0;
  S1.sp.bounces = 0;
  float d;
  V3 L;
  shade_pixel(S1, E, cm, px, py, pixel_seed(e, c, px, py, 0), d, L);
  o[14] = L.x; o[15] = L.y;
  shade_pixel(S0, E, cm, px, py, pixel_seed(e, c, px, py, 0), d, L);
  o[16] = L.x; o[17] = L.y; o[18] = L.z;
  uint32_t rs = pixel_seed(e, c, px, py, 0);
  o[19] = rnd01(rs);
  o[20] = cosine_dir(V3{0.0f, 0.0f, 1.0f}, 0.3f, 0.7f).x;
  o[21] = srgb_to_lin(0.5f);
  o[22] = fexp2(0.3f);
  o[23] = flog2(3.0f);
}
// shade_pixel 첫 표본을 그대로 따라가며 중간값을 적는다 (튕김마다 32 칸)
constexpr int kTraceN = 96;
static const char* kTraceName[32] = {"t", "inst", "tri", "u", "v", "alb.r", "alb.g", "alb.b", "ns.x", "ns.y", "ns.z",
                                     "li", "u1", "u2", "ls.r", "ls.g", "ls.b", "lsum.r", "lsum.g", "lsum.b", "acc.r", "acc.g",
                                     "acc.b", "thr.r", "thr.g", "thr.b", "nd.x", "nd.y", "nd.z", "po.x", "po.y", "po.z"};
EHD void probe_trace(const SceneView& S, const EnvView& E, const Camera& cm, int e, int c, int px, int py, float* o) {
  for (int k = 0; k < kTraceN; ++k) o[k] = 0.0f;
  const Ray r0 = camera_ray(cm, float(px) + 0.5f, float(py) + 0.5f);
  Hit h = trace(S, E, r0, cm.znear, cm.zfar);
  if (h.inst < 0) return;
  uint32_t rs = pcg(pixel_seed(e, c, px, py, 0) ^ pcg(0u * 0x9E3779B9u));
  Ray r = r0;
  float cone = 2.0f * cm.tanx / float(cm.w) * h.t * mag(r0.d);
  const V3 amb{S.sp.ambient[0], S.sp.ambient[1], S.sp.ambient[2]};
  const bool ao = S.sp.ao_range > 0.0f;
  V3 thr{1.0f, 1.0f, 1.0f}, acc{0.0f, 0.0f, 0.0f};
  for (int b = 0; b <= S.sp.bounces && b < 3; ++b) {  // shade_pixel 과 같은 순서 (shadow_lights = 1 가정)
    float* q = o + 32 * b;
    const Surf s = surface(S, E, r, h, cone);
    q[0] = h.t; q[1] = float(h.inst); q[2] = float(h.tri); q[3] = h.u; q[4] = h.v;
    q[5] = s.albedo.x; q[6] = s.albedo.y; q[7] = s.albedo.z; q[8] = s.ns.x; q[9] = s.ns.y; q[10] = s.ns.z;
    acc = acc + mulc(thr, s.emissive);
    const V3 po = s.p + s.ng * 1e-4f;
    q[29] = po.x; q[30] = po.y; q[31] = po.z;
    V3 lsum{0.0f, 0.0f, 0.0f};
    if (S.n_lights > 0) {
      V3 ls{0.0f, 0.0f, 0.0f};
      int li = int(rnd01(rs) * float(S.n_lights));
      li = li >= S.n_lights ? S.n_lights - 1 : li;
      const float u1 = rnd01(rs);
      const float u2 = rnd01(rs);
      q[11] = float(li); q[12] = u1; q[13] = u2;
      if (S.lights[li].visible) ls = ls + light_direct(S, E, li, po, s.ns, u1, u2);
      q[14] = ls.x; q[15] = ls.y; q[16] = ls.z;
      lsum = ls * float(S.n_lights);
    }
    if (!ao) lsum = lsum + amb;
    q[17] = lsum.x; q[18] = lsum.y; q[19] = lsum.z;
    acc = acc + mulc(thr, mulc(s.albedo, lsum));
    q[20] = acc.x; q[21] = acc.y; q[22] = acc.z;
    const V3 dome{S.sp.dome[0], S.sp.dome[1], S.sp.dome[2]};
    if (b == S.sp.bounces && !ao && !(dome.x > 0.0f || dome.y > 0.0f || dome.z > 0.0f)) break;
    const float b1 = rnd01(rs);
    const float b2 = rnd01(rs);
    const V3 nd = cosine_dir(s.ns, b1, b2);
    q[26] = nd.x; q[27] = nd.y; q[28] = nd.z;
    r = make_ray(po, nd);
    h = trace(S, E, r, 1e-4f, 1e30f);
    if (ao && (h.inst < 0 || h.t > S.sp.ao_range)) acc = acc + mulc(thr, mulc(s.albedo, amb));
    if (h.inst < 0) {
      acc = acc + mulc(thr, mulc(s.albedo, dome));
      break;
    }
    if (b == S.sp.bounces) break;
    cone = cone + 0.5f * h.t;
    thr = mulc(thr, s.albedo);
    q[23] = thr.x; q[24] = thr.y; q[25] = thr.z;
  }
}
__global__ void kProbe(SceneView S, gpu::Batch B, const Camera* cams, int ncam, int e, int c, int px, int py, float* o) {
  probe(S, gpu::env_view(B, e), cams[e * ncam + c], e, c, px, py, o);
}
__global__ void kProbeTrace(SceneView S, gpu::Batch B, const Camera* cams, int ncam, int e, int c, int px, int py, float* o) {
  probe_trace(S, gpu::env_view(B, e), cams[e * ncam + c], e, c, px, py, o);
}

int main(int argc, char** argv) {
  int envs = 64, ninst = 2000, spp = 1, bounces = 1, check = 4, reps = 5, use_gpu = 1, tonemap = 2;
  float ao = 0.0f;
  const char* dump = nullptr;
  int res[3] = {720, 480, 480};
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--envs") && i + 1 < argc) envs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--inst") && i + 1 < argc) ninst = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--spp") && i + 1 < argc) spp = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bounces") && i + 1 < argc) bounces = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--check") && i + 1 < argc) check = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) use_gpu = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ao") && i + 1 < argc) ao = float(atof(argv[++i]));
    else if (!strcmp(argv[i], "--tonemap") && i + 1 < argc) tonemap = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
    else if (!strcmp(argv[i], "--res") && i + 1 < argc) sscanf(argv[++i], "%d,%d,%d", &res[0], &res[1], &res[2]);
  }
  if (check > envs) check = envs;
  HostScene H;
  const int gbox = make_box(H), gsph = make_sphere(H, 24), gpl = make_plane(H);
  // 텍스처: 64x64 체크무늬
  H.texs.push_back(TexInfo{64, 64, 0, 1, 0});
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x) {
      const bool on = ((x / 8) + (y / 8)) & 1;
      H.texels.push_back(on ? 0xFFE0E0E0u : 0xFF303060u);
    }
  build_mips(H);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> U(0.0f, 1.0f);
  for (int m = 0; m < 6; ++m) {
    Material M{};
    M.albedo[0] = 0.2f + 0.7f * U(rng); M.albedo[1] = 0.2f + 0.7f * U(rng); M.albedo[2] = 0.2f + 0.7f * U(rng);
    M.tex_albedo = m == 0 ? 0 : -1;
    M.uv_scale[0] = M.uv_scale[1] = 1.0f;
    M.albedo_brightness = 1.0f;
    M.opacity = 1.0f;
    if (m == 5) { M.emissive[0] = 2.0f; M.emissive[1] = 1.5f; M.emissive[2] = 1.0f; }
    H.mats.push_back(M);
  }
  H.n_anchor = 16;
  // 인스턴스: 0 = 바닥, 나머지 상자·구
  auto add_inst = [&](int g, int anc, const Aff& rel, int m0, int m1) {
    InstInfo in{};
    in.geom = g; in.anchor = anc; in.slot_base = int32_t(H.slot_mat.size()); in.rel = rel;
    H.slot_mat.push_back(m0);
    H.slot_mat.push_back(m1);
    H.insts.push_back(in);
  };
  add_inst(gpl, 0, aff_trs(0, 0, 0, 0, 1), 0, 0);
  for (int i = 1; i < ninst; ++i) {
    const int anc = i % 16;
    const float s = 0.1f + 0.4f * U(rng);
    add_inst(U(rng) < 0.7f ? gbox : gsph, anc, aff_trs(U(rng) * 16 - 8, U(rng) * 16 - 8, s * 0.5f + U(rng) * 2.0f, U(rng) * 6.28f, s),
             int(U(rng) * 6) % 6, int(U(rng) * 6) % 6);
  }
  // 조명
  auto add_light = [&](int type, float r, float g, float b, const Aff& rel, float radius, float w, float h, float ang) {
    Light L{};
    L.type = type; L.anchor = -1; L.visible = 1;
    L.radiance[0] = r; L.radiance[1] = g; L.radiance[2] = b;
    L.radius = radius; L.width = w; L.height = h; L.angle = ang;
    L.rel = rel;
    H.lights.push_back(L);
  };
  add_light(kLightSphere, 800, 700, 600, aff_trs(0, 0, 6, 0, 1), 0.2f, 0, 0, 0);
  add_light(kLightRect, 30, 30, 40, aff_trs(3, 3, 5, 0.3f, 1), 0, 1.5f, 1.0f, 0);
  add_light(kLightDisk, 50, 40, 30, aff_trs(-4, 2, 4, 0, 1), 0.4f, 0, 0, 0);
  add_light(kLightDistant, 5000, 5000, 5000, Aff{{1, 0, 0, 0, 0, 0.8f, -0.6f, 0, 0, 0.6f, 0.8f, 0}}, 0, 0, 0, 0.53f);
  H.sp.ambient[0] = 0.05f; H.sp.ambient[1] = 0.05f; H.sp.ambient[2] = 0.06f;
  H.sp.exposure = 1.0f; H.sp.spp = spp; H.sp.shadow_lights = 1; H.sp.tonemap = tonemap; H.sp.bounces = bounces;
  H.sp.ao_range = ao; H.sp.white_scale = 8.0f;
  const SceneView SV = H.view();
  printf("합성 장면: 인스턴스 %d, 기하 %zu, 삼각형 %zu, BLAS 노드 %zu, 조명 %zu | 판 %d, 해상도 %d,%d,%d, spp %d, 튕김 %d\n",
         SV.n_inst, H.geoms.size(), H.tris.size(), H.blas_nodes.size(), H.lights.size(), envs, res[0], res[1], res[2],
         spp, bounces);
  // 판마다 기준 prim 자세와 카메라
  const int ncam = 3;
  std::vector<Aff> anchors(size_t(envs) * 16);
  std::vector<Camera> cams(size_t(envs) * ncam);
  for (int e = 0; e < envs; ++e) {
    for (int a = 0; a < 16; ++a)
      anchors[size_t(e) * 16 + a] = a == 0 ? aff_trs(0, 0, 0, 0, 1) : aff_trs(0.05f * e * std::sin(float(a)), 0.03f * e, 0, 0.01f * e * a, 1);
    const float ex = -9.0f + 0.05f * e;
    cams[size_t(e) * ncam + 0] = look_at(ex, -9, 4, 0, 0, 0.5f, res[0], res[0], 60);
    cams[size_t(e) * ncam + 1] = look_at(-2 + 0.02f * e, -3, 1.5f, 0, 0, 0.3f, res[1], res[1], 70);
    cams[size_t(e) * ncam + 2] = look_at(2, -3 - 0.02f * e, 1.5f, 0, 0, 0.3f, res[2], res[2], 70);
  }
  if (!use_gpu) {  // 층 1 만: 판 0 카메라 3 대를 그려 PPM 으로
    HostEnv HE;
    HE.resize(SV);
    for (int a = 0; a < 16; ++a) HE.anchor[a] = anchors[a];
    HE.build(SV);
    for (int c = 0; c < ncam; ++c) {
      const Camera& cm = cams[c];
      const size_t hw = size_t(cm.w) * cm.h;
      std::vector<float> hd(hw);
      std::vector<uint8_t> hr(hw * 3);
      auto t0 = std::chrono::steady_clock::now();
      render_host(SV, HE, cm, 0, c, 0, hd.data(), hr.data());
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      uint64_t hits = 0;
      double sb = 0;
      for (size_t p = 0; p < hw; ++p) { hits += hd[p] > 0; sb += hr[3 * p] + hr[3 * p + 1] + hr[3 * p + 2]; }
      printf("층 1 카메라 %d (%dx%d): %.3f s, 맞은 픽셀 %.1f%%, 평균 밝기 %.1f/255\n", c, cm.w, cm.h, dt,
             100.0 * hits / hw, sb / (3.0 * hw));
      if (dump) {
        char fn[512];
        snprintf(fn, sizeof fn, "%s/synth_cam%d.ppm", dump, c);
        FILE* f = fopen(fn, "wb");
        fprintf(f, "P6 %d %d 255\n", cm.w, cm.h);
        fwrite(hr.data(), 1, hr.size(), f);
        fclose(f);
      }
    }
    return 0;
  }
  // ---- 층 2
  gpu::DevScene DS;
  DS.init(H);
  gpu::Batch B = gpu::make_batch(DS.view, envs);
  RCK(cudaMemcpy(B.anchor, anchors.data(), anchors.size() * sizeof(Aff), cudaMemcpyHostToDevice));
  std::vector<uint32_t> vis(size_t(envs) * B.W, 0xFFFFFFFFu);
  RCK(cudaMemcpy(B.vis, vis.data(), vis.size() * 4, cudaMemcpyHostToDevice));
  Camera* dcams;
  RCK(cudaMalloc(&dcams, cams.size() * sizeof(Camera)));
  RCK(cudaMemcpy(dcams, cams.data(), cams.size() * sizeof(Camera), cudaMemcpyHostToDevice));
  float* ddep[3];
  uint8_t* drgb[3];
  for (int c = 0; c < ncam; ++c) {
    const size_t hw = size_t(res[c]) * res[c];
    RCK(cudaMalloc(&ddep[c], hw * envs * 4));
    RCK(cudaMalloc(&drgb[c], hw * envs * 3));
  }
  RCK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
  cudaEvent_t e0, e1, e2;
  cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  auto run = [&](int frame) {
    gpu::kBuild<<<envs, gpu::kBuildThreads>>>(DS.view, B);
    RCK(cudaGetLastError());
    cudaEventRecord(e1);
    for (int c = 0; c < ncam; ++c) {
      dim3 bs(16, 8), gs((res[c] + 15) / 16, (res[c] + 7) / 8, envs);
      gpu::kRender<<<gs, bs>>>(DS.view, B, dcams, ncam, c, frame, ddep[c], drgb[c]);
      RCK(cudaGetLastError());
    }
  };
  run(0);
  RCK(cudaDeviceSynchronize());
  float build_ms = 0, total_ms = 0;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(e0);
    run(0);
    cudaEventRecord(e2);
    RCK(cudaEventSynchronize(e2));
    float a, b;
    cudaEventElapsedTime(&a, e0, e1);
    cudaEventElapsedTime(&b, e0, e2);
    build_ms += a;
    total_ms += b;
  }
  build_ms /= reps;
  total_ms /= reps;
  const double pix = double(envs) * (double(res[0]) * res[0] + double(res[1]) * res[1] + double(res[2]) * res[2]);
  printf("층 2: 판 %d × 카메라 3 = %.3f ms (TLAS 짓기 %.3f ms) -> 판·프레임 %.0f/초, 광선(1차) %.2f G/초\n", envs, total_ms,
         build_ms, envs / (total_ms * 1e-3), pix / (total_ms * 1e-3) * 1e-9);
  // ---- 층 1 과 비교
  uint64_t n_dep = 0, bad_dep = 0, n_rgb = 0, bad_rgb = 0;
  int first_e = -1, first_c = -1, first_p = -1;
  double t1 = 0;
  for (int e = 0; e < check; ++e) {
    HostEnv HE;
    HE.resize(SV);
    for (int a = 0; a < 16; ++a) HE.anchor[a] = anchors[size_t(e) * 16 + a];
    HE.build(SV);
    // TLAS 도 비교 (자식 번호·상자 비트)
    std::vector<Node2> gt(B.T);
    std::vector<int32_t> go(SV.n_inst);
    RCK(cudaMemcpy(gt.data(), B.tlas + size_t(e) * B.T, B.T * sizeof(Node2), cudaMemcpyDeviceToHost));
    RCK(cudaMemcpy(go.data(), B.order + size_t(e) * B.I, SV.n_inst * 4, cudaMemcpyDeviceToHost));
    int tl_bad = 0;
    for (int i = 0; i < B.T; ++i) {
      const Node2& a = gt[i];
      const Node2& b = HE.tlas[i];
      if (a.c0 != b.c0 || a.c1 != b.c1 || memcmp(a.lo0, b.lo0, 48) != 0) ++tl_bad;
    }
    for (int i = 0; i < SV.n_inst; ++i) tl_bad += go[i] != HE.order[i];
    if (tl_bad) printf("  판 %d TLAS 다름 %d\n", e, tl_bad);
    for (int c = 0; c < ncam; ++c) {
      const Camera& cm = cams[size_t(e) * ncam + c];
      const size_t hw = size_t(cm.w) * cm.h;
      std::vector<float> hd(hw), gd(hw);
      std::vector<uint8_t> hr(hw * 3), gr(hw * 3);
      auto t0 = std::chrono::steady_clock::now();
      render_host(SV, HE, cm, e, c, 0, hd.data(), hr.data());
      t1 += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      RCK(cudaMemcpy(gd.data(), ddep[c] + e * hw, hw * 4, cudaMemcpyDeviceToHost));
      RCK(cudaMemcpy(gr.data(), drgb[c] + e * hw * 3, hw * 3, cudaMemcpyDeviceToHost));
      for (size_t p = 0; p < hw; ++p) {
        ++n_dep;
        if (memcmp(&hd[p], &gd[p], 4)) {
          if (!bad_dep && first_e < 0) { first_e = e; first_c = c; first_p = int(p); }
          ++bad_dep;
        }
        for (int k = 0; k < 3; ++k) {
          ++n_rgb;
          if (hr[3 * p + k] != gr[3 * p + k]) {
            if (first_e < 0) { first_e = e; first_c = c; first_p = int(p); }
            ++bad_rgb;
          }
        }
      }
      if (e == 0 && c == 0) {
        uint64_t hits = 0;
        double s = 0;
        for (size_t p = 0; p < hw; ++p) { hits += hd[p] > 0; s += hr[3 * p] + hr[3 * p + 1] + hr[3 * p + 2]; }
        printf("  판 0 카메라 0: 맞은 픽셀 %.1f%%, 평균 밝기 %.1f/255\n", 100.0 * hits / hw, s / (3.0 * hw));
      }
    }
  }
  printf("층 1 = 층 2 (판 %d 개): depth %" PRIu64 " 개 중 비트 다름 %" PRIu64 ", RGB %" PRIu64 " 개 중 다름 %" PRIu64
         " | 첫 다름 판 %d 카메라 %d 픽셀 %d | 층 1 CPU %.2f s (판 %d × 3 카메라, %u 스레드)\n",
         check, n_dep, bad_dep, n_rgb, bad_rgb, first_e, first_c, first_p, t1, check,
         std::thread::hardware_concurrency());
  if (first_e >= 0) {  // 첫 다름 픽셀을 단계별로 쪼개 어디서 갈라지는지
    const int e = first_e, c = first_c;
    const Camera& cm = cams[size_t(e) * ncam + c];
    const int px = first_p % cm.w, py = first_p / cm.w;
    HostEnv HE;
    HE.resize(SV);
    for (int a = 0; a < 16; ++a) HE.anchor[a] = anchors[size_t(e) * 16 + a];
    HE.build(SV);
    float hv[kProbeN], gv[kProbeN];
    const unsigned old = _mm_getcsr();
    _mm_setcsr(old | 0x8040u);
    probe(SV, HE.view(), cm, e, c, px, py, hv);
    _mm_setcsr(old);
    float* dg;
    RCK(cudaMalloc(&dg, sizeof gv));
    kProbe<<<1, 1>>>(DS.view, B, dcams, ncam, e, c, px, py, dg);
    RCK(cudaMemcpy(gv, dg, sizeof gv, cudaMemcpyDeviceToHost));
    for (int k = 0; k < kProbeN; ++k) {
      uint32_t a, b;
      memcpy(&a, &hv[k], 4);
      memcpy(&b, &gv[k], 4);
      printf("  probe[%2d] %-14s 층1 %.9g (%08x)  층2 %.9g (%08x) %s\n", k, kProbeName[k], hv[k], a, gv[k], b, a == b ? "" : "<- 다름");
    }
    float ht[kTraceN], gt2[kTraceN];
    _mm_setcsr(old | 0x8040u);
    probe_trace(SV, HE.view(), cm, e, c, px, py, ht);
    _mm_setcsr(old);
    float* dg2;
    RCK(cudaMalloc(&dg2, sizeof gt2));
    kProbeTrace<<<1, 1>>>(DS.view, B, dcams, ncam, e, c, px, py, dg2);
    RCK(cudaMemcpy(gt2, dg2, sizeof gt2, cudaMemcpyDeviceToHost));
    for (int k = 0; k < kTraceN; ++k) {
      uint32_t a, b;
      memcpy(&a, &ht[k], 4);
      memcpy(&b, &gt2[k], 4);
      if (a != b || k % 32 < 3)
        printf("  trace b%d %-7s 층1 %.9g (%08x)  층2 %.9g (%08x) %s\n", k / 32, kTraceName[k % 32], ht[k], a, gt2[k], b,
               a == b ? "" : "<- 다름");
    }
  }
  gpu::free_batch(B);
  return (bad_dep || bad_rgb) ? 1 : 0;
}
