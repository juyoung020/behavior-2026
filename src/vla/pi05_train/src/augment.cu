#include "augment.cuh"

#include <cmath>

namespace pi05t {

AugParams aug_params(const JaxKey& sample_key, bool wrist) {
  AugParams p;
  JaxKey jk;
  if (!wrist) {
    // optimized.Chain -> [GeometricChain(crop, resize, rotate), ColorspaceChain(jitter)]: split(key, 2)
    const JaxKey geo = jr_child(sample_key, 0), col = jr_child(sample_key, 1);
    const JaxKey kcrop = jr_child(geo, 0), krot = jr_child(geo, 2);  // GeometricChain: split(geo, 3)
    p.geo = 1;
    float c[2];
    jr_uniform(kcrop, c, 2, -6.0f, 6.0f);  // RandomCrop: center ~ U(-limit, limit), limit = (224 - 212) / 2
    p.cy = c[0];
    p.cx = c[1];
    const float tmin = (float)(-5.0 * M_PI / 180.0), tmax = (float)(5.0 * M_PI / 180.0);
    const bool do_rot = jr_uniform1(krot) < 1.0f;  // bernoulli(p=1)
    const float theta = (do_rot ? 1.0f : 0.0f) * jr_uniform1(krot, tmin, tmax);
    const float cs = cosf(theta), sn = sinf(theta);
    const float s = (float)(212.0 / 224.0);  // Resize: current (212) / output (224)
    p.t00 = s * cs;
    p.t01 = s * sn;
    p.t10 = s * -sn;
    p.t11 = s * cs;
    jk = jr_child(col, 0);  // ColorspaceChain: split(col, 1)
  } else {
    jk = jr_child(jr_child(sample_key, 0), 0);  // Chain -> ColorspaceChain -> ColorJitter
  }
  const std::vector<JaxKey> k = jr_split(jk, 5);
  p.bright = jr_uniform1(k[0], -0.3f, 0.3f);
  p.contrast = jr_uniform1(k[1], -0.4f, 0.4f);
  p.hue = jr_uniform1(k[2], -0.1f, 0.1f);
  p.apply = jr_uniform1(jk) < 0.5f;
  p.slant = tanf((p.contrast + 1.0f) * (float)(M_PI / 4));
  const float s2 = p.slant * p.slant;
  p.p1 = (p.slant - s2) / (2.0f * (1.0f - s2));
  p.p2 = 1.0f - p.p1;
  return p;
}

namespace {
__device__ __forceinline__ float src01(const uint8_t* u8, const float* f32, size_t i) {
  const float v = u8 ? __fsub_rn(__fmul_rn(__fdiv_rn((float)u8[i], 255.0f), 2.0f), 1.0f) : f32[i];
  return __fadd_rn(__fdiv_rn(v, 2.0f), 0.5f);  // image / 2.0 + 0.5
}
__device__ __forceinline__ float jmod6(float x) {  // jnp.mod(x, 6)
  float r = fmodf(x, 6.0f);
  if (r != 0.0f && r < 0.0f) r += 6.0f;
  return r;
}
}  // namespace

__global__ void augment_k(const uint8_t* u8, const float* f32, const AugParams* prm, float* out, int n) {
  const long long total = (long long)n * 3 * 224 * 224;
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    const int j = (int)(idx % 224), i = (int)((idx / 224) % 224);
    const long long img = idx / (224 * 224);  // sample * 3 + camera
    const AugParams p = prm[img];
    const size_t base = (size_t)img * 224 * 224 * 3;
    float px[3];
    if (p.geo) {
      const float gy = (float)i - 111.5f, gx = (float)j - 111.5f;
      const float y = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(p.t00, gy), __fmul_rn(p.t01, gx)), p.cy), 111.5f);
      const float x = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(p.t10, gy), __fmul_rn(p.t11, gx)), p.cx), 111.5f);
      const float fy = floorf(y), fx = floorf(x);
      const float wy1 = __fsub_rn(y, fy), wy0 = __fsub_rn(1.0f, wy1);
      const float wx1 = __fsub_rn(x, fx), wx0 = __fsub_rn(1.0f, wx1);
      const int y0 = (int)fy, x0 = (int)fx;
      for (int c = 0; c < 3; ++c) {
        float acc = 0.f;
        bool first = true;
        for (int a = 0; a < 2; ++a)
          for (int b = 0; b < 2; ++b) {
            const int yy = y0 + a, xx = x0 + b;
            const bool ok = yy >= 0 && yy < 224 && xx >= 0 && xx < 224;
            const float v = ok ? src01(u8, f32, base + ((size_t)yy * 224 + xx) * 3 + c) : 0.0f;
            const float t = __fmul_rn(__fmul_rn(a ? wy1 : wy0, b ? wx1 : wx0), v);
            acc = first ? t : __fadd_rn(acc, t);
            first = false;
          }
        px[c] = acc;
      }
    } else {
      for (int c = 0; c < 3; ++c) px[c] = src01(u8, f32, base + ((size_t)i * 224 + j) * 3 + c);
    }
    // ColorJitter.pixelwise (augmax colorspace.py:247-283, utils.py:57-83, functional/colorspace.py)
    float o[3] = {px[0], px[1], px[2]};
    if (p.apply) {
      int am = 0;
      float value = px[0], mn = px[0];
      for (int c = 1; c < 3; ++c) {
        if (px[c] > value) { value = px[c]; am = c; }
        mn = fminf(mn, px[c]);
      }
      const float range = __fsub_rn(value, mn);
      float hue = range == 0.0f
                      ? 0.0f
                      : __fdiv_rn(__fadd_rn((float)(2 * am), __fdiv_rn(__fsub_rn(px[(am + 1) % 3], px[(am + 2) % 3]), range)),
                                  6.0f);
      const float sat = value == 0.0f ? 0.0f : __fdiv_rn(range, value);
      value = p.bright < 0.0f ? __fmul_rn(value, __fadd_rn(1.0f, p.bright))
                              : __fadd_rn(__fmul_rn(value, __fsub_rn(1.0f, p.bright)), p.bright);
      if (value < p.p1) value = __fdiv_rn(value, p.slant);
      else if (value > p.p2) value = __fsub_rn(__fadd_rn(__fdiv_rn(value, p.slant), 1.0f), __fdiv_rn(1.0f, p.slant));
      else value = __fadd_rn(__fmul_rn(p.slant, __fsub_rn(value, 0.5f)), 0.5f);
      hue = __fadd_rn(hue, p.hue);
      const float nn[3] = {5.0f, 3.0f, 1.0f};
      for (int c = 0; c < 3; ++c) {
        const float k = jmod6(__fadd_rn(nn[c], __fmul_rn(hue, 6.0f)));
        const float m = fmaxf(0.0f, fminf(fminf(k, __fsub_rn(4.0f, k)), 1.0f));
        o[c] = __fsub_rn(value, __fmul_rn(__fmul_rn(value, sat), m));
      }
    }
    for (int c = 0; c < 3; ++c) out[base + ((size_t)i * 224 + j) * 3 + c] = __fsub_rn(__fmul_rn(o[c], 2.0f), 1.0f);
  }
}

void augment(const uint8_t* u8, const float* f32, const AugParams* params_dev, float* out, int n, cudaStream_t st) {
  const long long total = (long long)n * 3 * 224 * 224;
  augment_k<<<(int)std::min<long long>((total + 255) / 256, 8192), 256, 0, st>>>(u8, f32, params_dev, out, n);
}

}  // namespace pi05t
