// G3: native_render.h 구현 — RenderBatch(render 작업자, cuda/render/render_batch.cuh)를 감싼다.
#include "cuda/render/render_batch.cuh"
#include "tests/scene/native_render.h"

namespace nrender {

struct NativeRender {
  eng::rnd::gpu::RenderBatch rb;
  eng::rnd::Aff* dAnch = nullptr;
  size_t cap = 0;
  uint32_t* dVis = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
};

NativeRender* create(const eng::rnd::HostScene& H, int envs, const std::vector<eng::rnd::CamRig>& rigs, const std::vector<uint32_t>& vis) {
  NativeRender* r = new NativeRender;
  r->rb.init(H, envs, rigs);
  const size_t W = size_t(r->rb.B.W);
  std::vector<uint32_t> all(W * size_t(envs), 0xFFFFFFFFu);
  for (int e = 0; e < envs; ++e)
    for (size_t w = 0; w < W && w < vis.size(); ++w) all[size_t(e) * W + w] = vis[w];
  RCK(cudaMalloc(&r->dVis, all.size() * 4));
  RCK(cudaMemcpy(r->dVis, all.data(), all.size() * 4, cudaMemcpyHostToDevice));
  r->rb.set_visibility(r->dVis);
  RCK(cudaEventCreate(&r->e0));
  RCK(cudaEventCreate(&r->e1));
  return r;
}

void destroy(NativeRender* r) {
  if (!r) return;
  cudaFree(r->dAnch);
  cudaFree(r->dVis);
  cudaEventDestroy(r->e0);
  cudaEventDestroy(r->e1);
  delete r;
}

void setAnchors(NativeRender* r, const eng::rnd::Aff* host, size_t count) {
  if (count > r->cap) {
    cudaFree(r->dAnch);
    RCK(cudaMalloc(&r->dAnch, count * sizeof(eng::rnd::Aff)));
    r->cap = count;
  }
  RCK(cudaMemcpy(r->dAnch, host, count * sizeof(eng::rnd::Aff), cudaMemcpyHostToDevice));
  r->rb.set_anchors(r->dAnch);
}

float render(NativeRender* r, int frame) {
  cudaEventRecord(r->e0);
  r->rb.render(frame);
  cudaEventRecord(r->e1);
  RCK(cudaEventSynchronize(r->e1));
  float ms = 0;
  cudaEventElapsedTime(&ms, r->e0, r->e1);
  return ms;
}

int camW(NativeRender* r, int cam) { return r->rb.rigs[size_t(cam)].w; }
int camH(NativeRender* r, int cam) { return r->rb.rigs[size_t(cam)].h; }

void readRgb(NativeRender* r, int cam, int env, uint8_t* out) {
  const size_t hw = size_t(camW(r, cam)) * size_t(camH(r, cam));
  RCK(cudaMemcpy(out, r->rb.rgb(cam) + size_t(env) * hw * 3, hw * 3, cudaMemcpyDeviceToHost));
}
void readDepth(NativeRender* r, int cam, int env, float* out) {
  const size_t hw = size_t(camW(r, cam)) * size_t(camH(r, cam));
  RCK(cudaMemcpy(out, r->rb.depth(cam) + size_t(env) * hw, hw * 4, cudaMemcpyDeviceToHost));
}
const uint8_t* rgbDevice(NativeRender* r, int cam) { return r->rb.rgb(cam); }

}  // namespace nrender
