// G3: native_loop 의 렌더 쪽 (구현 native_render.cu, nvcc). 판 N 개 × 카메라 여러 대를 RenderBatch(cuda/render/render_batch.cuh) 한 번으로.
// 이 헤더는 CUDA 헤더 없이 .cpp 에서 include 한다.
#pragma once
#include <cstdint>
#include <vector>

#include "core/render/rig.h"
#include "core/render/rsc_io.h"

namespace nrender {

struct NativeRender;
// vis: 판 하나의 인스턴스 보임 비트 (모든 판 같게)
NativeRender* create(const eng::rnd::HostScene& H, int envs, const std::vector<eng::rnd::CamRig>& rigs, const std::vector<uint32_t>& vis);
void destroy(NativeRender* r);
// 기준 prim 행렬 (호스트, 판 × 기준 prim) -> 장치
void setAnchors(NativeRender* r, const eng::rnd::Aff* host, size_t count);
// 그리기 (frame = 잡음 씨앗에 들어가는 프레임 번호). 돌려줌: 이번 그리기 GPU 시간 ms (동기)
float render(NativeRender* r, int frame);
// 판 env, 카메라 c 결과를 호스트로
void readRgb(NativeRender* r, int cam, int env, uint8_t* out);    // h×w×3
void readDepth(NativeRender* r, int cam, int env, float* out);    // h×w
// 장치 버퍼 (정책 입력, G4): 카메라 c 의 판 E 개 RGB [E×h×w×3 u8]
const uint8_t* rgbDevice(NativeRender* r, int cam);
int camW(NativeRender* r, int cam);
int camH(NativeRender* r, int cam);

}  // namespace nrender
