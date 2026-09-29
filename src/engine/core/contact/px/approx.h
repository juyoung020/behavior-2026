// SSE 근사 역수(rcpps)·근사 역제곱근(rsqrtps) 흉내 — GPU 판(층 2)이 CPU PhysX 와 비트까지 같게.
// 실측(AMD Ryzen 9 9950X, Zen 5, tests/contact/test_approx_exhaustive 가 2^32 전수 확인):
//  - 정상수 입력 x = 2^e * 1.m 의 결과는 가수 m 의 위 12 비트만의 함수이고, 지수는 e 만큼 옮겨질 뿐이다
//    (rcp: 2^-e * R[m>>11], rsqrt: 2^-(e div 2) * S[e 홀짝][m>>11]). 결과 가수의 아래 11 비트는 늘 0.
//  - 0·비정규수 -> ±inf(부호 유지), ±inf -> ±0 (rsqrt(-inf)·음수 -> 0xffc00000), NaN -> 조용한 NaN, 결과가 비정규수 범위면 ±0.
//  - FTZ/DAZ 켬·끔과 무관.
// 표(rcp 4096 개, rsqrt 8192 개)는 실행하는 CPU 에서 진짜 명령으로 떠서 GPU 로 올린다 -> 인텔/AMD·세대가 달라도 그 CPU 와 같아진다
// (단 "위 12 비트만의 함수" 구조는 시험으로 확인한 CPU 에서만 보장. 다른 CPU 는 test_approx_exhaustive 를 먼저 돌릴 것).
#pragma once
#include <cstdint>

#include "core/contact/px/sse_emu.h"

namespace eng {
namespace px {

#if defined(EM_HOST_X86)
// 이 CPU 의 진짜 명령으로 표를 뜬다 (호스트 전용)
inline void buildApproxTables(ApproxTables& t) {
  for (uint32_t k = 0; k < 4096; ++k) {
    const float r = _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(em_u2f((127u << 23) | (k << 11)))));
    const uint32_t b = em_f2u(r);
    t.rcp[k] = uint16_t((((b >> 23) & 0xffu) == 127u ? 0x1000u : 0u) | ((b >> 11) & 0xfffu));
  }
  for (uint32_t par = 0; par < 2; ++par)
    for (uint32_t k = 0; k < 4096; ++k) {
      const float r = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(em_u2f(((127u + par) << 23) | (k << 11)))));
      const uint32_t b = em_f2u(r);
      t.rsq[(par << 12) | k] = uint16_t((((b >> 23) & 0xffu) == 127u ? 0x1000u : 0u) | ((b >> 11) & 0xfffu));
    }
}
#endif

#if defined(__CUDACC__)
// 이 번역 단위의 GPU 표(g_emApproxDev)를 채운다. 근사 명령을 쓰는 커널을 부르기 전에 한 번. (호스트 함수)
inline cudaError_t uploadApproxTables() {
#if !defined(__CUDA_ARCH__) && defined(EM_HOST_X86)
  ApproxTables t;
  buildApproxTables(t);
  return cudaMemcpyToSymbol(g_emApproxDev, &t, sizeof(t));
#else
  return cudaErrorNotSupported;
#endif
}
#endif

}  // namespace px
}  // namespace eng
