// torch CPU 의 float32 th.cos / th.sin (평가기 proprio 바닥 속도 회전, robot.py:1605) = Intel MKL VML vmsCos/vmsSin, 모드 VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE
// (torch 2.7 aten/src/ATen/cpu/vml.h, USE_MKL=ON). 09-30 확인: 무작위 200 만 개에서 torch 와 다름 0, 한 개씩 불러도 같음.
// SLEEF u10(core/common/sleef_trigf.h, SLEEF 와는 비트 같음)도 glibc cosf 도 torch 와 1~5 % 다르다 -> MKL 을 그대로 부른다(닫힌 소스라 옮길 수 없음).
// MKL 은 CPU 명령어 집합마다 다른 코드로 갈 수 있다 — 공식도 같은 영향을 받으므로 같은 CPU 에서 비교한다.
// 호스트 전용. 라이브러리 찾기: ENGINE_MKL_LIB (libmkl_rt.so.2 또는 torch 의 libtorch_cpu.so, MKL 이 정적으로 들어 있고 C 기호를 내보냄)
#pragma once
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace eng {
namespace omni {
namespace mkl {

typedef void (*VmsFn)(int64_t n, const float* a, float* y, int64_t mode);
constexpr int64_t kTorchMode = 0x2 | 0x140000 | 0x100;  // VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE

struct Trig {
  VmsFn cos = nullptr, sin = nullptr;
  bool load() {
    if (cos) return true;
    const char* cands[] = {getenv("ENGINE_MKL_LIB"), "libmkl_rt.so.2", "libmkl_rt.so"};
    for (const char* p : cands) {
      if (!p) continue;
      void* h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
      if (!h) continue;
      cos = (VmsFn)dlsym(h, "vmsCos");
      sin = (VmsFn)dlsym(h, "vmsSin");
      if (cos && sin) return true;
    }
    fprintf(stderr, "mkl_trig: MKL vmsCos/vmsSin 을 못 찾음 (ENGINE_MKL_LIB=<libmkl_rt.so.2 또는 torch/lib/libtorch_cpu.so>)\n");
    return false;
  }
};
inline Trig& trig() {
  static Trig t;
  t.load();
  return t;
}
inline float cosf(float x) {
  float y;
  trig().cos(1, &x, &y, kTorchMode);
  return y;
}
inline float sinf(float x) {
  float y;
  trig().sin(1, &x, &y, kTorchMode);
  return y;
}

}  // namespace mkl
}  // namespace omni
}  // namespace eng
