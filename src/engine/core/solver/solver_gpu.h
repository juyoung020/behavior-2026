// solver 판 N 개를 GPU 한 번 실행으로 (층 2, 문서 12.3 "G2a 풀이 GPU 진입", engine-solver-art).
// 호스트 판(SolverBoard, 호스트 포인터) N 개를 받아: 장치로 올리기 -> 블록 하나 = 판 하나로 solverStepPar + afterIntegration -> 결과 되받기.
// 판마다 solverStepHost + afterIntegrationHost 를 부른 것과 비트 같다 (같은 tgs_solver.h 코드, 같은 차례).
// 이 헤더는 CUDA 헤더 없이 .cpp 에서 include 할 수 있다. 구현 solver_gpu.cu 를 쓰는 쪽 대상에 넣고 엔진 CUDA 빌드 선택(ENGINE_CUDA_FLAGS:
// -fmad=false -prec-div=true -prec-sqrt=true -ftz=true -Xcompiler=-ffp-contract=off)으로 빌드한다.
//
// 넣는 것 (판마다, 호출 앞 상태 = solverStepHost 를 부르기 직전과 같게):
//   읽고 쓰는 것: bodies[nbBodies], cms[nbCMs], writebacks[c1d 가 가리키는 칸], bodySolverIndex[nbBodies+1], 섬에 든 관절체(artAt),
//                 마찰 arena(이번 스텝에 쓰는 쪽; 지난 쪽은 읽기만), 판 스칼라(error·통계·frictionCurIdx·constraints.size·plan)
//   읽는 것:     islands 와 섬 배열 4 개, patches/contacts(섬 접촉 관리자가 가리키는 범위), activatedCMs, resetCMs, c1d, jointData
//   작업 공간(vels·descs·arena·static 목록 등)은 호스트 것을 쓰지 않는다: 장치 용량은 이번 입력에서 셈하고, 넘치면(SV_ERR_POOL/DESC/ARENA/FRICTION/
//   PARTITION) 그 판 용량을 두 배로 늘려 전체를 다시 푼다(호스트 상태는 되받기 전까지 그대로라 다시 풀어도 같다).
// 되받는 것: 위 "읽고 쓰는 것" 전부. 관절체는 섬에 든 것만 딱 맞는 용량(artTightCaps)으로 옮겨 풀고 호스트 용량으로 다시 담는다(artRepack, 비트 불변).
// 섬에 없는 관절체·몸체의 다른 칸은 건드리지 않는다.
// 스레드: 호스트 스레드 하나에서 부른다. ctx 는 호출 사이 장치·고정(pinned) 버퍼를 다시 쓴다.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/solver/solver_io.h"

namespace eng {
namespace sv {

struct GpuSolveTimes {  // 지난 호출 (ms, 바이트)
  double packMs = 0, upMs = 0, kernelMs = 0, downMs = 0, unpackMs = 0, totalMs = 0;
  size_t upBytes = 0, downBytes = 0, devBytes = 0, pinBytes = 0;
  uint32_t retries = 0;
};

struct GpuSolveCtx {
  int threads = 32;                // 판(블록) 안 스레드 수 (1 이상; 결과는 스레드 수와 무관하게 같다)
  size_t stackBytes = 32 * 1024;   // 장치 스레드 스택 (관절체 풀이가 큰 지역 배열을 쓴다)
  uint32_t maxRetries = 6;
  bool hostEmulate = false;        // 진단: 장치 대신 호스트 버퍼·호스트 solverStep (담기·되받기만 확인). 처음 부르기 전에 정할 것
  GpuSolveTimes last;
  // ---- 내부 (호출 사이 재사용)
  uint8_t* dev = nullptr;
  size_t devCap = 0;
  uint8_t* pin = nullptr;
  size_t pinCap = 0;
  std::vector<float> grow;  // 판별 용량 배수 (넘치면 두 배)
  void* evt[4] = {nullptr, nullptr, nullptr, nullptr};
};

// 판 N 개를 한 번에 푼다. prms[i] 는 판 i 의 SolverParams. stream = cudaStream_t (nullptr = 기본 흐름).
// 돌려줌: true = 모든 판 풀림 (판 오류 비트는 각 boards[i]->error 에 — 용량 넘침이 끝까지 안 풀리면 그 비트가 남는다), false = CUDA 오류(메시지 stderr).
bool gpuSolveBatch(GpuSolveCtx& ctx, SolverBoard* const* boards, const SolverParams* const* prms, int n, void* stream = nullptr);
void gpuSolveFree(GpuSolveCtx& ctx);
// 같은 것을 함수 안 기본 ctx(스레드 32, 스택 32 KB)로. stream 에 cudaStream_t 를 그대로 넘기면 된다(포인터 형이라 void* 로 바뀐다).
bool gpuSolveBatch(SolverBoard* const* boards, const SolverParams* const* prms, int n, void* stream = nullptr);
const GpuSolveTimes& gpuSolveLastTimes();

}  // namespace sv
}  // namespace eng
