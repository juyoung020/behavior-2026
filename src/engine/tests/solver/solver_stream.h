// solver 입력 흐름 파일 (시험용, PhysX 없음): 층 1 시험(test_contact_solver --dump)이 PhysX 스냅샷에서 뽑은 스텝별 풀이 입력
// (섬 순서·접촉 관리자 입력·접촉 출력·활성화 간선·재운 몸체·Sc 층 깨움·numCountedInteractions)과 PhysX 결과를 쓴다.
// 층 2 시험(test_contact_solver_gpu)이 읽어 판 N 개를 GPU 에서 돌리고 층 1·PhysX 결과와 비트 비교한다.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/solver/tgs_solver.h"

namespace svs {
namespace sv = eng::sv;

struct Header {
  char magic[8];
  uint32_t version, nb, steps, stab;
  float gravity[3], dt, bounce, frictionOffset, correlation;
  uint32_t batchSize, articBatchSize, maxCMs;
  uint32_t bodySize, cmSize, patchSize, contactSize, islandSize;
  uint32_t maxArena, maxFriction, maxDescs;
};
struct StepCounts {
  uint32_t nIslands, nIB, nICM, nAct, nDeact, nWake, nPatches, nContacts;
};
struct Wake {
  uint32_t body;
  float value;
};
constexpr int RES_FLOATS = 14;  // 행위자 자세 7 + 선속도 3 + 각속도 3 + 깸 카운터 1

// 스텝 하나의 입력 (포인터는 호스트 벡터 또는 장치 버퍼)
struct StepView {
  StepCounts c;
  const sv::IslandIn* islands;
  const uint32_t* ib;
  const uint32_t* icm;
  const sv::SolverCM* cmIn;  // icm 과 같은 길이: 이번 스텝 입력 필드 (마찰 필드는 무시)
  const uint32_t* act;
  const uint32_t* deact;
  const Wake* wake;
  const uint32_t* numCounted;  // nb
  const sv::ContactPatchIn* patches;
  const sv::ContactIn* contacts;
};

// 스텝 입력을 판에 적용하고 한 스텝 푼다 (호스트·장치 공용)
SV_HD void runStep(sv::SolverBoard& B, const sv::SolverParams& prm, const StepView& v) {
  for (uint32_t k = 0; k < v.c.nWake; ++k) {  // Sc 층 internalWakeUpBase: 올리기만 (ScBodySim.cpp:541)
    eng::Body& b = B.bodies[v.wake[k].body];
    if (b.wakeCounter < v.wake[k].value) b.wakeCounter = v.wake[k].value;
  }
  for (uint32_t i = 0; i < B.nbBodies; ++i) B.bodies[i].numCountedInteractions = v.numCounted[i];
  for (uint32_t k = 0; k < v.c.nICM; ++k) {
    sv::SolverCM& m = B.cms[v.icm[k]];
    const sv::SolverCM& in = v.cmIn[k];
    const uint32_t fp = m.frictionPtr, fc = m.frictionCount;
    m = in;
    m.frictionPtr = fp;
    m.frictionCount = fc;
  }
  B.islands = v.islands;
  B.nbIslands = v.c.nIslands;
  B.islandBodies = v.ib;
  B.islandCMs = v.icm;
  B.activatedCMs = v.act;
  B.nbActivatedCMs = v.c.nAct;
  B.patches = v.patches;
  B.contacts = v.contacts;
  sv::solverStep(B, prm);
  sv::afterIntegration(B);
  sv::deactivateBodies(B, v.deact, v.c.nDeact);
}

SV_HD void bodyResult(const eng::Body& b, float* r) {
  const eng::Tf t = eng::getGlobalPose(b);
  r[0] = t.q.x; r[1] = t.q.y; r[2] = t.q.z; r[3] = t.q.w;
  r[4] = t.p.x; r[5] = t.p.y; r[6] = t.p.z;
  r[7] = b.linVel.x; r[8] = b.linVel.y; r[9] = b.linVel.z;
  r[10] = b.angVel.x; r[11] = b.angVel.y; r[12] = b.angVel.z;
  r[13] = b.wakeCounter;
}

// ---- 호스트: 파일 전체를 읽어 스텝별 위치표를 만든다
struct Stream {
  Header h;
  std::vector<eng::Body> bodies0;
  std::vector<uint8_t> data;          // 스텝 블록들 (그대로)
  std::vector<uint64_t> stepOffset;   // data 안 각 스텝 시작
  std::vector<float> pxResults;       // steps * nb * RES_FLOATS
};

inline size_t stepBytes(const StepCounts& c, uint32_t nb) {
  return sizeof(StepCounts) + c.nIslands * sizeof(sv::IslandIn) + c.nIB * 4 + c.nICM * 4 + c.nICM * sizeof(sv::SolverCM) + c.nAct * 4 +
         c.nDeact * 4 + c.nWake * sizeof(Wake) + nb * 4 + c.nPatches * sizeof(sv::ContactPatchIn) + c.nContacts * sizeof(sv::ContactIn);
}
// base 가 스텝 블록 시작을 가리킬 때 StepView 를 만든다 (장치 포인터여도 계산만 하므로 호스트에서 가능: counts 는 호스트 사본에서)
SV_HD StepView makeView(const uint8_t* base, const StepCounts& c, uint32_t nb) {
  StepView v;
  v.c = c;
  const uint8_t* p = base + sizeof(StepCounts);
  v.islands = reinterpret_cast<const sv::IslandIn*>(p); p += c.nIslands * sizeof(sv::IslandIn);
  v.ib = reinterpret_cast<const uint32_t*>(p); p += c.nIB * 4;
  v.icm = reinterpret_cast<const uint32_t*>(p); p += c.nICM * 4;
  v.cmIn = reinterpret_cast<const sv::SolverCM*>(p); p += c.nICM * sizeof(sv::SolverCM);
  v.act = reinterpret_cast<const uint32_t*>(p); p += c.nAct * 4;
  v.deact = reinterpret_cast<const uint32_t*>(p); p += c.nDeact * 4;
  v.wake = reinterpret_cast<const Wake*>(p); p += c.nWake * sizeof(Wake);
  v.numCounted = reinterpret_cast<const uint32_t*>(p); p += nb * 4;
  v.patches = reinterpret_cast<const sv::ContactPatchIn*>(p); p += c.nPatches * sizeof(sv::ContactPatchIn);
  v.contacts = reinterpret_cast<const sv::ContactIn*>(p);
  return v;
}

inline bool readStream(const char* path, Stream& S) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  if (fread(&S.h, sizeof(Header), 1, f) != 1 || memcmp(S.h.magic, "SVSTRM1", 7) != 0 || S.h.bodySize != sizeof(eng::Body) ||
      S.h.cmSize != sizeof(sv::SolverCM) || S.h.patchSize != sizeof(sv::ContactPatchIn)) {
    fclose(f);
    return false;
  }
  S.bodies0.resize(S.h.nb);
  if (fread(S.bodies0.data(), sizeof(eng::Body), S.h.nb, f) != S.h.nb) return false;
  S.pxResults.resize(size_t(S.h.steps) * S.h.nb * RES_FLOATS);
  for (uint32_t s = 0; s < S.h.steps; ++s) {
    StepCounts c;
    if (fread(&c, sizeof(c), 1, f) != 1) return false;
    const size_t n = stepBytes(c, S.h.nb);
    const size_t off = S.data.size();
    S.stepOffset.push_back(off);
    S.data.resize(off + ((n + 15) & ~size_t(15)));
    memcpy(S.data.data() + off, &c, sizeof(c));
    if (fread(S.data.data() + off + sizeof(c), 1, n - sizeof(c), f) != n - sizeof(c)) return false;
    if (fread(S.pxResults.data() + size_t(s) * S.h.nb * RES_FLOATS, sizeof(float), size_t(S.h.nb) * RES_FLOATS, f) != size_t(S.h.nb) * RES_FLOATS)
      return false;
  }
  fclose(f);
  return true;
}

}  // namespace svs
