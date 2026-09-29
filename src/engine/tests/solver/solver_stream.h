// solver 입력 흐름 파일 (시험용, PhysX 없음): 층 1 시험(test_contact_solver --dump)이 PhysX 스냅샷에서 뽑은 스텝별 풀이 입력
// (섬 순서·접촉 관리자 입력·접촉 출력·1D 제약(조인트)·활성화 간선·재운 몸체·Sc 층 깨움·numCountedInteractions)과 PhysX 결과를 쓴다.
// 층 2 시험(test_contact_solver_gpu)이 읽어 판 N 개를 GPU 에서 돌리고 층 1·PhysX 결과와 비트 비교한다.
// 스텝 블록 안 배열은 16 바이트 정렬(장치에서 alignas(16) 자료를 그대로 읽으므로).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// SVS_HOST_API 를 정의하면(PhysX 와 같은 번역 단위) solver_io.h 의 호스트 진입 함수를 부르고, 아니면 풀이 본체를 직접 넣는다(GPU).
#if defined(SVS_HOST_API)
#include "core/solver/solver_io.h"
#else
#include "core/solver/tgs_solver.h"
#endif

namespace svs {
namespace sv = eng::sv;
namespace jnt = eng::jnt;

struct Header {
  char magic[8];
  uint32_t version, nb, steps, stab;
  float gravity[3], dt, bounce, frictionOffset, correlation, lengthScale;
  uint32_t batchSize, articBatchSize, maxCMs, maxC1D;  // maxC1D = 제약 번호(Dy::Constraint::index) 최댓값 + 1
  uint32_t bodySize, cmSize, patchSize, contactSize, islandSize, c1dSize, d6Size;
  uint32_t maxArena, maxFriction, maxDescs;
};
struct StepCounts {
  uint32_t nIslands, nIB, nICM, nAct, nDeact, nWake, nPatches, nContacts, nC1D, nReset;
};
struct Wake {
  uint32_t body;
  float value;
};
constexpr int RES_FLOATS = 14;  // 행위자 자세 7 + 선속도 3 + 각속도 3 + 깸 카운터 1

SV_HD size_t al16(size_t x) { return (x + 15) & ~size_t(15); }

// 스텝 블록 안 배열 위치 (바이트)
struct StepLayout {
  size_t islands, ib, icm, cmIn, act, deact, wake, numCounted, patches, contacts, c1d, ic1d, jd, pxWb, reset, total;
};
SV_HD StepLayout stepLayout(const StepCounts& c, uint32_t nb) {
  StepLayout L;
  size_t p = al16(sizeof(StepCounts));
  L.islands = p; p = al16(p + c.nIslands * sizeof(sv::IslandIn));
  L.ib = p; p = al16(p + c.nIB * 4);
  L.icm = p; p = al16(p + c.nICM * 4);
  L.cmIn = p; p = al16(p + c.nICM * sizeof(sv::SolverCM));
  L.act = p; p = al16(p + c.nAct * 4);
  L.deact = p; p = al16(p + c.nDeact * 4);
  L.wake = p; p = al16(p + c.nWake * sizeof(Wake));
  L.numCounted = p; p = al16(p + size_t(nb) * 4);
  L.patches = p; p = al16(p + c.nPatches * sizeof(sv::ContactPatchIn));
  L.contacts = p; p = al16(p + c.nContacts * sizeof(sv::ContactIn));
  L.c1d = p; p = al16(p + c.nC1D * sizeof(sv::Constraint1DIn));
  L.ic1d = p; p = al16(p + c.nC1D * 4);
  L.jd = p; p = al16(p + c.nC1D * sizeof(jnt::D6Data));
  L.pxWb = p; p = al16(p + c.nC1D * sizeof(jnt::Writeback));  // PhysX 되쓰기 (이번 스텝 섬 제약 순서, 비교용)
  L.reset = p; p = al16(p + c.nReset * 4);
  L.total = p;
  return L;
}

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
  const sv::Constraint1DIn* c1d;  // nC1D (data = 이번 스텝 jd 번호, writeback = 제약 번호)
  const uint32_t* ic1d;           // 섬별 제약 목록 (c1d 번호)
  const jnt::D6Data* jd;          // nC1D
  const jnt::Writeback* pxWb;     // nC1D (PhysX 결과, 풀이는 안 읽음)
  const uint32_t* reset;          // nReset: Sc 층이 캐시 상태를 지운 접촉 관리자
};

SV_HD StepView makeView(const uint8_t* base, const StepCounts& c, uint32_t nb) {
  const StepLayout L = stepLayout(c, nb);
  StepView v;
  v.c = c;
  v.islands = reinterpret_cast<const sv::IslandIn*>(base + L.islands);
  v.ib = reinterpret_cast<const uint32_t*>(base + L.ib);
  v.icm = reinterpret_cast<const uint32_t*>(base + L.icm);
  v.cmIn = reinterpret_cast<const sv::SolverCM*>(base + L.cmIn);
  v.act = reinterpret_cast<const uint32_t*>(base + L.act);
  v.deact = reinterpret_cast<const uint32_t*>(base + L.deact);
  v.wake = reinterpret_cast<const Wake*>(base + L.wake);
  v.numCounted = reinterpret_cast<const uint32_t*>(base + L.numCounted);
  v.patches = reinterpret_cast<const sv::ContactPatchIn*>(base + L.patches);
  v.contacts = reinterpret_cast<const sv::ContactIn*>(base + L.contacts);
  v.c1d = reinterpret_cast<const sv::Constraint1DIn*>(base + L.c1d);
  v.ic1d = reinterpret_cast<const uint32_t*>(base + L.ic1d);
  v.jd = reinterpret_cast<const jnt::D6Data*>(base + L.jd);
  v.pxWb = reinterpret_cast<const jnt::Writeback*>(base + L.pxWb);
  v.reset = reinterpret_cast<const uint32_t*>(base + L.reset);
  return v;
}

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
  B.resetCMs = v.reset;
  B.nbResetCMs = v.c.nReset;
  B.patches = v.patches;
  B.contacts = v.contacts;
  B.c1d = v.c1d;
  B.nbC1D = v.c.nC1D;
  B.islandC1Ds = v.ic1d;
  B.jointData = v.jd;
#if defined(SVS_HOST_API)
  sv::solverStepHost(B, prm);
  sv::afterIntegrationHost(B);
  sv::deactivateBodiesHost(B, v.deact, v.c.nDeact);
#else
  sv::solverStep(B, prm);
  sv::afterIntegration(B);
  sv::deactivateBodies(B, v.deact, v.c.nDeact);
#endif
}

SV_HD void bodyResult(const eng::Body& b, float* r) {
  const eng::Tf t = eng::getGlobalPose(b);
  r[0] = t.q.x; r[1] = t.q.y; r[2] = t.q.z; r[3] = t.q.w;
  r[4] = t.p.x; r[5] = t.p.y; r[6] = t.p.z;
  r[7] = b.linVel.x; r[8] = b.linVel.y; r[9] = b.linVel.z;
  r[10] = b.angVel.x; r[11] = b.angVel.y; r[12] = b.angVel.z;
  r[13] = b.wakeCounter;
}

// ---- 호스트: 쓰기 (스텝 블록 하나를 정렬 배치로 만든다)
struct StepArrays {
  StepCounts c;
  const void *islands, *ib, *icm, *cmIn, *act, *deact, *wake, *numCounted, *patches, *contacts, *c1d, *ic1d, *jd, *pxWb, *reset;
};
inline std::vector<uint8_t> packStep(const StepArrays& a, uint32_t nb) {
  const StepLayout L = stepLayout(a.c, nb);
  std::vector<uint8_t> blk(L.total, 0);
  memcpy(blk.data(), &a.c, sizeof(StepCounts));
  auto put = [&](size_t off, const void* p, size_t n) { if (n) memcpy(blk.data() + off, p, n); };
  put(L.islands, a.islands, a.c.nIslands * sizeof(sv::IslandIn));
  put(L.ib, a.ib, a.c.nIB * 4);
  put(L.icm, a.icm, a.c.nICM * 4);
  put(L.cmIn, a.cmIn, a.c.nICM * sizeof(sv::SolverCM));
  put(L.act, a.act, a.c.nAct * 4);
  put(L.deact, a.deact, a.c.nDeact * 4);
  put(L.wake, a.wake, a.c.nWake * sizeof(Wake));
  put(L.numCounted, a.numCounted, size_t(nb) * 4);
  put(L.patches, a.patches, a.c.nPatches * sizeof(sv::ContactPatchIn));
  put(L.contacts, a.contacts, a.c.nContacts * sizeof(sv::ContactIn));
  put(L.c1d, a.c1d, a.c.nC1D * sizeof(sv::Constraint1DIn));
  put(L.ic1d, a.ic1d, a.c.nC1D * 4);
  put(L.jd, a.jd, a.c.nC1D * sizeof(jnt::D6Data));
  put(L.pxWb, a.pxWb, a.c.nC1D * sizeof(jnt::Writeback));
  put(L.reset, a.reset, a.c.nReset * 4);
  return blk;
}

// ---- 호스트: 파일 전체를 읽어 스텝별 위치표를 만든다
// 파일 = Header, 초기 몸체 nb 개, 스텝마다 (스텝 블록 total 바이트, PhysX 몸체 결과 nb * RES_FLOATS)
struct Stream {
  Header h;
  std::vector<eng::Body> bodies0;
  std::vector<uint8_t> data;          // 스텝 블록들 (그대로, 16 바이트 정렬)
  std::vector<uint64_t> stepOffset;   // data 안 각 스텝 시작
  std::vector<float> pxResults;       // steps * nb * RES_FLOATS
};

inline bool readStream(const char* path, Stream& S) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  if (fread(&S.h, sizeof(Header), 1, f) != 1 || memcmp(S.h.magic, "SVSTRM2", 7) != 0 || S.h.bodySize != sizeof(eng::Body) ||
      S.h.cmSize != sizeof(sv::SolverCM) || S.h.patchSize != sizeof(sv::ContactPatchIn) || S.h.islandSize != sizeof(sv::IslandIn) ||
      S.h.c1dSize != sizeof(sv::Constraint1DIn) || S.h.d6Size != sizeof(jnt::D6Data)) {
    fclose(f);
    return false;
  }
  S.bodies0.resize(S.h.nb);
  if (fread(S.bodies0.data(), sizeof(eng::Body), S.h.nb, f) != S.h.nb) return false;
  S.pxResults.resize(size_t(S.h.steps) * S.h.nb * RES_FLOATS);
  for (uint32_t s = 0; s < S.h.steps; ++s) {
    StepCounts c;
    if (fread(&c, sizeof(c), 1, f) != 1) return false;
    const size_t n = stepLayout(c, S.h.nb).total;
    const size_t off = S.data.size();
    S.stepOffset.push_back(off);
    S.data.resize(off + n);
    memcpy(S.data.data() + off, &c, sizeof(c));
    if (fread(S.data.data() + off + sizeof(c), 1, n - sizeof(c), f) != n - sizeof(c)) return false;
    if (fread(S.pxResults.data() + size_t(s) * S.h.nb * RES_FLOATS, sizeof(float), size_t(S.h.nb) * RES_FLOATS, f) != size_t(S.h.nb) * RES_FLOATS)
      return false;
  }
  fclose(f);
  return true;
}

}  // namespace svs
