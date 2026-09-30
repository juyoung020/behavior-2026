// 관절체 결합 시험의 입력 흐름 (test_art_solver --dump 가 쓰고 test_art_solver_gpu 가 읽는다). PhysX 없이 층 1·층 2 를 같은 입력으로 돌린다.
// 파일 = Header, 초기 몸체 nb, 초기 관절체 na(Articulation 통째), 관절체 입력 na(ArtInputs), 그 뒤 스텝마다 블록(16 바이트 정렬).
// 스텝 블록 = Counts + 배열들(아래 Layout 순서) — 스텝 입력(섬·접촉·1D·Sc 입력) + PhysX 결과(강체 RES, 관절체 결과, 1D 되쓰기).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/solver/solver_io.h"

namespace ast {
namespace sv = eng::sv;
namespace jnt = eng::jnt;

constexpr int RES_FLOATS = 14;  // 강체: 행위자 자세 7 + 선속도 3 + 각속도 3 + 깸 카운터 1

struct Header {
  char magic[8];  // "ARTSV1"
  uint32_t nb, na, steps, stab, last;
  float gravity[3], dt, bounce, frictionOffset, correlation, lengthScale;
  uint32_t batchSize, articBatchSize;
  uint32_t maxCMs, maxArena, maxFriction, maxDescs, maxC1D, staticCap;
  uint32_t sizeBody, sizeArt, sizeCM, sizeIsland, sizeC1D, sizeD6, sizeInputs;
  uint32_t maxLinks;           // 관절체 결과·Sc 입력 칸의 링크 수(관절체마다 이 칸만큼, 앞 nLinks 만 뜻 있음)
  uint32_t artResFloats;       // 관절체 하나 결과 칸 수 (링크 14 * maxLinks + dof 2 * maxDofs + 2)
  uint32_t pad[3];
};

struct Counts {
  uint32_t nIslands, nIB, nICM, nIA, nAct, nReset, nC1D, nPatches, nContacts, nDeact, nDeactArts, nWake, lateValid, pad[3];
};
struct Wake {
  uint32_t body;
  float value;
};

inline size_t al16(size_t x) { return (x + 15) & ~size_t(15); }

struct Layout {
  size_t islands, ib, icm, cmIn, ia, act, reset, c1d, ic1d, jd, patches, contacts, deact, deactArts, wake, numCounted, lateLinkWake, lateLinkCounted,
      lateArtWake, pxRes, pxArt, pxWb, total;
};
inline Layout layout(const Counts& c, const Header& h) {
  Layout L;
  size_t p = al16(sizeof(Counts));
  auto put = [&](size_t& f, size_t bytes) {
    f = p;
    p = al16(p + bytes);
  };
  put(L.islands, c.nIslands * sizeof(sv::IslandIn));
  put(L.ib, c.nIB * 4);
  put(L.icm, c.nICM * 4);
  put(L.cmIn, c.nICM * sizeof(sv::SolverCM));
  put(L.ia, c.nIA * 4);
  put(L.act, c.nAct * 4);
  put(L.reset, c.nReset * 4);
  put(L.c1d, c.nC1D * sizeof(sv::Constraint1DIn));
  put(L.ic1d, c.nC1D * 4);
  put(L.jd, c.nC1D * sizeof(jnt::D6Data));
  put(L.patches, c.nPatches * sizeof(sv::ContactPatchIn));
  put(L.contacts, c.nContacts * sizeof(sv::ContactIn));
  put(L.deact, c.nDeact * 4);
  put(L.deactArts, c.nDeactArts * 4);
  put(L.wake, c.nWake * sizeof(Wake));
  put(L.numCounted, size_t(h.nb) * 4);
  put(L.lateLinkWake, size_t(h.na) * h.maxLinks * 4);
  put(L.lateLinkCounted, size_t(h.na) * h.maxLinks * 4);
  put(L.lateArtWake, size_t(h.na) * 4);
  put(L.pxRes, size_t(h.nb) * RES_FLOATS * 4);
  put(L.pxArt, size_t(h.na) * h.artResFloats * 4);
  put(L.pxWb, c.nC1D * sizeof(jnt::Writeback));
  L.total = p;
  return L;
}

struct Stream {
  Header h;
  std::vector<uint8_t> bodies0, arts0, inputs;  // 바이트 그대로
  std::vector<uint8_t> data;
  std::vector<uint64_t> stepOffset;
};

inline bool readStream(const char* path, Stream& S) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  bool ok = fread(&S.h, sizeof(Header), 1, f) == 1 && !memcmp(S.h.magic, "ARTSV1", 7);
  if (ok) {
    S.bodies0.resize(size_t(S.h.nb) * S.h.sizeBody);
    S.arts0.resize(size_t(S.h.na) * S.h.sizeArt);
    S.inputs.resize(size_t(S.h.na) * S.h.sizeInputs);
    ok = (S.bodies0.empty() || fread(S.bodies0.data(), 1, S.bodies0.size(), f) == S.bodies0.size()) &&
         (S.arts0.empty() || fread(S.arts0.data(), 1, S.arts0.size(), f) == S.arts0.size()) &&
         (S.inputs.empty() || fread(S.inputs.data(), 1, S.inputs.size(), f) == S.inputs.size());
  }
  if (ok) {
    const long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    const long end = ftell(f);
    fseek(f, pos, SEEK_SET);
    S.data.resize(size_t(end - pos));
    ok = S.data.empty() || fread(S.data.data(), 1, S.data.size(), f) == S.data.size();
  }
  fclose(f);
  if (!ok) return false;
  size_t p = 0;
  for (uint32_t s = 0; s < S.h.steps; ++s) {
    if (p + sizeof(Counts) > S.data.size()) return false;
    S.stepOffset.push_back(p);
    Counts c;
    memcpy(&c, S.data.data() + p, sizeof(Counts));
    p += layout(c, S.h).total;
  }
  return p == S.data.size();
}

}  // namespace ast
