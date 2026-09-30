// 닫힌 고리 창 입력 흐름 (G2-0): 장면 파일에서 세운 env 에 스텝마다 넣는 창 입력을 파일 하나로.
// 쓰는 쪽 = 대조기(replay/g1_env.cpp, G1_ENV_REC=<파일>): 재생 중 PhysX 창에서 받은 입력을 그대로 적고, 스텝 뒤 우리 env 상태 요약값
//   (그 실행에서 PhysX 와 비트 같음이 확인된 값)을 함께 적는다.
// 읽는 쪽 = PhysX 없는 단독 실행기(tests/scene/env_run.cpp)와 N 판 GPU: 장면 파일 + 이 흐름만으로 envStep 을 돌려 요약값이 같아야 한다.
// 창 적용 차례는 대조기와 같다: 쌍 관리층 앞 연산(섬 호출은 모음) -> 기록 차례대로 바깥 섬 호출·우리 쌍 호출 자리 -> 남은 우리 호출
//   -> 깸 카운터·활성 차이 -> 건드린 몸체 -> 관절체 호출 -> 판 도중 조인트 상수 블록.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "env_solve.h"
#include "pairs_log.h"

namespace eng {
namespace scene {

// 관절체 호출 (replay G1ArtOp 와 같은 뜻): 0 드라이브 목표, 1 드라이브 목표 속도, 2 wakeUp, 3 putToSleep(아직 없음)
struct EnvArtOp {
  uint32_t art;
  uint8_t type, axis, pad[2];
  uint32_t link;  // 생성 순서 번호
  float v;
};
constexpr uint32_t kIslOurs = 0x80000000u;  // 창 차례 표의 op 에 이 비트 = "여기서 우리 쌍 관리층 호출 하나" (나머지 칸 = 기록된 PhysX 호출, 대조용)
struct EnvBodySet {                          // 건드린 몸체 (옮기지 않은 API) — 창 뒤 값
  uint64_t node;
  float b2w[7], b2a[7], lin[3], ang[3], wc;
  int32_t sc;  // Sc 행위자 (없으면 -1)
};
struct EnvWindow {
  uint64_t sim = 0;
  uint8_t first = 0, edit = 0, unsup = 0, pad = 0;  // edit(편집 창)·unsup(1 관절체 다시 맞춤, 2 깸 카운터 표 모양 바뀜): 단독 실행기가 재현 못 하는 창
  // 쌍 관리층 앞 연산: 표가 바뀐 창만 표를 담음
  uint8_t tables = 0;
  std::vector<PairsActorIn> actors;
  std::vector<ss::Shape> shapes;
  std::vector<PairsJoint> joints;
  std::vector<PairsPreOp> ops;
  // 섬: 기록 차례 (op 에 kIslOurs 비트면 우리 호출 자리). ADD_CONSTRAINT 는 newJoints 를 차례로 하나씩 (못 뜬 것은 actor0 = ~0 이고 간선 객체 0)
  std::vector<IslOp> isl;
  std::vector<SceneJoint> newJoints;
  std::vector<uint8_t> newJointOk;
  // 깸 카운터·활성: 지난 스텝 우리 값과 다른 칸만
  std::vector<HostBodyWake> wakeBodies;
  std::vector<HostArtWake> wakeArts;
  std::vector<std::pair<uint32_t, uint8_t>> active;
  std::vector<EnvBodySet> bodySets;
  std::vector<EnvArtOp> artOps;
  std::vector<std::pair<uint32_t, SceneJoint>> jointData;
  // 스텝 뒤 요약값: 몸체, 관절체, 깸 카운터·활성
  uint64_t dBody = 0, dArt = 0, dWake = 0;
};

// ---- 요약값 (FNV-1a 64)
struct EnvHash {
  uint64_t h = 1469598103934665603ull;
  void add(const void* p, size_t n) {
    const unsigned char* c = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
  }
  template <class T>
  void v(const T& x) { add(&x, sizeof(T)); }
};
inline void envDigest(const EnvStep& E, const EnvSolveImpl& S, uint64_t& dBody, uint64_t& dArt, uint64_t& dWake) {
  EnvHash hb, ha, hw;
  for (const Body& b : S.bodies) {
    hb.v(b.body2World);
    hb.v(b.linVel);
    hb.v(b.angVel);
    hb.v(b.wakeCounter);
    hb.v(b.solverWakeCounter);
    hb.v(b.sleepLinVelAcc);
    hb.v(b.sleepAngVelAcc);
  }
  for (const art::Articulation& a : S.arts) {
    for (uint32_t i = 0; i < a.nLinks; ++i) {
      const art::LinkBody& l = a.bodies[i];
      ha.v(l.body2World);
      ha.v(l.linVel);
      ha.v(l.angVel);
      ha.v(l.wakeCounter);
      ha.v(l.sleepLinVelAcc);
      ha.v(l.sleepAngVelAcc);
    }
    ha.add(&a.jointPosition[0], sizeof(float) * a.dofs);
    ha.add(&a.jointVelocity[0], sizeof(float) * a.dofs);
    ha.v(a.wakeCounter);
    ha.v(a.awake);
  }
  for (const HostBodyWake& w : E.wake.bodies) {
    hw.v(w.node);
    hw.v(w.wc);
    hw.v(w.solverWc);
    hw.v(w.solveWc);
  }
  for (const HostArtWake& w : E.wake.arts) {
    hw.v(w.node);
    hw.v(w.wc);
  }
  hw.add(E.active.data(), E.active.size());
  dBody = hb.h;
  dArt = ha.h;
  dWake = hw.h;
}

// ---- 관절체 호출 (replay/g1_art.cpp applyOps 와 같음)
inline void envApplyArtOp(art::Articulation& e, const EnvArtOp& o) {
  switch (o.type) {
    case 0: art::jointSetDriveTarget(e, o.link, o.axis, o.v, true); break;
    case 1: art::jointSetDriveVelocity(e, o.link, o.axis, o.v, true); break;
    case 2: {  // NpArticulationReducedCoordinate::wakeUp (:1161): 링크마다 깸 카운터 = 재설정 값, 관절체도
      const float reset = kWakeReset;
      for (uint32_t i = 0; i < e.nLinks; ++i) e.bodies[i].wakeCounter = reset;
      e.wakeCounter = reset;
      e.awake = 1;
      e.readyForSleep = 0;
      break;
    }
    default: break;
  }
}

// ---- 창 하나를 env 에 (대조기 g1_env_after 의 기록 창과 단독 실행기가 같은 함수를 쓴다)
struct EnvWinStats {
  uint64_t ext = 0, ours = 0, mismatch = 0, jointsAdded = 0, jointsRemoved = 0, jointsBad = 0;
};
// front: 쌍 관리층 앞 입력 표 (부르는 쪽이 들고 감 — W.tables 인 창만 바뀜). newJointIdx: 이번 창에 붙인 조인트 번호 (못 뜬 것은 ~0)
inline void envApplyWindow(EnvStep& E, ScScene& sc, EnvSolveImpl& S, ss::ScPairs& P, ig::IslandManager& M, const EnvWindow& W, PairsStep& front,
                           EnvWinStats& st, std::vector<uint32_t>* newJointIdx = nullptr) {
  LiveIslands& L = E.live;
  if (W.tables) {
    front.actors = W.actors;
    front.shapes = W.shapes;
    front.joints = W.joints;
  }
  front.ops = W.ops;
  L.clearStep();
  L.defer = true;
  pairsPreOps(P, front);
  L.defer = false;
  size_t gi = 0, nj = 0;
  if (!W.first) {
    for (const IslOp& r0 : W.isl) {
      if (r0.op & kIslOurs) {  // 기록의 쌍 관리층 호출 자리 -> 우리가 낸 호출
        IslOp r = r0;
        r.op &= ~kIslOurs;
        if (gi < L.out.size()) {
          if (!islSame(L.out[gi], r)) ++st.mismatch;
          L.apply(gi++);
          ++st.ours;
        } else {
          ++st.mismatch;
        }
      } else if (r0.op == ISL_ADD_CONSTRAINT) {  // 판 도중 새 조인트
        uint32_t obj = 0, k = ~0u;
        if (nj < W.newJoints.size() && nj < W.newJointOk.size() && W.newJointOk[nj]) {
          k = S.addJoint(W.newJoints[nj]);
          obj = 0x80000000u | k;
          ++st.jointsAdded;
        } else {
          ++st.jointsBad;
        }
        if (newJointIdx) newJointIdx->push_back(k);
        ++nj;
        if (ig::addConstraint(M, obj, islNode(r0.p), islNode(r0.q)) != r0.result) ++st.mismatch;
        ++st.ext;
      } else {
        if (r0.op == ISL_REMOVE_CONN && r0.c && 2 * r0.a + 1 < M.cpu.cap) {  // 조인트 간선 끊기 = 조인트 해제
          const uint32_t o = M.constraintOrCm[r0.a];
          if (o != ig::INVALID_EDGE && (o & 0x80000000u)) {
            S.removeJoint(o & 0x7fffffffu);
            ++st.jointsRemoved;
          }
        }
        islApplyExternal(M, r0);
        ++st.ext;
      }
    }
    for (; gi < L.out.size(); ++gi) {
      L.apply(gi);
      ++st.mismatch;
    }
  }  // 경계 simulate: 파일 섬 상태가 이미 창을 담았다 -> 우리 쌍 호출은 버림
  L.clearStep();
  // 깸 카운터·활성 (API 는 바깥: 창 뒤 값과 다른 칸)
  for (const HostBodyWake& w : W.wakeBodies)
    if (HostBodyWake* b = E.wake.body(w.node)) *b = w;
    else ++st.mismatch;
  for (const HostArtWake& w : W.wakeArts)
    if (HostArtWake* a = E.wake.art(w.node)) *a = w;
    else ++st.mismatch;
  if (E.active.size() < P.actors.size()) E.active.resize(P.actors.size(), 0);
  for (const auto& a : W.active)
    if (a.first < E.active.size()) E.active[a.first] = a.second;
  // 건드린 몸체 (옮기지 않은 API): 창 뒤 값 + Sc 칸
  for (const EnvBodySet& b : W.bodySets) {
    const int32_t bi = S.bodyOf(uint32_t(b.node & 0xffffffffu));
    if (bi < 0) { ++st.mismatch; continue; }
    Body& x = S.bodies[size_t(bi)];
    memcpy(&x.body2World, b.b2w, 28);
    memcpy(&x.body2Actor, b.b2a, 28);
    memcpy(&x.linVel, b.lin, 12);
    memcpy(&x.angVel, b.ang, 12);
    x.wakeCounter = b.wc;
    if (b.sc >= 0) {
      px::PxTransform t, a;
      memcpy(&t, b.b2w, 28);
      memcpy(&a, b.b2a, 28);
      sc.updateActorCached(b.sc, t, a, false);
    }
  }
  // 관절체 호출 (옮긴 API)
  for (const EnvArtOp& o : W.artOps)
    if (o.art < S.arts.size()) envApplyArtOp(S.arts[o.art], o);
    else ++st.mismatch;
  // 판 도중 조인트 상수 블록 (옮기지 않은 API: 이번 simulate 에 PhysX 가 쓴 값)
  for (const auto& jd : W.jointData)
    if (jd.first < S.joints.size()) {
      SceneJoint& J = S.joints[jd.first];
      J.flags = jd.second.flags;
      J.linBreakForce = jd.second.linBreakForce;
      J.angBreakForce = jd.second.angBreakForce;
      J.minResponseThreshold = jd.second.minResponseThreshold;
      J.data = jd.second.data;
    } else {
      ++st.mismatch;
    }
}

// ---- 파일 (머리 "ENVWIN01", 창마다 칸을 차례로)
namespace envwin_io {
template <class T>
inline void w(FILE* f, const T& x) { fwrite(&x, sizeof(T), 1, f); }
template <class T>
inline void wv(FILE* f, const std::vector<T>& v) {
  const uint64_t n = v.size();
  w(f, n);
  if (n) fwrite(v.data(), sizeof(T), n, f);
}
template <class T>
inline bool r(FILE* f, T& x) { return fread(&x, sizeof(T), 1, f) == 1; }
template <class T>
inline bool rv(FILE* f, std::vector<T>& v) {
  uint64_t n = 0;
  if (!r(f, n) || n > (1ull << 32)) return false;
  v.resize(size_t(n));
  return !n || fread(v.data(), sizeof(T), size_t(n), f) == size_t(n);
}
}  // namespace envwin_io

inline void envWriteWindow(FILE* f, const EnvWindow& W) {
  using namespace envwin_io;
  w(f, W.sim);
  w(f, W.first);
  w(f, W.edit);
  w(f, W.unsup);
  w(f, W.tables);
  if (W.tables) {
    wv(f, W.actors);
    wv(f, W.shapes);
    wv(f, W.joints);
  }
  wv(f, W.ops);
  const uint64_t ni = W.isl.size();
  w(f, ni);
  for (const IslOp& o : W.isl) {
    w(f, o.op);
    w(f, o.sim);
    w(f, o.a);
    w(f, o.b);
    w(f, o.c);
    w(f, o.p);
    w(f, o.q);
    w(f, o.result);
    wv(f, o.list);
  }
  wv(f, W.newJoints);
  wv(f, W.newJointOk);
  wv(f, W.wakeBodies);
  const uint64_t na = W.wakeArts.size();
  w(f, na);
  for (const HostArtWake& a : W.wakeArts) {
    w(f, a.node);
    w(f, a.wc);
    wv(f, a.links);
  }
  wv(f, W.active);
  wv(f, W.bodySets);
  wv(f, W.artOps);
  wv(f, W.jointData);
  w(f, W.dBody);
  w(f, W.dArt);
  w(f, W.dWake);
}
inline bool envReadWindow(FILE* f, EnvWindow& W) {
  using namespace envwin_io;
  W = EnvWindow{};
  if (!r(f, W.sim) || !r(f, W.first) || !r(f, W.edit) || !r(f, W.unsup) || !r(f, W.tables)) return false;
  if (W.tables && !(rv(f, W.actors) && rv(f, W.shapes) && rv(f, W.joints))) return false;
  if (!rv(f, W.ops)) return false;
  uint64_t ni = 0;
  if (!r(f, ni) || ni > (1ull << 32)) return false;
  W.isl.resize(size_t(ni));
  for (IslOp& o : W.isl)
    if (!(r(f, o.op) && r(f, o.sim) && r(f, o.a) && r(f, o.b) && r(f, o.c) && r(f, o.p) && r(f, o.q) && r(f, o.result) && rv(f, o.list))) return false;
  if (!(rv(f, W.newJoints) && rv(f, W.newJointOk) && rv(f, W.wakeBodies))) return false;
  uint64_t na = 0;
  if (!r(f, na) || na > (1ull << 32)) return false;
  W.wakeArts.resize(size_t(na));
  for (HostArtWake& a : W.wakeArts)
    if (!(r(f, a.node) && r(f, a.wc) && rv(f, a.links))) return false;
  return rv(f, W.active) && rv(f, W.bodySets) && rv(f, W.artOps) && rv(f, W.jointData) && r(f, W.dBody) && r(f, W.dArt) && r(f, W.dWake);
}
constexpr char kEnvWinMagic[8] = {'E', 'N', 'V', 'W', 'I', 'N', '0', '1'};

}  // namespace scene
}  // namespace eng
