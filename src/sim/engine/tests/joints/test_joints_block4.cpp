// 층 1 시험: TGS 1D 제약 4개 묶음(SIMD 4칸) 경로 = PhysX 5.6.1 비트 동일?
// 정답지: 즉시 모드 PxCreateJointConstraintsTGS 에 stride=4 머리를 주면 네 조인트 모두 행이 있을 때 Dy::setupSolverConstraintStep4 로
// 묶고(머리 형 DY_SC_TYPE_BLOCK_1D), 하나라도 행이 0 이면 하나씩(RB_1D) 만든다 (NpImmediateMode.cpp:1480-1541).
// PxSolveConstraintsTGS 는 머리 형에 따라 solve1D4 / solveConclude1D4 / writeBack1D4 (DyTGSDynamics.cpp:1266-1316) 를 부른다.
// 장면에서는 같은 분할에 독립 강체 조인트 4개가 모이면 이 경로다 (공식 radio: 정적-동적 고정 조인트 51개).
//   test_joints_block4 [--trials N] [--seed S] [--kind K]
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "PxImmediateMode.h"
#include "jeng_api.h"
#include "joint_cmds.h"

using namespace physx;
using namespace physx::immediate;
using namespace jcmd;
namespace J = eng::jnt;

static PxDefaultAllocator gAlloc;
struct QuietErr : PxErrorCallback {
  int n = 0;
  void reportError(PxErrorCode::Enum, const char*, const char*, int) override { n++; }
};
static QuietErr gErr;

static PxConstraintConnector* connectorOf(PxD6Joint* j) {
  const size_t al = alignof(PxConstraintConnector);
  const size_t off = (sizeof(PxD6Joint) + al - 1) & ~(al - 1);
  PxConstraintConnector* c = reinterpret_cast<PxConstraintConnector*>(reinterpret_cast<char*>(j) + off);
  PxU32 t = 0;
  if (c->getExternalReference(t) != static_cast<PxJoint*>(j)) {
    fprintf(stderr, "커넥터 위치 확인 실패\n");
    exit(2);
  }
  return c;
}

struct Alloc : PxConstraintAllocator {
  std::vector<void*> mem;
  PxU8* reserveConstraintData(const PxU32 size) override {
    void* p = aligned_alloc(16, (size + 15) & ~15u);
    memset(p, 0xcd, (size + 15) & ~15u);
    mem.push_back(p);
    return static_cast<PxU8*>(p);
  }
  PxU8* reserveFrictionData(const PxU32 size) override { return reserveConstraintData(size); }
  ~Alloc() { for (void* p : mem) free(p); }
};

struct Stat {
  const char* name;
  uint64_t cmp = 0, bad = 0;
  int64_t firstTrial = -1;
  void add(bool ok, int trial) {
    cmp++;
    if (!ok) { bad++; if (firstTrial < 0) firstTrial = trial; }
  }
};
static int gShown = 0;
static bool same(const void* a, const void* b, size_t n, const char* what, int trial, int idx) {
  if (!memcmp(a, b, n)) return true;
  if (gShown < 12) {
    gShown++;
    printf("[다름] 시도 %d %s #%d\n  PhysX:", trial, what, idx);
    const uint32_t* x = static_cast<const uint32_t*>(a);
    const uint32_t* y = static_cast<const uint32_t*>(b);
    for (size_t i = 0; i < n / 4; ++i) if (x[i] != y[i]) printf(" [%zu]%08x/%.9g", i, x[i], *reinterpret_cast<const float*>(&x[i]));
    printf("\n  엔진 :");
    for (size_t i = 0; i < n / 4; ++i) if (x[i] != y[i]) printf(" [%zu]%08x/%.9g", i, y[i], *reinterpret_cast<const float*>(&y[i]));
    printf("\n");
  }
  return false;
}
// 4개 묶음 블록 비교: 머리 pad0[3](바이트 1..3) 은 원본이 안 쓴다(할당기 쓰레기) — 그 바이트만 뺀다. 끝의 0 두 개(8 바이트)까지 본다.
static bool sameBlock4(const uint8_t* px, const uint8_t* en, uint32_t len, const char* what, int trial, int idx) {
  std::vector<uint8_t> a(px, px + len + 8), b(en, en + len + 8);
  a[1] = a[2] = a[3] = 0;
  b[1] = b[2] = b[3] = 0;
  return same(a.data(), b.data(), len + 8, what, trial, idx);
}

int main(int argc, char** argv) {
  int trials = 2000, seed = 7, onlyKind = -1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--kind") && i + 1 < argc) onlyKind = atoi(argv[++i]);
  }
  PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
  PxTolerancesScale tol(1.0f, 10.0f);
  PxPhysics* phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol);

  Stat sType{"묶음 판정(BLOCK_1D / 하나씩)"}, sBlock{"풀이 제약 블록"}, sBlockE{"엔진 셰이더로 만든 4묶음 블록"}, sVel{"풀이 후 몸체 속도·자세"},
      sAfter{"풀이 후 제약 블록"}, sWb{"되쓰기"};
  uint64_t nBlock4 = 0, nSingle = 0;
  Gen g(static_cast<uint32_t>(seed));

  for (int trial = 0; trial < trials; ++trial) {
    const int nGroups = 1 + g.i(3);
    const int nj = 4 * nGroups;
    const bool distinct = g.i(4) != 0;  // 실제 분할처럼 한 묶음 안 동적 몸체가 겹치지 않게 (아니면 아무렇게나 — 저장 순서까지 같아야 함)
    const int nb = distinct ? 8 * nGroups : 2 + g.i(6);
    std::vector<PxRigidDynamic*> actors(nb + 1, nullptr);
    std::vector<PxRigidBodyData> rbd(nb);
    std::vector<PxTransform> b2w(nb + 1, PxTransform(PxIdentity));
    for (int b = 1; b <= nb; ++b) {
      PxRigidDynamic* a = phys->createRigidDynamic(PxTransform(toPx(g.v(2.0f)), toPx(g.q())));
      a->setCMassLocalPose(PxTransform(toPx(g.v(0.1f)), toPx(g.q())));
      actors[b] = a;
      b2w[b] = PxTransform(toPx(g.v(2.0f)), toPx(g.q()));
      PxRigidBodyData& d = rbd[b - 1];
      memset(&d, 0, sizeof d);
      d.linearVelocity = toPx(g.v(2.0f));
      d.invMass = 1.0f / (0.1f + 10.0f * g.p());
      d.angularVelocity = toPx(g.v(3.0f));
      d.maxDepenetrationVelocity = 1e32f;
      d.invInertia = PxVec3(1.0f / (0.01f + g.p()), 1.0f / (0.01f + g.p()), 1.0f / (0.01f + g.p()));
      d.maxContactImpulse = 1e32f;
      d.body2World = b2w[b];
      d.angularDamping = 0.05f;
      d.maxLinearVelocitySq = 1e32f;
      d.maxAngularVelocitySq = 100.0f * 100.0f;
    }
    PxRigidStatic* stat = phys->createRigidStatic(PxTransform(toPx(g.v(1.0f)), toPx(g.q())));

    struct JP { PxD6Joint* px; J::D6Joint en; int a, b; };
    std::vector<JP> joints;
    for (int k = 0; k < nj; ++k) {
      const int kind = onlyKind >= 0 ? onlyKind : g.i(K_COUNT);
      int a, b;
      if (distinct) {
        a = 8 * (k / 4) + 2 * (k % 4) + 1;                          // 묶음마다 서로 다른 몸체 쌍
        const int r = g.i(3);
        b = r == 0 ? 0 : (r == 1 ? nb + 1 : a + 1);                 // 세계 / 정적 / 짝 동적
      } else {
        a = 1 + g.i(nb);
        b = g.i(nb + 2);
        if (b == a) b = 0;
      }
      const PxTransform f0(toPx(g.v(0.2f)), toPx(g.q())), f1(toPx(g.v(0.2f)), toPx(g.q()));
      PxRigidActor* act1 = (b == 0) ? nullptr : (b == nb + 1 ? static_cast<PxRigidActor*>(stat) : actors[b]);
      PxD6Joint* pj = PxD6JointCreate(*phys, actors[a], f0, act1, f1);
      const uint8_t kind1 = b == 0 ? J::ACTOR_NONE : (b == nb + 1 ? J::ACTOR_STATIC : J::ACTOR_DYNAMIC);
      const eng::Tf com0 = toE(actors[a]->getCMassLocalPose());
      const eng::Tf com1 = b == 0 ? eng::Tf{eng::qid(), {0, 0, 0}}
                                  : (b == nb + 1 ? J::comOf(J::ACTOR_STATIC, toE(stat->getGlobalPose())) : toE(actors[b]->getCMassLocalPose()));
      J::D6Joint ej = J::createD6(J::ACTOR_DYNAMIC, uint32_t(a), com0, toE(f0), kind1, uint32_t(b), com1, toE(f1), tol.length);
      for (const Cmd& c : g.joint(kind)) { applyPx(pj, c); applyEng(ej, c); }
      if (g.i(24) == 0) {  // 가끔 전부 자유(행 0개) — 그 묶음은 하나씩 경로로 떨어져야 한다
        for (int ax = 0; ax < 6; ++ax) { Cmd c{MOTION}; c.a = ax; c.b = 2; applyPx(pj, c); applyEng(ej, c); }
        for (int d = 0; d < 6; ++d) { Cmd c{DRIVE}; c.a = d; applyPx(pj, c); applyEng(ej, c); }
      }
      joints.push_back(JP{pj, ej, a, b == nb + 1 ? 0 : b});
    }

    // ---- 풀이 몸체 (test_joints_prep 에서 이미 비트 동일 확인 — 여기서는 PhysX 쪽을 우리 값으로 통일)
    const float simDt = 1.0f / 120.0f;
    const PxU32 posIters = 1 + g.i(8), velIters = 1 + g.i(4);
    const float stepDt = simDt / float(posIters);
    const float invStepDt = 1.0f / stepDt, invSimDt = 1.0f / simDt;
    const PxVec3 grav(0, 0, -9.81f);
    const bool gyro = g.i(3) == 0;
    std::vector<J::TgsBodyVel> ev(nb + 1);
    std::vector<J::TgsTxInertia> et(nb + 1);
    std::vector<J::TgsBodyData> ed(nb + 1);
    memset(ev.data(), 0, sizeof(J::TgsBodyVel) * (nb + 1));
    memset(et.data(), 0, sizeof(J::TgsTxInertia) * (nb + 1));
    memset(ed.data(), 0, sizeof(J::TgsBodyData) * (nb + 1));
    jeng::worldBody(ev[0], et[0], ed[0]);
    for (int b = 1; b <= nb; ++b) {
      const PxRigidBodyData& d = rbd[b - 1];
      eng::V3 lv = toE(d.linearVelocity), av = toE(d.angularVelocity);
      jeng::bodyCoreComputeUnconstrainedVelocity(toE(grav), simDt, d.linearDamping, d.angularDamping, 1.0f, d.maxLinearVelocitySq,
                                                 d.maxAngularVelocitySq, lv, av, false);
      jeng::copyToSolverBodyDataStep(lv, av, d.invMass, toE(d.invInertia), toE(d.body2World), -d.maxDepenetrationVelocity, d.maxContactImpulse,
                                     0xffffffffu, PX_MAX_F32, d.maxAngularVelocitySq, 0, false, ev[b], et[b], ed[b], simDt, gyro);
    }
    std::vector<PxTGSSolverBodyVel> pv(nb + 1);
    std::vector<PxTGSSolverBodyTxInertia> pt(nb + 1);
    std::vector<PxTGSSolverBodyData> pd(nb + 1);
    memcpy(pv.data(), ev.data(), sizeof(PxTGSSolverBodyVel) * (nb + 1));
    memcpy(pt.data(), et.data(), sizeof(PxTGSSolverBodyTxInertia) * (nb + 1));
    memcpy(pd.data(), ed.data(), sizeof(PxTGSSolverBodyData) * (nb + 1));

    // ---- 셰이더 행 (PhysX 셰이더 결과를 두 쪽에 똑같이) + 준비 설명
    Alloc alloc;
    std::vector<PxSolverConstraintDesc> descs(nj);
    std::vector<PxTGSSolverConstraintPrepDesc> preps(nj);
    std::vector<J::Writeback> pwb(nj), ewb(nj);
    memset(pwb.data(), 0, sizeof(J::Writeback) * nj);
    memset(ewb.data(), 0, sizeof(J::Writeback) * nj);
    std::vector<J::Row> prowsAll(J::MAX_CONSTRAINT_ROWS * nj), erowsAll(J::MAX_CONSTRAINT_ROWS * nj);
    std::vector<J::PrepIn> ein(nj);
    std::vector<PxTransform> fA(nj), fB(nj);
    for (int k = 0; k < nj; ++k) {
      const JP& jp = joints[k];
      J::Row* pr = &prowsAll[J::MAX_CONSTRAINT_ROWS * k];
      J::Row* er = &erowsAll[J::MAX_CONSTRAINT_ROWS * k];
      jeng::setupConstraintRows(pr, J::MAX_CONSTRAINT_ROWS);
      fA[k] = b2w[jp.a];
      fB[k] = jp.b ? b2w[jp.b] : PxTransform(PxIdentity);
      const bool extL = (jp.en.constraintFlags & J::CF_ENABLE_EXTENDED_LIMITS) != 0;
      PxVec3p off(0.0f), ca(0.0f), cb(0.0f);
      PxConstraintInvMassScale ims(1, 1, 1, 1);
      const PxU32 n = connectorOf(jp.px)->getPrep()(reinterpret_cast<Px1DConstraint*>(pr), off, J::MAX_CONSTRAINT_ROWS, ims,
                                                    connectorOf(jp.px)->prepareData(), fA[k], fB[k], extL, ca, cb);
      memcpy(er, pr, sizeof(J::Row) * J::MAX_CONSTRAINT_ROWS);
      PxSolverConstraintDesc& d = descs[k];
      memset(&d, 0, sizeof d);
      d.tgsBodyA = &pv[jp.a]; d.tgsBodyB = &pv[jp.b];
      d.bodyADataIndex = PxU32(jp.a); d.bodyBDataIndex = PxU32(jp.b);
      d.linkIndexA = PxSolverConstraintDesc::RIGID_BODY; d.linkIndexB = PxSolverConstraintDesc::RIGID_BODY;
      PxTGSSolverConstraintPrepDesc& p = preps[k];
      memset(&p, 0, sizeof p);
      p.invMassScales = ims;
      p.desc = &d;
      p.body0 = &pv[jp.a]; p.body1 = &pv[jp.b];
      p.body0TxI = &pt[jp.a]; p.body1TxI = &pt[jp.b];
      p.bodyData0 = &pd[jp.a]; p.bodyData1 = &pd[jp.b];
      p.bodyFrame0 = fA[k]; p.bodyFrame1 = fB[k];
      p.bodyState0 = PxSolverContactDesc::eDYNAMIC_BODY;
      p.bodyState1 = jp.b ? PxSolverContactDesc::eDYNAMIC_BODY : PxSolverContactDesc::eSTATIC_BODY;
      p.writeback = &pwb[k];
      p.rows = reinterpret_cast<Px1DConstraint*>(pr);
      p.numRows = n;
      p.linBreakForce = jp.en.linBreakForce; p.angBreakForce = jp.en.angBreakForce;
      p.minResponseThreshold = jp.en.minResponseThreshold;
      p.disablePreprocessing = (jp.en.constraintFlags & J::CF_DISABLE_PREPROCESSING) != 0;
      p.driveLimitsAreForces = (jp.en.constraintFlags & J::CF_DRIVE_LIMITS_ARE_FORCES) != 0;
      p.extendedLimits = extL;
      p.body0WorldOffset = off; p.cA2w = ca; p.cB2w = cb;

      J::PrepIn& e = ein[k];
      e.rows = er; e.numRows = n; e.invMassScales = J::InvMassScale{ims.linear0, ims.angular0, ims.linear1, ims.angular1};
      e.body0WorldOffset = toE(off); e.cA2w = toE(ca); e.cB2w = toE(cb);
      e.bodyFrame0 = toE(fA[k]); e.bodyFrame1 = toE(fB[k]);
      e.body0 = &ev[jp.a]; e.body1 = &ev[jp.b]; e.txI0 = &et[jp.a]; e.txI1 = &et[jp.b]; e.data0 = &ed[jp.a]; e.data1 = &ed[jp.b];
      e.linkIndexA = J::RIGID_BODY; e.linkIndexB = J::RIGID_BODY;
      e.linBreakForce = jp.en.linBreakForce; e.angBreakForce = jp.en.angBreakForce; e.minResponseThreshold = jp.en.minResponseThreshold;
      e.disablePreprocessing = p.disablePreprocessing; e.improvedSlerp = false; e.driveLimitsAreForces = p.driveLimitsAreForces;
      e.extendedLimits = extL; e.disableConstraint = false;
    }
    std::vector<PxConstraintBatchHeader> hdrs(nGroups);
    for (int q = 0; q < nGroups; ++q) { hdrs[q].startIndex = PxU32(4 * q); hdrs[q].stride = 4; hdrs[q].constraintType = 0; }
    const float biasCoef = 2.f * sqrtf(stepDt / simDt);
    PxCreateJointConstraintsTGS(hdrs.data(), PxU32(nGroups), preps.data(), alloc, stepDt, simDt, invStepDt, invSimDt, 1.0f);

    // ---- 엔진 준비 (같은 판정: 넷 다 행이 있으면 4묶음)
    std::vector<std::vector<uint8_t>> eblk(nj), eblk2(nGroups);
    std::vector<uint8_t*> eptr(nj, nullptr);
    std::vector<bool> isBlock(nGroups, false);
    for (int q = 0; q < nGroups; ++q) {
      uint32_t maxRows = 0;
      bool batchable = true;
      for (int a = 0; a < 4; ++a) {
        const uint32_t n = ein[4 * q + a].numRows;
        if (n == 0) batchable = false;
        maxRows = n > maxRows ? n : maxRows;
      }
      isBlock[q] = batchable;
      const uint16_t etype = batchable ? 8 : 2;  // DY_SC_TYPE_BLOCK_1D : DY_SC_TYPE_RB_1D
      sType.add(same(&hdrs[q].constraintType, &etype, 2, "묶음 형", trial, q), trial);
      if (batchable) {
        nBlock4++;
        eblk[4 * q].assign(jeng::blockLength4(maxRows) + 16, 0xcd);
        const uint32_t len = jeng::setupSolverConstraintStep4(&ein[4 * q], eblk[4 * q].data(), stepDt, simDt, invStepDt, invSimDt, maxRows, 1.0f, biasCoef);
        for (int a = 0; a < 4; ++a) eptr[4 * q + a] = eblk[4 * q].data();
        const uint32_t plen = uint32_t(descs[4 * q].constraintLengthOver16) * 16u;
        bool ok = same(&plen, &len, 4, "블록 길이", trial, q);
        if (ok) ok &= sameBlock4(descs[4 * q].constraint, eptr[4 * q], len, "4묶음 블록", trial, q);
        sBlock.add(ok, trial);
        // 엔진 셰이더(d6SolverPrep) + 준비를 한 번에 — 장면에서 쓰는 입구
        const J::D6Data* dd[4]; uint16_t fl[4]; float lb[4], ab[4], mr[4];
        eng::Tf tf0[4], tf1[4]; const eng::Tf* pf0[4]; const eng::Tf* pf1[4];
        const J::TgsBodyVel* b0[4]; const J::TgsBodyVel* b1[4]; const J::TgsTxInertia* t0[4]; const J::TgsTxInertia* t1[4];
        const J::TgsBodyData* d0[4]; const J::TgsBodyData* d1[4];
        for (int a = 0; a < 4; ++a) {
          const JP& jp = joints[4 * q + a];
          dd[a] = &J::prepareData(const_cast<J::D6Joint&>(jp.en));
          fl[a] = jp.en.constraintFlags; lb[a] = jp.en.linBreakForce; ab[a] = jp.en.angBreakForce; mr[a] = jp.en.minResponseThreshold;
          tf0[a] = toE(fA[4 * q + a]); tf1[a] = toE(fB[4 * q + a]); pf0[a] = &tf0[a]; pf1[a] = &tf1[a];
          b0[a] = &ev[jp.a]; b1[a] = &ev[jp.b]; t0[a] = &et[jp.a]; t1[a] = &et[jp.b]; d0[a] = &ed[jp.a]; d1[a] = &ed[jp.b];
        }
        std::vector<J::Row> rows4(J::MAX_CONSTRAINT_ROWS * 4);
        eblk2[q].assign(jeng::blockLength4(maxRows) + 16, 0xcd);
        const uint32_t len2 = jeng::prepareD6Step4(dd, fl, lb, ab, mr, pf0, pf1, b0, b1, t0, t1, d0, d1, rows4.data(), eblk2[q].data(), stepDt, simDt,
                                                   invStepDt, invSimDt, 1.0f, biasCoef);
        bool ok2 = same(&plen, &len2, 4, "엔진 셰이더 4묶음 길이", trial, q);
        if (ok2) ok2 &= sameBlock4(descs[4 * q].constraint, eblk2[q].data(), len2, "엔진 셰이더 4묶음 블록", trial, q);
        sBlockE.add(ok2, trial);
      } else {
        for (int a = 0; a < 4; ++a) {
          const int k = 4 * q + a;
          nSingle++;
          const uint32_t n = ein[k].numRows;
          eblk[k].assign(J::blockLength(n, false) + 16, 0xcd);
          eptr[k] = n ? eblk[k].data() : nullptr;
          if (n) jeng::setupSolverConstraintStep(ein[k], eblk[k].data(), stepDt, simDt, invStepDt, invSimDt, 1.0f, biasCoef);
          const uint32_t len = uint32_t(descs[k].constraintLengthOver16) * 16u;
          const uint32_t elen = eptr[k] ? J::blockLength(n, false) : 0u;
          bool ok = same(&len, &elen, 4, "블록 길이(하나씩)", trial, k);
          if (ok && len) ok &= same(descs[k].constraint, eptr[k], len, "블록(하나씩)", trial, k);
          sBlock.add(ok, trial);
        }
      }
    }

    // ---- 반복 풀이 + 적분 + 되쓰기 (즉시 모드 순서: 머리마다)
    PxSolveConstraintsTGS(hdrs.data(), PxU32(nGroups), descs.data(), pv.data(), pt.data(), PxU32(nb + 1), posIters, velIters, stepDt, invStepDt, 0,
                          nullptr, nullptr, nullptr);
    auto solveGroup = [&](int q, float elapsed) {
      if (isBlock[q]) {
        J::TgsBodyVel* bb[4][2];
        const J::TgsTxInertia* tt[4][2];
        for (int a = 0; a < 4; ++a) {
          const JP& jp = joints[4 * q + a];
          bb[a][0] = &ev[jp.a]; bb[a][1] = &ev[jp.b]; tt[a][0] = &et[jp.a]; tt[a][1] = &et[jp.b];
        }
        jeng::solve1DStep4(eptr[4 * q], bb, tt, elapsed);
      } else {
        for (int a = 0; a < 4; ++a) {
          const JP& jp = joints[4 * q + a];
          jeng::solve1DStep(eptr[4 * q + a], ev[jp.a], ev[jp.b], et[jp.a], et[jp.b], elapsed, false, true);
        }
      }
    };
    float elapsed = 0.0f;
    for (PxU32 it = 0; it < posIters; ++it) {
      const bool last = it + 1 == posIters;
      for (int q = 0; q < nGroups; ++q) {
        solveGroup(q, elapsed);
        if (last) {
          if (isBlock[q]) jeng::conclude1DStep4(eptr[4 * q]);
          else for (int a = 0; a < 4; ++a) jeng::conclude1DStep(eptr[4 * q + a]);
        }
      }
      for (int b = 0; b <= nb; ++b) jeng::integrateCoreStep(ev[b], et[b], stepDt);
      elapsed += stepDt;
    }
    for (PxU32 it = 0; it < velIters; ++it) {
      for (int q = 0; q < nGroups; ++q) {
        solveGroup(q, elapsed);
        if (it + 1 == velIters) {
          if (isBlock[q]) {
            J::Writeback* w[4] = {&ewb[4 * q], &ewb[4 * q + 1], &ewb[4 * q + 2], &ewb[4 * q + 3]};
            jeng::writeBack1D4(eptr[4 * q], w);
          } else {
            for (int a = 0; a < 4; ++a) jeng::writeBack1DStep(eptr[4 * q + a], &ewb[4 * q + a]);
          }
        }
      }
    }
    for (int b = 0; b <= nb; ++b) {
      bool ok = same(&pv[b], &ev[b], sizeof(PxTGSSolverBodyVel), "풀이 후 vel", trial, b);
      ok &= same(&pt[b], &et[b], sizeof(PxTGSSolverBodyTxInertia), "풀이 후 txI", trial, b);
      sVel.add(ok, trial);
    }
    for (int q = 0; q < nGroups; ++q) {
      if (isBlock[q]) {
        const uint32_t len = uint32_t(descs[4 * q].constraintLengthOver16) * 16u;
        sAfter.add(sameBlock4(descs[4 * q].constraint, eptr[4 * q], len, "풀이 후 4묶음 블록", trial, q), trial);
      } else {
        for (int a = 0; a < 4; ++a) {
          const int k = 4 * q + a;
          const uint32_t len = uint32_t(descs[k].constraintLengthOver16) * 16u;
          if (len) sAfter.add(same(descs[k].constraint, eptr[k], len, "풀이 후 블록", trial, k), trial);
        }
      }
      for (int a = 0; a < 4; ++a) sWb.add(same(&pwb[4 * q + a], &ewb[4 * q + a], sizeof(J::Writeback), "되쓰기", trial, 4 * q + a), trial);
    }

    for (auto& jp : joints) jp.px->release();
    for (int b = 1; b <= nb; ++b) actors[b]->release();
    stat->release();
  }

  printf("\nTGS 1D 4개 묶음 경로 비교 (시도 %d, 씨앗 %d): 4묶음 %" PRIu64 " 개, 하나씩 조인트 %" PRIu64 " 개\n", trials, seed, nBlock4, nSingle);
  Stat* all[] = {&sType, &sBlock, &sBlockE, &sVel, &sAfter, &sWb};
  bool ok = true;
  for (Stat* s : all) {
    printf("  %-34s 비교 %8" PRIu64 "  비트 다름 %6" PRIu64 "  첫 다름 시도 %" PRId64 "\n", s->name, s->cmp, s->bad, s->firstTrial);
    ok &= s->bad == 0;
  }
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
