// 층 1 시험 (함수 단위): D6 조인트 상수 블록 · 셰이더 행 · TGS 1D 준비 블록 · 반복 풀이 · 되쓰기 = PhysX 5.6.1 비트 동일?
// 정답지: PhysX 조인트 객체의 셰이더(PxConstraintConnector::getPrep)와 즉시 모드 API(PxCreateJointConstraintsTGS,
// PxSolveConstraintsTGS — 같은 Dy::setupSolverConstraintStep / solve1DStep / conclude1DStep / writeBack1DStep 을 부른다).
//   test_joints_prep [--trials N] [--seed S] [--kind K]
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// 엔진 함수는 jeng_api.cpp(다른 번역 단위, common/aos.h 포함)에 있다 — aos 와 PhysX 헤더는 한 파일에 같이 못 넣는다
#include "PxImmediateMode.h"
#include "jeng_api.h"
#include "joint_cmds.h"

using namespace physx;
using namespace physx::immediate;
using namespace jcmd;
namespace J = eng::jnt;

static_assert(sizeof(J::Row) == sizeof(Px1DConstraint), "Row 배치");
static_assert(sizeof(J::TgsBodyVel) == sizeof(PxTGSSolverBodyVel), "TgsBodyVel 배치");
static_assert(sizeof(J::TgsTxInertia) == sizeof(PxTGSSolverBodyTxInertia), "TgsTxInertia 배치");
static_assert(sizeof(J::TgsBodyData) == sizeof(PxTGSSolverBodyData), "TgsBodyData 배치");
static_assert(sizeof(J::Sc1DHeader) == 176 && sizeof(J::Sc1DRow) == 96 && sizeof(J::Sc1DRowExt) == 160, "풀이 제약 배치");
static_assert(sizeof(J::Writeback) == 32, "되쓰기 배치");

static PxDefaultAllocator gAlloc;
// PhysX 오류는 세기만 한다 ("Double pyramid mode not supported" 등 — 원본도 오류만 내고 그 행을 안 만든다. 우리도 같다)
struct QuietErr : PxErrorCallback {
  int n = 0;
  void reportError(PxErrorCode::Enum, const char*, const char*, int) override { n++; }
};
static QuietErr gErr;

// Ext::JointT<PxD6Joint, D6JointData, ...> : public PxD6Joint, public PxConstraintConnector (ExtJoint.h:97)
// -> 커넥터는 PxD6Joint 바로 뒤. getExternalReference 가 자기 조인트를 돌려주는지로 위치를 확인한다.
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
    if (n < 4) {
      printf(" 바이트 PhysX:");
      for (size_t i = 0; i < n; ++i) printf(" %02x", static_cast<const uint8_t*>(a)[i]);
      printf(" 엔진:");
      for (size_t i = 0; i < n; ++i) printf(" %02x", static_cast<const uint8_t*>(b)[i]);
    }
    printf("\n");
  }
  return false;
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

  Stat sData{"상수 블록(D6 데이터)"}, sRows{"셰이더 행"}, sMeta{"셰이더 부산물(행 수·오프셋·앵커)"}, sBody{"풀이 몸체 만들기"},
      sBlock{"풀이 제약 블록"}, sVel{"풀이 후 몸체 속도·자세"}, sAfter{"풀이 후 제약 블록"}, sWb{"되쓰기"};
  int kindCount[K_COUNT] = {0};
  Gen g(static_cast<uint32_t>(seed));

  for (int trial = 0; trial < trials; ++trial) {
    // ---- 몸체: 0 = 세계, 1..nb = 동적. 조인트 1..3 개 (사슬·세계 연결 섞음)
    const int nb = 1 + g.i(3);
    const int nj = 1 + g.i(3);
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
      d.linearDamping = 0.0f;
      d.angularDamping = 0.05f;
      d.maxLinearVelocitySq = 1e32f;
      d.maxAngularVelocitySq = 100.0f * 100.0f;
    }
    PxRigidStatic* stat = phys->createRigidStatic(PxTransform(toPx(g.v(1.0f)), toPx(g.q())));

    struct JP { PxD6Joint* px; J::D6Joint en; int a, b; int kind; };
    std::vector<JP> joints;
    for (int k = 0; k < nj; ++k) {
      int kind = onlyKind >= 0 ? onlyKind : g.i(K_COUNT);
      kindCount[kind]++;
      const int a = 1 + g.i(nb);
      int b = g.i(nb + 2);            // 0 = 세계(null), nb+1 = 정적 행위자
      if (b == a) b = 0;
      const PxTransform f0(toPx(g.v(0.2f)), toPx(g.q())), f1(toPx(g.v(0.2f)), toPx(g.q()));
      PxRigidActor* act1 = (b == 0) ? nullptr : (b == nb + 1 ? static_cast<PxRigidActor*>(stat) : actors[b]);
      PxD6Joint* pj = PxD6JointCreate(*phys, actors[a], f0, act1, f1);
      const uint8_t kind1 = b == 0 ? J::ACTOR_NONE : (b == nb + 1 ? J::ACTOR_STATIC : J::ACTOR_DYNAMIC);
      const eng::Tf com0 = toE(actors[a]->getCMassLocalPose());
      const eng::Tf com1 = b == 0 ? eng::Tf{eng::qid(), {0, 0, 0}}
                                  : (b == nb + 1 ? J::comOf(J::ACTOR_STATIC, toE(stat->getGlobalPose())) : toE(actors[b]->getCMassLocalPose()));
      J::D6Joint ej = J::createD6(J::ACTOR_DYNAMIC, uint32_t(a), com0, toE(f0), kind1, uint32_t(b), com1, toE(f1), tol.length);
      for (const Cmd& c : g.joint(kind)) { applyPx(pj, c); applyEng(ej, c); }
      joints.push_back(JP{pj, ej, a, b == nb + 1 ? 0 : b, kind});  // 정적 행위자는 풀이에서 세계 몸체(0)
    }

    // ---- 1) 상수 블록 (prepareData)
    std::vector<const void*> pxData(nj);
    for (int k = 0; k < nj; ++k) {
      PxConstraintConnector* conn = connectorOf(joints[k].px);
      pxData[k] = conn->prepareData();
      const J::D6Data& e = J::prepareData(joints[k].en);
      J::D6Data p;
      memcpy(&p, pxData[k], sizeof p);
      bool ok = true;
#define CMPF(f) ok &= same(&p.f, &e.f, sizeof(p.f), "D6." #f, trial, k)
      CMPF(invMassScale); CMPF(c2b[0].q); CMPF(c2b[0].p); CMPF(c2b[1].q); CMPF(c2b[1].p); CMPF(motion); CMPF(distanceLimit);
      CMPF(linearLimitX); CMPF(linearLimitY); CMPF(linearLimitZ); CMPF(twistLimit); CMPF(swingLimit); CMPF(pyramidSwingLimit);
      CMPF(drive); CMPF(drivePosition); CMPF(driveLinearVelocity); CMPF(driveAngularVelocity); CMPF(locked); CMPF(limited);
      CMPF(driving); CMPF(distanceMinDist); CMPF(mUseDistanceLimit); CMPF(mUseNewLinearLimits); CMPF(mUseConeLimit);
      CMPF(mUsePyramidLimits); CMPF(angularDriveConfig);
#undef CMPF
      const uint16_t pf = uint16_t(uint32_t(joints[k].px->getConstraintFlags()));
      const uint16_t ef = uint16_t(joints[k].en.constraintFlags & ~J::CF_GPU_COMPATIBLE);  // 공개 getFlags 는 eGPU_COMPATIBLE 을 가린다 (NpConstraint.cpp:50)
      ok &= same(&pf, &ef, 2, "제약 플래그", trial, k);
      float lf, af;
      joints[k].px->getBreakForce(lf, af);
      ok &= same(&lf, &joints[k].en.linBreakForce, 4, "break", trial, k) && same(&af, &joints[k].en.angBreakForce, 4, "breakT", trial, k);
      sData.add(ok, trial);
    }

    // ---- 2) 셰이더 행 (같은 몸체 자세)
    for (int k = 0; k < nj; ++k) {
      PxConstraintConnector* conn = connectorOf(joints[k].px);
      PxConstraintSolverPrep prep = conn->getPrep();
      const bool ext = (joints[k].en.constraintFlags & J::CF_ENABLE_EXTENDED_LIMITS) != 0;
      const PxTransform bA = b2w[joints[k].a];
      const PxTransform bB = joints[k].b ? b2w[joints[k].b] : PxTransform(PxIdentity);
      J::Row pr[J::MAX_CONSTRAINT_ROWS], er[J::MAX_CONSTRAINT_ROWS];
      jeng::setupConstraintRows(pr, J::MAX_CONSTRAINT_ROWS);
      jeng::setupConstraintRows(er, J::MAX_CONSTRAINT_ROWS);
      PxVec3p off(0.0f), ca(0.0f), cb(0.0f);
      PxConstraintInvMassScale ims(1, 1, 1, 1);
      const PxU32 n = prep(reinterpret_cast<Px1DConstraint*>(pr), off, J::MAX_CONSTRAINT_ROWS, ims, pxData[k], bA, bB, ext, ca, cb);
      const J::PrepOut o = jeng::d6SolverPrep(er, joints[k].en.data, toE(bA), toE(bB), ext);
      bool ok = same(&n, &o.numRows, 4, "행 수", trial, k);
      ok &= same(&off.x, &o.body0WorldOffset, 12, "body0WorldOffset", trial, k);
      ok &= same(&ca.x, &o.cA2w, 12, "cA2w", trial, k) && same(&cb.x, &o.cB2w, 12, "cB2w", trial, k);
      ok &= same(&ims, &o.invMassScale, 16, "invMassScale", trial, k);
      sMeta.add(ok, trial);
      for (PxU32 r = 0; r < n && r < o.numRows; ++r) sRows.add(same(&pr[r], &er[r], sizeof(J::Row), "행", trial, int(r)), trial);
    }

    // ---- 3) 풀이 몸체 (PxConstructSolverBodiesTGS vs 우리 copyToSolverBodyDataStep)
    const float simDt = 1.0f / 120.0f;
    const PxU32 posIters = 1 + g.i(8), velIters = 1 + g.i(4);
    const float stepDt = simDt / float(posIters);
    const float invStepDt = 1.0f / stepDt, invSimDt = 1.0f / simDt;
    const PxVec3 grav(0, 0, -9.81f);
    std::vector<PxTGSSolverBodyVel> pv(nb + 1);
    std::vector<PxTGSSolverBodyTxInertia> pt(nb + 1);
    std::vector<PxTGSSolverBodyData> pd(nb + 1);
    memset(pv.data(), 0, sizeof(PxTGSSolverBodyVel) * (nb + 1));
    memset(pt.data(), 0, sizeof(PxTGSSolverBodyTxInertia) * (nb + 1));
    memset(pd.data(), 0, sizeof(PxTGSSolverBodyData) * (nb + 1));
    const bool gyro = g.i(3) == 0;
    PxConstructSolverBodiesTGS(rbd.data(), pv.data() + 1, pt.data() + 1, pd.data() + 1, PxU32(nb), grav, simDt, gyro);
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
      jeng::copyToSolverBodyDataStep(lv, av, d.invMass, toE(d.invInertia), toE(d.body2World), -d.maxDepenetrationVelocity,
                                           d.maxContactImpulse, 0xffffffffu, PX_MAX_F32, d.maxAngularVelocitySq, 0, false, ev[b], et[b],
                                           ed[b], simDt, gyro);
      bool ok = same(&pv[b], &ev[b], sizeof(PxTGSSolverBodyVel), "풀이몸체 vel", trial, b);
      ok &= same(&pt[b], &et[b], sizeof(PxTGSSolverBodyTxInertia), "풀이몸체 txI", trial, b);
      ok &= same(&pd[b], &ed[b], sizeof(PxTGSSolverBodyData), "풀이몸체 data", trial, b);
      sBody.add(ok, trial);
    }
    // PhysX 쪽도 세계 몸체를 장면과 같게(운동학 아님) 둔다 — 두 쪽 입력을 우리 몸체 값으로 통일
    memcpy(pv.data(), ev.data(), sizeof(PxTGSSolverBodyVel) * (nb + 1));
    memcpy(pt.data(), et.data(), sizeof(PxTGSSolverBodyTxInertia) * (nb + 1));
    memcpy(pd.data(), ed.data(), sizeof(PxTGSSolverBodyData) * (nb + 1));

    // ---- 4) 풀이 제약 블록: PhysX 셰이더 행을 두 쪽에 똑같이 넣고 setupSolverConstraintStep 비교
    Alloc alloc;
    std::vector<PxSolverConstraintDesc> descs(nj);
    std::vector<PxTGSSolverConstraintPrepDesc> preps(nj);
    std::vector<PxConstraintBatchHeader> hdrs(nj);
    std::vector<J::Writeback> pwb(nj), ewb(nj);
    memset(pwb.data(), 0, sizeof(J::Writeback) * nj);
    memset(ewb.data(), 0, sizeof(J::Writeback) * nj);
    std::vector<std::vector<uint8_t>> eblk(nj);
    std::vector<uint8_t*> eptr(nj, nullptr);
    std::vector<J::Row> prowsAll(J::MAX_CONSTRAINT_ROWS * nj), erowsAll(J::MAX_CONSTRAINT_ROWS * nj);
    const float biasCoef = 2.f * sqrtf(stepDt / simDt);  // 즉시 모드 식 (NpImmediateMode.cpp:1486)
    for (int k = 0; k < nj; ++k) {
      const JP& jp = joints[k];
      J::Row* pr = &prowsAll[J::MAX_CONSTRAINT_ROWS * k];
      J::Row* er = &erowsAll[J::MAX_CONSTRAINT_ROWS * k];
      jeng::setupConstraintRows(pr, J::MAX_CONSTRAINT_ROWS);
      const PxTransform bA = b2w[jp.a];
      const PxTransform bB = jp.b ? b2w[jp.b] : PxTransform(PxIdentity);
      const bool extL = (jp.en.constraintFlags & J::CF_ENABLE_EXTENDED_LIMITS) != 0;
      PxVec3p off(0.0f), ca(0.0f), cb(0.0f);
      PxConstraintInvMassScale ims(1, 1, 1, 1);
      const PxU32 n = connectorOf(jp.px)->getPrep()(reinterpret_cast<Px1DConstraint*>(pr), off, J::MAX_CONSTRAINT_ROWS, ims, pxData[k], bA, bB,
                                                    extL, ca, cb);
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
      p.bodyFrame0 = bA; p.bodyFrame1 = bB;
      p.bodyState0 = PxSolverContactDesc::eDYNAMIC_BODY;
      p.bodyState1 = jp.b ? PxSolverContactDesc::eDYNAMIC_BODY : PxSolverContactDesc::eSTATIC_BODY;
      p.writeback = &pwb[k];
      p.rows = reinterpret_cast<Px1DConstraint*>(pr);
      p.numRows = n;
      p.linBreakForce = jp.en.linBreakForce; p.angBreakForce = jp.en.angBreakForce;
      p.minResponseThreshold = jp.en.minResponseThreshold;
      p.disablePreprocessing = (jp.en.constraintFlags & J::CF_DISABLE_PREPROCESSING) != 0;
      p.improvedSlerp = false;
      p.driveLimitsAreForces = (jp.en.constraintFlags & J::CF_DRIVE_LIMITS_ARE_FORCES) != 0;
      p.extendedLimits = extL;
      p.disableConstraint = false;
      p.body0WorldOffset = off; p.cA2w = ca; p.cB2w = cb;
      hdrs[k].startIndex = PxU32(k); hdrs[k].stride = 1; hdrs[k].constraintType = 0;

      J::PrepIn e;
      e.rows = er; e.numRows = n; e.invMassScales = J::InvMassScale{ims.linear0, ims.angular0, ims.linear1, ims.angular1};
      e.body0WorldOffset = toE(off); e.cA2w = toE(ca); e.cB2w = toE(cb);
      e.bodyFrame0 = toE(bA); e.bodyFrame1 = toE(bB);
      e.body0 = &ev[jp.a]; e.body1 = &ev[jp.b]; e.txI0 = &et[jp.a]; e.txI1 = &et[jp.b]; e.data0 = &ed[jp.a]; e.data1 = &ed[jp.b];
      e.linkIndexA = J::RIGID_BODY; e.linkIndexB = J::RIGID_BODY;
      e.linBreakForce = jp.en.linBreakForce; e.angBreakForce = jp.en.angBreakForce; e.minResponseThreshold = jp.en.minResponseThreshold;
      e.disablePreprocessing = p.disablePreprocessing; e.improvedSlerp = false; e.driveLimitsAreForces = p.driveLimitsAreForces;
      e.extendedLimits = extL; e.disableConstraint = false;
      eblk[k].assign(J::blockLength(n, false) + 16, 0xcd);
      eptr[k] = n ? eblk[k].data() : nullptr;
      if (n) jeng::setupSolverConstraintStep(e, eblk[k].data(), stepDt, simDt, invStepDt, invSimDt, 1.0f, biasCoef);
    }
    PxCreateJointConstraintsTGS(hdrs.data(), PxU32(nj), preps.data(), alloc, stepDt, simDt, invStepDt, invSimDt, 1.0f);
    for (int k = 0; k < nj; ++k) {
      const uint32_t len = uint32_t(descs[k].constraintLengthOver16) * 16u;
      const uint32_t elen = eptr[k] ? J::blockLength(preps[k].numRows, false) : 0u;
      bool ok = same(&len, &elen, 4, "블록 길이", trial, k);
      if (ok && len) ok &= same(descs[k].constraint, eptr[k], len, "블록", trial, k);
      sBlock.add(ok, trial);
    }

    // ---- 5) 반복 풀이 + 적분 + 되쓰기 (즉시 모드 루프와 같은 순서)
    PxSolveConstraintsTGS(hdrs.data(), PxU32(nj), descs.data(), pv.data(), pt.data(), PxU32(nb + 1), posIters, velIters, stepDt, invStepDt, 0,
                          nullptr, nullptr, nullptr);
    float elapsed = 0.0f;
    for (PxU32 it = 0; it < posIters; ++it) {
      const bool last = it + 1 == posIters;
      for (int k = 0; k < nj; ++k) {
        const JP& jp = joints[k];
        jeng::solve1DStep(eptr[k], ev[jp.a], ev[jp.b], et[jp.a], et[jp.b], elapsed, false, true);
        if (last) jeng::conclude1DStep(eptr[k]);
      }
      for (int b = 0; b <= nb; ++b) jeng::integrateCoreStep(ev[b], et[b], stepDt);
      elapsed += stepDt;
    }
    for (PxU32 it = 0; it < velIters; ++it) {
      for (int k = 0; k < nj; ++k) {
        const JP& jp = joints[k];
        jeng::solve1DStep(eptr[k], ev[jp.a], ev[jp.b], et[jp.a], et[jp.b], elapsed, false, true);
        if (it + 1 == velIters) jeng::writeBack1DStep(eptr[k], &ewb[k]);
      }
    }
    for (int b = 0; b <= nb; ++b) {
      bool ok = same(&pv[b], &ev[b], sizeof(PxTGSSolverBodyVel), "풀이 후 vel", trial, b);
      ok &= same(&pt[b], &et[b], sizeof(PxTGSSolverBodyTxInertia), "풀이 후 txI", trial, b);
      sVel.add(ok, trial);
    }
    for (int k = 0; k < nj; ++k) {
      const uint32_t len = uint32_t(descs[k].constraintLengthOver16) * 16u;
      if (len) sAfter.add(same(descs[k].constraint, eptr[k], len, "풀이 후 블록", trial, k), trial);
      if (len) sWb.add(same(&pwb[k], &ewb[k], sizeof(J::Writeback), "되쓰기", trial, k), trial);
    }

    for (auto& jp : joints) jp.px->release();
    for (int b = 1; b <= nb; ++b) actors[b]->release();
    stat->release();
  }

  printf("\nD6 조인트 함수 단위 비교 (시도 %d, 씨앗 %d)\n  종류별 조인트 수:", trials, seed);
  for (int k = 0; k < K_COUNT; ++k) printf(" %s %d", kKindNames[k], kindCount[k]);
  printf("\n");
  Stat* all[] = {&sData, &sRows, &sMeta, &sBody, &sBlock, &sVel, &sAfter, &sWb};
  bool ok = true;
  for (Stat* s : all) {
    printf("  %-32s 비교 %8" PRIu64 "  비트 다름 %6" PRIu64 "  첫 다름 시도 %" PRId64 "\n", s->name, s->cmp, s->bad, s->firstTrial);
    ok &= s->bad == 0;
  }
  printf("  (PhysX 오류 보고 %d 건: 지원 안 되는 설정 경고 — 원본도 그 행을 안 만들고 우리도 같다)\n", gErr.n);
  printf("%s\n", ok ? "결과: 전부 비트 동일" : "결과: 불일치 있음");
  phys->release();
  fnd->release();
  return ok ? 0 : 3;
}
