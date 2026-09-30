// OVD 재생기 (층 0 정답지): 공식 평가기에서 뜬 OmniPVD 기록을 우리 PhysX 5.6.1 빌드로 처음부터 다시 만들고
// 같은 API 호출을 같은 순서로 넣은 뒤, simulate 마다 공식이 기록한 결과(자세·속도·관절값)와 비트 단위로 비교한다.
//   ovd_replay <file.ovd> [--convex convex.bin] [--filters filters.txt] [--sidelog sidelog.bin] [--threads N]
//              [--csv per_frame.csv] [--max-frames N] [--verbose]
// 원리 (PhysX 소스 기준)
//   - 객체는 만들어질 때 "create + 그 순간 속성 전부" 가 기록되고(OmniPvdPxSampler::onObjectAdd), 이어서
//     PxPhysics.<목록> add 가 온다 -> 그 시점에 우리 쪽 객체를 같은 인자로 만든다.
//   - 그 뒤의 set 은 API 호출이다 -> 같은 setter 를 부른다.
//   - simulate() 는 PxScene.elapsedTime set 으로 표시된다(NpScene.cpp:2839). 그 뒤 다음 stopFrame 까지는
//     fetchResults 가 적은 결과(NpSceneFetchResults.cpp:277-470) -> 적용하지 않고 비교만 한다.
//   - OVD 에 안 남는 호출(applyCache 계열, wakeUp/putToSleep)은 capture 의 sidelog 로 채운다.
#include <algorithm>
#include <array>
#include <unordered_set>
#include <sstream>
#include <fstream>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "OmniPvdDefines.h"
#include "PxPhysicsAPI.h"
#include "omnipvd/PxOmniPvd.h"
#include "OmniPvdFileWriteStream.h"
#include "OmniPvdWriter.h"
#include "omni_filter.h"
#include "ovd.h"
#include "sidecar.h"
#include "core/omni/controllers.h"
#include "core/omni/states.h"
#include "core/omni/bddl.h"
#include "core/omni/agframe.h"
#include "core/omni/assisted_grasp.h"

using namespace physx;

static PxDefaultAllocator gAlloc;
struct ErrCb : PxErrorCallback {
  int n = 0;
  void reportError(PxErrorCode::Enum code, const char* msg, const char* file, int line) override {
    if (++n <= 40) fprintf(stderr, "[PhysX %d] %s (%s:%d)\n", int(code), msg, file, line);
  }
} gErr;

// ------------------------------------------------------------------------------------------------ 정규화의 원상(原像)
// PhysX 의 몇몇 setter 는 입력 자세를 getNormalized() 한 뒤 저장하고, OVD 에는 정규화된 값이 남는다
// (setGlobalPose NpRigidDynamic.cpp:106, setCMassLocalPose :185, PxShape::setLocalPose NpShape.cpp:345,
//  createRigid* NpPhysics.cpp:454/522, createLink NpArticulationReducedCoordinate.cpp:987, PxJoint::setLocalPose ExtJoint.h:239).
// 기록값을 그대로 다시 넣으면 한 번 더 정규화되어 마지막 비트가 바뀔 수 있다. 그래서 getNormalized() 결과가
// 기록값과 비트까지 같은 입력을 찾아 넣는다 (내부 상태는 정규화된 값에만 의존하므로 이러면 원본과 같다).
static uint64_t g_prenorm_calls = 0, g_prenorm_fail = 0;
static bool qeq(const PxQuat& a, const PxQuat& b) { return memcmp(&a, &b, sizeof(PxQuat)) == 0; }
static PxQuat prenorm(const PxQuat& r) {
  g_prenorm_calls++;
  if (qeq(r.getNormalized(), r)) return r;
  for (int k = 1; k <= 64; ++k)
    for (int sg = -1; sg <= 1; sg += 2) {
      const float s = 1.0f + float(sg * k) * 5.9604645e-08f;  // 2^-24 단위로 전체 크기를 조금씩
      PxQuat q(r.x * s, r.y * s, r.z * s, r.w * s);
      if (qeq(q.getNormalized(), r)) return q;
    }
  float c[4] = {r.x, r.y, r.z, r.w};
  for (int a = -2; a <= 2; ++a)
    for (int b = -2; b <= 2; ++b)
      for (int d = -2; d <= 2; ++d)
        for (int e = -2; e <= 2; ++e) {
          int st[4] = {a, b, d, e};
          float t[4];
          for (int i = 0; i < 4; ++i) {
            t[i] = c[i];
            for (int j = 0; j < std::abs(st[i]); ++j) t[i] = std::nextafter(t[i], st[i] > 0 ? INFINITY : -INFINITY);
          }
          PxQuat q(t[0], t[1], t[2], t[3]);
          if (qeq(q.getNormalized(), r)) return q;
        }
  g_prenorm_fail++;
  return r;
}
static PxTransform prenorm(const PxTransform& t) { return PxTransform(t.p, prenorm(t.q)); }

// S3 v1: OmniGibson RigidContactAPI 의 입력(텐서 접촉 뷰 get_contact_force_matrix·get_net_contact_forces)을 PhysX 접촉 보고로 만든다.
// 서브스텝마다 쌍(액터 둘)별 충격 벡터 합, 액터별 알짜 충격 합. 판정은 0 이 아닌지만 본다(usd_utils.py:127·189).
struct S3ContactCb : physx::PxSimulationEventCallback {
  std::map<std::pair<const void*, const void*>, physx::PxVec3> pair_imp;  // (작은 주소, 큰 주소) -> 합 (방향은 판정에 무관)
  std::unordered_map<const void*, physx::PxVec3> net;
  std::vector<physx::PxContactPairPoint> buf;
  std::map<std::pair<const void*, const void*>, physx::PxVec3> first_pt;  // 쌍의 첫 접촉점 (첫 PxContactPair 의 첫 점)
  std::map<std::pair<const void*, const void*>, physx::PxVec3> last_first_pt;  // 직전 simulate 몫 (다음 서브스텝의 AG 가 읽음)
  void clear() { pair_imp.clear(); net.clear(); last_first_pt.swap(first_pt); first_pt.clear(); }
  void onContact(const physx::PxContactPairHeader& h, const physx::PxContactPair* pairs, physx::PxU32 n) override {
    using namespace physx;
    if (h.flags & (PxContactPairHeaderFlag::eREMOVED_ACTOR_0 | PxContactPairHeaderFlag::eREMOVED_ACTOR_1)) return;
    for (PxU32 i = 0; i < n; ++i) {
      const PxContactPair& cp = pairs[i];
      if (!cp.contactCount) continue;
      buf.resize(cp.contactCount);
      const PxU32 np = cp.extractContacts(buf.data(), cp.contactCount);
      PxVec3 sum(0.0f);
      for (PxU32 k = 0; k < np; ++k) sum += buf[k].impulse;
      const void* a = h.actors[0];
      const void* b = h.actors[1];
      // PxVec3 기본 생성자는 값을 채우지 않는다 -> 0 으로 넣고 더한다
      net.emplace(a, physx::PxVec3(0.0f)).first->second += sum;
      net.emplace(b, physx::PxVec3(0.0f)).first->second -= sum;
      auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
      pair_imp.emplace(key, physx::PxVec3(0.0f)).first->second += sum;
      if (np) first_pt.emplace(key, buf[0].position);
    }
  }
  void onConstraintBreak(physx::PxConstraintInfo*, physx::PxU32) override {}
  void onWake(physx::PxActor**, physx::PxU32) override {}
  void onSleep(physx::PxActor**, physx::PxU32) override {}
  void onTrigger(physx::PxTriggerPair*, physx::PxU32) override {}
  void onAdvance(const physx::PxRigidBody* const*, const physx::PxTransform*, const physx::PxU32) override {}
};

// ------------------------------------------------------------------------------------------------ 재생기
// 진단 전용 접촉 보고 (--trace-obj 가 있을 때만): 이름에 trace_sub 가 든 액터가 낀 쌍의 접촉점·분리 거리를 찍는다.
// 알림 플래그(eNOTIFY_*)만 더하고 풀이 플래그는 건드리지 않는다.
struct DiagContactCb : physx::PxSimulationEventCallback {
  std::string sub;
  const uint64_t* sims = nullptr;
  int printed = 0;
  void onContact(const physx::PxContactPairHeader& h, const physx::PxContactPair* pairs, physx::PxU32 n) override {
    using namespace physx;
    for (PxU32 i = 0; i < n && printed < 60; ++i) {
      const PxContactPair& cp = pairs[i];
      const char* a = h.actors[0] && h.actors[0]->getName() ? h.actors[0]->getName() : "?";
      const char* b = h.actors[1] && h.actors[1]->getName() ? h.actors[1]->getName() : "?";
      PxContactPairPoint pts[16];
      const PxU32 np = cp.extractContacts(pts, 16);
      float minsep = 1e30f, maximp = 0;
      for (PxU32 k = 0; k < np; ++k) { minsep = std::min(minsep, pts[k].separation); maximp = std::max(maximp, pts[k].impulse.magnitude()); }
      fprintf(stderr, "[접촉] sim %llu %s <-> %s 점 %u 최소분리 %.6g 최대충격 %.6g\n", (unsigned long long)(sims ? *sims : 0), a, b, np, minsep, maximp);
      printed++;
    }
  }
  void onConstraintBreak(physx::PxConstraintInfo*, physx::PxU32) override {}
  void onWake(physx::PxActor**, physx::PxU32) override {}
  void onSleep(physx::PxActor**, physx::PxU32) override {}
  void onTrigger(physx::PxTriggerPair*, physx::PxU32) override {}
  void onAdvance(const physx::PxRigidBody* const*, const physx::PxTransform*, const physx::PxU32) override {}
};

struct Obj {
  uint32_t cls = 0;
  std::string name;
  bool done = false;
  PxBase* px = nullptr;
  PxScene* scene = nullptr;
  bool is_geom = false;
  std::unordered_map<uint32_t, std::vector<uint8_t>> pend;  // 만들 때 값
};

struct Stat {
  uint64_t compared = 0, bitdiff = 0;
  double maxdiff = 0;
  int64_t first_bit = -1, first_over = -1;
};

class Replayer {
 public:
  ovd::File& F;
  PxFoundation* fnd = nullptr;
  PxPhysics* phys = nullptr;
  PxDefaultCpuDispatcher* disp = nullptr;
  PxCookingParams* cook = nullptr;
  engine::FilterSpec spec;
  const engine::FilterSpec* specp = &spec;
  engine::OmniFilterCallback filter_cb;
  DiagContactCb diag_cb;
  std::unordered_map<uint64_t, engine::ConvexData> convex;
  std::vector<engine::SideView> sviews;
  std::vector<engine::SideCall> scalls;
  size_t scall_next = 0;
  std::unordered_map<uint64_t, Obj> objs;
  std::unordered_map<std::string, uint32_t> A;  // "Class.attr" -> handle
  // 핸들(메모리 주소)은 지웠다 다시 만든 객체에 재사용된다(09-29 Linux radio: 관절체 조인트 121 개 생성·121 개 삭제).
  // 그래서 대응표는 (핸들, 몇 번째로 만든 것인가) 로 적는다: gkey. 값도 같은 꼴(당시의 몇 번째).
  std::unordered_map<uint64_t, uint64_t> link_parent;  // gkey(link) -> gkey(parent link) (prescan)
  std::unordered_map<uint64_t, uint64_t> joint_child;  // gkey(art joint) -> gkey(child link)
  std::unordered_map<uint64_t, uint64_t> joint_parent; // gkey(art joint) -> gkey(parent link) (점검용)
  std::unordered_map<uint64_t, uint32_t> gen_run;      // 재생 중 핸들별로 지금 몇 번째 객체인가
  static uint64_t gkey(uint64_t h, uint32_t g) { return (uint64_t(g) << 56) | (h & 0x00FFFFFFFFFFFFFFull); }
  static uint64_t graw(uint64_t k) { return k & 0x00FFFFFFFFFFFFFFull; }
  uint64_t gnow(uint64_t h) { auto it = gen_run.find(h); return gkey(h, it == gen_run.end() ? 0 : it->second); }
  // D6 드라이브: OVD 는 드라이브 값을 조인트와 따로 된 값 객체(PxD6JointDrive)에 남긴다. 핸들 = 조인트 자료의 drive 배열 주소 + 종류
  //   (ExtD6Joint.cpp:76 omniPvdCreateDriveObjectHandle, :254 omniPvdSetDriveData 네 값 한 묶음, :512 값 먼저·연결 나중).
  // 연결(PxD6Joint.driveX..driveSwing2 = 핸들)과 값이 둘 다 모인 순간 setDrive 한 번.
  struct D6DriveRef { PxD6Joint* joint = nullptr; int type = -1; bool have = false; PxD6JointDrive v; };
  std::unordered_map<uint64_t, D6DriveRef> d6drive;
  std::map<std::string, uint64_t> unsupported;
  std::map<std::string, uint64_t> applied;
  std::map<std::string, Stat> stats;  // 항목별 비교
  uint64_t sims = 0;
  uint64_t side_offset = 0;  // --side-offset: 이 OVD 파일 앞의 simulate 수 (meta.post_step_count - 이 파일의 simulate 수)
  int64_t max_frames = -1;
  bool verbose = false;
  int verbose_max = 12;
  bool diag_no_self_collision = false;
  bool contact_report_all = false;
  std::set<size_t> side_skip;  // --side-skip i,j,... : 곁기록 번호(sidelog.npz 순서)를 건너뜀 (진단)
  bool dbg_drive = getenv("OVD_DBG_DRIVE") != nullptr;  // --contact-report-all: omni 접촉 보고 쌍 처리(알림 플래그 + 정적/키네마틱 쌍 풀이 끔)를 모든 쌍에  // --diag-no-self-collision: 모든 관절체 자기 충돌 끔 (원인 가르기 진단용, 결과 비교용 아님)
  FILE* csv = nullptr;
  double frame_max = 0;
  uint64_t frame_bitdiff = 0, frame_cmp = 0;
  int64_t first_div_frame = -1;
  std::string first_div_what;
  uint64_t convex_exact = 0, convex_approx = 0;
  std::unordered_map<std::string, PxArticulationReducedCoordinate*> art_by_name;
  std::unordered_map<std::string, PxRigidActor*> actor_by_name;
  std::unordered_map<PxArticulationReducedCoordinate*, PxArticulationCache*> caches;

  explicit Replayer(ovd::File& f) : F(f) {}

  uint32_t attr(const char* c, const char* a) {
    auto it = A.find(std::string(c) + "." + a);
    return it == A.end() ? 0xffffffffu : it->second;
  }
  const std::vector<uint8_t>* pv(Obj& o, const char* c, const char* a) {
    auto it = o.pend.find(attr(c, a));
    return it == o.pend.end() ? nullptr : &it->second;
  }
  template <class T> bool pget(Obj& o, const char* c, const char* a, T& out) {
    auto v = pv(o, c, a);
    if (!v || v->size() < sizeof(T)) return false;
    memcpy(&out, v->data(), sizeof(T));
    return true;
  }
  const char* cname(uint32_t c) { return F.classes[c].name.c_str(); }
  bool isa(const Obj& o, const char* c) { return F.is_a(o.cls, F.cls(c)); }

  std::string record_path;
  std::string trace_sub;  // --trace-obj: 이름(관절 조인트는 자식 링크 이름)에 이 글자가 든 객체의 set 을 simulate 전까지 모두 찍는다 (진단용)
  std::unordered_map<uint64_t, uint64_t> shape_owner;  // 진단: gkey(모양) -> gkey(붙은 액터)
  std::vector<std::pair<uint64_t, std::string>> trace_buf;  // (gkey, 한 줄) — 이름은 끝에 푼다 (이름은 만든 뒤 set 으로 온다)
  void trace_event(const ovd::Event& e, const char* tag) {
    if (trace_sub.empty() || sims > 2) return;
    const ovd::AttrInfo& ai = F.attrs[e.attr];
    std::string line = std::string("[추적 sim") + std::to_string(sims) + " " + tag + "] " + F.classes[ai.cls].name + "." + ai.name + " =";
    const uint8_t* d = F.data(e);
    char buf[32];
    if (e.data_len < 4) { for (uint32_t k = 0; k < e.data_len; ++k) { snprintf(buf, sizeof buf, " 0x%02x", d[k]); line += buf; } }
    for (uint32_t k = 0; k + 4 <= e.data_len && k < 64; k += 4) { float f; memcpy(&f, d + k, 4); snprintf(buf, sizeof buf, " %.9g", f); line += buf; }
    trace_buf.emplace_back(gnow(e.obj), line);
  }
  std::string gname_of(uint64_t k) {  // 지금도 같은 몇 번째 객체면 그 이름
    const uint64_t raw = graw(k);
    auto it = objs.find(raw);
    if (it == objs.end() || gnow(raw) != k) return "";
    return it->second.name;
  }
  void dump_state_before_first_sim() {  // --trace-obj 진단: 첫 simulate 직전 우리 PhysX 상태 (링크·조인트·모양 거르개)
    for (auto& kv : objs) {
      const Obj& o = kv.second;
      if (!o.px) continue;
      auto* l = o.px->is<PxArticulationLink>();
      if (!l) continue;
      const char* nm = l->getName();
      if (!nm || std::string(nm).find(trace_sub) == std::string::npos) continue;
      PxTransform g = l->getGlobalPose(), c = l->getCMassLocalPose();
      PxVec3 I = l->getMassSpaceInertiaTensor();
      fprintf(stderr, "[상태] 링크 %s 자세 q(%.9g %.9g %.9g %.9g) p(%.9g %.9g %.9g) 질량 %.9g 관성 (%.9g %.9g %.9g) 질량중심 p(%.9g %.9g %.9g) q(%.9g %.9g %.9g %.9g)\n",
              nm, g.q.x, g.q.y, g.q.z, g.q.w, g.p.x, g.p.y, g.p.z, l->getMass(), I.x, I.y, I.z, c.p.x, c.p.y, c.p.z, c.q.x, c.q.y, c.q.z, c.q.w);
      fprintf(stderr, "[상태]   최대침투속도 %.9g CFM %.9g 최대접촉충격 %.9g 선감쇠 %.9g 각감쇠 %.9g 최대선속 %.9g 최대각속 %.9g 몸체플래그 0x%x\n",
              l->getMaxDepenetrationVelocity(), l->getCfmScale(), l->getMaxContactImpulse(), l->getLinearDamping(), l->getAngularDamping(),
              l->getMaxLinearVelocity(), l->getMaxAngularVelocity(), uint32_t(PxU16(l->getRigidBodyFlags())));
      {
        PxAggregate* ag = l->getArticulation().getAggregate();
        fprintf(stderr, "[상태]   묶음(aggregate) %p 자기충돌 %d 액터 %u / 최대 %u\n", (void*)ag, ag ? int(ag->getSelfCollision()) : -1, ag ? ag->getNbActors() : 0u,
                ag ? ag->getMaxNbActors() : 0u);
      }
      fprintf(stderr, "[상태]   관절체 플래그 %u 링크 수 %u 반복 %u\n", uint32_t(PxU8(l->getArticulation().getArticulationFlags())), l->getArticulation().getNbLinks(), 0u);
      if (auto* j = l->getInboundJoint()) {
        PxTransform pp = j->getParentPose(), cp = j->getChildPose();
        fprintf(stderr, "[상태]   부모 링크 %s\n", j->getParentArticulationLink().getName() ? j->getParentArticulationLink().getName() : "(이름 없음)");
        fprintf(stderr, "[상태]   조인트 종류 %d 부모틀 q(%.9g %.9g %.9g %.9g) p(%.9g %.9g %.9g) 자식틀 q(%.9g %.9g %.9g %.9g) p(%.9g %.9g %.9g) 마찰 %.9g\n",
                int(j->getJointType()), pp.q.x, pp.q.y, pp.q.z, pp.q.w, pp.p.x, pp.p.y, pp.p.z, cp.q.x, cp.q.y, cp.q.z, cp.q.w, cp.p.x, cp.p.y, cp.p.z,
                j->getFrictionCoefficient());
        for (int i = 0; i < 6; ++i) {
          auto ax = PxArticulationAxis::Enum(i);
          if (j->getMotion(ax) == PxArticulationMotion::eLOCKED) continue;
          PxArticulationLimit L = j->getLimitParams(ax);
          PxArticulationDrive D = j->getDriveParams(ax);
          fprintf(stderr, "[상태]   축%d 운동 %d 위치 %.9g 속도 %.9g 한계[%.9g, %.9g] k %.9g c %.9g max %.9g 종류 %d 목표 %.9g 아머처 %.9g\n", i,
                  int(j->getMotion(ax)), j->getJointPosition(ax), j->getJointVelocity(ax), L.low, L.high, D.stiffness, D.damping, D.maxForce,
                  int(D.driveType), j->getDriveTarget(ax), j->getArmature(ax));
        }
      }
      PxShape* sh[16];
      PxU32 ns = l->getShapes(sh, 16);
      {  // 같은 관절체의 다른 링크 모양과 겹침(침투 깊이) — 자기 충돌 진단
        PxArticulationReducedCoordinate& art = l->getArticulation();
        std::vector<PxArticulationLink*> links(art.getNbLinks());
        art.getLinks(links.data(), PxU32(links.size()));
        for (PxU32 k = 0; k < ns; ++k) {
          const PxTransform pk = l->getGlobalPose() * sh[k]->getLocalPose();
          for (PxArticulationLink* o2 : links) {
            if (o2 == l) continue;
            PxShape* sh2[32];
            PxU32 n2 = o2->getShapes(sh2, 32);
            for (PxU32 m = 0; m < n2; ++m) {
              const PxTransform pm = o2->getGlobalPose() * sh2[m]->getLocalPose();
              PxVec3 dir;
              PxF32 depth = 0;
              if (PxGeometryQuery::computePenetration(dir, depth, sh[k]->getGeometry(), pk, sh2[m]->getGeometry(), pm))
                fprintf(stderr, "[상태]   겹침 모양 %u <-> %s 모양 %u 깊이 %.6g 방향 (%.3g %.3g %.3g) 접촉거리 %.4g/%.4g\n", k,
                        o2->getName() ? o2->getName() : "?", m, depth, dir.x, dir.y, dir.z, sh[k]->getContactOffset(), sh2[m]->getContactOffset());
            }
          }
        }
      }
      for (PxU32 k = 0; k < ns; ++k) {
        PxFilterData f = sh[k]->getSimulationFilterData();
        fprintf(stderr, "[상태]   모양 %u 종류 %d 거르개 %u %u %u %u 플래그 %u\n", k, int(sh[k]->getGeometry().getType()), f.word0, f.word1, f.word2,
                f.word3, uint32_t(sh[k]->getFlags()));
        const PxTransform lp = sh[k]->getLocalPose();
        fprintf(stderr, "[상태]     국소 q(%.9g %.9g %.9g %.9g) p(%.9g %.9g %.9g) 접촉거리 %.9g 쉼거리 %.9g\n", lp.q.x, lp.q.y, lp.q.z, lp.q.w, lp.p.x, lp.p.y,
                lp.p.z, sh[k]->getContactOffset(), sh[k]->getRestOffset());
        if (sh[k]->getGeometry().getType() == PxGeometryType::eCONVEXMESH) {
          const PxConvexMeshGeometry& g = static_cast<const PxConvexMeshGeometry&>(sh[k]->getGeometry());
          fprintf(stderr, "[상태]     척도 (%.9g %.9g %.9g) 척도회전 (%.9g %.9g %.9g %.9g) 꼭짓점 %u 면 %u 형상플래그 %u\n", g.scale.scale.x, g.scale.scale.y,
                  g.scale.scale.z, g.scale.rotation.x, g.scale.rotation.y, g.scale.rotation.z, g.scale.rotation.w, g.convexMesh->getNbVertices(),
                  g.convexMesh->getNbPolygons(), uint32_t(g.meshFlags));
        }
      }
    }
  }
  // --dump-art <관절체 이름> <파일>: 첫 simulate 직전 그 관절체의 PhysX 값을 텍스트로 (공식 쪽 텐서 뷰 덤프 state_pre<N>.npz 와 비교용).
  //   한 줄 = "키 번호 값..." (링크 번호 = PhysX 링크 순서, dof 번호 = 링크 순서 x 풀린 축 = 텐서 API 순서, 모양 번호 = 링크 순서 x 모양 순서)
  std::string dump_art_name, dump_art_file;
  int64_t dump_art_at = -1;  // 전체 simulate 번호 (곁기록·공식 --dump-at-pre 와 같은 셈). 기본: 이 파일 첫 simulate
  void dump_art() {
    PxArticulationReducedCoordinate* art = nullptr;
    auto ia = art_by_name.find(dump_art_name);
    if (ia != art_by_name.end()) art = ia->second;
    if (!art) { fprintf(stderr, "dump-art: 관절체 %s 없음\n", dump_art_name.c_str()); return; }
    FILE* f = fopen(dump_art_file.c_str(), "w");
    if (!f) return;
    PxU32 pi = 0, vi = 0;
    art->getSolverIterationCounts(pi, vi);
    fprintf(f, "art_iters 0 %u %u\n", pi, vi);
    fprintf(f, "art_sleep 0 %.9g %.9g %.9g\n", art->getSleepThreshold(), art->getStabilizationThreshold(), art->getWakeCounter());
    fprintf(f, "art_flags 0 %u\n", uint32_t(PxU8(art->getArticulationFlags())));
    std::vector<PxArticulationLink*> links(art->getNbLinks());
    art->getLinks(links.data(), PxU32(links.size()));
    // 관절값은 캐시로 읽는다 (텐서 API 와 같은 길; applyCache 직후 getJointPosition 은 다음 simulate 전까지 옛 값일 수 있다).
    // 캐시 dof 번호 = 링크 low-level 번호 순서로 관절 dof 수를 더한 것 (PxArticulationReducedCoordinate.h:230)
    PxArticulationCache* kc = art->createCache();
    art->copyInternalStateToCache(*kc, PxArticulationCacheFlag::eALL);
    std::vector<PxU32> dof_off(links.size() + 1, 0);
    {
      std::vector<PxU32> ndof(links.size(), 0);
      for (auto* l : links) ndof[l->getLinkIndex()] = l->getInboundJointDof();
      for (size_t k = 0; k < links.size(); ++k) dof_off[k + 1] = dof_off[k] + ndof[k];
    }
    fprintf(f, "root_cache 0 %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", kc->rootLinkData->transform.p.x, kc->rootLinkData->transform.p.y,
            kc->rootLinkData->transform.p.z, kc->rootLinkData->transform.q.x, kc->rootLinkData->transform.q.y, kc->rootLinkData->transform.q.z,
            kc->rootLinkData->transform.q.w);
    int dof = 0, shp = 0;
    for (size_t i = 0; i < links.size(); ++i) {
      PxArticulationLink* l = links[i];
      const PxTransform c = l->getCMassLocalPose(), g = l->getGlobalPose();
      const PxVec3 I = l->getMassSpaceInertiaTensor();
      fprintf(f, "link_name %zu %s\n", i, l->getName() ? l->getName() : "?");
      fprintf(f, "mass %zu %.9g\n", i, l->getMass());
      fprintf(f, "com %zu %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", i, c.p.x, c.p.y, c.p.z, c.q.x, c.q.y, c.q.z, c.q.w);
      fprintf(f, "inertia_diag %zu %.9g %.9g %.9g\n", i, I.x, I.y, I.z);
      fprintf(f, "pose %zu %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", i, g.p.x, g.p.y, g.p.z, g.q.x, g.q.y, g.q.z, g.q.w);
      fprintf(f, "body %zu %.9g %.9g %.9g %.9g %.9g %.9g 0x%x\n", i, l->getLinearDamping(), l->getAngularDamping(), l->getMaxLinearVelocity(),
              l->getMaxAngularVelocity(), l->getMaxDepenetrationVelocity(), l->getCfmScale(), uint32_t(PxU16(l->getRigidBodyFlags())));
      fprintf(f, "actor_flags %zu 0x%x %u\n", i, uint32_t(PxU8(l->getActorFlags())), uint32_t(l->getDominanceGroup()));
      if (auto* j = l->getInboundJoint()) {
        fprintf(f, "joint %zu type %d friction %.9g maxvel %.9g\n", i, int(j->getJointType()), j->getFrictionCoefficient(), j->getMaxJointVelocity());
        int kd = 0;  // 이 관절 안에서 몇 번째 풀린 축
        for (int a = 0; a < 6; ++a) {
          auto ax = PxArticulationAxis::Enum(a);
          if (j->getMotion(ax) == PxArticulationMotion::eLOCKED) continue;
          const PxU32 ci = dof_off[l->getLinkIndex()] + kd++;
          const float cpos = kc->jointPosition[ci], cvel = kc->jointVelocity[ci];
          const PxArticulationLimit L = j->getLimitParams(ax);
          const PxArticulationDrive D = j->getDriveParams(ax);
          const PxJointFrictionParams FP = j->getFrictionParams(ax);
          fprintf(f, "dof %d link %zu axis %d motion %d lim %.9g %.9g k %.9g c %.9g maxf %.9g dtype %d env %.9g %.9g %.9g %.9g arm %.9g maxdofvel %.9g fr %.9g %.9g %.9g pos %.9g vel %.9g tgt %.9g tvel %.9g\n",
                  dof, i, a, int(j->getMotion(ax)), L.low, L.high, D.stiffness, D.damping, D.maxForce, int(D.driveType), D.envelope.maxEffort,
                  D.envelope.maxActuatorVelocity, D.envelope.velocityDependentResistance, D.envelope.speedEffortGradient, j->getArmature(ax),
                  j->getMaxJointVelocity(ax), FP.staticFrictionEffort, FP.dynamicFrictionEffort, FP.viscousFrictionCoefficient, cpos,
                  cvel, j->getDriveTarget(ax), j->getDriveVelocity(ax));
          dof++;
        }
      }
      PxShape* sh[64];
      const PxU32 ns = l->getShapes(sh, 64);
      for (PxU32 k = 0; k < ns; ++k) {
        PxMaterial* m = nullptr;
        sh[k]->getMaterials(&m, 1);
        const PxTransform lp = sh[k]->getLocalPose();
        fprintf(f, "shape %d link %zu off %.9g %.9g torsion %.9g %.9g flags 0x%x lp %.9g %.9g %.9g %.9g %.9g %.9g %.9g", shp, i, sh[k]->getContactOffset(),
                sh[k]->getRestOffset(), sh[k]->getTorsionalPatchRadius(), sh[k]->getMinTorsionalPatchRadius(), uint32_t(PxU8(sh[k]->getFlags())), lp.p.x,
                lp.p.y, lp.p.z, lp.q.x, lp.q.y, lp.q.z, lp.q.w);
        if (m)
          fprintf(f, " mat %.9g %.9g %.9g comb %d %d flags 0x%x damp %.9g", m->getStaticFriction(), m->getDynamicFriction(), m->getRestitution(),
                  int(m->getFrictionCombineMode()), int(m->getRestitutionCombineMode()), uint32_t(PxU16(m->getFlags())), m->getDamping());
        fputc('\n', f);
        shp++;
      }
    }
    kc->release();
    fclose(f);
    printf("dump-art: %s -> %s (링크 %zu, dof %d, 모양 %d)\n", dump_art_name.c_str(), dump_art_file.c_str(), links.size(), dof, shp);
  }
  // ------------------------------------------------------------------ S2 닫힌 고리 (문서 15절): --ctrl <s1 폴더>
  // 에피소드 시작 뒤로는 로봇 드라이브 목표를 OVD 기록 대신 omni 제어기(core/omni/controllers.h)가 "우리 PhysX 상태"로 계산해 넣는다.
  // 입력: s1_setup.txt·s1_actions.bin (capture/export_s1.py). 공식처럼 스텝 첫 서브스텝에 apply_action(바닥·뿌리 링크 자세),
  // 서브스텝마다 step(관절 위치) -> setDriveTarget / setDriveVelocity (텐서 API 가 부르는 것과 같은 setter, 늘 부름).
  struct CtrlSetup {
    uint64_t episode_start = 0;
    uint32_t substeps = 4, T = 0, A = 0;
    std::string base_link, root_link;
    int n_dof = 0;
    std::vector<std::string> dof_link;
    std::vector<float> sign, acts;
    eng::omni::ctrl::R1ProConfig cfg{};
    eng::omni::ctrl::R1ProState st{};
    std::vector<PxArticulationJointReducedCoordinate*> joint;
    std::vector<int> axis;
    std::unordered_set<const void*> robot_joints;
    std::unordered_map<const void*, std::array<std::vector<uint8_t>, 2>> rec;  // 조인트 -> 이번 서브스텝 공식 (위치, 속도) 목표 6 칸
    uint64_t cmp_n = 0, cmp_bad = 0;
    int64_t first_bad_step = -1;
    std::string first_bad;
    int64_t last_t = -1;
    bool on = false;
    uint64_t suppressed = 0, applied_steps = 0;
    // 잡기 보조(AG) 조인트가 이 서브스텝 직전에 생기거나 없어지면 OmniGibson 은 pre-physics 안에서 update_handles 를 부르고,
    // ControllableObjectViewAPI 가 새로 만들어지며 쓰기 캐시가 비어 그 서브스텝의 flush_control 이 아무것도 안 쓴다
    // (simulator.py:1579 step_all → robot.post_step(AG) → :1587 flush_control, robot.py:2060 _release_grasp update_handles).
    // 제어기 상태는 그대로 나아가고 드라이브 목표만 앞 서브스텝 값이 남는다.
    bool joint_event = false;
    // --free: 에피소드 시작 뒤로는 OVD 입력(set·create·destroy·목록)을 넣지 않고 제어기(--ctrl) 입력만으로 돈다 (S4 준비:
    // 엔진이 에피소드 동안 스스로 만들어야 하는 입력이 무엇인지 센다). 출력 비교는 그대로.
    bool free_run = false;
    std::set<uint64_t> skip_objs_dummy;
    std::map<std::string, uint64_t> free_skipped;
    uint64_t dropped_flush = 0;
  } C;

  bool load_ctrl(const std::string& dir) {
    std::ifstream f(dir + "/s1_setup.txt");
    if (!f) return false;
    std::map<std::string, std::vector<int>> groups;
    float bl[12] = {};
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream is(line);
      std::string k;
      is >> k;
      if (k == "episode_start_post") is >> C.episode_start;
      else if (k == "substeps") is >> C.substeps;
      else if (k == "base_link") is >> C.base_link;
      else if (k == "root_link") is >> C.root_link;
      else if (k == "n_dof") { is >> C.n_dof; C.dof_link.resize(C.n_dof); C.sign.resize(C.n_dof); }
      else if (k == "group") { std::string g; int d; is >> g; while (is >> d) groups[g].push_back(d); }
      else if (k == "base_limits") { for (float& x : bl) { std::string t; is >> t; x = strtof(t.c_str(), nullptr); } }
      else if (k == "dof") {
        int d, sg, has;
        std::string link, lo, hi, vlo, vhi;
        is >> d >> link >> sg >> lo >> hi >> vlo >> vhi >> has;
        C.dof_link[d] = link;
        C.sign[d] = float(sg);
        C.cfg.pos_lo[d] = strtof(lo.c_str(), nullptr);
        C.cfg.pos_hi[d] = strtof(hi.c_str(), nullptr);
        C.cfg.vel_lo[d] = strtof(vlo.c_str(), nullptr);
        C.cfg.vel_hi[d] = strtof(vhi.c_str(), nullptr);
        C.cfg.has_limit[d] = uint8_t(has);
      }
    }
    auto& c = C.cfg;
    c.n_dof = C.n_dof;
    auto cp = [&](const char* g, int* dst, int n) { auto& v = groups[g]; for (int i = 0; i < n && i < int(v.size()); ++i) dst[i] = v[i]; };
    cp("base", c.base_dof, 3); cp("trunk", c.trunk_dof, 4); cp("arm_left", c.arm_dof[0], 7); cp("arm_right", c.arm_dof[1], 7);
    cp("gripper_left", c.grip_dof[0], 2); cp("gripper_right", c.grip_dof[1], 2);
    for (int k = 0; k < 3; ++k) { c.base_in_lo[k] = bl[k]; c.base_in_hi[k] = bl[3 + k]; c.base_out_lo[k] = bl[6 + k]; c.base_out_hi[k] = bl[9 + k]; }
    eng::omni::ctrl::init(c);
    eng::omni::ctrl::reset(C.st);
    FILE* fa = fopen((dir + "/s1_actions.bin").c_str(), "rb");
    if (!fa || fread(&C.T, 4, 1, fa) != 1 || fread(&C.A, 4, 1, fa) != 1) return false;
    C.acts.resize(size_t(C.T) * C.A);
    if (fread(C.acts.data(), 4, C.acts.size(), fa) != C.acts.size()) return false;
    fclose(fa);
    C.on = true;
    return true;
  }
  bool ctrl_bind() {  // dof -> 조인트·축 (자식 링크 이름으로), 한 번
    if (!C.joint.empty()) return true;
    C.joint.assign(C.n_dof, nullptr);
    C.axis.assign(C.n_dof, 0);
    for (int d = 0; d < C.n_dof; ++d) {
      auto il = actor_by_name.find(C.dof_link[d]);
      if (il == actor_by_name.end()) { C.joint.clear(); return false; }
      auto* l = il->second->is<PxArticulationLink>();
      auto* j = l ? l->getInboundJoint() : nullptr;
      if (!j) { C.joint.clear(); return false; }
      C.joint[d] = j;
      for (int a = 0; a < 6; ++a) if (j->getMotion(PxArticulationAxis::Enum(a)) != PxArticulationMotion::eLOCKED) { C.axis[d] = a; break; }
      C.robot_joints.insert(j);
    }
    return true;
  }
  // OVD 의 로봇 드라이브 목표 쓰기를 건너뛸지 (에피소드 시작 뒤, 닫힌 고리가 대신 넣는다)
  bool ctrl_suppress(PxBase* b, const std::string& an) {
    if (!C.on || (an != "driveTarget" && an != "driveVelocity")) return false;
    if (sims + side_offset + 1 <= C.episode_start) return false;
    if (!ctrl_bind() || !C.robot_joints.count(b)) return false;
    C.suppressed++;
    C.rec[b][an == "driveTarget" ? 0 : 1] = cur_set_data;  // 공식 값 (비교용)
    return true;
  }
  // simulate 직전: 이번 서브스텝 목표를 계산해 넣는다
  // ------------------------------------------------------------------ S3 판정 (문서 15절): --s3 <폴더> (capture/export_s3.py)
  // 스텝 끝(마지막 서브스텝 뒤)마다 OmniGibson ToggledOn 갱신을 core/omni/states.h 로 한다: 접촉 단계(공식 접촉 행렬 입력, v0),
  // 표식-손가락 겹침(우리 PhysX 링크 자세 → pose_to_mat), 누적 시간·뒤집기. 그 값으로 core/omni/bddl.h 가 목표를 판정해
  // 공식 ToggledOn 값·시간(toggle 기록)과 trace 의 goal_satisfied 에 스텝마다 비트 비교한다.
  struct S3Setup {
    bool on = false;
    int S = 0, O = 0, nr = 0, nc = 0;
    std::vector<std::string> obj_bddl;
    std::vector<std::string> mk_parent;
    std::vector<float> mk_off, mk_rad;
    std::vector<std::string> fg_path;
    std::vector<std::vector<float>> fg_pts;
    std::vector<std::vector<int32_t>> fg_tri;
    std::vector<std::pair<int, int>> pairs;
    std::vector<std::vector<uint8_t>> with;  // 물체마다 (nc,)
    std::vector<uint8_t> qrow;
    uint64_t episode_start = 0;
    int substeps = 4, steps = 0;
    struct Row {
      uint64_t post;
      std::vector<uint8_t> val;
      std::vector<float> time;
      std::vector<uint8_t> cm, ccm;
      bool has_cm;
    };
    std::vector<Row> rows;
    std::vector<std::string> goals;
    eng::omni::bddl::Compiled G;
    std::vector<int> atom_obj;  // 원자 -> 물체 칸 (toggled_on 만), -1 = 모름
    std::vector<uint8_t> val;
    std::vector<float> time;
    bool init = false;
    uint64_t n_steps = 0, bad_val = 0, bad_time = 0, bad_goal = 0, n_goal_true = 0;
    int64_t first_bad = -1;
    std::string first_bad_what;
    std::string last_goal = "[]";
    // v1: 접촉 행렬을 우리 PhysX 로 (--s3-own-contact)
    bool own = false;
    std::vector<std::string> rowpath, colpath;
    std::vector<const PxRigidActor*> body;  // 행 몸체(0..nr-1) 뒤에 열 몸체
    std::vector<int32_t> row_to_rigid, col_to_rigid, b2r;
    int R_ext = 0;
    std::vector<float> pend_tf, pend_net, pend_imp, prev_tf;
    int n_pend = 0;
    std::vector<uint8_t> cm, ccm;
    uint64_t cm_steps = 0, cm_bad = 0;
    int64_t cm_first_bad = -1;
  } S3;
  S3ContactCb s3_cb;

  static float hexf(const std::string& h) {
    uint32_t u = uint32_t(strtoul(h.c_str(), nullptr, 16));
    float f;
    memcpy(&f, &u, 4);
    return f;
  }

  bool load_s3(const std::string& dir) {
    std::ifstream f(dir + "/s3_setup.txt");
    if (!f) return false;
    std::string line, bddl_path;
    std::vector<std::pair<int, int>> fsz;
    while (std::getline(f, line)) {
      std::istringstream is(line);
      std::string k;
      is >> k;
      if (k == "so") {
        is >> S3.S >> S3.O;
        S3.obj_bddl.assign(S3.S * S3.O, "-");
        S3.with.assign(S3.S * S3.O, {});
      } else if (k == "obj") {
        int i;
        std::string n, b;
        is >> i >> n >> b;
        S3.obj_bddl[i] = b;
      } else if (k == "marker") {
        int i;
        std::string par, a, b, c, r;
        is >> i >> par >> a >> b >> c >> r;
        if (int(S3.mk_parent.size()) <= i) {
          S3.mk_parent.resize(i + 1);
          S3.mk_off.resize((i + 1) * 3);
          S3.mk_rad.resize(i + 1);
        }
        S3.mk_parent[i] = par;
        S3.mk_off[i * 3] = hexf(a);
        S3.mk_off[i * 3 + 1] = hexf(b);
        S3.mk_off[i * 3 + 2] = hexf(c);
        S3.mk_rad[i] = hexf(r);
      } else if (k == "finger") {
        int i, np_, nt;
        std::string pth;
        is >> i >> pth >> np_ >> nt;
        S3.fg_path.push_back(pth);
        fsz.push_back({np_, nt});
      } else if (k == "pair") {
        int a, b;
        is >> a >> b;
        S3.pairs.push_back({a, b});
      } else if (k == "rowcol") {
        is >> S3.nr >> S3.nc;
      } else if (k == "with") {
        int i, c;
        is >> i;
        S3.with[i].assign(S3.nc, 0);
        while (is >> c) S3.with[i][c] = 1;
      } else if (k == "qrow") {
        S3.qrow.assign(S3.nr, 0);
        int r;
        while (is >> r) S3.qrow[r] = 1;
      } else if (k == "rowpath" || k == "colpath") {
        int i;
        std::string pth;
        is >> i >> pth;
        auto& v = k == "rowpath" ? S3.rowpath : S3.colpath;
        if (int(v.size()) <= i) v.resize(i + 1);
        v[i] = pth;
      } else if (k == "episode_start") {
        std::string t;
        is >> S3.episode_start >> t >> S3.substeps >> t >> S3.steps >> t >> bddl_path;
      }
    }
    FILE* fm = fopen((dir + "/s3_meshes.bin").c_str(), "rb");
    if (!fm) return false;
    for (auto& z : fsz) {
      std::vector<float> p(size_t(z.first) * 3);
      std::vector<int32_t> t(size_t(z.second) * 3);
      if (fread(p.data(), 4, p.size(), fm) != p.size() || fread(t.data(), 4, t.size(), fm) != t.size()) {
        fclose(fm);
        return false;
      }
      S3.fg_pts.push_back(std::move(p));
      S3.fg_tri.push_back(std::move(t));
    }
    fclose(fm);
    FILE* fr = fopen((dir + "/s3_rows.bin").c_str(), "rb");
    uint32_t n = 0;
    if (!fr || fread(&n, 4, 1, fr) != 1) return false;
    const int SO = S3.S * S3.O, RC = S3.nr * S3.nc;
    for (uint32_t i = 0; i < n; ++i) {
      S3Setup::Row r;
      r.val.resize(SO);
      r.time.resize(SO);
      r.cm.resize(RC);
      std::vector<uint8_t> ccm(RC);
      uint8_t has = 0;
      bool ok = fread(&r.post, 8, 1, fr) == 1 && fread(r.val.data(), 1, SO, fr) == size_t(SO) &&
                fread(r.time.data(), 4, SO, fr) == size_t(SO) && fread(r.cm.data(), 1, RC, fr) == size_t(RC) &&
                fread(ccm.data(), 1, RC, fr) == size_t(RC) && fread(&has, 1, 1, fr) == 1;
      if (!ok) {
        fclose(fr);
        return false;
      }
      r.has_cm = has != 0;
      r.ccm = ccm;
      S3.rows.push_back(std::move(r));
    }
    fclose(fr);
    std::ifstream fg(dir + "/s3_goal.txt");
    while (std::getline(fg, line)) S3.goals.push_back(line);
    std::ifstream fb(bddl_path);
    if (!fb) {
      fprintf(stderr, "S3: BDDL 을 못 읽음 %s\n", bddl_path.c_str());
      return false;
    }
    std::stringstream ss;
    ss << fb.rdbuf();
    S3.G = eng::omni::bddl::compile_problem(eng::omni::bddl::parse_problem(ss.str()));
    for (auto& at : S3.G.atoms) {
      int o = -1;
      if (at.pred == "toggled_on" && at.args.size() == 1)
        for (int k = 0; k < SO; ++k)
          if (S3.obj_bddl[k] == at.args[0]) o = k;
      S3.atom_obj.push_back(o);
    }
    S3.on = true;
    return true;
  }

  const S3Setup::Row* s3_row_at(uint64_t post) const {  // post 이하 마지막 기록 (기록은 바뀔 때만 남음)
    const S3Setup::Row* r = nullptr;
    for (auto& x : S3.rows) {
      if (x.post <= post) r = &x;
      else break;
    }
    return r;
  }
  bool s3_mat(const std::string& path, eng::omni::wf::M44& M) {
    auto it = actor_by_name.find(path);
    if (it == actor_by_name.end()) return false;
    const PxTransform t = it->second->getGlobalPose();
    const float p7[7] = {t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w};
    M = eng::omni::wf::pose_to_mat(p7);
    return true;
  }
  bool s3_bind_bodies() {
    if (!S3.body.empty()) return true;
    const PxRigidActor* none = nullptr;
    for (auto& p : S3.rowpath) {
      auto it = actor_by_name.find(p);
      if (it == actor_by_name.end()) { fprintf(stderr, "S3: 행 몸체 없음 %s\n", p.c_str()); return false; }
      S3.body.push_back(it->second);
    }
    S3.row_to_rigid.clear();
    for (int r = 0; r < S3.nr; ++r) S3.row_to_rigid.push_back(r);
    S3.b2r.assign(S3.nr, 0);
    for (int r = 0; r < S3.nr; ++r) S3.b2r[r] = r;
    S3.R_ext = S3.nr;
    for (auto& p : S3.colpath) {
      auto it = actor_by_name.find(p);
      const PxRigidActor* a = it == actor_by_name.end() ? none : it->second;
      const PxRigidDynamic* d = a ? a->is<PxRigidDynamic>() : nullptr;
      const bool dyn = a && (a->is<PxArticulationLink>() || (d && !(d->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC)));
      if (!dyn) { S3.col_to_rigid.push_back(-1); continue; }  // 키네마틱·정적 = 열에 몸체 없음 (_CONTACT_MATRIX_COLS_HAS_RIGID_BODY)
      int b = -1;
      for (size_t k = 0; k < S3.body.size(); ++k) if (S3.body[k] == a) b = int(k);
      if (b < 0) { b = int(S3.body.size()); S3.body.push_back(a); S3.b2r.push_back(S3.R_ext++); }
      S3.col_to_rigid.push_back(b);
    }
    return true;
  }
  void s3_collect_substep() {  // 방금 끝난 서브스텝의 몸체 자세·알짜 충격·쌍 충격 (read_from_physx 자리)
    const int B = int(S3.body.size()), C = S3.nc;
    const float dt = last_dt;
    for (int b = 0; b < B; ++b) {
      const PxTransform t = S3.body[b]->getGlobalPose();
      const float v[7] = {t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w};
      S3.pend_tf.insert(S3.pend_tf.end(), v, v + 7);
    }
    std::vector<float> net(size_t(S3.R_ext) * 3, 0.0f), imp(size_t(S3.R_ext) * C * 3, 0.0f);
    for (int b = 0; b < B; ++b) {
      const int r = S3.b2r[b];
      if (r < 0) continue;
      auto it = s3_cb.net.find(S3.body[b]);
      if (it != s3_cb.net.end()) { net[r * 3] = it->second.x / dt; net[r * 3 + 1] = it->second.y / dt; net[r * 3 + 2] = it->second.z / dt; }
    }
    for (int r = 0; r < S3.nr; ++r)
      for (int c = 0; c < C; ++c) {
        auto ic = actor_by_name.find(S3.colpath[c]);
        if (ic == actor_by_name.end()) continue;
        const void* a = S3.body[r];
        const void* b = ic->second;
        auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        auto ip = s3_cb.pair_imp.find(key);
        if (ip == s3_cb.pair_imp.end()) continue;
        float* v = &imp[(size_t(r) * C + c) * 3];
        v[0] = ip->second.x / dt; v[1] = ip->second.y / dt; v[2] = ip->second.z / dt;
      }
    if (getenv("S3_DEBUG_PAIR")) {  // 진단: 손가락 행 × 열 칸의 서브스텝별 힘 크기
      const int64_t st = int64_t((sims + side_offset - S3.episode_start - 1) / S3.substeps);
      const int64_t lo = atoll(getenv("S3_DEBUG_PAIR")), hi = lo + 30;
      if (st >= lo && st <= hi)
        for (int r = 0; r < S3.nr; ++r)
          for (int c = 0; c < C; ++c) {
            const float* v = &imp[(size_t(r) * C + c) * 3];
            if (v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f)
              fprintf(stderr, "[S3 힘] 스텝 %lld sim %llu 행 %d 열 %d F (%.6g %.6g %.6g) net_r (%.6g %.6g %.6g) dt %.6g\n", (long long)st,
                      (unsigned long long)(sims + side_offset), r, c, v[0], v[1], v[2], net[r * 3], net[r * 3 + 1], net[r * 3 + 2], double(last_dt));
          }
    }
    S3.pend_net.insert(S3.pend_net.end(), net.begin(), net.end());
    S3.pend_imp.insert(S3.pend_imp.end(), imp.begin(), imp.end());
    S3.n_pend++;
  }
  void s3_update_contact() {  // RigidContactAPI.update (usd_utils.py:1511) 의 부분 행렬
    namespace st = eng::omni::st;
    const int B = int(S3.body.size());
    st::ContactIn in{S3.pend_tf.data(), S3.prev_tf.data(), S3.pend_net.data(), S3.b2r.data(), B, S3.R_ext, 1e-6f, 1e-4f};
    for (int r = 0; r < S3.nr; ++r)
      for (int c = 0; c < S3.nc; ++c)
        st::contact_update_rc(in, S3.pend_imp.data(), S3.nc, S3.row_to_rigid.data(), S3.col_to_rigid.data(), S3.n_pend, r, c,
                              S3.cm.data(), S3.ccm.data());
    std::vector<float> nt = S3.prev_tf;
    for (int b = 0; b < B; ++b) st::body_transform_update(in, S3.n_pend, b, nt.data());
    S3.prev_tf = nt;
    S3.pend_tf.clear(); S3.pend_net.clear(); S3.pend_imp.clear(); S3.n_pend = 0;
  }
  void s3_after_simulate() {
    if (!S3.on) return;
    const uint64_t g = sims + side_offset;  // 방금 끝난 simulate 의 전체 번호 = post 번호
    if (S3.own && S3.init && g > S3.episode_start) s3_collect_substep();
    if (AG.on && RC.init && g > S3.episode_start) {
      rca_collect();
      if ((g - S3.episode_start) % S3.substeps == 0) rca_update();
    }
    if (S3.own) s3_cb.clear();
    if (g < S3.episode_start || (g - S3.episode_start) % S3.substeps != 0) return;
    const int SO = S3.S * S3.O;
    const S3Setup::Row* off = s3_row_at(g);
    if (!off) return;
    if (!S3.init) {  // 에피소드 시작 시점 공식 값·시간에서 출발
      S3.val = off->val;
      S3.time = off->time;
      if (S3.own) {
        if (!s3_bind_bodies()) { S3.own = false; } else {
          S3.cm = off->cm;
          S3.ccm = off->ccm;
          S3.prev_tf.clear();
          for (auto* a : S3.body) {
            const PxTransform t = a->getGlobalPose();
            const float v[7] = {t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w};
            S3.prev_tf.insert(S3.prev_tf.end(), v, v + 7);
          }
        }
      }
      S3.init = true;
      return;
    }
    const uint8_t* cm_in = off->cm.data();
    if (S3.own) {
      s3_update_contact();
      cm_in = S3.cm.data();
      if (AG.on && RC.init && RC.rows.size() == size_t(S3.nr)) {  // 교차 확인: AG 용 접촉 행렬(모든 열)의 판정 열 부분 = S3 v1 값
        for (int r = 0; r < S3.nr; ++r)
          for (int c = 0; c < S3.nc; ++c) {
            auto ia = actor_by_name.find(S3.colpath[c]);
            if (ia == actor_by_name.end()) continue;
            auto ic = RC.col_of.find(ia->second);
            if (ic == RC.col_of.end()) continue;
            if (RC.cm[size_t(r) * RC.cols.size() + ic->second] != S3.cm[size_t(r) * S3.nc + c]) RC_bad++;
          }
        RC_checked++;
      }
      S3.cm_steps++;
      if (off->has_cm && memcmp(S3.cm.data(), off->cm.data(), S3.cm.size())) {
        S3.cm_bad++;
        if (S3.cm_first_bad < 0) S3.cm_first_bad = int64_t((g - S3.episode_start) / S3.substeps) - 1;
        if (S3.cm_bad <= 10)
          for (size_t k = 0; k < S3.cm.size(); ++k)
            if (S3.cm[k] != off->cm[k])
              fprintf(stderr, "[S3 접촉] 스텝 %lld 행 %s 열 %s: 우리 %d 공식 %d (현재 우리 %d 공식 %d)\n",
                      (long long)((g - S3.episode_start) / S3.substeps) - 1, S3.rowpath[k / S3.nc].c_str(), S3.colpath[k % S3.nc].c_str(),
                      S3.cm[k], off->cm[k], S3.ccm[k], off->ccm[k]);
      }
    }
    const int64_t t = int64_t((g - S3.episode_start) / S3.substeps) - 1;  // 방금 끝난 평가 스텝
    namespace st = eng::omni::st;
    std::vector<int32_t> mask(SO, 0);
    for (int k = 0; k < SO; ++k)
      mask[k] = (S3.own || off->has_cm) ? st::toggle_contact(S3.qrow.data(), cm_in, S3.nr, S3.nc, S3.with[k].data()) : 0;
    for (auto& pr : S3.pairs) {
      const int k = pr.first, f = pr.second;
      if (mask[k] != 1 || S3.fg_tri[f].empty()) continue;
      eng::omni::wf::M44 Mp, Mf;
      if (!s3_mat(S3.mk_parent[k], Mp) || !s3_mat(S3.fg_path[f], Mf)) continue;
      if (st::toggle_marker_overlap(Mp, &S3.mk_off[k * 3], S3.mk_rad[k], Mf, S3.fg_pts[f].data(), S3.fg_tri[f].data(),
                                    int(S3.fg_tri[f].size() / 3)))
        mask[k] = 2;
    }
    for (int k = 0; k < SO; ++k) st::toggle_set_value(&S3.val[k], &mask[k], &S3.time[k], 0.15f, float(1.0 / 30.0));
    std::vector<uint8_t> atom(S3.G.atoms.size(), 0), node(S3.G.nodes.size(), 0), head(S3.G.heads.size(), 0);
    for (size_t a = 0; a < atom.size(); ++a) atom[a] = S3.atom_obj[a] >= 0 ? S3.val[S3.atom_obj[a]] : 0;
    eng::omni::bddl::eval_goal(S3.G.nodes.data(), int(S3.G.nodes.size()), S3.G.kids.data(), S3.G.heads.data(),
                               int(S3.G.heads.size()), atom.data(), node.data(), head.data());
    std::string gs = "[";
    bool first = true;
    for (size_t h = 0; h < head.size(); ++h)
      if (head[h]) {
        gs += (first ? "" : ", ") + std::to_string(h);
        first = false;
      }
    gs += "]";
    S3.last_goal = gs;
    if (gs != "[]") S3.n_goal_true++;
    S3.n_steps++;
    auto bad = [&](const std::string& w) {
      if (S3.first_bad < 0) {
        S3.first_bad = t;
        S3.first_bad_what = w;
      }
    };
    // 공식 기록은 다음 simulate 직전에 적힌다 -> 에피소드 마지막 스텝 뒤 값은 기록이 없다 (그 스텝은 목표만 비교)
    const bool have_off = off->post == g || &off[0] != &S3.rows.back();
    for (int k = 0; k < SO && have_off; ++k) {
      if (S3.val[k] != off->val[k]) {
        S3.bad_val++;
        bad("value " + std::to_string(k));
      }
      uint32_t a, b;
      memcpy(&a, &S3.time[k], 4);
      memcpy(&b, &off->time[k], 4);
      if (a != b) {
        S3.bad_time++;
        bad("time " + std::to_string(k));
      }
    }
    if (t >= 0 && t < int64_t(S3.goals.size()) && gs != S3.goals[t]) {
      S3.bad_goal++;
      bad("goal " + gs + " vs " + S3.goals[t]);
    }
  }
  void s3_report() {
    if (AG.on) printf("AG 접촉 행렬 교차 확인: 스텝 %" PRIu64 ", S3 v1 과 다른 칸 %" PRIu64 "\n", RC_checked, RC_bad);
    if (AG.on) printf("AG(--ag): 시도 %" PRIu64 ", 잡음 %" PRIu64 ", 놓음 %" PRIu64 ", 무거워서 안 잡음 %" PRIu64 ", 접촉점 없어 안 잡음 %" PRIu64 "\n", AG.n_try, AG.n_grasp, AG.n_release, AG.n_skip_type, AG.n_skip_contact);
    if (!S3.on) return;
    if (S3.own) printf("S3 접촉 행렬(--s3-own-contact, 우리 PhysX 접촉 보고): 평가 스텝 %" PRIu64 ", 공식과 다른 스텝 %" PRIu64 ", 첫 다름 %lld\n", S3.cm_steps, S3.cm_bad, (long long)S3.cm_first_bad);
    const std::string fb = S3.first_bad < 0 ? std::string("없음") : "스텝 " + std::to_string(S3.first_bad) + " " + S3.first_bad_what;
    printf("S3 판정(--s3): 평가 스텝 %" PRIu64 " (목표 참 %" PRIu64 "), ToggledOn 값 다름 %" PRIu64 ", 시간 다름 %" PRIu64
           ", 목표 다름 %" PRIu64 ", 첫 다름 %s\n",
           S3.n_steps, S3.n_goal_true, S3.bad_val, S3.bad_time, S3.bad_goal, fb.c_str());
  }

  // ------------------------------------------------------------------ 보조 잡기(AG) 관절 만들기 (robot.py:3497 _maybe_establish_grasp ~ :3570)
  // 단계 A(09-30): 언제·무엇을 잡는지는 옵션으로 주고(공식 기록에서), 관절 틀·생성 호출만 엔진이 한다 -> 물리 비트 비교로 틀·호출 순서 확인.
  struct AGForce {
    int64_t post = -1;  // 이 번호의 simulate 직전 서브스텝에서 (= 곁기록 post 번호 + 1 의 simulate)
    std::string eef, target;
    std::vector<std::string> fingers;
    float scale1[3] = {1.0f, 1.0f, 1.0f};
    bool done = false;
    PxD6Joint* joint = nullptr;
    PxTransform lp0, lp1;
  } AGF;
  bool ag_contact_point(const PxRigidActor* target, PxVec3& out) {  // _find_finger_contact_position (손가락 순서대로 첫 점)
    for (auto& fp : AGF.fingers) {
      auto it = actor_by_name.find(fp);
      if (it == actor_by_name.end()) continue;
      const void* a = target;
      const void* b = it->second;
      auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
      auto ip = s3_cb.last_first_pt.find(key);
      if (ip != s3_cb.last_first_pt.end()) { out = ip->second; return true; }
    }
    return false;
  }
  void ag_create(const PxRigidActor* eef, const PxRigidActor* target, const PxVec3& contact) {
    namespace agf = eng::omni::agf;
    const PxTransform te = eef->getGlobalPose(), tt = target->getGlobalPose();
    const float c[3] = {contact.x, contact.y, contact.z};
    const float ep[3] = {te.p.x, te.p.y, te.p.z}, eq[4] = {te.q.x, te.q.y, te.q.z, te.q.w};
    const float tp[3] = {tt.p.x, tt.p.y, tt.p.z}, tq[4] = {tt.q.x, tt.q.y, tt.q.z, tt.q.w};
    const float one[3] = {1.0f, 1.0f, 1.0f};
    float p0[3], q0[4], p1[3], q1[4];
    agf::grasp_frame(c, ep, eq, one, p0, q0);
    agf::grasp_frame(c, tp, tq, AGF.scale1, p1, q1);
    // omni: USD localPos 에 몸체 척도를 곱해 PhysX 국소 자세로 (추정 — OVD 값과 비교)
    for (int i = 0; i < 3; ++i) p1[i] = p1[i] * AGF.scale1[i];
    AGF.lp0 = PxTransform(PxVec3(p0[0], p0[1], p0[2]), PxQuat(q0[0], q0[1], q0[2], q0[3]));
    AGF.lp1 = PxTransform(PxVec3(p1[0], p1[1], p1[2]), PxQuat(q1[0], q1[1], q1[2], q1[3]));
    PxD6Joint* j = PxD6JointCreate(*phys, const_cast<PxRigidActor*>(eef), prenorm(AGF.lp0), const_cast<PxRigidActor*>(target), prenorm(AGF.lp1));
    // 공식 OVD 의 생성 뒤 호출 순서 (radio 시연 0, simulate 4465 직전): setMotion x6(잠금) -> setBreakForce -> 제약 플래그 0 -> DRIVE_LIMITS_ARE_FORCES -> 이름
    for (int a = 0; a < 6; ++a) j->setMotion(PxD6Axis::Enum(a), PxD6Motion::eLOCKED);
    j->setBreakForce(PX_MAX_F32, PX_MAX_F32);
    j->setConstraintFlags(PxConstraintFlags());
    j->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
    static std::string nm;
    nm = AGF.eef + "/ag_constraint";
    j->setName(nm.c_str());
    AGF.joint = j;
    C.joint_event = true;  // update_handles -> 이 서브스텝 flush 비움
    fprintf(stderr, "[AG] 관절 만듦 post %lld: lp0 q(%.9g %.9g %.9g %.9g) p(%.9g %.9g %.9g) lp1 q(%.9g %.9g %.9g %.9g) p(%.9g %.9g %.9g)\n",
            (long long)(sims + side_offset), AGF.lp0.q.x, AGF.lp0.q.y, AGF.lp0.q.z, AGF.lp0.q.w, AGF.lp0.p.x, AGF.lp0.p.y, AGF.lp0.p.z,
            AGF.lp1.q.x, AGF.lp1.q.y, AGF.lp1.q.z, AGF.lp1.q.w, AGF.lp1.p.x, AGF.lp1.p.y, AGF.lp1.p.z);
  }
  // ------------------------------------------------------------------ AG 단계 B: 언제·무엇을 잡나 (--ag <폴더>, ag_setup.txt)
  // robot.py:835 _handle_assisted_grasping 을 서브스텝마다 (제어기 step 뒤, flush 전). 판단은 core/omni/assisted_grasp.h.
  // 입력: 손가락 접촉 = RigidContactAPI 접촉 행렬(손가락 행 × 장면 모든 몸체 열, 렌더 스텝 끝마다 갱신 — 아래 RCA),
  //       광선 = PhysX 장면 질의 raycast 가장 가까운 것(omni psqi.raycast_closest), 링크 자세 = PhysX.
  struct AGArm {
    std::string name, eef;
    std::string finger[2];
    std::vector<int> grip_dof;
    std::vector<float> grip_hi;
    std::vector<std::pair<std::string, std::array<float, 3>>> start, end;
    eng::omni::ag::ArmState st;
    PxJoint* joint = nullptr;
  };
  struct AGLink { std::string obj, name, path; float mass = 0.0f; bool has_mass = false; bool dynamic = false; bool nonfixed = false; };
  struct AGObj { bool fixed_base = false; std::string root; };
  struct AGSetup {
    bool on = false;
    std::string robot;
    std::set<std::string> robot_links;
    std::vector<AGArm> arms;
    std::map<std::string, AGObj> objs;
    std::map<std::string, AGLink> links;  // 링크 경로 -> 정보
    uint64_t n_try = 0, n_grasp = 0, n_release = 0, n_skip_type = 0, n_skip_contact = 0;
  } AG;
  // RigidContactAPI 흉내 (손가락 행 × 모든 몸체 열). usd_utils.py:189 _update_contact_matrices_kernel 과 같은 규칙.
  struct RCA {
    bool init = false;
    std::vector<const PxRigidActor*> rows;     // 손가락 (팔 순서, 손가락 순서)
    std::vector<const PxRigidActor*> cols;     // 장면 몸체 (/World/scene_0/<물체>/<링크>)
    std::vector<std::string> col_name;
    std::vector<int32_t> col_to_rigid;         // 동적이면 몸체 번호, 아니면 -1
    std::vector<const PxRigidActor*> bodies;   // 동적 몸체 (자세·알짜 충격을 서브스텝마다)
    std::vector<int32_t> row_to_rigid, b2r;
    std::vector<float> pend_tf, pend_net, pend_imp, prev_tf;
    int n_pend = 0;
    std::vector<uint8_t> cm, ccm;
    std::unordered_map<const void*, int> col_of;
  } RC;
  uint64_t RC_bad = 0, RC_checked = 0;
  bool load_ag(const std::string& dir) {
    std::ifstream f(dir + "/ag_setup.txt");
    if (!f) return false;
    std::string line;
    auto arm_of = [&](const std::string& n) -> AGArm& {
      for (auto& a : AG.arms) if (a.name == n) return a;
      AG.arms.push_back(AGArm());
      AG.arms.back().name = n;
      return AG.arms.back();
    };
    while (std::getline(f, line)) {
      std::istringstream is(line);
      std::string k;
      is >> k;
      if (k == "robot") is >> AG.robot;
      else if (k == "robotlink") { std::string p; is >> p; AG.robot_links.insert(p); }
      else if (k == "arm") { std::string n; is >> n; AGArm& a = arm_of(n); is >> a.eef >> a.finger[0] >> a.finger[1]; }
      else if (k == "grip") { std::string n, hi; int d; is >> n >> d >> hi; AGArm& a = arm_of(n); a.grip_dof.push_back(d); a.grip_hi.push_back(strtof(hi.c_str(), nullptr)); }
      else if (k == "start" || k == "end") {
        std::string n, lp, x, y, z;
        is >> n >> lp >> x >> y >> z;
        AGArm& a = arm_of(n);
        std::array<float, 3> v = {hexf(x), hexf(y), hexf(z)};
        (k == "start" ? a.start : a.end).push_back({lp, v});
      } else if (k == "obj") { std::string op, root; int fb; is >> op >> fb >> root; AG.objs[op] = AGObj{fb != 0, root}; }
      else if (k == "link") {
        AGLink l;
        std::string m;
        int dyn, nf;
        is >> l.obj >> l.name >> l.path >> m >> dyn >> nf;
        l.has_mass = m != "-";
        if (l.has_mass) l.mass = hexf(m);
        l.dynamic = dyn != 0;
        l.nonfixed = nf != 0;
        AG.links[l.path] = l;
      }
    }
    AG.on = !AG.arms.empty();
    return AG.on;
  }
  bool rca_init() {
    if (RC.init) return true;
    for (auto& a : AG.arms)
      for (auto& fp : a.finger) {
        auto it = actor_by_name.find(fp);
        if (it == actor_by_name.end()) return false;
        RC.rows.push_back(it->second);
      }
    std::vector<std::pair<std::string, const PxRigidActor*>> all;
    for (auto& kv : actor_by_name) {
      const std::string& n = kv.first;
      if (n.rfind("/World/scene_0/", 0) != 0) continue;
      if (std::count(n.begin(), n.end(), '/') != 4) continue;  // /World/scene_0/<물체>/<링크>
      all.push_back({n, kv.second});
    }
    std::sort(all.begin(), all.end());
    for (auto& nb : all) {
      const PxRigidActor* a = nb.second;
      const PxRigidDynamic* d = a->is<PxRigidDynamic>();
      const bool dyn = a->is<PxArticulationLink>() || (d && !(d->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC));
      RC.col_of[a] = int(RC.cols.size());
      RC.cols.push_back(a);
      RC.col_name.push_back(nb.first);
      if (dyn) { RC.col_to_rigid.push_back(int(RC.bodies.size())); RC.bodies.push_back(a); }
      else RC.col_to_rigid.push_back(-1);
    }
    for (auto* r : RC.rows) {
      int b = -1;
      for (size_t k = 0; k < RC.bodies.size(); ++k) if (RC.bodies[k] == r) b = int(k);
      if (b < 0) { b = int(RC.bodies.size()); RC.bodies.push_back(r); }
      RC.row_to_rigid.push_back(b);
    }
    RC.b2r.resize(RC.bodies.size());
    for (size_t b = 0; b < RC.bodies.size(); ++b) RC.b2r[b] = int32_t(b);  // 동적 몸체는 모두 행 (알짜 충격 번호 = 몸체 번호)
    RC.cm.assign(RC.rows.size() * RC.cols.size(), 0);
    RC.ccm = RC.cm;
    for (auto* a : RC.bodies) {
      const PxTransform t = a->getGlobalPose();
      const float v[7] = {t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w};
      RC.prev_tf.insert(RC.prev_tf.end(), v, v + 7);
    }
    RC.init = true;
    return true;
  }
  void rca_collect() {  // 서브스텝 끝 (read_from_physx)
    const size_t B = RC.bodies.size(), R = RC.rows.size(), Cn = RC.cols.size();
    const float dt = last_dt;
    for (auto* a : RC.bodies) {
      const PxTransform t = a->getGlobalPose();
      const float v[7] = {t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w};
      RC.pend_tf.insert(RC.pend_tf.end(), v, v + 7);
    }
    const size_t n0 = RC.pend_net.size();
    RC.pend_net.resize(n0 + B * 3, 0.0f);
    for (size_t b = 0; b < B; ++b) {
      auto it = s3_cb.net.find(RC.bodies[b]);
      if (it != s3_cb.net.end()) { RC.pend_net[n0 + b * 3] = it->second.x / dt; RC.pend_net[n0 + b * 3 + 1] = it->second.y / dt; RC.pend_net[n0 + b * 3 + 2] = it->second.z / dt; }
    }
    // 충격 행렬은 손가락 행만 (N, R, C, 3)
    const size_t i0 = RC.pend_imp.size();
    RC.pend_imp.resize(i0 + R * Cn * 3, 0.0f);
    for (size_t r = 0; r < R; ++r) {
      const void* a = RC.rows[r];
      const size_t rb = r;
      for (auto& kv : s3_cb.pair_imp) {
        const void* other = kv.first.first == a ? kv.first.second : (kv.first.second == a ? kv.first.first : nullptr);
        if (!other) continue;
        auto ic = RC.col_of.find(other);
        if (ic == RC.col_of.end()) continue;
        float* v = &RC.pend_imp[i0 + (rb * Cn + size_t(ic->second)) * 3];
        v[0] = kv.second.x / dt; v[1] = kv.second.y / dt; v[2] = kv.second.z / dt;
      }
    }
    RC.n_pend++;
  }
  void rca_update() {  // 렌더 스텝 끝 (RigidContactAPI.update, usd_utils.py:189 와 같은 규칙을 손가락 행만)
    namespace st = eng::omni::st;
    const int B = int(RC.bodies.size()), Cn = int(RC.cols.size());
    st::ContactIn in{RC.pend_tf.data(), RC.prev_tf.data(), RC.pend_net.data(), RC.b2r.data(), B, B, 1e-6f, 1e-4f};
    for (size_t r = 0; r < RC.rows.size(); ++r) {
      const int rb = RC.row_to_rigid[r];
      for (int c = 0; c < Cn; ++c) {
        const size_t k = r * Cn + c;
        const int cb = RC.col_to_rigid[c];
        int last_awake = -1;
        uint8_t any_contact = 0;
        for (int i = 0; i < RC.n_pend; ++i) {
          const bool ra = st::body_awake_at_step(in, i, rb);
          const bool ca = cb >= 0 ? st::body_awake_at_step(in, i, cb) : false;
          if (ra || ca) {
            last_awake = i;
            const float* v = &RC.pend_imp[((size_t(i) * RC.rows.size() + r) * Cn + c) * 3];
            if (v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f) any_contact = 1;
          }
        }
        if (last_awake >= 0) {
          const float* v = &RC.pend_imp[((size_t(last_awake) * RC.rows.size() + r) * Cn + c) * 3];
          RC.ccm[k] = (v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f) ? 1 : 0;
          RC.cm[k] = any_contact;
        } else {
          RC.cm[k] = RC.ccm[k];
        }
      }
    }
    std::vector<float> nt = RC.prev_tf;
    for (int b = 0; b < B; ++b) st::body_transform_update(in, RC.n_pend, b, nt.data());
    RC.prev_tf = nt;
    RC.pend_tf.clear(); RC.pend_net.clear(); RC.pend_imp.clear(); RC.n_pend = 0;
  }
  // 광선: 링크 틀 점 -> 세계 (link_pos + quat2mat(q) @ p, torch float32 추정), 시작 x 끝 모든 쌍, 끝점 + 1e-8
  std::set<std::string> ag_raycast(const AGArm& a) {
    std::set<std::string> hits;
    std::vector<PxVec3> S, E;
    auto world = [&](const std::string& lp, const std::array<float, 3>& p, PxVec3& out) {
      auto it = actor_by_name.find(lp);
      if (it == actor_by_name.end()) return false;
      const PxTransform t = it->second->getGlobalPose();
      float R[9];
      const float q[4] = {t.q.x, t.q.y, t.q.z, t.q.w};
      eng::omni::agf::quat2mat(q, R);
      out = PxVec3(t.p.x + (R[0] * p[0] + R[1] * p[1] + R[2] * p[2]), t.p.y + (R[3] * p[0] + R[4] * p[1] + R[5] * p[2]),
                   t.p.z + (R[6] * p[0] + R[7] * p[1] + R[8] * p[2]));
      return true;
    };
    for (auto& sp : a.start) { PxVec3 w; if (world(sp.first, sp.second, w)) S.push_back(w); }
    for (auto& ep : a.end) { PxVec3 w; if (world(ep.first, ep.second, w)) E.push_back(w + PxVec3(1e-8f)); }
    PxScene* sc = last_scene;
    if (!sc) return hits;
    for (auto& e : E)
      for (auto& st0 : S) {
        PxVec3 d = e - st0;
        const float dist = eng::omni::ag::torch_norm3(d.x, d.y, d.z);
        if (!(dist > 0.0f)) continue;
        d = PxVec3(d.x / dist, d.y / dist, d.z / dist);
        PxRaycastBuffer buf;
        if (sc->raycast(st0, d, dist, buf) && buf.hasBlock && buf.block.actor) {
          const char* n = buf.block.actor->getName();
          if (n && std::string(n).find(AG.robot) == std::string::npos) hits.insert(n);
        }
      }
    return hits;
  }
  void ag_decide(const eng::omni::ctrl::DriveTargets& out) {
    namespace ag = eng::omni::ag;
    if (!AG.on || !rca_init()) return;
    ag::Params P;
    for (size_t ai = 0; ai < AG.arms.size(); ++ai) {
      AGArm& a = AG.arms[ai];
      bool applying = false;
      for (size_t k = 0; k < a.grip_dof.size(); ++k)
        if (out.set_pos[a.grip_dof[k]] && out.pos[a.grip_dof[k]] < a.grip_hi[k]) applying = true;
      int in_hand = -1;
      ag::CandidateIn cin{};
      std::vector<std::string> cand_path;
      if (a.st.obj_in_hand < 0 && applying) {
        // 손가락 접촉 (current_only=False 접촉 행렬), 로봇 링크 제외
        std::map<std::string, int> fingers_of;
        for (int f = 0; f < 2; ++f) {
          const size_t r = ai * 2 + f;
          for (size_t c = 0; c < RC.cols.size(); ++c)
            if (RC.cm[r * RC.cols.size() + c] && !AG.robot_links.count(RC.col_name[c])) fingers_of[RC.col_name[c]]++;
        }
        const std::set<std::string> rays = ag_raycast(a);
        auto ie = actor_by_name.find(a.eef);
        if (ie != actor_by_name.end()) {
          const PxTransform te = ie->second->getGlobalPose();
          const float eef[3] = {te.p.x, te.p.y, te.p.z};
          for (auto& kv : fingers_of) {
            if (cin.n >= ag::kMaxCand) break;
            auto il = AG.links.find(kv.first);
            auto ia = actor_by_name.find(kv.first);
            const int i = cin.n++;
            cin.link[i] = i;
            cin.fingers[i] = uint8_t(kv.second);
            cin.ray_hit[i] = rays.count(kv.first) ? 1 : 0;
            cin.obj[i] = il != AG.links.end() ? 0 : -1;
            cin.dynamic[i] = il != AG.links.end() && il->second.dynamic ? 1 : 0;
            if (ia != actor_by_name.end()) {
              const PxTransform tl = ia->second->getGlobalPose();
              cin.pos[i][0] = tl.p.x; cin.pos[i][1] = tl.p.y; cin.pos[i][2] = tl.p.z;
            }
            cand_path.push_back(kv.first);
          }
          const int best = ag::calculate_in_hand(cin, eef);
          in_hand = best;
        }
      }
      int target = -1;
      const ag::Event ev = ag::step_arm(P, a.st, applying, in_hand, &target);
      if (ev == ag::EV_RELEASE) {
        if (a.joint) { a.joint->release(); a.joint = nullptr; }
        C.joint_event = true;  // _release_grasp -> update_handles
        AG.n_release++;
      } else if (ev == ag::EV_TRY_GRASP && target >= 0) {
        AG.n_try++;
        const std::string& lp = cand_path[size_t(target)];
        const AGLink& L = AG.links[lp];
        const AGObj& O = AG.objs[L.obj];
        const int jt = ag::joint_type(P, L.mass, O.fixed_base, L.name == O.root, L.nonfixed);
        if (jt == 0) { AG.n_skip_type++; continue; }
        auto ie = actor_by_name.find(a.eef), it = actor_by_name.find(lp);
        PxVec3 cp;
        AGF.fingers = {a.finger[0], a.finger[1]};
        AGF.eef = a.eef;
        if (ie == actor_by_name.end() || it == actor_by_name.end() || !ag_contact_point(it->second, cp)) { AG.n_skip_contact++; continue; }
        if (jt == 2) fprintf(stderr, "[AG] 구 관절(SphericalJoint) 은 아직 고정 관절로 만든다 — 공식 생성 순서 확인 필요\n");
        ag_create(ie->second, it->second, cp);
        a.joint = AGF.joint;
        ag::grasp_established(a.st, 0, 0);
        AG.n_grasp++;
        fprintf(stderr, "[AG] 잡음 post %llu 팔 %s 대상 %s\n", (unsigned long long)(sims + side_offset), a.name.c_str(), lp.c_str());
      }
    }
  }
  void ag_before_simulate() {  // 제어기 step 다음, 목표 쓰기 전 (simulator.py:1579~1587 순서)
    if (AGF.post < 0 || AGF.done) return;
    if (int64_t(sims + side_offset) != AGF.post) return;
    AGF.done = true;
    auto ie = actor_by_name.find(AGF.eef), it = actor_by_name.find(AGF.target);
    if (ie == actor_by_name.end() || it == actor_by_name.end()) { fprintf(stderr, "[AG] 몸체 없음\n"); return; }
    PxVec3 cp;
    if (!ag_contact_point(it->second, cp)) { fprintf(stderr, "[AG] 접촉점 없음 -> 안 잡음\n"); return; }
    ag_create(ie->second, it->second, cp);
  }
  std::set<uint64_t> skip_objs;
  bool in_free(bool out_block) const { return C.free_run && C.on && !out_block && sims + side_offset >= C.episode_start; }
  bool is_ext_joint_cls(uint32_t cls) {
    const std::string n = cname(cls);
    return n == "PxFixedJoint" || n == "PxD6Joint" || n == "PxSphericalJoint" || n == "PxRevoluteJoint" || n == "PxPrismaticJoint" ||
           n == "PxDistanceJoint";
  }
  void ctrl_before_simulate() {
    if (!C.on) return;
    const uint64_t g = sims + side_offset + 1;  // 곧 할 simulate 의 전체 번호
    if (g <= C.episode_start) { C.joint_event = false; return; }
    const int64_t k = int64_t(g - C.episode_start - 1), t = k / C.substeps;
    if (t >= int64_t(C.T) || !ctrl_bind()) return;
    namespace ct = eng::omni::ctrl;
    if (t != C.last_t) {
      auto ib = actor_by_name.find(C.base_link), ir = actor_by_name.find(C.root_link);
      if (ib == actor_by_name.end() || ir == actor_by_name.end()) return;
      const PxTransform bp = ib->second->getGlobalPose(), rp = ir->second->getGlobalPose();
      const float base_p[3] = {bp.p.x, bp.p.y, bp.p.z}, base_q[4] = {bp.q.x, bp.q.y, bp.q.z, bp.q.w};
      const float root_p[3] = {rp.p.x, rp.p.y, rp.p.z}, root_q[4] = {rp.q.x, rp.q.y, rp.q.z, rp.q.w};
      ct::apply_action(C.cfg, C.st, &C.acts[size_t(t) * C.A], base_p, base_q, root_p, root_q);
      C.last_t = t;
      C.applied_steps++;
    }
    float q[ct::kMaxDof] = {};
    for (int d = 0; d < C.n_dof; ++d) q[d] = C.sign[d] * C.joint[d]->getJointPosition(PxArticulationAxis::Enum(C.axis[d]));
    ct::DriveTargets out{};
    ct::step(C.cfg, C.st, q, out);
    ag_before_simulate();
    ag_decide(out);
    if (C.joint_event) {  // 이 서브스텝은 flush 가 비었다 (위 설명)
      C.joint_event = false;
      C.dropped_flush++;
      C.rec.clear();
      return;
    }
    // 공식 순서: 위치 목표 한 번에(set_dof_position_targets), 그다음 속도 목표 (dof 순서)
    for (int d = 0; d < C.n_dof; ++d) if (out.set_pos[d]) C.joint[d]->setDriveTarget(PxArticulationAxis::Enum(C.axis[d]), C.sign[d] * out.pos[d]);
    for (int d = 0; d < C.n_dof; ++d) if (out.set_vel[d]) C.joint[d]->setDriveVelocity(PxArticulationAxis::Enum(C.axis[d]), C.sign[d] * out.vel[d]);
    // 자체 일관성: 우리 상태로 계산한 목표 vs 같은 서브스텝 공식 목표
    for (int d = 0; d < C.n_dof; ++d) {
      auto ir = C.rec.find(C.joint[d]);
      if (ir == C.rec.end()) continue;
      for (int w = 0; w < 2; ++w) {
        if (!(w == 0 ? out.set_pos[d] : out.set_vel[d]) || ir->second[w].size() < 24) continue;
        const float ours = C.sign[d] * (w == 0 ? out.pos[d] : out.vel[d]);
        float r; memcpy(&r, ir->second[w].data() + 4 * C.axis[d], 4);
        C.cmp_n++;
        if (memcmp(&ours, &r, 4) && !C.cmp_bad++) {
          C.first_bad_step = t;
          char b[200]; snprintf(b, sizeof b, "스텝 %lld dof %d %s: 우리 %.9g 공식 %.9g", (long long)t, d, w == 0 ? "위치" : "속도", ours, r);
          C.first_bad = b;
        }
      }
    }
    C.rec.clear();
  }
  // OVD 에 안 남는 단일 액터 플래그: PxActor::setActorFlag(한 개)는 OVD 에 기록하지 않는다(NpActorTemplate.h:193 setActorFlagInternal —
  // OMNI_PVD_SET 은 setActorFlags(여러 개) 쪽에만). omni 는 physxRigidBody:disableGravity 를 setActorFlag(eDISABLE_GRAVITY) 로 넣는다
  // (UsdInterface.cpp:487, PhysXRigidBodyPropertiesUpdate.cpp:904). 곁 파일(이름 목록)로 첫 simulate 전에 넣는다.
  std::vector<std::string> gravity_off;
  bool gravity_applied = false;
  void apply_gravity_off() {
    size_t n = 0, miss = 0;
    for (auto& name : gravity_off) {
      auto it = actor_by_name.find(name);
      if (it == actor_by_name.end()) { miss++; continue; }
      it->second->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
      n++;
    }
    gravity_applied = true;
    printf("중력 끔(곁 파일): %zu 액터, 못 찾음 %zu\n", n, miss);
  }
  void trace_flush() {
    for (auto& t : trace_buf) {
      std::string nm = gname_of(t.first);
      auto ic = joint_child.find(t.first);
      if (ic != joint_child.end()) nm = "joint->" + gname_of(ic->second);
      auto is = shape_owner.find(t.first);
      if (is != shape_owner.end()) nm = "shape@" + gname_of(is->second);
      if (nm.find(trace_sub) != std::string::npos) { fputs(nm.c_str(), stderr); fputc(' ', stderr); fputs(t.second.c_str(), stderr); fputc('\n', stderr); }
    }
  }
  PxOmniPvd* rec_pvd = nullptr;
  bool init(int threads) {
    for (auto& kv : F.attrs) A[F.classes[kv.second.cls].name + "." + kv.second.name] = kv.first;
    fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    // 척도: OVD 의 PxPhysics.tolerancesScale (length, speed)
    PxTolerancesScale tol;
    for (auto& e : F.events)
      if (e.cmd == ovd::kSet && F.attr_name(e.attr) == "PxPhysics.tolerancesScale" && e.data_len >= 8) {
        memcpy(&tol.length, F.data(e), 4);
        memcpy(&tol.speed, F.data(e) + 4, 4);
        break;
      }
    // --record: 재생 쪽도 OVD 로 남겨 원본 OVD 와 명령 단위로 비교한다 (ovd_diff)
    PxOmniPvd* opvd = nullptr;
    if (!record_path.empty()) {
      opvd = PxCreateOmniPvd(*fnd);
      if (opvd) {
        OmniPvdFileWriteStream* fs = opvd->getFileWriteStream();
        fs->setFileName(record_path.c_str());
        opvd->getWriter()->setWriteStream(static_cast<OmniPvdWriteStream&>(*fs));
      }
    }
    phys = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, tol, true, nullptr, opvd);
    if (opvd) opvd->startSampling();
    rec_pvd = opvd;
    PxInitExtensions(*phys, nullptr);
    disp = PxDefaultCpuDispatcherCreate(threads);
    cook = new PxCookingParams(tol);
    cook->buildGPUData = false;            // omni.physx/plugins/Setup.cpp:869
    cook->buildTriangleAdjacencies = true;
    prescan();
    return phys != nullptr;
  }

  void prescan() {
    const uint32_t a_parent = attr("PxArticulationJointReducedCoordinate", "parentLink");
    const uint32_t a_child = attr("PxArticulationJointReducedCoordinate", "childLink");
    std::unordered_map<uint64_t, uint64_t> jparent;
    std::unordered_map<uint64_t, uint32_t> gen;
    auto gk = [&](uint64_t h) { auto it = gen.find(h); return gkey(h, it == gen.end() ? 0 : it->second); };
    for (auto& e : F.events) {
      if (e.cmd == ovd::kCreate) { gen[e.obj]++; continue; }
      if (e.cmd != ovd::kSet) continue;
      if (e.attr == a_child && e.data_len == 8) { uint64_t v; memcpy(&v, F.data(e), 8); const uint64_t k = gk(e.obj); if (v && !joint_child.count(k)) joint_child[k] = gk(v); }
      if (e.attr == a_parent && e.data_len == 8) { uint64_t v; memcpy(&v, F.data(e), 8); const uint64_t k = gk(e.obj); if (v && !jparent.count(k)) jparent[k] = gk(v); }
    }
    for (auto& kv : joint_child) {
      auto it = jparent.find(kv.first);
      if (it != jparent.end()) link_parent[kv.second] = it->second;
    }
    joint_parent = jparent;
  }

  // ------------------------------------------------------------------ 객체 만들기
  PxGeometryHolder geometry(uint64_t h, bool& ok) {
    ok = false;
    auto it = objs.find(h);
    if (it == objs.end()) return PxGeometryHolder();
    Obj& g = it->second;
    std::string c = cname(g.cls);
    if (c == "PxBoxGeometry") {
      PxVec3 he; if (!pget(g, "PxBoxGeometry", "halfExtents", he)) return PxGeometryHolder();
      ok = true; return PxGeometryHolder(PxBoxGeometry(he));
    }
    if (c == "PxSphereGeometry") {
      float r; if (!pget(g, "PxSphereGeometry", "radius", r)) return PxGeometryHolder();
      ok = true; return PxGeometryHolder(PxSphereGeometry(r));
    }
    if (c == "PxCapsuleGeometry") {
      float r, hh; if (!pget(g, "PxCapsuleGeometry", "radius", r) || !pget(g, "PxCapsuleGeometry", "halfHeight", hh)) return PxGeometryHolder();
      ok = true; return PxGeometryHolder(PxCapsuleGeometry(r, hh));
    }
    if (c == "PxPlaneGeometry") { ok = true; return PxGeometryHolder(PxPlaneGeometry()); }
    if (c == "PxConvexMeshGeometry") {
      PxVec3 s; uint64_t m;
      if (!pget(g, "PxConvexMeshGeometry", "scale", s) || !pget(g, "PxConvexMeshGeometry", "convexMesh", m)) return PxGeometryHolder();
      auto im = objs.find(m);
      if (im == objs.end() || !im->second.px) return PxGeometryHolder();
      ok = true;  // OVD 에 척도 회전·geometry flags 가 없다 -> 항등 회전, flags 0 (추정; worldBounds 비교로 확인)
      return PxGeometryHolder(PxConvexMeshGeometry(static_cast<PxConvexMesh*>(im->second.px), PxMeshScale(s)));
    }
    if (c == "PxTriangleMeshGeometry") {
      PxVec3 s; uint64_t m;
      if (!pget(g, "PxTriangleMeshGeometry", "scale", s) || !pget(g, "PxTriangleMeshGeometry", "triangleMesh", m)) return PxGeometryHolder();
      auto im = objs.find(m);
      if (im == objs.end() || !im->second.px) return PxGeometryHolder();
      ok = true;
      return PxGeometryHolder(PxTriangleMeshGeometry(static_cast<PxTriangleMesh*>(im->second.px), PxMeshScale(s)));
    }
    unsupported["geometry:" + c]++;
    return PxGeometryHolder();
  }

  void make_scene(Obj& o) {
    PxTolerancesScale tol = phys->getTolerancesScale();
    { float ts[2]; if (pget(o, "PxScene", "tolerancesScale", ts)) { tol.length = ts[0]; tol.speed = ts[1]; } }
    PxSceneDesc sd(tol);
    uint32_t u32; float f; PxVec3 v;
    if (pget(o, "PxScene", "flags", u32)) sd.flags = PxSceneFlags(u32);
    sd.flags &= ~PxSceneFlags(PxSceneFlag::eENABLE_GPU_DYNAMICS | PxSceneFlag::eENABLE_DIRECT_GPU_API);
    if (pget(o, "PxScene", "solverType", u32)) sd.solverType = PxSolverType::Enum(u32);
    if (pget(o, "PxScene", "broadPhaseType", u32)) sd.broadPhaseType = PxBroadPhaseType::Enum(u32);
    if (sd.broadPhaseType == PxBroadPhaseType::eGPU) sd.broadPhaseType = PxBroadPhaseType::ePABP;
    if (pget(o, "PxScene", "kineKineFilteringMode", u32)) sd.kineKineFilteringMode = PxPairFilteringMode::Enum(u32);
    if (pget(o, "PxScene", "staticKineFilteringMode", u32)) sd.staticKineFilteringMode = PxPairFilteringMode::Enum(u32);
    if (pget(o, "PxScene", "gravity", v)) sd.gravity = v;
    if (pget(o, "PxScene", "bounceThresholdVelocity", f)) sd.bounceThresholdVelocity = f;
    if (pget(o, "PxScene", "frictionOffsetThreshold", f)) sd.frictionOffsetThreshold = f;
    if (pget(o, "PxScene", "frictionCorrelationDistance", f)) sd.frictionCorrelationDistance = f;
    if (pget(o, "PxScene", "solverBatchSize", u32)) sd.solverBatchSize = u32;
    if (pget(o, "PxScene", "solverArticulationBatchSize", u32)) sd.solverArticulationBatchSize = u32;
    // nbContactDataBlocks / maxNbContactDataBlocks 는 OVD 에 "지금 쓰는 블록 수"(처음 0)로 남는다 -> 설명값이 아니라 기본값 유지
    if (pget(o, "PxScene", "maxBiasCoefficient", f)) sd.maxBiasCoefficient = f;
    if (pget(o, "PxScene", "contactReportStreamBufferSize", u32)) sd.contactReportStreamBufferSize = u32;
    if (pget(o, "PxScene", "ccdMaxPasses", u32)) sd.ccdMaxPasses = u32;
    if (pget(o, "PxScene", "ccdThreshold", f)) sd.ccdThreshold = f;
    if (pget(o, "PxScene", "ccdMaxSeparation", f)) sd.ccdMaxSeparation = f;
    if (pget(o, "PxScene", "wakeCounterResetValue", f)) sd.wakeCounterResetValue = f;
    if (pget(o, "PxScene", "limitsMaxNbActors", u32)) sd.limits.maxNbActors = u32;
    if (pget(o, "PxScene", "limitsMaxNbBodies", u32)) sd.limits.maxNbBodies = u32;
    if (pget(o, "PxScene", "limitsMaxNbStaticShapes", u32)) sd.limits.maxNbStaticShapes = u32;
    if (pget(o, "PxScene", "limitsMaxNbDynamicShapes", u32)) sd.limits.maxNbDynamicShapes = u32;
    if (pget(o, "PxScene", "limitsMaxNbAggregates", u32)) sd.limits.maxNbAggregates = u32;
    if (pget(o, "PxScene", "limitsMaxNbConstraints", u32)) sd.limits.maxNbConstraints = u32;
    if (pget(o, "PxScene", "limitsMaxNbRegions", u32)) sd.limits.maxNbRegions = u32;
    if (pget(o, "PxScene", "limitsMaxNbBroadPhaseOverlaps", u32)) sd.limits.maxNbBroadPhaseOverlaps = u32;
    PxBounds3 b;
    if (pget(o, "PxScene", "sanityBounds", b)) sd.sanityBounds = b;
    if (pget(o, "PxScene", "contactPairSlabSize", u32)) sd.contactPairSlabSize = u32;
    sd.cpuDispatcher = disp;
    sd.filterShader = engine::OmniFilterShader;
    sd.filterShaderData = &specp;
    sd.filterShaderDataSize = sizeof(specp);
    sd.filterCallback = &filter_cb;
    if (!trace_sub.empty()) { diag_cb.sub = trace_sub; diag_cb.sims = &sims; sd.simulationEventCallback = &diag_cb; filter_cb.diag_sub = trace_sub; }
    if (contact_report_all) { filter_cb.report_all = true; if (!sd.simulationEventCallback) sd.simulationEventCallback = &diag_cb; diag_cb.sims = &sims; }
    if (S3.own) { filter_cb.report_all = true; sd.simulationEventCallback = &s3_cb; }  // omni: 접촉 보고 prim ≈ 모든 몸체
    o.scene = phys->createScene(sd);
    o.px = nullptr;
    if (verbose) fprintf(stderr, "[scene] flags=0x%x solver=%d bp=%d\n", uint32_t(sd.flags), int(sd.solverType), int(sd.broadPhaseType));
  }

  void make_material(Obj& o) {
    float sf = 0.5f, df = 0.5f, r = 0.5f;
    pget(o, "PxMaterial", "staticFriction", sf);
    pget(o, "PxMaterial", "dynamicFriction", df);
    pget(o, "PxMaterial", "restitution", r);
    PxMaterial* m = phys->createMaterial(sf, df, r);
    // 만든 순간 값이 생성 기본값과 다를 때만 setter (원본에 없던 호출을 끼우지 않는다)
    uint32_t u; float f;
    if (auto fv = pv(o, "PxMaterial", "flags")) {
      PxMaterialFlags fl(PxU16(flagv(fv->data(), uint32_t(fv->size()))));
      if (fl != m->getFlags()) m->setFlags(fl);
    }
    if (pget(o, "PxMaterial", "frictionCombineMode", u) && PxCombineMode::Enum(u) != m->getFrictionCombineMode()) m->setFrictionCombineMode(PxCombineMode::Enum(u));
    if (pget(o, "PxMaterial", "restitutionCombineMode", u) && PxCombineMode::Enum(u) != m->getRestitutionCombineMode()) m->setRestitutionCombineMode(PxCombineMode::Enum(u));
    if (pget(o, "PxMaterial", "dampingCombineMode", u) && PxCombineMode::Enum(u) != m->getDampingCombineMode()) m->setDampingCombineMode(PxCombineMode::Enum(u));
    if (pget(o, "PxMaterial", "damping", f)) { float c = m->getDamping(); if (memcmp(&c, &f, 4)) m->setDamping(f); }
    o.px = m;
  }

  void make_convex(Obj& o) {
    auto v = pv(o, "PxConvexMesh", "verts");
    if (!v) { unsupported["convex:no verts"]++; return; }
    uint64_t h = engine::hash_bytes(v->data(), v->size());
    auto it = convex.find(h);
    if (it != convex.end()) {
      const engine::ConvexData& c = it->second;
      std::vector<PxHullPolygon> polys(c.polys.size());
      for (size_t i = 0; i < polys.size(); ++i) {
        memcpy(polys[i].mPlane, c.polys[i].plane, 16);
        polys[i].mNbVerts = c.polys[i].nverts;
        polys[i].mIndexBase = c.polys[i].base;
      }
      std::vector<PxU32> idx(c.indices.begin(), c.indices.end());
      PxConvexMeshDesc d;
      d.points.count = PxU32(c.verts.size() / 3); d.points.stride = 12; d.points.data = c.verts.data();
      d.polygons.count = PxU32(polys.size()); d.polygons.stride = sizeof(PxHullPolygon); d.polygons.data = polys.data();
      d.indices.count = PxU32(idx.size()); d.indices.stride = 4; d.indices.data = idx.data();
      o.px = PxCreateConvexMesh(*cook, d, phys->getPhysicsInsertionCallback());
      if (o.px) { convex_exact++; return; }
    }
    // 보조 파일에 없으면 꼭짓점에서 다시 볼록 껍질 계산 (근사: 면 평면식이 원본과 비트 같다는 보장 없음)
    PxConvexMeshDesc d;
    d.points.count = PxU32(v->size() / 12); d.points.stride = 12; d.points.data = v->data();
    d.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    d.vertexLimit = 255;
    o.px = PxCreateConvexMesh(*cook, d, phys->getPhysicsInsertionCallback());
    convex_approx++;
    if (o.px) approx_meshes.insert(o.px);
  }
  std::set<const PxBase*> approx_meshes;  // 보조 파일에서 못 찾아 다시 계산한 볼록 메시
  void report_approx_users() {  // 근사 메시를 쓰는 액터 이름 (원인 가르기용)
    std::map<std::string, int> users;
    for (auto& kv : objs) {
      if (!kv.second.px) continue;
      auto* a = kv.second.px->is<PxRigidActor>();
      if (!a) continue;
      PxShape* sh[64];
      PxU32 ns = a->getShapes(sh, 64);
      for (PxU32 k = 0; k < ns; ++k)
        if (sh[k]->getGeometry().getType() == PxGeometryType::eCONVEXMESH) {
          const PxConvexMeshGeometry& g = static_cast<const PxConvexMeshGeometry&>(sh[k]->getGeometry());
          if (approx_meshes.count(g.convexMesh)) users[a->getName() ? a->getName() : "?"]++;
        }
    }
    for (auto& u : users) printf("  근사 볼록 메시 사용: %s (%d 개 모양)\n", u.first.c_str(), u.second);
  }

  void make_trimesh(Obj& o) {
    auto v = pv(o, "PxTriangleMesh", "verts");
    auto t = pv(o, "PxTriangleMesh", "tris");
    if (!v || !t) { unsupported["trimesh:no data"]++; return; }
    PxTriangleMeshDesc d;
    d.points.count = PxU32(v->size() / 12); d.points.stride = 12; d.points.data = v->data();
    d.triangles.count = PxU32(t->size() / 12); d.triangles.stride = 12; d.triangles.data = t->data();
    o.px = PxCreateTriangleMesh(*cook, d, phys->getPhysicsInsertionCallback());
  }

  void make_shape(Obj& o) {
    uint64_t gh = 0;
    if (!pget(o, "PxShape", "geom", gh)) { unsupported["shape:no geom"]++; return; }
    bool ok;
    PxGeometryHolder g = geometry(gh, ok);
    if (!ok) {  // 형상 값이 비었거나 메시 객체가 없음 -> 어느 종류인지 남긴다
      auto ig = objs.find(gh);
      unsupported[std::string("shape:geom unsupported ") + (ig != objs.end() ? cname(ig->second.cls) : "?")]++;
      return;
    }
    auto mv = pv(o, "PxShape", "materials");
    std::vector<PxMaterial*> mats;
    if (mv)
      for (size_t i = 0; i + 8 <= mv->size(); i += 8) {
        uint64_t mh; memcpy(&mh, mv->data() + i, 8);
        auto im = objs.find(mh);
        if (im != objs.end() && im->second.px) mats.push_back(static_cast<PxMaterial*>(im->second.px));
      }
    if (mats.empty()) { unsupported["shape:no material"]++; return; }
    uint8_t excl = 1; pget(o, "PxShape", "isExclusive", excl);
    uint32_t flags = uint32_t(PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE);
    { auto fv = pv(o, "PxShape", "shapeFlags"); if (fv) { flags = 0; memcpy(&flags, fv->data(), fv->size() < 4 ? fv->size() : 4); } }
    PxShape* s = phys->createShape(g.any(), mats.data(), PxU16(mats.size()), excl != 0, PxShapeFlags(PxU8(flags)));
    o.px = s;
    if (!s) return;
    // 생성 순간 값이 기본값과 다를 수 있는 것들 (createShape 인자에 없는 것)
    PxTransform lp; float f; PxFilterData fd;
    auto differs = [](const void* a, const void* b, size_t n) { return memcmp(a, b, n) != 0; };
    if (pget(o, "PxShape", "localPose", lp)) { PxTransform c = s->getLocalPose(); if (differs(&c, &lp, sizeof lp)) s->setLocalPose(prenorm(lp)); }
    if (pget(o, "PxShape", "contactOffset", f)) { float c = s->getContactOffset(); if (differs(&c, &f, 4)) s->setContactOffset(f); }
    if (pget(o, "PxShape", "restOffset", f)) { float c = s->getRestOffset(); if (differs(&c, &f, 4)) s->setRestOffset(f); }
    if (pget(o, "PxShape", "simulationFilterData", fd)) { PxFilterData c = s->getSimulationFilterData(); if (differs(&c, &fd, sizeof fd)) s->setSimulationFilterData(fd); }
    if (pget(o, "PxShape", "queryFilterData", fd)) { PxFilterData c = s->getQueryFilterData(); if (differs(&c, &fd, sizeof fd)) s->setQueryFilterData(fd); }
  }

  PxTransform pose_of(Obj& o) {
    PxTransform p(PxIdentity);
    pget(o, "PxRigidActor", "globalPose", p);
    return p;
  }

  void make_actor(Obj& o) {
    std::string c = cname(o.cls);
    if (c == "PxRigidStatic") o.px = phys->createRigidStatic(prenorm(pose_of(o)));
    else if (c == "PxRigidDynamic") o.px = phys->createRigidDynamic(prenorm(pose_of(o)));
  }

  void make_aggregate(Obj& o) {
    uint32_t maxShapes = 0; uint8_t self = 0; pget(o, "PxAggregate", "maxNbShapes", maxShapes); pget(o, "PxAggregate", "selfCollision", self);
    uint32_t maxActors = 256;  // OVD 에 없음 (용량일 뿐 결과와 무관)
    o.px = phys->createAggregate(maxActors, maxShapes ? maxShapes : 256, PxGetAggregateFilterHint(PxAggregateType::eGENERIC, self != 0));
  }

  void make_articulation(Obj& o) { o.px = phys->createArticulationReducedCoordinate(); }

  void make_link(uint64_t art_h, uint64_t link_h) {
    Obj& l = objs[link_h];
    auto ia = objs.find(art_h);
    if (ia == objs.end() || !ia->second.px) { unsupported["link:no articulation"]++; return; }
    auto* art = static_cast<PxArticulationReducedCoordinate*>(ia->second.px);
    PxArticulationLink* parent = nullptr;
    auto ip = link_parent.find(gnow(link_h));
    if (ip != link_parent.end()) {
      auto ol = objs.find(graw(ip->second));
      if (ol != objs.end()) parent = static_cast<PxArticulationLink*>(ol->second.px);
    }
    l.px = art->createLink(parent, prenorm(pose_of(l)));
    l.done = true;
  }

  PxJoint* make_ext_joint(Obj& o) {
    std::string c = cname(o.cls);
    uint64_t a0 = 0, a1 = 0;
    PxTransform p0(PxIdentity), p1(PxIdentity);
    pget(o, "PxJoint", "actor0", a0); pget(o, "PxJoint", "actor1", a1);
    pget(o, "PxJoint", "actor0LocalPose", p0); pget(o, "PxJoint", "actor1LocalPose", p1);
    p0 = prenorm(p0); p1 = prenorm(p1);
    auto act = [&](uint64_t h) -> PxRigidActor* {
      auto it = objs.find(h);
      return (h && it != objs.end()) ? static_cast<PxRigidActor*>(it->second.px) : nullptr;
    };
    PxJoint* j = nullptr;
    if (c == "PxFixedJoint") j = PxFixedJointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxD6Joint") j = PxD6JointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxRevoluteJoint") j = PxRevoluteJointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxPrismaticJoint") j = PxPrismaticJointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxSphericalJoint") j = PxSphericalJointCreate(*phys, act(a0), p0, act(a1), p1);
    else if (c == "PxDistanceJoint") j = PxDistanceJointCreate(*phys, act(a0), p0, act(a1), p1);
    else unsupported["joint:" + c]++;
    return j;
  }

  // 만들 때 값 중 생성 인자에 없는 것을 setter 로 (관절 조인트·조인트·흉내 관절)
  void close_cluster(uint64_t h) {
    auto it = objs.find(h);
    if (it == objs.end() || it->second.done) return;
    Obj& o = it->second;
    std::string c = cname(o.cls);
    if (c == "PxArticulationJointReducedCoordinate") {
      auto ic = joint_child.find(gnow(h));
      if (ic != joint_child.end()) {
        auto ol = objs.find(graw(ic->second));
        if (ol != objs.end() && ol->second.px) {
          o.px = static_cast<PxArticulationLink*>(ol->second.px)->getInboundJoint();
          o.done = true;
          if (!trace_sub.empty() && ol->second.name.find(trace_sub) != std::string::npos && o.px) {
            auto* jj = static_cast<PxArticulationJointReducedCoordinate*>(o.px);
            PxArticulationLimit L0 = jj->getLimitParams(PxArticulationAxis::eTWIST);
            fprintf(stderr, "[묶음] %s 조인트 %p 묶음 직후 한계 축0 [%.9g, %.9g] 운동 %d\n", ol->second.name.c_str(), (void*)jj, L0.low, L0.high, int(jj->getMotion(PxArticulationAxis::eTWIST)));
          }
        }
      }
      return;
    }
    if (c == "PxArticulationMimicJoint") {
      uint64_t ah, ja, jb; uint32_t axa, axb; float gear = 0, off = 0, nf = 0, dr = 0;
      if (pget(o, c.c_str(), "articulation", ah) && pget(o, c.c_str(), "jointA", ja) && pget(o, c.c_str(), "jointB", jb) &&
          pget(o, c.c_str(), "axisA", axa) && pget(o, c.c_str(), "axisB", axb)) {
        pget(o, c.c_str(), "gearRatio", gear); pget(o, c.c_str(), "offset", off);
        pget(o, c.c_str(), "naturalFrequency", nf); pget(o, c.c_str(), "dampingRatio", dr);
        auto* art = static_cast<PxArticulationReducedCoordinate*>(objs[ah].px);
        auto* JA = static_cast<PxArticulationJointReducedCoordinate*>(objs[ja].px);
        auto* JB = static_cast<PxArticulationJointReducedCoordinate*>(objs[jb].px);
        if (art && JA && JB)
          o.px = art->createMimicJoint(*JA, PxArticulationAxis::Enum(axa), *JB, PxArticulationAxis::Enum(axb), gear, off, nf, dr);
        if (!o.px) unsupported["mimic joint create failed"]++;
        else if (verbose) fprintf(stderr, "[mimic] 만듦 art=%p A=%p B=%p\n", (void*)art, (void*)JA, (void*)JB);
      } else unsupported["mimic joint missing attrs"]++;
      o.done = true;
      return;
    }
    if (F.is_a(o.cls, F.cls("PxJoint"))) {
      PxJoint* j = make_ext_joint(o);
      o.px = j;
      o.done = true;
      if (j) {
        uint64_t ch = 0;
        if (pget(o, "PxJoint", "constraint", ch)) { Obj& co = objs[ch]; co.px = j->getConstraint(); co.done = true; }
        // 만든 순간 값(cluster)은 생성 기본값 + 생성 인자뿐이다. 기본값이 아닌 것은 뒤따르는 set(API 호출)으로 온다.
      }
      return;
    }
    if (c == "PxD6JointDrive") {  // 값 객체: 만들 때 값(네 개)을 모아 두고, 조인트 쪽 연결이 오면 setDrive
      o.done = true;
      D6DriveRef& r = d6drive[h];
      float st = r.v.stiffness, dm = r.v.damping, fl = r.v.forceLimit;
      uint32_t fg = uint32_t(r.v.flags);
      const bool any = pget(o, "PxD6JointDrive", "stiffness", st) | pget(o, "PxD6JointDrive", "damping", dm) |
                       pget(o, "PxD6JointDrive", "forceLimit", fl) | pget(o, "PxD6JointDrive", "flags", fg);
      if (any) {
        r.v.stiffness = st; r.v.damping = dm; r.v.forceLimit = fl; r.v.flags = PxD6JointDriveFlags(fg);
        r.have = true;
        d6_push(h);
      }
      o.pend.clear();
      return;
    }
  }

  void apply_joint_cluster(Obj& o, PxJoint* j) {
    // 생성 인자에 없는 속성을 한 번에 (기본값이면 같은 값을 다시 넣는 것이라 무해)
    for (auto& kv : o.pend) {
      const std::string an = F.attrs[kv.first].name;
      if (an == "actor0" || an == "actor1" || an == "actor0LocalPose" || an == "actor1LocalPose" || an == "constraint" ||
          an == "type" || an == "concreteTypeName")
        continue;
      apply_set(o, kv.first, kv.second.data(), uint32_t(kv.second.size()), j);
    }
  }

  // ------------------------------------------------------------------ setter (API 호출 재현)
  template <class T> static T as(const uint8_t* p) { T v; memcpy(&v, p, sizeof(T)); return v; }
  // 플래그 속성은 타입마다 1/2/4 바이트 -> 길이만큼만 읽는다
  static uint32_t flagv(const uint8_t* p, uint32_t n) { uint32_t v = 0; memcpy(&v, p, n < 4 ? n : 4); return v; }
  static std::string cstr(const uint8_t* p, uint32_t n) { while (n && p[n - 1] == 0) --n; return std::string(reinterpret_cast<const char*>(p), n); }

  std::vector<uint8_t> cur_set_data;
  void apply_set(Obj& o, uint32_t ah, const uint8_t* p, uint32_t n, PxBase* target = nullptr) {
    cur_set_data.assign(p, p + n);
    const ovd::AttrInfo& ai = F.attrs[ah];
    const std::string cl = F.classes[ai.cls].name, an = ai.name;
    const std::string key = cl + "." + an;
    PxBase* b = target ? target : o.px;
    bool ok = true;
    if (cl == "PxScene") {
      PxScene* s = o.scene;
      if (!s) ok = false;
      else if (an == "gravity") s->setGravity(as<PxVec3>(p));
      else if (an == "bounceThresholdVelocity") s->setBounceThresholdVelocity(as<float>(p));
      else if (an == "flags") {
        PxSceneFlags want(flagv(p, n)), have = s->getFlags();
        for (uint32_t bit = 1; bit; bit <<= 1)
          if ((uint32_t(want) ^ uint32_t(have)) & bit) s->setFlag(PxSceneFlag::Enum(bit), (uint32_t(want) & bit) != 0);
      } else if (an == "frictionOffsetThreshold") s->setFrictionOffsetThreshold(as<float>(p));
      else if (an == "frictionCorrelationDistance") s->setFrictionCorrelationDistance(as<float>(p));
      else if (an == "ccdMaxPasses") s->setCCDMaxPasses(as<uint32_t>(p));
      else if (an == "ccdThreshold") s->setCCDThreshold(as<float>(p));
      else if (an == "ccdMaxSeparation") s->setCCDMaxSeparation(as<float>(p));
      else if (an == "maxBiasCoefficient") s->setMaxBiasCoefficient(as<float>(p));
      else ok = false;
    } else if (cl == "PxActor") {
      auto* a = b ? b->is<PxActor>() : nullptr;
      if (!a) ok = false;
      else if (an == "name") { o.name = cstr(p, n); a->setName(o.name.c_str()); register_name(o, a); }
      else if (an == "flags") a->setActorFlags(PxActorFlags(PxU8(flagv(p, n))));
      else if (an == "dominance") a->setDominanceGroup(p[0]);
      else if (an == "environmentID") a->setEnvironmentID(as<uint32_t>(p));
      else if (an == "worldBounds" || an == "type" || an == "ownerClient") {}
      else ok = false;
    } else if (cl == "PxRigidActor") {
      auto* a = b ? b->is<PxRigidActor>() : nullptr;
      if (!a) ok = false;
      else if (an == "globalPose") {
        if (a->is<PxArticulationLink>()) ok = false;  // 링크는 setGlobalPose 가 없다 (루트 자세는 applyCache/sidelog)
        else a->setGlobalPose(prenorm(as<PxTransform>(p)));
      } else ok = false;
    } else if (cl == "PxRigidBody") {
      auto* rb = b ? b->is<PxRigidBody>() : nullptr;
      auto* rd = b ? b->is<PxRigidDynamic>() : nullptr;
      if (!rb) ok = false;
      else if (an == "cMassLocalPose") rb->setCMassLocalPose(prenorm(as<PxTransform>(p)));
      else if (an == "mass") rb->setMass(as<float>(p));
      else if (an == "massSpaceInertiaTensor") rb->setMassSpaceInertiaTensor(as<PxVec3>(p));
      else if (an == "linearDamping") rb->setLinearDamping(as<float>(p));
      else if (an == "angularDamping") rb->setAngularDamping(as<float>(p));
      else if (an == "linearVelocity") { if (rd) rd->setLinearVelocity(as<PxVec3>(p)); else ok = false; }
      else if (an == "angularVelocity") { if (rd) rd->setAngularVelocity(as<PxVec3>(p)); else ok = false; }
      else if (an == "maxLinearVelocity") rb->setMaxLinearVelocity(as<float>(p));
      else if (an == "maxAngularVelocity") rb->setMaxAngularVelocity(as<float>(p));
      else if (an == "rigidBodyFlags") rb->setRigidBodyFlags(PxRigidBodyFlags(PxU16(flagv(p, n))));
      else if (an == "minAdvancedCCDCoefficient") rb->setMinCCDAdvanceCoefficient(as<float>(p));
      else if (an == "maxDepenetrationVelocity") rb->setMaxDepenetrationVelocity(as<float>(p));
      else if (an == "maxContactImpulse") rb->setMaxContactImpulse(as<float>(p));
      else if (an == "contactSlopCoefficient") rb->setContactSlopCoefficient(as<float>(p));
      else if (an == "force" || an == "torque") {}  // fetchResults 뒤 0 으로 되돌리는 기록일 뿐 (addForce 는 OVD 에 안 남음)
      else ok = false;
    } else if (cl == "PxRigidDynamic") {
      auto* rd = b ? b->is<PxRigidDynamic>() : nullptr;
      if (!rd) ok = false;
      else if (an == "sleepThreshold") rd->setSleepThreshold(as<float>(p));
      else if (an == "stabilizationThreshold") rd->setStabilizationThreshold(as<float>(p));
      else if (an == "rigidDynamicLockFlags") rd->setRigidDynamicLockFlags(PxRigidDynamicLockFlags(PxU8(flagv(p, n))));
      else if (an == "wakeCounter") rd->setWakeCounter(as<float>(p));
      else if (an == "positionIterations") { PxU32 a0, v0; rd->getSolverIterationCounts(a0, v0); rd->setSolverIterationCounts(as<uint32_t>(p), v0); }
      else if (an == "velocityIterations") { PxU32 a0, v0; rd->getSolverIterationCounts(a0, v0); rd->setSolverIterationCounts(a0, as<uint32_t>(p)); }
      else if (an == "contactReportThreshold") rd->setContactReportThreshold(as<float>(p));
      else if (an == "isSleeping") {}
      else ok = false;
    } else if (cl == "PxArticulationLink") {
      auto* l = b ? b->is<PxArticulationLink>() : nullptr;
      if (!l) ok = false;
      else if (an == "CFMScale") l->setCfmScale(as<float>(p));
      else if (an == "inboundJoint" || an == "articulation" || an == "inboundJointDOF") {}
      else ok = false;
    } else if (cl == "PxArticulationReducedCoordinate") {
      auto* a = b ? b->is<PxArticulationReducedCoordinate>() : nullptr;
      if (!a) ok = false;
      else if (an == "name") { o.name = cstr(p, n); a->setName(o.name.c_str()); art_by_name[o.name] = a; }
      else if (an == "positionIterations") { PxU32 a0, v0; a->getSolverIterationCounts(a0, v0); a->setSolverIterationCounts(as<uint32_t>(p), v0); }
      else if (an == "velocityIterations") { PxU32 a0, v0; a->getSolverIterationCounts(a0, v0); a->setSolverIterationCounts(a0, as<uint32_t>(p)); }
      else if (an == "sleepThreshold") a->setSleepThreshold(as<float>(p));
      else if (an == "stabilizationThreshold") a->setStabilizationThreshold(as<float>(p));
      else if (an == "wakeCounter") a->setWakeCounter(as<float>(p));
      else if (an == "articulationFlags") {
        PxArticulationFlags fl(PxU8(flagv(p, n)));
        if (diag_no_self_collision) fl |= PxArticulationFlag::eDISABLE_SELF_COLLISION;  // 진단 전용
        a->setArticulationFlags(fl);
      }
      else if (an == "isSleeping" || an == "worldBounds" || an == "dofs") {}
      else ok = false;
    } else if (cl == "PxArticulationJointReducedCoordinate") {
      auto* j = static_cast<PxArticulationJointReducedCoordinate*>(b);
      if (!j) ok = false;
      else if (ctrl_suppress(b, an)) ok = true;
      else ok = apply_art_joint(j, an, p, n);
    } else if (cl == "PxShape") {
      auto* s = b ? b->is<PxShape>() : nullptr;
      if (!s) ok = false;
      else if (an == "localPose") s->setLocalPose(prenorm(as<PxTransform>(p)));
      else if (an == "contactOffset") s->setContactOffset(as<float>(p));
      else if (an == "restOffset") s->setRestOffset(as<float>(p));
      else if (an == "torsionalPatchRadius") s->setTorsionalPatchRadius(as<float>(p));
      else if (an == "minTorsionalPatchRadius") s->setMinTorsionalPatchRadius(as<float>(p));
      else if (an == "shapeFlags") { s->setFlags(PxShapeFlags(PxU8(flagv(p, n)))); mark_refilter(s); }
      else if (an == "simulationFilterData") { s->setSimulationFilterData(as<PxFilterData>(p)); mark_refilter(s); }
      else if (an == "queryFilterData") s->setQueryFilterData(as<PxFilterData>(p));
      else if (an == "materials") {
        std::vector<PxMaterial*> mats;
        for (uint32_t i = 0; i + 8 <= n; i += 8) { auto im = objs.find(as<uint64_t>(p + i)); if (im != objs.end() && im->second.px) mats.push_back(static_cast<PxMaterial*>(im->second.px)); }
        if (!mats.empty()) s->setMaterials(mats.data(), PxU16(mats.size()));
      } else if (an == "densityForFluid" || an == "isExclusive") {}
      else if (an == "geom") {
        bool gok; PxGeometryHolder g = geometry(as<uint64_t>(p), gok);
        if (gok) s->setGeometry(g.any()); else ok = false;
      } else ok = false;
    } else if (cl == "PxMaterial") {
      auto* m = static_cast<PxMaterial*>(b);
      if (!m) ok = false;
      else if (an == "staticFriction") m->setStaticFriction(as<float>(p));
      else if (an == "dynamicFriction") m->setDynamicFriction(as<float>(p));
      else if (an == "restitution") m->setRestitution(as<float>(p));
      else if (an == "damping") m->setDamping(as<float>(p));
      else if (an == "flags") m->setFlags(PxMaterialFlags(PxU16(flagv(p, n))));
      else if (an == "frictionCombineMode") m->setFrictionCombineMode(PxCombineMode::Enum(as<uint32_t>(p)));
      else if (an == "restitutionCombineMode") m->setRestitutionCombineMode(PxCombineMode::Enum(as<uint32_t>(p)));
      else if (an == "dampingCombineMode") m->setDampingCombineMode(PxCombineMode::Enum(as<uint32_t>(p)));
      else ok = false;
    } else if (cl == "PxJoint" || F.is_a(ai.cls, F.cls("PxJoint"))) {
      ok = apply_ext_joint(o, static_cast<PxJoint*>(b), cl, an, p, n);
    } else if (cl == "PxAggregate") {
      if (an == "selfCollision" || an == "environmentID" || an == "maxNbShapes" || an == "scene") {} else ok = false;
    } else if (cl == "PxArticulationMimicJoint") {
      auto* m = static_cast<PxArticulationMimicJoint*>(b);
      if (!m) ok = false;
      else if (an == "gearRatio") m->setGearRatio(as<float>(p));
      else if (an == "offset") m->setOffset(as<float>(p));
      else if (an == "naturalFrequency") m->setNaturalFrequency(as<float>(p));
      else if (an == "dampingRatio") m->setDampingRatio(as<float>(p));
      else ok = false;
    } else ok = false;
    if (ok) applied[key]++;
    else unsupported["set:" + key]++;
  }

  bool apply_art_joint(PxArticulationJointReducedCoordinate* j, const std::string& an, const uint8_t* p, uint32_t n) {
    const int na = int(n / 4);
    auto f = [&](int i) { return as<float>(p + 4 * i); };
    auto u = [&](int i) { return as<uint32_t>(p + 4 * i); };
    auto AX = [](int i) { return PxArticulationAxis::Enum(i); };
    // 원본은 같은 값도 setter 를 부른다(omni 가 eFIX 로 한 번 놓고 제 종류로 다시 놓음). setter 는 관절체를 더럽힘 표시하므로 늘 부른다.
    if (an == "type") { j->setJointType(PxArticulationJointType::Enum(u(0))); return true; }
    if (an == "motion") { for (int i = 0; i < na && i < 6; ++i) if (j->getMotion(AX(i)) != PxArticulationMotion::Enum(u(i))) j->setMotion(AX(i), PxArticulationMotion::Enum(u(i))); return true; }
    if (an == "parentTranslation") { PxTransform t = j->getParentPose(); t.p = as<PxVec3>(p); j->setParentPose(t); return true; }
    if (an == "parentRotation") { PxTransform t = j->getParentPose(); t.q = as<PxQuat>(p); j->setParentPose(t); return true; }
    if (an == "childTranslation") { PxTransform t = j->getChildPose(); t.p = as<PxVec3>(p); j->setChildPose(t); return true; }
    if (an == "childRotation") { PxTransform t = j->getChildPose(); t.q = as<PxQuat>(p); j->setChildPose(t); return true; }
    if (an == "frictionCoefficient") { j->setFrictionCoefficient(f(0)); return true; }
    if (an == "maxJointVelocity") { j->setMaxJointVelocity(f(0)); return true; }
    if (an == "name") { static std::vector<std::unique_ptr<std::string>> keep; keep.emplace_back(new std::string(reinterpret_cast<const char*>(p), n)); j->setName(keep.back()->c_str()); return true; }
    if (an == "concreteTypeName" || an == "parentLink" || an == "childLink" || an == "jointForce") return true;
    // 축 6 개짜리 배열: 값이 바뀐 축만 setter 로 (원본은 축 하나씩 부른다)
    auto changed = [&](int i, float cur) { float v = f(i); return memcmp(&v, &cur, 4) != 0; };
    // 같은 값이어도 원본 setter 가 불렸다(OVD 에 set 이 남음) -> 바뀐 축이 없으면 첫 풀린 축에 같은 값을 다시 넣어 부수효과를 낸다.
    //   setJointPosition/Velocity: 관절체 위치·속도 더럽힘 -> 다음 simulate 에서 링크 자세를 관절값으로 다시 계산 (NpArticulationJointReducedCoordinate.cpp)
    //   setArmature: 관절 더럽힘. (09-30 S0: 로봇이 simulate 1 부터 다르던 원인)
    auto first_free = [&]() { for (int i = 0; i < 6; ++i) if (j->getMotion(AX(i)) != PxArticulationMotion::eLOCKED) return i; return -1; };
    if (an == "armature") {
      bool any = false;
      for (int i = 0; i < na && i < 6; ++i) if (changed(i, j->getArmature(AX(i)))) { j->setArmature(AX(i), f(i)); any = true; }
      if (!any) { const int k = first_free(); const int a = k < 0 ? 0 : k; j->setArmature(AX(a), j->getArmature(AX(a))); }
      return true;
    }
    if (an == "jointPosition" || an == "jointVelocity") {
      const bool pos = an == "jointPosition";
      bool any = false;
      for (int i = 0; i < na && i < 6; ++i) {
        if (j->getMotion(AX(i)) == PxArticulationMotion::eLOCKED) continue;
        if (changed(i, pos ? j->getJointPosition(AX(i)) : j->getJointVelocity(AX(i)))) { if (pos) j->setJointPosition(AX(i), f(i)); else j->setJointVelocity(AX(i), f(i)); any = true; }
      }
      const int k = first_free();
      if (!any && k >= 0) { if (pos) j->setJointPosition(AX(k), j->getJointPosition(AX(k))); else j->setJointVelocity(AX(k), j->getJointVelocity(AX(k))); }
      return true;
    }
    if (an == "limitLow" || an == "limitHigh") {
      for (int i = 0; i < na && i < 6; ++i) {
        PxArticulationLimit l = j->getLimitParams(AX(i));
        float& d = (an == "limitLow") ? l.low : l.high;
        if (memcmp(&d, p + 4 * i, 4)) { d = f(i); j->setLimitParams(AX(i), l); }
      }
      return true;
    }
    if (an == "driveStiffness" || an == "driveDamping" || an == "driveMaxForce" || an == "driveType" || an == "driveMaxEffort" ||
        an == "driveMaxActuatorVelocity" || an == "driveVelocityDependentResistance" || an == "driveSpeedEffortGradient") {
      for (int i = 0; i < na && i < 6; ++i) {
        PxArticulationDrive d = j->getDriveParams(AX(i));
        bool ch = false;
        auto setf = [&](float& dst) { if (memcmp(&dst, p + 4 * i, 4)) { dst = f(i); ch = true; } };
        if (an == "driveStiffness") setf(d.stiffness);
        else if (an == "driveDamping") setf(d.damping);
        else if (an == "driveMaxForce") setf(d.maxForce);
        else if (an == "driveMaxEffort") setf(d.envelope.maxEffort);
        else if (an == "driveMaxActuatorVelocity") setf(d.envelope.maxActuatorVelocity);
        else if (an == "driveVelocityDependentResistance") setf(d.envelope.velocityDependentResistance);
        else if (an == "driveSpeedEffortGradient") setf(d.envelope.speedEffortGradient);
        else if (an == "driveType") { if (uint32_t(d.driveType) != u(i)) { d.driveType = PxArticulationDriveType::Enum(u(i)); ch = true; } }
        if (ch) j->setDriveParams(AX(i), d);
      }
      return true;
    }
    // 값이 같아도 원본은 setter 를 불렀다(자동 깨우기 autoWakeInternal 이 걸린다, NpArticulationJointReducedCoordinate.cpp:400)
    // -> 바뀐 축이 없으면 축 0 에 같은 값을 다시 넣어 부수효과를 똑같이 낸다.
    if (an == "driveTarget") {
      bool any = false;
      for (int i = 0; i < na && i < 6; ++i) if (changed(i, j->getDriveTarget(AX(i)))) { j->setDriveTarget(AX(i), f(i)); any = true; }
      if (!any) j->setDriveTarget(AX(0), j->getDriveTarget(AX(0)));
      return true;
    }
    if (an == "driveVelocity") {
      bool any = false;
      for (int i = 0; i < na && i < 6; ++i) if (changed(i, j->getDriveVelocity(AX(i)))) { j->setDriveVelocity(AX(i), f(i)); any = true; }
      if (!any) j->setDriveVelocity(AX(0), j->getDriveVelocity(AX(0)));
      return true;
    }
    if (an == "staticFrictionEffort" || an == "dynamicFrictionEffort" || an == "viscousFrictionCoefficient") {
      for (int i = 0; i < na && i < 6; ++i) {
        PxJointFrictionParams fp = j->getFrictionParams(AX(i));
        float& d = an == "staticFrictionEffort" ? fp.staticFrictionEffort : (an == "dynamicFrictionEffort" ? fp.dynamicFrictionEffort : fp.viscousFrictionCoefficient);
        if (memcmp(&d, p + 4 * i, 4)) { d = f(i); j->setFrictionParams(AX(i), fp); }
      }
      return true;
    }
    if (an == "maxJointDofVelocity") { for (int i = 0; i < na && i < 6; ++i) if (changed(i, j->getMaxJointVelocity(AX(i)))) j->setMaxJointVelocity(AX(i), f(i)); return true; }
    return false;
  }

  // ------------------------------------------------------------------ 묶음 setter
  // 원본 API 한 번이 속성 여러 개를 남기는 것들 (PhysX 소스에서 확인):
  //   setSolverIterationCounts -> positionIterations, velocityIterations (NpRigidDynamic.cpp:543, NpArticulationReducedCoordinate.cpp:1010)
  //   PxJoint::setLocalPose -> actor0LocalPose, actor1LocalPose (ExtJoint.h:245), setBreakForce -> breakForce, breakTorque
  //   setParentPose / setChildPose -> *Translation, *Rotation (NpArticulationJointReducedCoordinate.cpp:498)
  //   setLimitParams -> limitLow, limitHigh (:307), setFrictionParams -> 3 개 (:220), setDriveParams -> 8 개 (:345~)
  //   PxD6Joint::setDriveVelocity -> driveLinVelocity, driveAngVelocity
  const std::vector<std::string>* group_of(Obj& o, uint32_t ah) {
    static const std::vector<std::string> iters = {"positionIterations", "velocityIterations"};
    static const std::vector<std::string> jlocal = {"actor0LocalPose", "actor1LocalPose"};
    static const std::vector<std::string> jbreak = {"breakForce", "breakTorque"};
    static const std::vector<std::string> ppose = {"parentTranslation", "parentRotation"};
    static const std::vector<std::string> cpose = {"childTranslation", "childRotation"};
    static const std::vector<std::string> limits = {"limitLow", "limitHigh"};
    static const std::vector<std::string> fric = {"staticFrictionEffort", "dynamicFrictionEffort", "viscousFrictionCoefficient"};
    static const std::vector<std::string> drive = {"driveStiffness", "driveDamping", "driveMaxForce", "driveMaxEffort",
                                                   "driveMaxActuatorVelocity", "driveVelocityDependentResistance",
                                                   "driveSpeedEffortGradient", "driveType"};
    static const std::vector<std::string> d6vel = {"driveLinVelocity", "driveAngVelocity"};
    static const std::vector<std::string> d6drv = {"stiffness", "damping", "forceLimit", "flags"};  // setDrive 한 번 (ExtD6Joint.cpp:254)
    const ovd::AttrInfo& ai = F.attrs[ah];
    const std::string& cl = F.classes[ai.cls].name;
    const std::string& an = ai.name;
    auto in = [&](const std::vector<std::string>& g) { for (auto& x : g) if (x == an) return true; return false; };
    if ((cl == "PxRigidDynamic" || cl == "PxArticulationReducedCoordinate") && in(iters)) return &iters;
    if (cl == "PxJoint" && in(jlocal)) return &jlocal;
    if (cl == "PxJoint" && in(jbreak)) return &jbreak;
    if (cl == "PxD6Joint" && in(d6vel)) return &d6vel;
    if (cl == "PxD6JointDrive" && in(d6drv)) return &d6drv;
    if (cl == "PxArticulationJointReducedCoordinate") {
      if (in(ppose)) return &ppose;
      if (in(cpose)) return &cpose;
      if (in(limits)) return &limits;
      if (in(fric)) return &fric;
      if (in(drive)) return &drive;
    }
    (void)o;
    return nullptr;
  }

  void apply_group(Obj& o, const std::vector<std::string>& g, std::unordered_map<std::string, const ovd::Event*>& v) {
    auto has = [&](const char* n) { return v.count(n) != 0; };
    auto dat = [&](const char* n) { return F.data(*v[n]); };
    auto AX = [](int i) { return PxArticulationAxis::Enum(i); };
    const std::string first = g[0];
    if (first == "stiffness") {  // PxD6JointDrive 값 객체 (px 없음): 값 갱신 후 연결돼 있으면 setDrive
      const uint64_t h = v.begin()->second->obj;
      D6DriveRef& r = d6drive[h];
      if (has("stiffness")) r.v.stiffness = as<float>(dat("stiffness"));
      if (has("damping")) r.v.damping = as<float>(dat("damping"));
      if (has("forceLimit")) r.v.forceLimit = as<float>(dat("forceLimit"));
      if (has("flags")) r.v.flags = PxD6JointDriveFlags(flagv(dat("flags"), v["flags"]->data_len));
      r.have = true;
      d6_push(h);
      return;
    }
    PxBase* b = o.px;
    if (!b) { unsupported["group:no object " + first]++; return; }
    if (first == "positionIterations") {
      PxU32 pi, vi;
      if (auto* rd = b->is<PxRigidDynamic>()) {
        rd->getSolverIterationCounts(pi, vi);
        if (has("positionIterations")) pi = as<uint32_t>(dat("positionIterations"));
        if (has("velocityIterations")) vi = as<uint32_t>(dat("velocityIterations"));
        rd->setSolverIterationCounts(pi, vi);
      } else if (auto* ar = b->is<PxArticulationReducedCoordinate>()) {
        ar->getSolverIterationCounts(pi, vi);
        if (has("positionIterations")) pi = as<uint32_t>(dat("positionIterations"));
        if (has("velocityIterations")) vi = as<uint32_t>(dat("velocityIterations"));
        ar->setSolverIterationCounts(pi, vi);
      }
      applied["group:iterations"]++;
      return;
    }
    if (first == "actor0LocalPose") {
      auto* j = static_cast<PxJoint*>(b);
      bool any = false;
      for (int a = 0; a < 2; ++a) {
        const char* n = a ? "actor1LocalPose" : "actor0LocalPose";
        if (!has(n)) continue;
        PxTransform want = as<PxTransform>(dat(n)), cur = j->getLocalPose(PxJointActorIndex::Enum(a));
        if (memcmp(&want, &cur, sizeof want)) { j->setLocalPose(PxJointActorIndex::Enum(a), prenorm(want)); any = true; }
      }
      if (!any) j->setLocalPose(PxJointActorIndex::eACTOR0, j->getLocalPose(PxJointActorIndex::eACTOR0));
      applied["group:jointLocalPose"]++;
      return;
    }
    if (first == "breakForce") {
      auto* j = static_cast<PxJoint*>(b);
      float f0, t0; j->getBreakForce(f0, t0);
      if (has("breakForce")) f0 = as<float>(dat("breakForce"));
      if (has("breakTorque")) t0 = as<float>(dat("breakTorque"));
      j->setBreakForce(f0, t0);
      applied["group:breakForce"]++;
      return;
    }
    if (first == "driveLinVelocity") {
      auto* d = static_cast<PxJoint*>(b)->is<PxD6Joint>();
      if (!d) return;
      PxVec3 l, a; d->getDriveVelocity(l, a);
      if (has("driveLinVelocity")) l = as<PxVec3>(dat("driveLinVelocity"));
      if (has("driveAngVelocity")) a = as<PxVec3>(dat("driveAngVelocity"));
      d->setDriveVelocity(l, a);
      applied["group:d6DriveVelocity"]++;
      return;
    }
    auto* j = static_cast<PxArticulationJointReducedCoordinate*>(b);
    if (first == "parentTranslation" || first == "childTranslation") {
      const bool par = first == "parentTranslation";
      PxTransform t = par ? j->getParentPose() : j->getChildPose();
      if (has(par ? "parentTranslation" : "childTranslation")) t.p = as<PxVec3>(dat(par ? "parentTranslation" : "childTranslation"));
      if (has(par ? "parentRotation" : "childRotation")) t.q = as<PxQuat>(dat(par ? "parentRotation" : "childRotation"));
      if (par) j->setParentPose(t); else j->setChildPose(t);  // 입력 그대로 기록되므로 원상 찾기 불필요
      applied[par ? "group:parentPose" : "group:childPose"]++;
      return;
    }
    auto fa = [&](const char* n, int i) { return as<float>(dat(n) + 4 * i); };
    auto same = [](float a, float b) { return memcmp(&a, &b, 4) == 0; };
    if (first == "limitLow") {
      if (!trace_sub.empty()) {
        const char* cn = j->getChildArticulationLink().getName();
        if (cn && std::string(cn).find(trace_sub) != std::string::npos) {
          PxArticulationLimit L0 = j->getLimitParams(AX(0));
          fprintf(stderr, "[한계] %s sim %llu 전 [%.9g, %.9g] low있음 %d high있음 %d\n", cn, (unsigned long long)sims, L0.low, L0.high, int(has("limitLow")), int(has("limitHigh")));
        }
      }
      bool any = false;
      for (int i = 0; i < 6; ++i) {
        PxArticulationLimit l = j->getLimitParams(AX(i));
        PxArticulationLimit w = l;
        if (has("limitLow")) w.low = fa("limitLow", i);
        if (has("limitHigh")) w.high = fa("limitHigh", i);
        if (!same(w.low, l.low) || !same(w.high, l.high)) { j->setLimitParams(AX(i), w); any = true; }
      }
      if (!any) j->setLimitParams(AX(0), j->getLimitParams(AX(0)));
      applied["group:limitParams"]++;
      return;
    }
    if (first == "staticFrictionEffort") {
      bool any = false;
      for (int i = 0; i < 6; ++i) {
        PxJointFrictionParams c = j->getFrictionParams(AX(i)), w = c;
        if (has("staticFrictionEffort")) w.staticFrictionEffort = fa("staticFrictionEffort", i);
        if (has("dynamicFrictionEffort")) w.dynamicFrictionEffort = fa("dynamicFrictionEffort", i);
        if (has("viscousFrictionCoefficient")) w.viscousFrictionCoefficient = fa("viscousFrictionCoefficient", i);
        if (memcmp(&c, &w, sizeof c)) { j->setFrictionParams(AX(i), w); any = true; }
      }
      if (!any) j->setFrictionParams(AX(0), j->getFrictionParams(AX(0)));
      applied["group:frictionParams"]++;
      return;
    }
    if (first == "driveStiffness") {
      bool any = false;
      for (int i = 0; i < 6; ++i) {
        PxArticulationDrive c = j->getDriveParams(AX(i)), w = c;
        if (has("driveStiffness")) w.stiffness = fa("driveStiffness", i);
        if (has("driveDamping")) w.damping = fa("driveDamping", i);
        if (has("driveMaxForce")) w.maxForce = fa("driveMaxForce", i);
        if (has("driveMaxEffort")) w.envelope.maxEffort = fa("driveMaxEffort", i);
        if (has("driveMaxActuatorVelocity")) w.envelope.maxActuatorVelocity = fa("driveMaxActuatorVelocity", i);
        if (has("driveVelocityDependentResistance")) w.envelope.velocityDependentResistance = fa("driveVelocityDependentResistance", i);
        if (has("driveSpeedEffortGradient")) w.envelope.speedEffortGradient = fa("driveSpeedEffortGradient", i);
        if (has("driveType")) w.driveType = PxArticulationDriveType::Enum(as<uint32_t>(dat("driveType") + 4 * i));
        const bool ch = !same(c.stiffness, w.stiffness) || !same(c.damping, w.damping) || !same(c.maxForce, w.maxForce) ||
                        !same(c.envelope.maxEffort, w.envelope.maxEffort) ||
                        !same(c.envelope.maxActuatorVelocity, w.envelope.maxActuatorVelocity) ||
                        !same(c.envelope.velocityDependentResistance, w.envelope.velocityDependentResistance) ||
                        !same(c.envelope.speedEffortGradient, w.envelope.speedEffortGradient) || c.driveType != w.driveType;
        if (ch) { j->setDriveParams(AX(i), w); any = true; }
        if (dbg_drive && i == 0) {
          const char* cn = j->getChildArticulationLink().getName();
          if (cn && strstr(cn, "torso_link1"))
            fprintf(stderr, "[drv] sim %llu torso_link1 has_k %d 받은 k %.9g 적용 전 %.9g 후 %.9g j=%p\n", (unsigned long long)sims, int(has("driveStiffness")),
                    has("driveStiffness") ? fa("driveStiffness", 0) : -1.0f, c.stiffness, j->getDriveParams(AX(0)).stiffness, (void*)j);
        }
      }
      if (!any) j->setDriveParams(AX(0), j->getDriveParams(AX(0)));
      applied["group:driveParams"]++;
      return;
    }
    unsupported["group:" + first]++;
  }

  bool apply_ext_joint(Obj& o, PxJoint* j, const std::string& cl, const std::string& an, const uint8_t* p, uint32_t n) {
    if (!j) return false;
    if (cl == "PxJoint") {
      if (an == "actor0LocalPose") { j->setLocalPose(PxJointActorIndex::eACTOR0, prenorm(as<PxTransform>(p))); return true; }
      if (an == "actor1LocalPose") { j->setLocalPose(PxJointActorIndex::eACTOR1, prenorm(as<PxTransform>(p))); return true; }
      if (an == "breakForce") { float f0, t0; j->getBreakForce(f0, t0); j->setBreakForce(as<float>(p), t0); return true; }
      if (an == "breakTorque") { float f0, t0; j->getBreakForce(f0, t0); j->setBreakForce(f0, as<float>(p)); return true; }
      if (an == "constraintFlags") { j->setConstraintFlags(PxConstraintFlags(PxU16(flagv(p, n)))); return true; }
      if (an == "invMassScale0") { j->setInvMassScale0(as<float>(p)); return true; }
      if (an == "invInertiaScale0") { j->setInvInertiaScale0(as<float>(p)); return true; }
      if (an == "invMassScale1") { j->setInvMassScale1(as<float>(p)); return true; }
      if (an == "invInertiaScale1") { j->setInvInertiaScale1(as<float>(p)); return true; }
      if (an == "name") { o.name = cstr(p, n); j->setName(o.name.c_str()); return true; }
      if (an == "actor0" || an == "actor1" || an == "constraint" || an == "type" || an == "concreteTypeName") return true;
      return false;
    }
    if (cl == "PxFixedJoint") return true;
    if (cl == "PxD6Joint") {
      auto* d = j->is<PxD6Joint>();
      if (!d) return false;
      if (an == "motions") { for (uint32_t i = 0; i < n / 4 && i < 6; ++i) d->setMotion(PxD6Axis::Enum(i), PxD6Motion::Enum(as<uint32_t>(p + 4 * i))); return true; }
      if (an == "drivePosition") { d->setDrivePosition(prenorm(as<PxTransform>(p))); return true; }
      if (an == "driveLinVelocity") { PxVec3 l, a; d->getDriveVelocity(l, a); d->setDriveVelocity(as<PxVec3>(p), a); return true; }
      if (an == "driveAngVelocity") { PxVec3 l, a; d->getDriveVelocity(l, a); d->setDriveVelocity(l, as<PxVec3>(p)); return true; }
      if (an == "twistAngle" || an == "swingYAngle" || an == "swingZAngle") return true;  // 결과값
      // 드라이브 연결: 속성 이름 순서 = PxD6Drive 번호 (eX..eSWING2, PxD6Joint.h:154)
      static const char* kDrive[8] = {"driveX", "driveY", "driveZ", "driveSwing", "driveTwist", "driveSlerp", "driveSwing1", "driveSwing2"};
      for (int t = 0; t < 8; ++t)
        if (an == kDrive[t]) {
          uint64_t h = 0;
          if (n >= 8) memcpy(&h, p, 8);
          if (h) { D6DriveRef& r = d6drive[h]; r.joint = d; r.type = t; d6_push(h); }
          return true;  // 0 = 연결 끊음(설정 바꿈 때 omniPvdClearDriveData) — API 호출 아님
        }
      if (an == "angularDriveConfig") {
        const auto c = PxD6AngularDriveConfig::Enum(as<uint32_t>(p));
        if (d->getAngularDriveConfig() != c) d->setAngularDriveConfig(c);  // 바뀔 때만 (ExtD6Joint.cpp:550 과 같은 조건)
        return true;
      }
      // 한계(선·비틀기·흔들기·거리): BEHAVIOR radio 기록에는 없다(ovd_dump). 나오면 여기서 못 옮김으로 센다
      return false;
    }
    if (cl == "PxRevoluteJoint" || cl == "PxPrismaticJoint" || cl == "PxSphericalJoint") {
      if (an == "angle" || an == "velocity" || an == "position" || an == "swingYAngle" || an == "swingZAngle") return true;  // 결과값
      return false;
    }
    return false;
  }

  // 연결과 값이 모였으면 setDrive. 지금 각 구동 설정에서 허용 안 되는 종류는 PhysX checked 빌드처럼 건너뛴다
  // (ExtD6Joint.cpp:150 isDriveTypeAllowed — 원본에서도 그 호출은 아무 일도 안 한다).
  void d6_push(uint64_t h) {
    auto it = d6drive.find(h);
    if (it == d6drive.end() || !it->second.joint || !it->second.have) return;
    const D6DriveRef& r = it->second;
    const int t = r.type;
    const auto cfg = r.joint->getAngularDriveConfig();
    bool ok = t <= 2;
    if (cfg == PxD6AngularDriveConfig::eSWING_TWIST) ok |= (t == 4 || t == 6 || t == 7);
    else if (cfg == PxD6AngularDriveConfig::eSLERP) ok |= (t == 5);
    else ok |= (t == 3 || t == 4 || t == 5);
    if (!ok) { applied["PxD6JointDrive(설정상 불가, 건너뜀)"]++; return; }
    r.joint->setDrive(PxD6Drive::Enum(t), r.v);
    applied["PxD6Joint.setDrive"]++;
  }

  void register_name(Obj& o, PxActor* a) {
    if (auto* ra = a->is<PxRigidActor>()) actor_by_name[o.name] = ra;
  }

  // ------------------------------------------------------------------ 목록 add/remove (구조 변경)
  size_t cur_event = 0;  // run() 이 지금 처리 중인 명령 번호
  PxAggregate* pending_agg = nullptr;  // 만들었지만 아직 장면에 안 들어간 묶음 (관절체 구성원 복원용)
  void list_op(const ovd::Event& e, bool add) {
    const ovd::AttrInfo& ai = F.attrs[e.attr];
    const std::string key = F.classes[ai.cls].name + "." + ai.name;
    uint64_t item = 0;
    if (e.data_len >= 8) memcpy(&item, F.data(e), 8);
    Obj& owner = objs[e.obj];
    auto itm = objs.find(item);
    if (key.rfind("PxPhysics.", 0) == 0) {
      if (!add || itm == objs.end()) return;
      Obj& o = itm->second;
      if (o.done) return;
      const std::string list = ai.name;
      if (list == "scenes") make_scene(o);
      else if (list == "materials") make_material(o);
      else if (list == "convexMeshes") make_convex(o);
      else if (list == "triangleMeshes") make_trimesh(o);
      else if (list == "shapes") make_shape(o);
      else if (list == "rigidDynamics" || list == "rigidStatics") make_actor(o);
      else if (list == "aggregates") { make_aggregate(o); pending_agg = static_cast<PxAggregate*>(o.px); }
      else if (list == "articulations") make_articulation(o);
      else if (list == "constraints") { return; }  // 조인트가 만든다
      else { unsupported["physics list:" + list]++; }
      o.done = true;
      return;
    }
    if (key == "PxArticulationReducedCoordinate.links") { if (add) make_link(e.obj, item); return; }
    if (itm == objs.end()) {
      if (key == "PxScene.constraints") return;  // 조인트가 지워진 뒤 오는 제거 기록
      unsupported["list item unknown:" + key]++;
      return;
    }
    Obj& o = itm->second;
    if (key == "PxRigidActor.shapes") {
      auto* a = owner.px ? owner.px->is<PxRigidActor>() : nullptr;
      auto* s = o.px ? o.px->is<PxShape>() : nullptr;
      if (!a || !s) { unsupported["attach missing"]++; return; }
      if (add) a->attachShape(*s); else a->detachShape(*s);
      if (add && !trace_sub.empty()) shape_owner[gnow(item)] = gnow(e.obj);
      filters_dirty = true;
      return;
    }
    if (key == "PxScene.actors") {
      PxScene* sc = owner.scene;
      auto* a = o.px ? o.px->is<PxActor>() : nullptr;
      if (!sc && !add) return;  // 장면을 지운 뒤 따라오는 제거 기록
      if (!sc || !a) { unsupported[std::string("scene actor missing:") + (sc ? "" : "noscene ") + (a ? "" : cname(o.cls))]++; return; }
      if (add) { if (!a->getScene() && !a->getAggregate()) sc->addActor(*a); }
      else if (a->getScene() && !a->getAggregate()) sc->removeActor(*a);
      return;
    }
    if (key == "PxScene.articulations") {
      auto* a = o.px ? o.px->is<PxArticulationReducedCoordinate>() : nullptr;
      if (!owner.scene || !a) return;
      // OVD 는 묶음 구성원(addArticulation)을 남기지 않는다. omni.physx 는 관절체마다 묶음 하나를 만들어 관절체를 넣은 뒤 장면에 넣는다
      // (omni.physx UsdInterface.cpp:5942 addArticulationToScene, selfCollision = physxArticulation:enabledSelfCollisions).
      // 기록 순서: 묶음 생성 -> PxPhysics.aggregates -> PxScene.articulations(이 관절체) -> PxScene.aggregates.
      // 그래서 아직 장면에 안 들어간 빈 묶음이 있으면 관절체를 그 묶음에 넣는다(묶음이 장면에 들어갈 때 함께 들어간다).
      if (add && pending_agg && pending_agg->getNbActors() == 0 && !a->getScene() && !a->getAggregate()) {
        pending_agg->addArticulation(*a);
        applied["agg:관절체를 묶음에"]++;
        return;
      }
      if (add) { if (!a->getScene() && !a->getAggregate()) owner.scene->addArticulation(*a); }
      else if (a->getScene() && !a->getAggregate()) owner.scene->removeArticulation(*a);
      return;
    }
    if (key == "PxScene.aggregates") {
      auto* g = static_cast<PxAggregate*>(o.px);
      if (!owner.scene || !g) { unsupported["agg:장면/묶음 없음"]++; return; }
      if (add) owner.scene->addAggregate(*g); else owner.scene->removeAggregate(*g);
      if (add && pending_agg == g) pending_agg = nullptr;
      return;
    }
    if (key == "PxAggregate.actors") {
      auto* g = static_cast<PxAggregate*>(owner.px);
      if (!g) { unsupported["agg:묶음 객체 없음"]++; return; }
      if (!o.px) { unsupported["agg:구성원 객체 없음 " + std::string(cname(o.cls))]++; return; }
      applied[std::string("agg:") + (add ? "넣기 " : "빼기 ") + cname(o.cls)]++;
      if (auto* art = o.px->is<PxArticulationReducedCoordinate>()) { if (add) g->addArticulation(*art); else g->removeArticulation(*art); return; }
      if (auto* l = o.px->is<PxArticulationLink>()) {  // 관절체를 묶음에 넣으면 링크마다 기록된다 -> 첫 링크에서 관절체째로
        PxArticulationReducedCoordinate& art = l->getArticulation();
        if (add && !art.getAggregate()) g->addArticulation(art);
        if (!add && art.getAggregate()) g->removeArticulation(art);
        return;
      }
      if (auto* a = o.px->is<PxActor>()) { if (add) g->addActor(*a); else g->removeActor(*a); }
      return;
    }
    if (key == "PxScene.constraints") return;  // 조인트 액터가 장면에 들어가면 자동
    unsupported["list:" + key]++;
  }

  void destroy(uint64_t h) {
    auto it = objs.find(h);
    if (it == objs.end()) return;
    Obj& o = it->second;
    std::string c = cname(o.cls);
    if (o.px) {
      if (c == "PxArticulationLink" || c == "PxArticulationJointReducedCoordinate" || c == "PxConstraint") {}  // 주인이 지운다
      else if (F.is_a(o.cls, F.cls("PxJoint"))) static_cast<PxJoint*>(o.px)->release();
      else if (c == "PxArticulationMimicJoint") static_cast<PxArticulationMimicJoint*>(o.px)->release();
      else o.px->release();
    }
    if (o.scene) o.scene->release();
    objs.erase(it);
  }

  // ------------------------------------------------------------------ sidelog 적용 (OVD 에 없는 호출)
  PxArticulationCache* cache_of(PxArticulationReducedCoordinate* a) {
    auto it = caches.find(a);
    if (it != caches.end()) return it->second;
    PxArticulationCache* c = a->createCache();
    caches[a] = c;
    return c;
  }

  void apply_side(const engine::SideCall& c) {
    const engine::SideView& v = sviews[c.view];
    const std::string& m = c.method;
    auto arti = [&](uint32_t i) -> PxArticulationReducedCoordinate* {
      if (i >= v.prims.size()) return nullptr;
      auto ia = art_by_name.find(v.prims[i]);
      if (ia != art_by_name.end()) return ia->second;
      auto il = actor_by_name.find(v.prims[i]);  // 강체 뷰의 루트 링크
      if (il != actor_by_name.end()) if (auto* l = il->second->is<PxArticulationLink>()) return &l->getArticulation();
      return nullptr;
    };
    std::vector<uint32_t> idx = c.idx;
    if (idx.empty()) for (uint32_t i = 0; i < v.prims.size(); ++i) idx.push_back(i);
    for (uint32_t i : idx) {
      if (m == "set_dof_positions" || m == "set_dof_velocities" || m == "set_dof_actuation_forces") {
        auto* a = arti(i);
        if (!a) { unsupported["side:no art " + (i < v.prims.size() ? v.prims[i] : std::string("?"))]++; continue; }
        PxArticulationCache* ca = cache_of(a);
        const PxU32 nd = a->getDofs();
        PxReal* dst = m == "set_dof_positions" ? ca->jointPosition : (m == "set_dof_velocities" ? ca->jointVelocity : ca->jointForce);
        for (PxU32 j = 0; j < nd; ++j) {
          float s = (i < v.signs.size() && j < v.signs[i].size() && v.signs[i][j] < 0) ? -1.0f : 1.0f;
          float x = c.data[size_t(i) * v.max_dofs + j];
          dst[j] = s < 0 ? -x : x;
        }
        a->applyCache(*ca, m == "set_dof_positions" ? PxArticulationCacheFlag::ePOSITION
                                                   : (m == "set_dof_velocities" ? PxArticulationCacheFlag::eVELOCITY : PxArticulationCacheFlag::eFORCE));
        if (dbg_drive && c.after >= 40 && c.after <= 44 && v.prims[i].find("r1pro") != std::string::npos)
          fprintf(stderr, "[side] after %llu %s 값0 %.9g 값1 %.9g (dof %u, 뷰 max_dofs %u)\n", (unsigned long long)c.after, m.c_str(), c.data[size_t(i) * v.max_dofs],
                  c.data[size_t(i) * v.max_dofs + 1], nd, v.max_dofs);
        applied["side:" + m]++;
      } else if (m == "set_root_transforms" || (m == "set_transforms" && v.kind == 1)) {
        const float* s = &c.data[size_t(i) * 7];
        PxTransform t(PxVec3(s[0], s[1], s[2]), PxQuat(s[3], s[4], s[5], s[6]));
        auto* a = arti(i);
        if (a) { PxArticulationCache* ca = cache_of(a); ca->rootLinkData->transform = t; a->applyCache(*ca, PxArticulationCacheFlag::eROOT_TRANSFORM); applied["side:" + m]++; }
        // 강체 동체는 setGlobalPose 로 OVD 에 이미 남는다
      } else if (m == "set_root_velocities" || (m == "set_velocities" && v.kind == 1)) {
        const float* s = &c.data[size_t(i) * 6];
        auto* a = arti(i);
        if (a) {
          PxArticulationCache* ca = cache_of(a);
          ca->rootLinkData->worldLinVel = PxVec3(s[0], s[1], s[2]);
          ca->rootLinkData->worldAngVel = PxVec3(s[3], s[4], s[5]);
          a->applyCache(*ca, PxArticulationCacheFlag::eROOT_VELOCITIES);
          applied["side:" + m]++;
        }
      } else if (m == "add_force") {  // 강체 동체에 힘 (PxForceMode::eFORCE, 자동 깨우기)
        auto ib = actor_by_name.find(v.prims[i]);
        if (ib != actor_by_name.end()) if (auto* rd = ib->second->is<PxRigidDynamic>()) {
          const float* s3 = &c.data[size_t(i) * 3];
          rd->addForce(PxVec3(s3[0], s3[1], s3[2]));
          applied["side:add_force"]++;
        }
      } else if (m == "wake_up" || m == "put_to_sleep") {
        const std::string& path = v.prims[i];
        auto ia = art_by_name.find(path);
        if (ia != art_by_name.end()) { if (m == "wake_up") ia->second->wakeUp(); else ia->second->putToSleep(); applied["side:" + m]++; continue; }
        auto ib = actor_by_name.find(path);
        if (ib != actor_by_name.end()) if (auto* rd = ib->second->is<PxRigidDynamic>()) { if (m == "wake_up") rd->wakeUp(); else rd->putToSleep(); applied["side:" + m]++; }
      } else {
        // 목표값(드라이브)·질량 등은 OVD 에 이미 남는다
        applied["side-skip:" + m]++;
      }
    }
  }

  void apply_side_until(uint64_t after) {
    while (scall_next < scalls.size() && scalls[scall_next].after <= after) {
      if (side_skip.count(scall_next)) { scall_next++; applied["side:건너뜀(--side-skip)"]++; continue; }
      apply_side(scalls[scall_next++]);
    }
  }

  // ------------------------------------------------------------------ 거르개: resetFiltering 따라 하기 + 경로 기반 표 풀기
  // omni.physx 는 장면 안 액터의 모양 filterData / 모양 플래그를 바꾸면 resetFiltering(actor) 를 부른다
  // (InternalFilteredPairs.cpp:159, InternalActor.cpp:258~, PhysXCollisionPropertiesUpdate.cpp:52). 이 호출은 OVD 에 안 남는다.
  bool refilter_on = true;
  std::vector<PxRigidActor*> refilter;
  bool filters_dirty = true;
  void mark_refilter(PxShape* s) {
    filters_dirty = true;
    PxRigidActor* a = s->getActor();
    if (refilter_on && a && a->getScene()) refilter.push_back(a);
  }
  void do_refilter() {
    std::set<PxRigidActor*> done;
    for (PxRigidActor* a : refilter)
      if (done.insert(a).second && a->getScene()) { a->getScene()->resetFiltering(*a); applied["refilter"]++; }
    refilter.clear();
  }
  std::vector<PxShape*> shapes_of_prim(const std::string& path) {
    std::vector<PxShape*> out;
    auto add_actor = [&](PxRigidActor* a) {
      PxU32 n = a->getNbShapes();
      size_t b = out.size();
      out.resize(b + n);
      a->getShapes(out.data() + b, n);
    };
    const std::string pre = path + "/";
    bool any = false;
    for (auto& kv : actor_by_name)
      if (kv.first == path || kv.first.compare(0, pre.size(), pre) == 0) { add_actor(kv.second); any = true; }
    if (!any) {  // 충돌체 prim 이면 그것을 품은 강체(이름이 경로의 앞부분인 가장 긴 액터)
      size_t best = 0;
      PxRigidActor* ba = nullptr;
      for (auto& kv : actor_by_name)
        if (kv.first.size() > best && path.compare(0, kv.first.size() + 1, kv.first + "/") == 0) { best = kv.first.size(); ba = kv.second; }
      if (ba) add_actor(ba);
    }
    return out;
  }
  // omni.physx 의 거른 쌍 표(PhysXSetup::mFilteredPairs)는 프로세스 전체에 하나이고 커지기만 한다: handleFilteringPair 가
  // 모양 포인터 해시(word1)로 쌍을 넣고(InternalFilteredPairs.cpp:137), PxPhysics 를 다시 만들어도 비우지 않는다.
  // 새 PhysX 인스턴스가 옛 주소를 재사용하면 옛 쌍이 새 모양 쌍을 우연히 거른다. 그래서 같은 기록의 모든 OVD 파일에서
  // (관계 prim 쌍 x 그 파일에서 그 prim 모양에 붙었던 모든 word1) 을 모아 표에 더한다.  --filter-history <ovd> [...]
  size_t add_filter_history(const std::string& path) {
    ovd::File H;
    std::string err;
    if (!ovd::load(path, H, err)) { fprintf(stderr, "filter-history 를 못 읽음: %s\n", path.c_str()); return 0; }
    uint32_t a_name = 0, a_sfd = 0, a_shapes = 0;
    for (auto& kv : H.attrs) {
      const std::string& c = H.classes[kv.second.cls].name;
      if (c == "PxActor" && kv.second.name == "name") a_name = kv.first;
      if (c == "PxShape" && kv.second.name == "simulationFilterData") a_sfd = kv.first;
      if (c == "PxRigidActor" && kv.second.name == "shapes") a_shapes = kv.first;
    }
    std::unordered_map<uint64_t, std::string> actor_name;             // 액터 핸들 -> 이름 (핸들 재사용은 이름 덮어쓰기로 충분)
    std::unordered_map<uint64_t, uint64_t> shape_actor;               // 모양 -> 액터
    std::vector<std::pair<uint64_t, uint32_t>> w1;                    // (모양, word1) 모두
    for (const ovd::Event& e : H.events) {
      if (e.cmd == ovd::kSet && e.attr == a_name) actor_name[e.obj] = H.str(e);
      else if (e.cmd == ovd::kAddToList && e.attr == a_shapes && e.data_len >= 8) { uint64_t s; memcpy(&s, H.data(e), 8); shape_actor[s] = e.obj; }
      else if (e.cmd == ovd::kSet && e.attr == a_sfd && e.data_len >= 16) { uint32_t w[4]; memcpy(w, H.data(e), 16); if (w[1]) w1.emplace_back(e.obj, w[1]); }
    }
    // 이름 -> word1 들
    std::unordered_map<std::string, std::set<uint32_t>> by_name;
    for (auto& p : w1) {
      auto ia = shape_actor.find(p.first);
      if (ia == shape_actor.end()) continue;
      auto in = actor_name.find(ia->second);
      if (in != actor_name.end()) by_name[in->second].insert(p.second);
    }
    auto of_prim = [&](const std::string& prim) {  // shapes_of_prim 과 같은 규칙: 이름 = prim 또는 prim/ 아래, 없으면 prim 을 품은 가장 긴 액터
      std::set<uint32_t> out;
      const std::string pre = prim + "/";
      bool any = false;
      for (auto& kv : by_name)
        if (kv.first == prim || kv.first.compare(0, pre.size(), pre) == 0) { out.insert(kv.second.begin(), kv.second.end()); any = true; }
      if (!any) {
        const std::string* best = nullptr;
        for (auto& kv : by_name)
          if ((!best || kv.first.size() > best->size()) && prim.compare(0, kv.first.size() + 1, kv.first + "/") == 0) best = &kv.first;
        if (best) out = by_name[*best];
      }
      return out;
    };
    const size_t before = spec.filtered_pairs.size();
    for (auto& r : spec.rels) {
      const auto a = of_prim(r.first), b = of_prim(r.second);
      for (uint32_t x : a) for (uint32_t y : b) spec.filtered_pairs.insert(engine::pair_key(x, y));
    }
    return spec.filtered_pairs.size() - before;
  }
  void resolve_filters() {
    filters_dirty = false;
    if (spec.groups.empty() && spec.rels.empty()) return;
    std::unordered_map<std::string, std::set<uint32_t>> gid;
    for (auto& g : spec.groups)
      for (auto& inc : g.includes)
        for (PxShape* s : shapes_of_prim(inc)) { uint32_t w2 = s->getSimulationFilterData().word2; if (w2) gid[g.path].insert(w2); }
    const size_t before = spec.group_pairs.size();
    for (auto& g : spec.groups)
      for (auto& fpath : g.filtered)
        for (uint32_t a : gid[g.path]) for (uint32_t b : gid[fpath]) spec.group_pairs.insert(engine::pair_key(a, b));
    const size_t before_p = spec.filtered_pairs.size();
    for (auto& r : spec.rels) {
      auto sa = shapes_of_prim(r.first), sb = shapes_of_prim(r.second);
      for (PxShape* x : sa)
        for (PxShape* y : sb) {
          uint32_t a = x->getSimulationFilterData().word1, b = y->getSimulationFilterData().word1;
          if (a && b) spec.filtered_pairs.insert(engine::pair_key(a, b));
        }
    }
    if (verbose)
      fprintf(stderr, "[filter] sim %llu: group pairs %zu(+%zu), filtered pairs %zu(+%zu)\n", (unsigned long long)sims,
              spec.group_pairs.size(), spec.group_pairs.size() - before, spec.filtered_pairs.size(),
              spec.filtered_pairs.size() - before_p);
  }

  // ------------------------------------------------------------------ 비교
  int shown = 0;
  std::string cur_obj;
  std::map<std::string, int64_t> obj_first_diff;  // 물체(/World/scene_0/<이름>) -> 첫 다른 simulate
  void cmp(const std::string& what, const float* ours, const uint8_t* rec, int n) {
    Stat& s = stats[what];
    if (verbose && shown < verbose_max && memcmp(ours, rec, 4 * n)) {
      shown++;
      fprintf(stderr, "[다름] sim %" PRIu64 " %s %s\n   우리:", sims, what.c_str(), cur_obj.c_str());
      for (int i = 0; i < n; ++i) fprintf(stderr, " %.9g", ours[i]);
      fprintf(stderr, "\n   공식:");
      for (int i = 0; i < n; ++i) { float r; memcpy(&r, rec + 4 * i, 4); fprintf(stderr, " %.9g", r); }
      fprintf(stderr, "\n");
    }
    double m = 0;
    bool bit = memcmp(ours, rec, 4 * n) != 0;
    for (int i = 0; i < n; ++i) { float r; memcpy(&r, rec + 4 * i, 4); double d = std::fabs(double(ours[i]) - double(r)); if (d > m || std::isnan(d)) m = std::isnan(d) ? INFINITY : d; }
    s.compared++;
    frame_cmp++;
    if (bit) {
      s.bitdiff++;
      frame_bitdiff++;
      {  // 객체별 첫 다름 (관절 조인트는 자식 링크 이름으로 이미 cur_obj 에 들어 있다)
        std::string key = cur_obj;
        const size_t cut = key.find("/", key.find("/World/scene_0/") == 0 ? 15 : 0);  // 물체 단위: /World/scene_0/<물체>
        if (key.rfind("joint->", 0) == 0) key = key.substr(7);
        const size_t s0 = key.find("/World/scene_0/");
        if (s0 != std::string::npos) { const size_t e = key.find('/', s0 + 15); key = key.substr(s0, e == std::string::npos ? std::string::npos : e - s0); }
        (void)cut;
        if (!obj_first_diff.count(key)) obj_first_diff[key] = int64_t(sims);
      }
      if (s.first_bit < 0) s.first_bit = int64_t(sims);
      if (first_div_frame < 0) { first_div_frame = int64_t(sims); first_div_what = what; }
    }
    if (m > s.maxdiff) s.maxdiff = m;
    if (m > frame_max) frame_max = m;
  }

  void compare_output(const ovd::Event& e) {
    const ovd::AttrInfo& ai = F.attrs[e.attr];
    const std::string& an = ai.name;
    auto it = objs.find(e.obj);
    if (it == objs.end() || !it->second.px) return;
    PxBase* b = it->second.px;
    const std::string cl = cname(it->second.cls);
    cur_obj = it->second.name;
    if (an == "globalPose" && e.data_len == 28) {
      if (auto* a = b->is<PxRigidActor>()) { PxTransform t = a->getGlobalPose(); cmp(cl + ".pose", &t.q.x, F.data(e), 7); }
    } else if ((an == "linearVelocity" || an == "angularVelocity") && e.data_len == 12) {
      if (auto* rb = b->is<PxRigidBody>()) { PxVec3 v = an == "linearVelocity" ? rb->getLinearVelocity() : rb->getAngularVelocity(); cmp(cl + "." + an, &v.x, F.data(e), 3); }
    } else if ((an == "jointPosition" || an == "jointVelocity") && e.data_len == 24) {
      auto* j = static_cast<PxArticulationJointReducedCoordinate*>(b);
      cur_obj = std::string("joint->") + (j->getChildArticulationLink().getName() ? j->getChildArticulationLink().getName() : "?");
      if (verbose && shown < 12) {  // 다를 때 조인트 설정을 함께 본다 (한계·드라이브·운동)
        static std::set<const void*> dumped;
        float v0[6];
        for (int i = 0; i < 6; ++i) v0[i] = an == "jointPosition" ? j->getJointPosition(PxArticulationAxis::Enum(i)) : j->getJointVelocity(PxArticulationAxis::Enum(i));
        if (memcmp(v0, F.data(e), 24) && dumped.insert(j).second) {
          fprintf(stderr, "[조인트] %s 종류=%d\n", cur_obj.c_str(), int(j->getJointType()));
          for (int i = 0; i < 6; ++i) {
            auto ax = PxArticulationAxis::Enum(i);
            if (j->getMotion(ax) == PxArticulationMotion::eLOCKED) continue;
            PxArticulationLimit L = j->getLimitParams(ax);
            PxArticulationDrive D = j->getDriveParams(ax);
            fprintf(stderr, "   축%d 운동=%d 한계[%.9g, %.9g] 드라이브 k=%.9g c=%.9g max=%.9g 종류=%d 목표=%.9g 목표속도=%.9g 아머처=%.9g 최대속도=%.9g\n", i,
                    int(j->getMotion(ax)), L.low, L.high, D.stiffness, D.damping, D.maxForce, int(D.driveType), j->getDriveTarget(ax),
                    j->getDriveVelocity(ax), j->getArmature(ax), j->getMaxJointVelocity(ax));
          }
        }
      }
      float v[6];
      for (int i = 0; i < 6; ++i) v[i] = an == "jointPosition" ? j->getJointPosition(PxArticulationAxis::Enum(i)) : j->getJointVelocity(PxArticulationAxis::Enum(i));
      cmp("joint." + an, v, F.data(e), 6);
    } else if (an == "wakeCounter" && e.data_len == 4) {
      float w;
      if (auto* rd = b->is<PxRigidDynamic>()) w = rd->getWakeCounter();
      else if (auto* a = b->is<PxArticulationReducedCoordinate>()) w = a->getWakeCounter();
      else return;
      cmp(cl + ".wakeCounter", &w, F.data(e), 1);
    } else if (an == "worldBounds" && e.data_len == 24) {
      PxBounds3 bb;
      if (auto* a = b->is<PxActor>()) bb = a->getWorldBounds();
      else if (auto* ar = b->is<PxArticulationReducedCoordinate>()) bb = ar->getWorldBounds();
      else return;
      cmp("worldBounds", &bb.minimum.x, F.data(e), 6);
    }
  }

  void end_frame() {
    if (csv) fprintf(csv, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.9g\n", sims, frame_cmp, frame_bitdiff, frame_max);
    frame_max = 0; frame_bitdiff = 0; frame_cmp = 0;
  }

  // ------------------------------------------------------------------ 본 반복
  // 이어 돌기(엔진 바인딩 engine_capi.cpp): 사건 번호·묶음 상태를 멤버로 두고 run() 은 max_frames 에서 멈췄다가 다시 부르면 잇는다
  uint64_t run_open = 0;
  bool run_out_block = false;
  uint64_t run_out_ctx = 0;
  size_t run_i = 0;
  PxScene* last_scene = nullptr;
  float last_dt = 1.0f / 120.0f;
  void run() {
    uint64_t& open = run_open;         // 만들 때 값을 모으는 중인 객체
    bool& out_block = run_out_block;   // simulate 뒤 결과 구간
    uint64_t& out_ctx = run_out_ctx;
    const uint32_t a_elapsed = attr("PxScene", "elapsedTime");
    for (size_t& i = run_i; i < F.events.size(); ++i) {
      const ovd::Event& e = F.events[i];
      // 만든 순간 값 묶음(cluster): create 바로 뒤, 같은 객체의 set 이 이어지는 동안. 같은 속성이 두 번 나오면
      // 두 번째부터는 API 호출이다 (생성 기록은 속성마다 한 번씩만 쓴다: OmniPvdPxSampler::stream*).
      if (e.cmd == ovd::kSet) trace_event(e, (e.obj == open && !objs[e.obj].done && !objs[e.obj].pend.count(e.attr)) ? "생성값" : "set");
      if (e.cmd == ovd::kSet && e.obj == open && !objs[e.obj].done && !objs[e.obj].pend.count(e.attr)) {
        objs[e.obj].pend[e.attr].assign(F.data(e), F.data(e) + e.data_len);
        continue;
      }
      if (open) { close_cluster(open); open = 0; }
      if (in_free(out_block) && e.cmd != ovd::kStopFrame && !(e.cmd == ovd::kSet && e.attr == a_elapsed)) {
        std::string what = e.cmd == ovd::kSet ? std::string("set:") + cname(objs.count(e.obj) ? objs[e.obj].cls : 0) + "." + F.attrs[e.attr].name
                         : e.cmd == ovd::kCreate ? std::string("create:") + cname(e.cls) : std::string("cmd:") + std::to_string(int(e.cmd));
        C.free_skipped[what]++;
        if (e.cmd == ovd::kSet || e.cmd == ovd::kAddToList || e.cmd == ovd::kRemoveFromList) continue;  // 생성·파괴는 넣는다(객체 대응 유지, 개수만 보고)
        if ((AGF.post >= 0 || AG.on) && e.cmd == ovd::kCreate && is_ext_joint_cls(e.cls)) { C.free_skipped["AG 관절 생성(엔진이 만듦)"]++; skip_objs.insert(e.obj); continue; }
        // 파괴: 에피소드 동안 OVD 의 파괴는 잡기 관절(엔진이 한다) 아니면 기록 끝의 장면 해체(Kit 종료)뿐이다 -> 넣지 않는다
        if (e.cmd == ovd::kDestroy) { C.free_skipped["파괴(넣지 않음)"]++; continue; }
      }
      switch (e.cmd) {
        case ovd::kCreate: {
          gen_run[e.obj]++;
          if (sims > 0 && is_ext_joint_cls(e.cls)) C.joint_event = true;
          Obj o;
          o.cls = e.cls;
          o.name = F.str(e);
          objs[e.obj] = o;
          open = e.obj;
          break;
        }
        case ovd::kDestroy:
          if (sims > 0) { auto io = objs.find(e.obj); if (io != objs.end() && is_ext_joint_cls(io->second.cls)) C.joint_event = true; }
          destroy(e.obj);
          break;
        case ovd::kAddToList:
        case ovd::kRemoveFromList:
          if (out_block) break;
          cur_event = i;
          list_op(e, e.cmd == ovd::kAddToList);
          break;
        case ovd::kSet: {
          if (e.attr == a_elapsed) {
            Obj& so = objs[e.obj];
            if (!so.scene) { unsupported["simulate without scene"]++; break; }
            apply_side_until(sims + side_offset);  // 곁기록 번호는 프로세스 전체 simulate 수 (앞선 PxPhysics 의 OVD 파일 몫 = side_offset)
            do_refilter();
            if (filters_dirty) resolve_filters();
            float dt; memcpy(&dt, F.data(e), 4);
            if (sims == 0 && !trace_sub.empty()) dump_state_before_first_sim();
            if (!gravity_off.empty() && !gravity_applied) apply_gravity_off();
            if (int64_t(sims + side_offset + 1) == dump_art_at && !dump_art_name.empty()) dump_art();
            ctrl_before_simulate();
            so.scene->simulate(dt);
            so.scene->fetchResults(true);
            last_scene = so.scene;
            last_dt = dt;
            sims++;
            s3_after_simulate();
            out_block = true;
            out_ctx = e.obj;  // 프레임 명령의 문맥 = 장면 핸들 (OmniPvdPxSampler.cpp:101)
            break;
          }
          if (out_block) { compare_output(e); break; }
          auto it = objs.find(e.obj);
          if (it == objs.end()) { unsupported["set unknown obj"]++; break; }
          if (!it->second.done) { it->second.pend[e.attr].assign(F.data(e), F.data(e) + e.data_len); break; }
          // API 한 번이 속성 여러 개를 남기는 경우(묶음)는 한 번의 호출로 되돌린다
          const std::vector<std::string>* grp = group_of(it->second, e.attr);
          if (grp) {
            std::unordered_map<std::string, const ovd::Event*> vals;
            size_t k = i;
            while (k < F.events.size()) {
              const ovd::Event& x = F.events[k];
              if (x.cmd != ovd::kSet || x.obj != e.obj) break;
              const std::string& xn = F.attrs[x.attr].name;
              bool in = false;
              for (auto& g : *grp) in |= (g == xn);
              if (!in || vals.count(xn)) break;
              vals[xn] = &x;
              ++k;
            }
            apply_group(it->second, *grp, vals);
            i = k - 1;
            break;
          }
          apply_set(it->second, e.attr, F.data(e), e.data_len);
          break;
        }
        case ovd::kStopFrame:
          if (out_block && e.ctx == out_ctx) {
            out_block = false;
            end_frame();
            if (max_frames >= 0 && int64_t(sims) >= max_frames) return;
          }
          break;
        default:
          break;
      }
    }
  }

  // simulate 수가 target 이 될 때까지 (프레임 끝에서 멈춤). OVD 사건이 다 떨어지면 제어기 입력만으로 직접 simulate 한다(자유 실행).
  void run_until(int64_t target) {
    max_frames = target;
    run();
    while (run_i >= F.events.size() && last_scene && int64_t(sims) < target) {
      apply_side_until(sims + side_offset);
      ctrl_before_simulate();
      last_scene->simulate(last_dt);
      last_scene->fetchResults(true);
      sims++;
      s3_after_simulate();
    }
  }
  // 바깥(파이썬 정책)에서 온 행동 한 스텝을 붙인다 (--ctrl 설정의 기록 행동 대신)
  void push_action(const float* a) {
    C.acts.insert(C.acts.end(), a, a + C.A);
    C.T++;
  }

  // 묶음 점검: OVD 조인트마다 묶인 PhysX 조인트의 자식 링크 이름 = OVD 가 말하는 자식 링크 이름인가 (지금 살아 있는 객체만)
  void check_joint_binding() {
    uint64_t n = 0, bad = 0, unbound = 0;
    for (auto& kv : objs) {
      const Obj& o = kv.second;
      if (cname(o.cls) != std::string("PxArticulationJointReducedCoordinate")) continue;
      auto ic = joint_child.find(gnow(kv.first));
      if (ic == joint_child.end()) continue;
      const std::string want = gname_of(ic->second);
      if (!o.px) { unbound++; continue; }
      const char* got = static_cast<PxArticulationJointReducedCoordinate*>(o.px)->getChildArticulationLink().getName();
      n++;
      auto ip = joint_parent.find(gnow(kv.first));
      if (ip != joint_parent.end()) {
        const std::string wantp = gname_of(ip->second);
        const char* gotp = static_cast<PxArticulationJointReducedCoordinate*>(o.px)->getParentArticulationLink().getName();
        if (wantp != (gotp ? gotp : "") && bad++ < 5)
          fprintf(stderr, "[부모 틀림] 자식 %s: OVD 부모 %s <-> 우리 부모 %s\n", want.c_str(), wantp.c_str(), gotp ? gotp : "(없음)");
      }
      if (want != (got ? got : "")) {
        if (bad++ < 5) fprintf(stderr, "[묶음 틀림] OVD 자식 %s <-> 묶인 PhysX 자식 %s\n", want.c_str(), got ? got : "(없음)");
      }
    }
    printf("조인트 묶음 점검: %" PRIu64 " 개 중 틀림 %" PRIu64 ", 안 묶임 %" PRIu64 "\n", n, bad, unbound);
  }

  void report() {
    check_joint_binding();
    {
      std::vector<std::pair<int64_t, std::string>> v;
      for (auto& kv : obj_first_diff) v.emplace_back(kv.second, kv.first);
      std::sort(v.begin(), v.end());
      printf("물체별 첫 다른 simulate (%zu 물체):\n", v.size());
      for (size_t i = 0; i < v.size() && i < 25; ++i) printf("  %6" PRId64 "  %s\n", v[i].first, v[i].second.c_str());
    }
    report_approx_users();
    printf("\n== 재생 결과 ==\nsimulate %" PRIu64 " 번, 볼록 메시 원본 %" PRIu64 " / 근사 %" PRIu64 ", 정규화 원상 %" PRIu64
           " 번 중 못 찾음 %" PRIu64 "\n", sims, convex_exact, convex_approx, g_prenorm_calls, g_prenorm_fail);
    printf("첫 비트 불일치: %s\n", first_div_frame < 0 ? "없음 (전 구간 비트 동일)" : (std::to_string(first_div_frame) + " 번째 simulate, " + first_div_what).c_str());
    printf("%-44s %10s %10s %12s %10s\n", "항목", "비교 수", "비트 다름", "최대|차|", "첫 다름");
    for (auto& kv : stats)
      printf("%-44s %10" PRIu64 " %10" PRIu64 " %12.3e %10" PRId64 "\n", kv.first.c_str(), kv.second.compared, kv.second.bitdiff, kv.second.maxdiff, kv.second.first_bit);
    if (!unsupported.empty()) {
      printf("\n지원 안 한 것 (재생에서 빠짐 -> 불일치 원인 후보)\n");
      for (auto& kv : unsupported) printf("  %8" PRIu64 "  %s\n", kv.second, kv.first.c_str());
    }
    if (verbose) {
      printf("\n적용한 호출\n");
      for (auto& kv : applied) printf("  %8" PRIu64 "  %s\n", kv.second, kv.first.c_str());
    }
  }
};

#ifndef OVD_REPLAY_NO_MAIN
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ovd_replay <file.ovd> [--convex f] [--filters f] [--sidelog f] [--threads N] [--csv f] [--max-frames N] [--verbose]\n");
    return 2;
  }
  ovd::File F;
  std::string err;
  if (!ovd::load(argv[1], F, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  Replayer R(F);
  int threads = 4;
  std::string convex, filters, sidelog, csv;
  std::vector<std::string> filter_hist;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--convex" && i + 1 < argc) convex = argv[++i];
    else if (a == "--filters" && i + 1 < argc) filters = argv[++i];
    else if (a == "--sidelog" && i + 1 < argc) sidelog = argv[++i];
    else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
    else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    else if (a == "--max-frames" && i + 1 < argc) R.max_frames = atoll(argv[++i]);
    else if (a == "--verbose") R.verbose = true;
    else if (a == "--verbose-max" && i + 1 < argc) R.verbose_max = atoi(argv[++i]);
    else if (a == "--record" && i + 1 < argc) R.record_path = argv[++i];
    else if (a == "--no-refilter") R.refilter_on = false;
    else if (a == "--trace-obj" && i + 1 < argc) R.trace_sub = argv[++i];
    else if (a == "--diag-no-self-collision") R.diag_no_self_collision = true;
    else if (a == "--contact-report-all") R.contact_report_all = true;
    else if (a == "--side-skip" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) R.side_skip.insert(size_t(atoll(t.c_str()))); }
    else if (a == "--gravity-off" && i + 1 < argc) { std::ifstream gf(argv[++i]); std::string ln; while (std::getline(gf, ln)) if (!ln.empty()) R.gravity_off.push_back(ln); }
    else if (a == "--ctrl" && i + 1 < argc) { if (!R.load_ctrl(argv[++i])) { fprintf(stderr, "--ctrl 입력을 못 읽음\n"); return 1; } }
    else if (a == "--free") R.C.free_run = true;
    else if (a == "--s3-own-contact") R.S3.own = true;
    else if (a == "--ag" && i + 1 < argc) { if (!R.load_ag(argv[++i])) { fprintf(stderr, "--ag 입력을 못 읽음 (export_ag.py)\n"); return 1; } }
    else if (a == "--ag-force" && i + 4 < argc) {
      R.AGF.post = atoll(argv[++i]); R.AGF.eef = argv[++i]; R.AGF.target = argv[++i];
      std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) R.AGF.fingers.push_back(t);
      if (i + 1 < argc && argv[i + 1][0] != '-') { std::stringstream s2(argv[++i]); for (int k = 0; k < 3 && std::getline(s2, t, ','); ++k) R.AGF.scale1[k] = strtof(t.c_str(), nullptr); }
    }
    else if (a == "--s3" && i + 1 < argc) { if (!R.load_s3(argv[++i])) { fprintf(stderr, "--s3 입력을 못 읽음\n"); return 1; } }
    else if (a == "--dump-art" && i + 2 < argc) { R.dump_art_name = argv[++i]; R.dump_art_file = argv[++i]; }
    else if (a == "--dump-art-at" && i + 1 < argc) R.dump_art_at = atoll(argv[++i]);
    else if (a == "--filter-history") { while (i + 1 < argc && argv[i + 1][0] != '-') filter_hist.push_back(argv[++i]); }
    else if (a == "--side-offset" && i + 1 < argc) R.side_offset = strtoull(argv[++i], nullptr, 10);
  }
  if (!convex.empty() && !engine::read_convex_bin(convex, R.convex)) fprintf(stderr, "convex 보조 파일을 못 읽음: %s\n", convex.c_str());
  if (!filters.empty() && !engine::read_filters(filters, R.spec)) fprintf(stderr, "filters 파일을 못 읽음: %s\n", filters.c_str());
  if (!sidelog.empty() && !engine::read_sidelog(sidelog, R.sviews, R.scalls)) fprintf(stderr, "sidelog 를 못 읽음: %s\n", sidelog.c_str());
  if (!csv.empty()) { R.csv = fopen(csv.c_str(), "w"); if (R.csv) fprintf(R.csv, "simulate,compared,bitdiff,maxabs\n"); }
  if (!R.init(threads)) return 1;
  if (R.dump_art_at < 0) R.dump_art_at = int64_t(R.side_offset) + 1;
  for (auto& h : filter_hist) printf("거른 쌍 역사 %s: +%zu 쌍\n", h.c_str(), R.add_filter_history(h));
  // 앞선 PhysX 인스턴스(omni.physx 는 stop/play 때 장면을 통째로 다시 만든다) 몫의 곁기록은 버린다:
  // 그 쓰기의 결과는 USD 에 되쓰여(updateToUsd) 이 OVD 의 생성 값에 이미 들어 있다.
  size_t side_dropped = 0;
  while (R.scall_next < R.scalls.size() && R.scalls[R.scall_next].after < R.side_offset) { R.scall_next++; side_dropped++; }
  if (R.side_offset) printf("곁기록: 앞선 인스턴스 몫 %zu 건 버림 (post < %" PRIu64 ")\n", side_dropped, R.side_offset);
  R.run();
  if (!R.trace_sub.empty()) R.trace_flush();
  R.s3_report();
  if (R.C.free_run) { printf("자유 실행(--free): 에피소드 동안 넣지 않은 OVD 입력\n"); for (auto& kv : R.C.free_skipped) printf("  %8" PRIu64 "  %s\n", kv.second, kv.first.c_str()); }
  if (R.C.on && R.C.dropped_flush) printf("닫힌 고리: 잡기 조인트 생성·해제로 목표를 안 쓴 서브스텝 %" PRIu64 "\n", R.C.dropped_flush);
  if (R.C.on) printf("닫힌 고리(--ctrl): 제어기 스텝 %" PRIu64 ", 건너뛴 OVD 드라이브 목표 %" PRIu64 ", 공식 목표와 비교 %" PRIu64 " 다름 %" PRIu64 " 첫 다름 %s\n", R.C.applied_steps, R.C.suppressed, R.C.cmp_n, R.C.cmp_bad, R.C.first_bad.empty() ? "없음" : R.C.first_bad.c_str());
  R.report();
  if (R.csv) fclose(R.csv);
  if (R.rec_pvd) {  // 기록 마무리 (파일 닫기)
    for (auto& kv : R.objs) if (kv.second.scene) { kv.second.scene->release(); kv.second.scene = nullptr; }
    R.phys->release();
    R.rec_pvd->release();
  }
  printf("PhysX 오류/경고 %d 건\n", gErr.n);
  return R.first_div_frame < 0 ? 0 : 3;
}
#endif  // OVD_REPLAY_NO_MAIN
