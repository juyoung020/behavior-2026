// 엔진 파이썬 바인딩의 C 입구 (S4 v0, docs/엔진_자체구현.md 15절). ctypes 로 부른다: src/engine/eval/backend_engine.py
//
// 지금 엔진 = PhysX 비계 + 우리 모듈(제어기 core/omni/controllers.h, 판정 core/omni/states.h·bddl.h).
// 에피소드 시작 전(인스턴스 불러오기·가라앉히기·복원)은 행동과 무관하므로 그 판의 공식 기록(OVD + 곁기록)을 "장면 추출물"로 재생하고,
// 에피소드부터는 OVD 입력을 넣지 않고(--free) 바깥에서 받은 행동 -> 제어기 -> PhysX 로만 돈다. OVD 는 비교용으로만 읽는다
// (같은 행동이면 매 프레임 비트 비교가 그대로 돈다, OVD 가 끝나면 직접 simulate).
#define OVD_REPLAY_NO_MAIN
#include "ovd_replay.cpp"

namespace {
struct Engine {
  ovd::File F;
  std::unique_ptr<Replayer> R;
  std::string log;
};
}  // namespace

extern "C" {

// rec_dir: 기록 폴더 (convex.bin·filters.txt·sidelog.bin·gravity_off.txt·s1_setup.txt, 있으면 s3_*)
// ovd: 가장 큰 OVD (에피소드가 든 PhysX 인스턴스), side_offset: 앞선 인스턴스 simulate 수
void* ee_open(const char* rec_dir, const char* ovd_path, uint64_t side_offset, int threads, int with_s3) {
  auto* E = new Engine();
  std::string err;
  if (!ovd::load(ovd_path, E->F, err)) {
    fprintf(stderr, "ee_open: %s\n", err.c_str());
    delete E;
    return nullptr;
  }
  E->R.reset(new Replayer(E->F));
  Replayer& R = *E->R;
  const std::string d = rec_dir;
  R.side_offset = side_offset;
  if (!engine::read_convex_bin(d + "/convex.bin", R.convex)) fprintf(stderr, "ee_open: convex.bin 없음\n");
  if (!engine::read_filters(d + "/filters.txt", R.spec)) fprintf(stderr, "ee_open: filters.txt 없음\n");
  if (!engine::read_sidelog(d + "/sidelog.bin", R.sviews, R.scalls)) fprintf(stderr, "ee_open: sidelog.bin 없음\n");
  {
    std::ifstream gf(d + "/gravity_off.txt");
    std::string ln;
    while (std::getline(gf, ln))
      if (!ln.empty()) R.gravity_off.push_back(ln);
  }
  if (!R.load_ctrl(d)) {
    fprintf(stderr, "ee_open: s1_setup.txt·s1_actions.bin 을 못 읽음 (export_s1.py)\n");
    delete E;
    return nullptr;
  }
  R.C.acts.clear();  // 행동은 바깥에서 (ee_step)
  R.C.T = 0;
  R.C.free_run = true;
  if (with_s3) {
    if (!R.load_s3(d)) fprintf(stderr, "ee_open: s3 입력 없음 (판정 끔)\n");
    else R.S3.own = true;  // S3 v1: 판정 접촉 행렬을 우리 PhysX 접촉 보고로 (기록의 접촉 행렬은 비교용)
  }
  // AG 단계 B: ag_setup.txt(export_ag.py) 가 있으면 엔진이 잡기를 판단한다 (없으면 기록의 AG 관절을 그대로 — v0)
  if (R.load_ag(d)) {
    if (!R.S3.own) fprintf(stderr, "ee_open: AG 는 접촉 보고가 필요 -> s3 입력(export_s3.py)도 있어야 한다
");
  }
  if (!R.init(threads)) {
    delete E;
    return nullptr;
  }
  R.dump_art_at = -1;
  while (R.scall_next < R.scalls.size() && R.scalls[R.scall_next].after < R.side_offset) R.scall_next++;
  // 에피소드 시작 직전까지 (장면 추출물 재생)
  R.run_until(int64_t(R.C.episode_start) - int64_t(R.side_offset));
  return E;
}

int ee_n_dof(void* h) { return static_cast<Engine*>(h)->R->C.n_dof; }
int ee_action_dim(void* h) { return int(static_cast<Engine*>(h)->R->C.A); }
int ee_substeps(void* h) { return int(static_cast<Engine*>(h)->R->C.substeps); }
uint64_t ee_sims(void* h) { return static_cast<Engine*>(h)->R->sims; }

// 한 평가 스텝 = 서브스텝 substeps 번 (행동은 float32 A 개)
void ee_step(void* h, const float* action) {
  Replayer& R = *static_cast<Engine*>(h)->R;
  R.push_action(action);
  R.run_until(int64_t(R.sims) + R.C.substeps);
}

// 로봇 관절값: 텐서 API dof 순서(s1_setup 의 dof 줄), 부호 반영. 반환 = dof 수 (묶기 실패 시 0)
int ee_joint_state(void* h, float* q, float* v) {
  Replayer& R = *static_cast<Engine*>(h)->R;
  if (!R.ctrl_bind()) return 0;
  for (int d = 0; d < R.C.n_dof; ++d) {
    const auto ax = PxArticulationAxis::Enum(R.C.axis[d]);
    q[d] = R.C.sign[d] * R.C.joint[d]->getJointPosition(ax);
    v[d] = R.C.sign[d] * R.C.joint[d]->getJointVelocity(ax);
  }
  return R.C.n_dof;
}

// 몸체(링크·강체) 세계 자세 (x,y,z, qx,qy,qz,qw) 와 속도 (선 3, 각 3). 이름 = prim 경로. 없으면 0
int ee_body_pose(void* h, const char* name, float* p7) {
  Replayer& R = *static_cast<Engine*>(h)->R;
  auto it = R.actor_by_name.find(name);
  if (it == R.actor_by_name.end()) return 0;
  const PxTransform t = it->second->getGlobalPose();
  const float o[7] = {t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w};
  memcpy(p7, o, sizeof o);
  return 1;
}
int ee_body_vel(void* h, const char* name, float* v6) {
  Replayer& R = *static_cast<Engine*>(h)->R;
  auto it = R.actor_by_name.find(name);
  if (it == R.actor_by_name.end()) return 0;
  PxVec3 l(0.0f), a(0.0f);
  if (auto* rb = it->second->is<PxRigidBody>()) {
    l = rb->getLinearVelocity();
    a = rb->getAngularVelocity();
  }
  const float o[6] = {l.x, l.y, l.z, a.x, a.y, a.z};
  memcpy(v6, o, sizeof o);
  return 1;
}

// 판정: 마지막 평가 스텝의 목표 문자열 (trace goal_satisfied 와 같은 꼴, 예 "[]" "[0]")
int ee_goal(void* h, char* buf, int n) {
  const std::string& g = static_cast<Engine*>(h)->R->S3.last_goal;
  snprintf(buf, size_t(n), "%s", g.c_str());
  return static_cast<Engine*>(h)->R->S3.on ? 1 : 0;
}

// 공식 OVD 와의 비트 비교 요약을 stdout 에
void ee_report(void* h) {
  Replayer& R = *static_cast<Engine*>(h)->R;
  R.s3_report();
  R.report();
  fflush(stdout);
}

void ee_close(void* h) { delete static_cast<Engine*>(h); }

}  // extern "C"
