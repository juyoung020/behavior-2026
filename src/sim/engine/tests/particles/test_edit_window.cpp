// 층 1 시험 (물리 결합): 편집 창(edit_window.h)의 입자 계 뜨기·되돌리기가 공식 재적재(test_reload 정답)와 같은 자세를 몸체 API 로 쓰는지,
// 그리고 편집 차례(뜨기 → 무덤 → 지우기 → 되돌리기 → 넣기)가 약속대로인지. 몸체 API·틀은 가짜 (ScScene 편집 번호는 리드 g1_sc 가 PhysX 와 대조).
//   ./test_edit_window ~/engine-data/particles/reload_real_onion
#include <cstdio>
#include <cstring>
#include <string>

#include "core/particles/edit_window.h"
#include "tests/omni/npy.h"

using namespace eng;
using namespace eng::particles;

struct FakeBody : BodyApi {
  std::vector<Tf> pose;
  std::vector<std::string> log;
  Tf actorPose(int32_t h) override { return pose[size_t(h)]; }
  void setActorPose(int32_t h, const Tf& p) override {
    pose[size_t(h)] = p;
    log.push_back("pose " + std::to_string(h));
  }
  void velocity(int32_t, V3& l, V3& a) override { l = a = V3{0, 0, 0}; }
  void setVelocity(int32_t h, const V3&, const V3&) override { log.push_back("vel " + std::to_string(h)); }
  void bodyAdded(int32_t h, const Body&) override { log.push_back("add " + std::to_string(h)); }
  void bodyRemoved(int32_t h) override { log.push_back("rm " + std::to_string(h)); }
};
struct NoSpawn : SpawnResolver {
  bool half(int32_t, int32_t, SpawnSource&) override { return false; }
  bool particle(int32_t, int32_t, SpawnSource&) override { return false; }
};
struct LogBase : scene::EditWindow {
  std::vector<std::string>* log;
  void dumpState(scene::EnvStep&) override { log->push_back("base dump"); }
  void loadState(scene::EnvStep&) override { log->push_back("base load"); }
};

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  auto L = [&](const char* k) {
    Npy a;
    if (!npy_load(std::string(argv[1]) + "/" + k + ".npy", a)) exit(1);
    return a;
  };
  Npy nn = L("n"), tfs = L("tfs"), off = L("off"), out = L("out"), sp = L("scene_pose");
  const int32_t* n = nn.as<int32_t>();
  const float* tf = tfs.as<float>();
  const float* w = out.as<float>();
  long k0 = 0, bad = 0, total = 0;
  bool orderOk = true;
  for (int c = 0; c < (int)nn.shape[0]; ++c) {
    FakeBody B;
    std::vector<ObjectRt> objs(1);
    std::vector<ParticleSystemRt> sys(1);
    for (int i = 0; i < n[c]; ++i) {
      B.pose.push_back(TransitionEditWindow::tf7(tf + 7 * (k0 + i)));
      sys[0].actors.push_back(i);
    }
    memcpy(sys[0].off, off.as<float>() + 3 * c, 12);
    B.pose.push_back(Tf{Q{0, 0, 0, 1}, V3{1, 2, 3}});  // 지울 물체 (행위자 n)
    objs[0].actors.push_back(n[c]);
    std::vector<Edit> edits(2);
    edits[0].kind = EDIT_REMOVE_BEGIN;
    edits[0].poses.push_back(Pose7{{100, 100, 100}, {0, 0, 0, 1}});
    edits[1].kind = EDIT_REMOVE_OBJECT;
    edits[1].object = 0;
    NoSpawn ns;
    LogBase base;
    base.log = &B.log;
    TransitionEditWindow W;
    W.base = &base, W.body = &B, W.spawn = &ns, W.objects = &objs, W.systems = &sys, W.edits = &edits;
    memcpy(W.scenePose, sp.as<float>() + 32 * c, 64);
    memcpy(W.scenePoseInv, sp.as<float>() + 32 * c + 16, 64);
    // envEditWindow 와 같은 차례 (물리 한 스텝·ScScene 지우기는 빼고 — 가짜 장면)
    scene::EnvStep E;  // 쓰이지 않음 (지우기·넣기 없음)
    W.dumpState(E);
    W.teleportToGrave(E);
    orderOk = orderOk && W.extraPhysicsStep();
    for (const Edit& e : edits)  // W.edit 대신 (ScScene 없음): 지울 행위자 알림만
      if (e.kind == EDIT_REMOVE_OBJECT)
        for (int32_t h : objs[size_t(e.object)].actors) B.bodyRemoved(h);
    W.loadState(E);
    for (int i = 0; i < n[c]; ++i) {
      const Tf& g = B.pose[size_t(i)];
      const float got[7] = {g.p.x, g.p.y, g.p.z, g.q.x, g.q.y, g.q.z, g.q.w};
      bad += memcmp(got, w + 7 * (k0 + i), 28) != 0;
      ++total;
    }
    // 차례: (입자 뜨기) base dump → 무덤(pose n) → rm n → 입자 자세 n 개 → 입자 속도 n 개 → base load
    // 계 등록부가 물체 등록부보다 먼저: base dump 는 입자 뒤, base load 는 입자 자세·속도 뒤
    std::vector<std::string> want = {"base dump", "pose " + std::to_string(n[c]), "rm " + std::to_string(n[c])};
    for (int i = 0; i < n[c]; ++i) want.push_back("pose " + std::to_string(i));
    for (int i = 0; i < n[c]; ++i) want.push_back("vel " + std::to_string(i));
    want.push_back("base load");
    orderOk = orderOk && B.log == want && W.failed == 0;
    k0 += n[c];
  }
  printf("편집 창 입자 재적재: 입자 %ld  자세 다름 %ld  차례 %s\n", total, bad, orderOk ? "맞음" : "다름");
  const bool ok = total > 0 && !bad && orderOk;
  printf(ok ? "비트 동일\n" : "실패\n");
  return ok ? 0 : 1;
}
