// S1 (docs/엔진_자체구현.md 15절): 제어기 오프라인 비교. PhysX 없음.
// Linux 공식 OVD 에서 매 서브스텝 직전의 관절 위치·링크 자세를 읽고, 그 실행이 쓴 행동으로 omni 제어기(core/omni/controllers.h)를 돌려
// 나온 드라이브 목표를 같은 서브스텝의 OVD driveTarget / driveVelocity 쓰기(= 공식 텐서 API 가 PhysX 에 넣은 값)와 비트 비교한다.
//   s1_ctrl <기록 폴더> [OVD 파일]     (입력: s1_setup.txt, s1_actions.bin — capture/export_s1.py)
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/omni/controllers.h"
#include "ovd.h"

using namespace eng::omni;

struct Setup {
  uint64_t episode_start = 0, post_total = 0;
  uint32_t substeps = 4;
  std::string base_link, root_link;
  int n_dof = 0;
  std::map<std::string, std::vector<int>> groups;
  float base_lim[12] = {};
  std::vector<std::string> dof_link;
  std::vector<float> sign;
  ctrl::R1ProConfig cfg{};
};

static bool read_setup(const std::string& path, Setup& s) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream is(line);
    std::string k;
    is >> k;
    if (k == "episode_start_post") is >> s.episode_start;
    else if (k == "post_step_count") is >> s.post_total;
    else if (k == "substeps") is >> s.substeps;
    else if (k == "base_link") is >> s.base_link;
    else if (k == "root_link") is >> s.root_link;
    else if (k == "n_dof") { is >> s.n_dof; s.dof_link.resize(s.n_dof); s.sign.resize(s.n_dof); }
    else if (k == "group") { std::string g; int d; is >> g; while (is >> d) s.groups[g].push_back(d); }
    else if (k == "base_limits") { for (float& x : s.base_lim) { std::string t; is >> t; x = strtof(t.c_str(), nullptr); } }
    else if (k == "dof") {
      int d, sg, has;
      std::string link, lo, hi, vlo, vhi;
      is >> d >> link >> sg >> lo >> hi >> vlo >> vhi >> has;
      s.dof_link[d] = link;
      s.sign[d] = float(sg);
      s.cfg.pos_lo[d] = strtof(lo.c_str(), nullptr);
      s.cfg.pos_hi[d] = strtof(hi.c_str(), nullptr);
      s.cfg.vel_lo[d] = strtof(vlo.c_str(), nullptr);
      s.cfg.vel_hi[d] = strtof(vhi.c_str(), nullptr);
      s.cfg.has_limit[d] = uint8_t(has);
    }
  }
  ctrl::R1ProConfig& c = s.cfg;
  c.n_dof = s.n_dof;
  auto cp = [&](const char* g, int* dst, int n) {
    auto& v = s.groups[g];
    for (int i = 0; i < n && i < int(v.size()); ++i) dst[i] = v[i];
  };
  cp("base", c.base_dof, 3);
  cp("trunk", c.trunk_dof, 4);
  cp("arm_left", c.arm_dof[0], 7);
  cp("arm_right", c.arm_dof[1], 7);
  cp("gripper_left", c.grip_dof[0], 2);
  cp("gripper_right", c.grip_dof[1], 2);
  for (int k = 0; k < 3; ++k) {
    c.base_in_lo[k] = s.base_lim[k];
    c.base_in_hi[k] = s.base_lim[3 + k];
    c.base_out_lo[k] = s.base_lim[6 + k];
    c.base_out_hi[k] = s.base_lim[9 + k];
  }
  ctrl::init(c);
  return s.n_dof > 0;
}

struct Tally {
  uint64_t n = 0, bad = 0;
  int64_t first_g = -1;
  std::string first;
};

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: s1_ctrl <기록 폴더> [OVD 파일]\n"); return 2; }
  const std::string dir = argv[1];
  Setup S;
  if (!read_setup(dir + "/s1_setup.txt", S)) { fprintf(stderr, "s1_setup.txt 를 못 읽음\n"); return 1; }
  std::vector<float> acts;
  uint32_t T = 0, A = 0;
  {
    FILE* f = fopen((dir + "/s1_actions.bin").c_str(), "rb");
    if (!f || fread(&T, 4, 1, f) != 1 || fread(&A, 4, 1, f) != 1) { fprintf(stderr, "s1_actions.bin 을 못 읽음\n"); return 1; }
    acts.resize(size_t(T) * A);
    if (fread(acts.data(), 4, acts.size(), f) != acts.size()) { fprintf(stderr, "행동 짧음\n"); return 1; }
    fclose(f);
  }
  std::string ovd_path = argc > 2 ? argv[2] : "";
  if (ovd_path.empty()) { fprintf(stderr, "OVD 파일을 주시오 (가장 큰 *_rec.ovd)\n"); return 2; }
  ovd::File F;
  std::string err;
  if (!ovd::load(ovd_path, F, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }

  // 속성 핸들
  auto attr = [&](const char* c, const char* a) -> uint32_t {
    for (auto& kv : F.attrs) if (kv.second.name == a && F.classes[kv.second.cls].name == c) return kv.first;
    return 0;
  };
  const uint32_t a_elapsed = attr("PxScene", "elapsedTime");
  const uint32_t a_name = attr("PxActor", "name");
  const uint32_t a_pose = attr("PxRigidActor", "globalPose");
  const uint32_t a_child = attr("PxArticulationJointReducedCoordinate", "childLink");
  const uint32_t a_motion = attr("PxArticulationJointReducedCoordinate", "motion");
  const uint32_t a_jpos = attr("PxArticulationJointReducedCoordinate", "jointPosition");
  const uint32_t a_dt = attr("PxArticulationJointReducedCoordinate", "driveTarget");
  const uint32_t a_dv = attr("PxArticulationJointReducedCoordinate", "driveVelocity");
  uint64_t nsim = 0;
  for (auto& e : F.events) if (e.cmd == ovd::kSet && e.attr == a_elapsed) nsim++;
  const uint64_t offset = S.post_total - nsim;  // 이 OVD 파일 앞 PhysX 인스턴스들이 쓴 simulate 수
  printf("OVD simulate %" PRIu64 ", 앞 몫 %" PRIu64 ", 에피소드 시작 %" PRIu64 ", 서브스텝 %u, 행동 %u x %u\n", nsim, offset, S.episode_start, S.substeps, T, A);

  // 상태 (핸들은 재사용되므로 create 때마다 지운다)
  std::unordered_map<uint64_t, std::string> name;       // 링크 핸들 -> 이름
  std::unordered_map<std::string, uint64_t> link_by_name;
  std::unordered_map<uint64_t, uint64_t> child;          // 조인트 -> 자식 링크 핸들
  std::unordered_map<uint64_t, float[7]> pose;           // 링크 -> q(xyzw) p
  std::unordered_map<uint64_t, float[6]> jpos;           // 조인트 -> 축별 위치
  std::unordered_map<uint64_t, int> jaxis;               // 조인트 -> 잠기지 않은 축
  std::unordered_map<std::string, int> dof_of_link;
  for (int d = 0; d < S.n_dof; ++d) dof_of_link[S.dof_link[d]] = d;

  ctrl::R1ProState st{};
  ctrl::reset(st);
  ctrl::DriveTargets out{};
  int64_t last_g = -1, last_t = -1;
  uint64_t sims = 0;
  Tally tpos, tvel;
  uint64_t unmapped = 0, before_episode = 0;
  int shown = 0;

  auto dof_of_joint = [&](uint64_t j) -> int {
    auto ic = child.find(j);
    if (ic == child.end()) return -1;
    auto in = name.find(ic->second);
    if (in == name.end()) return -1;
    auto id = dof_of_link.find(in->second);
    return id == dof_of_link.end() ? -1 : id->second;
  };
  auto joint_of_dof = [&](int d) -> uint64_t {  // 느리지만 명확하게: 자식 링크 이름이 맞는 조인트
    for (auto& kv : child) {
      auto in = name.find(kv.second);
      if (in != name.end() && in->second == S.dof_link[d]) return kv.first;
    }
    return 0;
  };

  for (size_t i = 0; i < F.events.size(); ++i) {
    const ovd::Event& e = F.events[i];
    if (e.cmd == ovd::kCreate) {
      name.erase(e.obj); child.erase(e.obj); pose.erase(e.obj); jpos.erase(e.obj); jaxis.erase(e.obj);
      const std::string nm = F.str(e);
      if (!nm.empty()) { name[e.obj] = nm; link_by_name[nm] = e.obj; }
      continue;
    }
    if (e.cmd != ovd::kSet) continue;
    const uint8_t* p = F.data(e);
    if (e.attr == a_elapsed) { sims++; continue; }
    if (e.attr == a_name) { std::string nm = F.str(e); name[e.obj] = nm; link_by_name[nm] = e.obj; continue; }
    if (e.attr == a_pose && e.data_len == 28) { memcpy(pose[e.obj], p, 28); continue; }
    if (e.attr == a_child && e.data_len == 8) { uint64_t c; memcpy(&c, p, 8); child[e.obj] = c; continue; }
    if (e.attr == a_motion && e.data_len == 24) {
      uint32_t m[6];
      memcpy(m, p, 24);
      for (int k = 0; k < 6; ++k) if (m[k] != 0) { jaxis[e.obj] = k; break; }
      continue;
    }
    if (e.attr == a_jpos && e.data_len == 24) { memcpy(jpos[e.obj], p, 24); continue; }
    if ((e.attr != a_dt && e.attr != a_dv) || e.data_len != 24) continue;
    // 로봇 조인트의 드라이브 목표 쓰기 -> 이번 서브스텝 = 다음 simulate
    const int d0 = dof_of_joint(e.obj);
    if (d0 < 0) { unmapped++; continue; }
    const int64_t g = int64_t(offset + sims + 1);  // 곧 일어날 simulate 의 전체 번호 (1 부터)
    if (g <= int64_t(S.episode_start)) { before_episode++; continue; }
    const int64_t k = g - int64_t(S.episode_start) - 1;
    const int64_t t = k / S.substeps;
    if (t >= int64_t(T)) continue;
    if (g != last_g) {  // 이번 서브스텝 제어기 계산 (한 번)
      if (t != last_t) {  // 스텝 첫 서브스텝: robot.apply_action (자세는 지금 = 이전 simulate 뒤)
        auto ib = link_by_name.find(S.base_link);
        auto ir = link_by_name.find(S.root_link);
        if (ib == link_by_name.end() || ir == link_by_name.end() || !pose.count(ib->second) || !pose.count(ir->second)) {
          fprintf(stderr, "바닥/뿌리 링크 자세를 못 찾음 (스텝 %" PRId64 ")\n", t);
          return 1;
        }
        const float* bp = pose[ib->second];
        const float* rp = pose[ir->second];
        const float base_p[3] = {bp[4], bp[5], bp[6]}, base_q[4] = {bp[0], bp[1], bp[2], bp[3]};
        const float root_p[3] = {rp[4], rp[5], rp[6]}, root_q[4] = {rp[0], rp[1], rp[2], rp[3]};
        ctrl::apply_action(S.cfg, st, &acts[size_t(t) * A], base_p, base_q, root_p, root_q);
        last_t = t;
      }
      float q[ctrl::kMaxDof] = {};
      for (int d = 0; d < S.n_dof; ++d) {
        const uint64_t j = joint_of_dof(d);
        if (!j || !jpos.count(j) || !jaxis.count(j)) { fprintf(stderr, "dof %d 관절값 없음\n", d); return 1; }
        q[d] = S.sign[d] * jpos[j][jaxis[j]];
      }
      ctrl::step(S.cfg, st, q, out);
      last_g = g;
    }
    // 비교: 이 조인트의 축 값
    const int ax = jaxis.count(e.obj) ? jaxis[e.obj] : 0;
    float rec;
    memcpy(&rec, p + 4 * ax, 4);
    const bool is_pos = e.attr == a_dt;
    if (is_pos ? !out.set_pos[d0] : !out.set_vel[d0]) continue;
    const float ours = S.sign[d0] * (is_pos ? out.pos[d0] : out.vel[d0]);
    Tally& ty = is_pos ? tpos : tvel;
    ty.n++;
    if (memcmp(&ours, &rec, 4)) {
      ty.bad++;
      if (ty.first_g < 0) {
        ty.first_g = g;
        char b[200];
        snprintf(b, sizeof b, "스텝 %" PRId64 " 서브 %" PRId64 " dof %d(%s): 우리 %.9g 공식 %.9g", t, k % S.substeps, d0, S.dof_link[d0].c_str(), ours, rec);
        ty.first = b;
      }
      if (shown++ < 8) printf("[다름] %s 스텝 %" PRId64 " 서브 %" PRId64 " dof %d: 우리 %.9g 공식 %.9g\n", is_pos ? "위치목표" : "속도목표", t, k % S.substeps, d0, ours, rec);
    }
  }
  printf("위치 목표: 비교 %" PRIu64 ", 다름 %" PRIu64 "%s%s\n", tpos.n, tpos.bad, tpos.bad ? ", 첫 다름 " : "", tpos.first.c_str());
  printf("속도 목표: 비교 %" PRIu64 ", 다름 %" PRIu64 "%s%s\n", tvel.n, tvel.bad, tvel.bad ? ", 첫 다름 " : "", tvel.first.c_str());
  printf("(에피소드 전 쓰기 %" PRIu64 " 건 제외, 로봇 아닌 조인트 %" PRIu64 " 건)\n", before_episode, unmapped);
  return (tpos.bad || tvel.bad) ? 3 : 0;
}
