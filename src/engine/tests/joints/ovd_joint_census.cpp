// joints 도구: 공식 기록(OVD)에서 D6 조인트가 무엇과 무엇을 잇는지 센다 (강체-강체 / 강체-링크 / 링크-정적 …),
// 생성·삭제 프레임(보조 잡기처럼 도중에 붙였다 떼는 것), 설정값 분포. 옮길 경로(강체 1D·관절체 1D·4개 묶음)의 우선순위를 정하는 데 쓴다.
//   ovd_joint_census <file.ovd>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "replay/ovd.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ovd_joint_census <file.ovd>\n");
    return 1;
  }
  ovd::File F;
  std::string err;
  if (!ovd::load(argv[1], F, err)) {
    fprintf(stderr, "읽기 실패: %s\n", err.c_str());
    return 1;
  }
  std::unordered_map<uint64_t, uint32_t> clsOf;           // 객체 -> 클래스
  std::unordered_map<uint64_t, std::string> nameOf;
  std::unordered_map<uint64_t, uint64_t> linkArt;          // 링크 -> 관절체
  std::unordered_map<uint64_t, std::vector<uint64_t>> jointActors;  // 조인트 -> actor0, actor1 (마지막 값)
  std::unordered_map<uint64_t, uint32_t> jointFlags, jointCreate, jointDestroy;
  std::unordered_map<uint64_t, std::string> jointMotions;
  std::unordered_map<uint64_t, uint32_t> rbFlags;
  uint32_t frame = 0;
  const uint32_t cD6 = F.cls("PxD6Joint");
  for (const ovd::Event& e : F.events) {
    if (e.cmd == ovd::kStartFrame) { frame++; continue; }
    if (e.cmd == ovd::kCreate) {
      clsOf[e.obj] = e.cls;
      if (e.cls == cD6) jointCreate[e.obj] = frame;
      continue;
    }
    if (e.cmd == ovd::kDestroy) {
      if (clsOf.count(e.obj) && clsOf[e.obj] == cD6) jointDestroy[e.obj] = frame;
      continue;
    }
    if (e.cmd != ovd::kSet) continue;
    const std::string an = F.attr_name(e.attr);
    if (an == "PxActor.name" || an == "PxJoint.name") nameOf[e.obj] = F.str(e);
    else if (an == "PxJoint.actor0" || an == "PxJoint.actor1") {
      uint64_t v = 0;
      memcpy(&v, F.data(e), 8);
      auto& a = jointActors[e.obj];
      a.resize(2);
      a[an == "PxJoint.actor0" ? 0 : 1] = v;
    } else if (an == "PxJoint.constraintFlags") {
      uint32_t v = 0;
      memcpy(&v, F.data(e), e.data_len < 4 ? e.data_len : 4);
      jointFlags[e.obj] = v;
    } else if (an == "PxD6Joint.motions") {
      std::string s;
      for (uint32_t i = 0; i < e.data_len / 4; ++i) { uint32_t m; memcpy(&m, F.data(e) + 4 * i, 4); s += char('0' + m); }
      jointMotions[e.obj] = s;
    } else if (an == "PxArticulationLink.articulation") {
      uint64_t v = 0;
      memcpy(&v, F.data(e), 8);
      linkArt[e.obj] = v;
    } else if (an == "PxRigidBody.rigidBodyFlags") {
      uint32_t v = 0;
      memcpy(&v, F.data(e), e.data_len < 4 ? e.data_len : 4);
      rbFlags[e.obj] = v;
    }
  }
  auto kind = [&](uint64_t a) -> std::string {
    if (!a) return "세계";
    auto it = clsOf.find(a);
    if (it == clsOf.end()) return "?";
    const std::string& c = F.classes[it->second].name;
    if (c == "PxRigidDynamic") return (rbFlags.count(a) && (rbFlags[a] & 1)) ? "운동학" : "동적";
    if (c == "PxRigidStatic") return "정적";
    if (c == "PxArticulationLink") return "링크";
    return c;
  };
  std::map<std::string, int> pairCount, motionCount, flagCount;
  std::map<uint64_t, int> perActor0;
  int midCreated = 0, midDestroyed = 0;
  printf("D6 조인트 %zu 개 (프레임 %u)\n", jointActors.size(), frame);
  for (auto& kv : jointActors) {
    const uint64_t j = kv.first;
    const std::string k0 = kind(kv.second[0]), k1 = kind(kv.second[1]);
    std::string pk = k0 + "-" + k1;
    if (k0 == "링크" && k1 == "링크") pk += (linkArt[kv.second[0]] == linkArt[kv.second[1]]) ? "(같은 관절체)" : "(다른 관절체)";
    pairCount[pk]++;
    motionCount[jointMotions[j]]++;
    flagCount[std::to_string(jointFlags[j])]++;
    perActor0[kv.second[0]]++;
    if (jointCreate[j] > 1) midCreated++;
    if (jointDestroy.count(j)) midDestroyed++;
  }
  printf("  잇는 쌍 (actor0-actor1):\n");
  for (auto& kv : pairCount) printf("    %-28s %d\n", kv.first.c_str(), kv.second);
  printf("  motions(X Y Z 비틀기 스윙1 스윙2, 0 잠금 1 한계 2 자유):\n");
  for (auto& kv : motionCount) printf("    %-10s %d\n", kv.first.c_str(), kv.second);
  printf("  constraintFlags:\n");
  for (auto& kv : flagCount) printf("    %-10s %d\n", kv.first.c_str(), kv.second);
  int shared = 0;
  for (auto& kv : perActor0) if (kv.second > 1) shared += kv.second;
  printf("  actor0 을 여럿이 나눠 쓰는 조인트 %d 개, 도중 생성 %d, 도중 삭제 %d\n", shared, midCreated, midDestroyed);
  printf("  예시:\n");
  int shown = 0;
  for (auto& kv : jointActors) {
    if (shown++ >= 12) break;
    printf("    %s  [%s] %s  <->  [%s] %s\n", nameOf[kv.first].c_str(), kind(kv.second[0]).c_str(), nameOf[kv.second[0]].c_str(),
           kind(kv.second[1]).c_str(), nameOf[kv.second[1]].c_str());
  }
  return 0;
}
