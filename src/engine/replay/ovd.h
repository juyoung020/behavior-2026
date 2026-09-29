// OmniPVD(.ovd) 파일을 통째로 메모리에 읽어 명령 목록으로 만든다.
// PhysX 의 PVDRuntime(libPVDRuntime_64.so) 리더를 그대로 쓴다 — 형식을 우리가 다시 해석하지 않는다.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ovd {

enum Cmd : uint8_t {
  kInvalid = 0, kRegClass, kRegEnum, kRegAttr, kRegClassAttr, kRegList, kSet, kAddToList, kRemoveFromList,
  kCreate, kDestroy, kStartFrame, kStopFrame, kMessage
};

struct ClassInfo {
  std::string name;
  uint32_t base = 0;
};

struct AttrInfo {
  uint32_t cls = 0;           // 이 속성을 가진 클래스
  std::string name;
  uint32_t type = 0;          // OmniPvdDataType
  uint32_t n_elems = 0;       // 고정 길이 배열이면 원소 수 (가변이면 0)
  uint32_t enum_class = 0;    // 플래그 워드의 열거 클래스
  bool is_list = false;       // unique list (집합)
  bool is_class_attr = false; // 중첩 구조체 속성
  uint32_t attr_class = 0;
};

struct Event {
  Cmd cmd;
  uint8_t depth = 0;          // 속성 핸들 깊이 (중첩 속성)
  uint32_t cls = 0;           // create: 클래스, set: 0
  uint32_t attr = 0;          // set/add/remove: 속성 핸들
  uint64_t ctx = 0;
  uint64_t obj = 0;           // 원래 프로세스의 포인터 값
  uint64_t data_off = 0;      // blob 안 위치
  uint32_t data_len = 0;
  uint64_t time = 0;          // frame start/stop 시각 (프레임 번호)
};

struct File {
  uint32_t ver_major = 0, ver_minor = 0, ver_patch = 0;
  std::unordered_map<uint32_t, ClassInfo> classes;
  std::unordered_map<uint32_t, AttrInfo> attrs;
  std::vector<Event> events;
  std::vector<uint8_t> blob;            // set 데이터, 객체 이름, 메시지
  std::unordered_map<std::string, uint32_t> class_by_name;

  const uint8_t* data(const Event& e) const { return blob.data() + e.data_off; }
  std::string str(const Event& e) const { return std::string(reinterpret_cast<const char*>(data(e)), e.data_len); }
  // 클래스 이름으로 찾기 (없으면 0)
  uint32_t cls(const std::string& n) const {
    auto it = class_by_name.find(n);
    return it == class_by_name.end() ? 0 : it->second;
  }
  bool is_a(uint32_t c, uint32_t base) const;  // 상속 포함
  std::string attr_name(uint32_t a) const;     // "Class.attr"
};

// libPVDRuntime_64.so 경로 (비우면 PHYSX_ROOT 빌드 폴더 기본값)
bool load(const std::string& path, File& out, std::string& err, const std::string& lib = "");

}  // namespace ovd
