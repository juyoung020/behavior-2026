#include "ovd.h"

#include <cstdlib>
#include <cstring>

#include "OmniPvdFileReadStream.h"
#include "OmniPvdLoader.h"
#include "OmniPvdReader.h"

namespace ovd {

bool File::is_a(uint32_t c, uint32_t base) const {
  for (int guard = 0; c && guard < 64; ++guard) {
    if (c == base) return true;
    auto it = classes.find(c);
    if (it == classes.end()) return false;
    c = it->second.base;
  }
  return false;
}

std::string File::attr_name(uint32_t a) const {
  auto it = attrs.find(a);
  if (it == attrs.end()) return "?attr" + std::to_string(a);
  auto ic = classes.find(it->second.cls);
  return (ic == classes.end() ? std::string("?") : ic->second.name) + "." + it->second.name;
}

static std::string default_lib() {
  const char* home = getenv("HOME");
  return std::string(home ? home : "") +
         "/engine-deps/physx-107.3-omni/physx/bin/linux.x86_64/checked/libPVDRuntime_64.so";
}

bool load(const std::string& path, File& out, std::string& err, const std::string& lib_in) {
  static OmniPvdLoader loader;  // 프로세스 끝까지 유지
  if (!loader.mLibraryHandle) {
    std::string lib = lib_in.empty() ? default_lib() : lib_in;
    if (!loader.loadOmniPvd(lib.c_str())) {
      err = "PVDRuntime 을 못 불러옴: " + lib;
      return false;
    }
  }
  OmniPvdReader* rd = loader.mCreateOmniPvdReader();
  OmniPvdFileReadStream* fs = loader.mCreateOmniPvdFileReadStream();
  fs->setFileName(path.c_str());
  if (!fs->openFile()) {
    err = "파일을 못 엶: " + path;
    return false;
  }
  rd->setReadStream(*fs);
  OmniPvdVersionType ma = 0, mi = 0, pa = 0;
  if (!rd->startReading(ma, mi, pa)) {
    err = "OVD 머리말을 못 읽음";
    return false;
  }
  out.ver_major = ma; out.ver_minor = mi; out.ver_patch = pa;
  out.events.reserve(1 << 20);

  auto put_blob = [&](const void* p, size_t n, Event& e) {
    e.data_off = out.blob.size();
    e.data_len = static_cast<uint32_t>(n);
    const uint8_t* b = static_cast<const uint8_t*>(p);
    out.blob.insert(out.blob.end(), b, b + n);
  };

  for (;;) {
    OmniPvdCommand::Enum c = rd->getNextCommand();
    if (c == OmniPvdCommand::eINVALID) break;
    Event e{};
    e.cmd = static_cast<Cmd>(c);
    switch (c) {
      case OmniPvdCommand::eREGISTER_CLASS: {
        ClassInfo ci;
        ci.name = rd->getClassName();
        ci.base = rd->getBaseClassHandle();
        out.classes[rd->getClassHandle()] = ci;
        out.class_by_name[ci.name] = rd->getClassHandle();
        e.cls = rd->getClassHandle();
        break;
      }
      case OmniPvdCommand::eREGISTER_ATTRIBUTE: {
        AttrInfo a;
        a.cls = rd->getClassHandle();
        a.name = rd->getAttributeName();
        a.type = rd->getAttributeDataType();
        if (a.type == OmniPvdDataType::eFLAGS_WORD) a.enum_class = rd->getEnumClassHandle();
        else if (a.type != OmniPvdDataType::eENUM_VALUE) a.n_elems = rd->getAttributeNumberElements();
        out.attrs[rd->getAttributeHandle()] = a;
        e.attr = rd->getAttributeHandle();
        break;
      }
      case OmniPvdCommand::eREGISTER_CLASS_ATTRIBUTE: {
        AttrInfo a;
        a.cls = rd->getClassHandle();
        a.name = rd->getAttributeName();
        a.is_class_attr = true;
        a.attr_class = rd->getAttributeClassHandle();
        out.attrs[rd->getAttributeHandle()] = a;
        e.attr = rd->getAttributeHandle();
        break;
      }
      case OmniPvdCommand::eREGISTER_UNIQUE_LIST_ATTRIBUTE: {
        AttrInfo a;
        a.cls = rd->getClassHandle();
        a.name = rd->getAttributeName();
        a.type = rd->getAttributeDataType();
        a.is_list = true;
        out.attrs[rd->getAttributeHandle()] = a;
        e.attr = rd->getAttributeHandle();
        break;
      }
      case OmniPvdCommand::eSET_ATTRIBUTE:
      case OmniPvdCommand::eADD_TO_UNIQUE_LIST_ATTRIBUTE:
      case OmniPvdCommand::eREMOVE_FROM_UNIQUE_LIST_ATTRIBUTE:
        e.ctx = rd->getContextHandle();
        e.obj = rd->getObjectHandle();
        e.attr = rd->getAttributeHandle();
        put_blob(rd->getAttributeDataPointer(), rd->getAttributeDataLength(), e);
        break;
      case OmniPvdCommand::eCREATE_OBJECT: {
        e.ctx = rd->getContextHandle();
        e.obj = rd->getObjectHandle();
        e.cls = rd->getClassHandle();
        const char* nm = rd->getObjectName();
        put_blob(nm, nm ? strlen(nm) : 0, e);
        break;
      }
      case OmniPvdCommand::eDESTROY_OBJECT:
        e.ctx = rd->getContextHandle();
        e.obj = rd->getObjectHandle();
        break;
      case OmniPvdCommand::eSTART_FRAME:
        e.ctx = rd->getContextHandle();
        e.time = rd->getFrameTimeStart();
        break;
      case OmniPvdCommand::eSTOP_FRAME:
        e.ctx = rd->getContextHandle();
        e.time = rd->getFrameTimeStop();
        break;
      case OmniPvdCommand::eRECORD_MESSAGE: {
        const char *msg = nullptr, *file = nullptr;
        uint32_t line = 0, type = 0;
        OmniPvdClassHandle h = 0;
        rd->getMessageData(msg, file, line, type, h);
        std::string s = std::string(msg ? msg : "") + " @" + (file ? file : "") + ":" + std::to_string(line);
        e.ctx = rd->getContextHandle();
        put_blob(s.data(), s.size(), e);
        break;
      }
      default:
        break;
    }
    out.events.push_back(e);
  }
  fs->closeFile();
  loader.mDestroyOmniPvdReader(*rd);
  loader.mDestroyOmniPvdFileReadStream(*fs);
  return true;
}

}  // namespace ovd
