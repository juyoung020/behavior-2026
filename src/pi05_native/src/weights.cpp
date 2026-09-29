#include "weights.h"

#include <cstring>
#include <sstream>

#ifdef _WIN32
#define PI05_FSEEK _fseeki64
#else
#define PI05_FSEEK fseeko
#endif

namespace pi05 {

WeightFile::~WeightFile() {
  if (f_) fclose(f_);
}

bool WeightFile::open(const std::string& path, std::string* err) {
  f_ = fopen(path.c_str(), "rb");
  if (!f_) { *err = "cannot open " + path; return false; }
  char magic[8];
  uint64_t mlen = 0, data_off = 0;
  if (fread(magic, 1, 8, f_) != 8 || memcmp(magic, "PI05W\0\0\1", 8) != 0) { *err = "bad magic in " + path; return false; }
  if (fread(&mlen, 8, 1, f_) != 1 || fread(&data_off, 8, 1, f_) != 1) { *err = "short header"; return false; }
  std::string manifest(mlen, '\0');
  if (fread(&manifest[0], 1, mlen, f_) != mlen) { *err = "short manifest"; return false; }
  std::istringstream in(manifest);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string kind;
    ls >> kind;
    if (kind == "cfg") {
      std::string key, v;
      ls >> key;
      std::vector<std::string> vals;
      while (ls >> v) vals.push_back(v);
      cfg_.emplace(key, vals);
    } else if (kind == "t") {
      TensorInfo t;
      std::string dt;
      int nd = 0;
      ls >> t.name >> dt >> t.offset >> t.nbytes >> nd;
      for (int i = 0; i < nd; ++i) { int64_t d; ls >> d; t.shape.push_back(d); }
      if (dt == "bf16") t.dtype = DType::BF16;
      else if (dt == "f32") t.dtype = DType::F32;
      else if (dt == "f64") t.dtype = DType::F64;
      else if (dt == "i32") t.dtype = DType::I32;
      else if (dt == "u8") t.dtype = DType::U8;
      else { *err = "bad dtype " + dt; return false; }
      t.offset += data_off;
      index_[t.name] = tensors_.size();
      tensors_.push_back(std::move(t));
    }
  }
  return true;
}

const TensorInfo* WeightFile::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

bool WeightFile::read_range(const TensorInfo& t, uint64_t off, uint64_t n, void* dst, std::string* err) const {
  if (off + n > t.nbytes) { *err = "range outside " + t.name; return false; }
  if (PI05_FSEEK(f_, (int64_t)(t.offset + off), SEEK_SET) != 0) { *err = "seek failed"; return false; }
  if (fread(dst, 1, n, f_) != n) { *err = "short read " + t.name; return false; }
  return true;
}

bool WeightFile::read(const TensorInfo& t, void* dst, std::string* err) const {
  return read_range(t, 0, t.nbytes, dst, err);
}

const std::vector<std::string>* WeightFile::cfg(const std::string& key) const {
  auto it = cfg_.find(key);
  return it == cfg_.end() ? nullptr : &it->second;
}

int WeightFile::cfg_int(const std::string& key, int def) const {
  auto v = cfg(key);
  return (v && !v->empty()) ? std::stoi((*v)[0]) : def;
}

std::string WeightFile::cfg_str(const std::string& key, const std::string& def) const {
  auto v = cfg(key);
  return (v && !v->empty()) ? (*v)[0] : def;
}

}  // namespace pi05
