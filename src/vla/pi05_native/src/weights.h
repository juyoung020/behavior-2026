// Reader for our raw weight file (.pi05w), written by tools/export_weights.py.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace pi05 {

enum class DType { BF16, F32, F64, I32, U8 };

struct TensorInfo {
  std::string name;
  DType dtype;
  uint64_t offset = 0;  // absolute file offset
  uint64_t nbytes = 0;
  std::vector<int64_t> shape;
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

class WeightFile {
 public:
  ~WeightFile();
  bool open(const std::string& path, std::string* err);
  const TensorInfo* find(const std::string& name) const;
  bool read(const TensorInfo& t, void* dst, std::string* err) const;
  // Reads [off, off+n) bytes of a tensor.
  bool read_range(const TensorInfo& t, uint64_t off, uint64_t n, void* dst, std::string* err) const;
  // Config lines "cfg key v1 v2 ..."; repeated keys keep every occurrence.
  const std::vector<std::string>* cfg(const std::string& key) const;
  int cfg_int(const std::string& key, int def) const;
  std::string cfg_str(const std::string& key, const std::string& def) const;
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const std::multimap<std::string, std::vector<std::string>>& cfg_all() const { return cfg_; }

 private:
  FILE* f_ = nullptr;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, size_t> index_;
  std::multimap<std::string, std::vector<std::string>> cfg_;
};

}  // namespace pi05
