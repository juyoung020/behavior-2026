// SentencePiece BPE encoder (PaliGemma / Gemma tokenizer), written from the algorithm in
// sentencepiece bpe_model.cc (Model::Encode) + sentencepiece_processor.cc (byte fallback).
// Only what the PaliGemma model uses: identity normalizer, escape_whitespaces, no dummy prefix,
// user-defined symbols (frozen, longest prefix match), unused-piece resegmentation, byte fallback.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pi05 {

struct SvHash {  // heterogeneous lookup: find(string_view) without building a std::string
  using is_transparent = void;
  size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};
template <class V>
using SvMap = std::unordered_map<std::string, V, SvHash, std::equal_to<>>;

class Tokenizer {
 public:
  // Parses a serialized sentencepiece ModelProto. Returns false (and sets err) on anything unexpected.
  bool load(const uint8_t* data, size_t n, std::string* err);
  // Encodes UTF-8 text. add_bos prepends bos_id.
  std::vector<int> encode(std::string_view text, bool add_bos) const;
  int bos_id() const { return bos_id_; }
  int eos_id() const { return eos_id_; }
  int vocab_size() const { return (int)pieces_.size(); }
  const std::string& piece(int id) const { return pieces_[id].piece; }

 private:
  enum Type { NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, BYTE = 6, UNUSED = 5 };
  struct Piece { std::string piece; float score = 0; int type = NORMAL; };
  std::vector<Piece> pieces_;
  SvMap<int> normal_;    // NORMAL, USER_DEFINED, UNUSED (sentencepiece pieces_)
  SvMap<int> reserved_;  // CONTROL, UNKNOWN, BYTE (reserved_id_map_)
  SvMap<int> user_;      // user-defined symbols for the prefix matcher
  std::vector<int> user_lens_;                     // distinct byte lengths of user symbols, descending
  bool user_first_[256] = {};                      // first bytes that can start a user symbol
  int byte_ids_[256];
  int unk_id_ = 0, bos_id_ = 1, eos_id_ = 2;
  bool byte_fallback_ = false;

  int piece_to_id(std::string_view s) const;
  bool is_unused(int id) const { return id >= 0 && pieces_[id].type == UNUSED; }
};

}  // namespace pi05
