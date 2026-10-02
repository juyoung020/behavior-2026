#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <queue>

namespace pi05 {
namespace {

// ---- minimal protobuf wire-format reader -------------------------------------------------------
struct PB {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  uint64_t varint() {
    uint64_t v = 0;
    for (int s = 0; s < 64 && p < end; s += 7) {
      uint8_t b = *p++;
      v |= uint64_t(b & 0x7f) << s;
      if (!(b & 0x80)) return v;
    }
    ok = false;
    return 0;
  }
  // Reads the next field header; returns false at end.
  bool next(uint32_t* field, uint32_t* wire) {
    if (p >= end) return false;
    uint64_t k = varint();
    *field = uint32_t(k >> 3);
    *wire = uint32_t(k & 7);
    return ok;
  }
  // For wire type 2 returns the payload slice; skips other wire types.
  bool bytes(const uint8_t** b, size_t* n) {
    uint64_t len = varint();
    if (!ok || len > size_t(end - p)) return ok = false;
    *b = p;
    *n = size_t(len);
    p += len;
    return true;
  }
  void skip(uint32_t wire) {
    const uint8_t* b;
    size_t n;
    switch (wire) {
      case 0: varint(); break;
      case 1: p += 8; break;
      case 2: bytes(&b, &n); break;
      case 5: p += 4; break;
      default: ok = false;
    }
    if (p > end) ok = false;
  }
};

size_t utf8_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;  // malformed: one byte (sentencepiece treats it as one char)
}

}  // namespace

bool Tokenizer::load(const uint8_t* data, size_t n, std::string* err) {
  pieces_.clear();
  normal_.clear();
  reserved_.clear();
  user_.clear();
  PB pb{data, data + n};
  uint32_t f, w;
  bool identity = false, add_dummy_prefix = true, remove_extra_ws = true, escape_ws = true;
  int model_type = 1;
  while (pb.next(&f, &w)) {
    const uint8_t* b;
    size_t len;
    if (w != 2) { pb.skip(w); continue; }
    if (!pb.bytes(&b, &len)) break;
    PB sub{b, b + len};
    uint32_t sf, sw;
    if (f == 1) {  // SentencePiece
      Piece pc;
      while (sub.next(&sf, &sw)) {
        if (sf == 1 && sw == 2) { const uint8_t* s; size_t sl; sub.bytes(&s, &sl); pc.piece.assign((const char*)s, sl); }
        else if (sf == 2 && sw == 5) { memcpy(&pc.score, sub.p, 4); sub.p += 4; }
        else if (sf == 3 && sw == 0) pc.type = int(sub.varint());
        else sub.skip(sw);
      }
      pieces_.push_back(std::move(pc));
    } else if (f == 2) {  // TrainerSpec
      while (sub.next(&sf, &sw)) {
        if (sw == 0) {
          uint64_t v = sub.varint();
          if (sf == 3) model_type = int(v);
          else if (sf == 35) byte_fallback_ = v != 0;
          else if (sf == 40) unk_id_ = int(v);
          else if (sf == 41) bos_id_ = int(v);
          else if (sf == 42) eos_id_ = int(v);
        } else sub.skip(sw);
      }
    } else if (f == 3) {  // NormalizerSpec
      while (sub.next(&sf, &sw)) {
        if (sf == 1 && sw == 2) { const uint8_t* s; size_t sl; sub.bytes(&s, &sl); identity = std::string((const char*)s, sl) == "identity"; }
        else if (sf == 2 && sw == 2) { const uint8_t* s; size_t sl; sub.bytes(&s, &sl); if (sl) identity = false; }
        else if (sw == 0) {
          uint64_t v = sub.varint();
          if (sf == 3) add_dummy_prefix = v != 0;
          else if (sf == 4) remove_extra_ws = v != 0;
          else if (sf == 5) escape_ws = v != 0;
        } else sub.skip(sw);
      }
    }
    if (!sub.ok) { *err = "bad protobuf submessage"; return false; }
  }
  if (!pb.ok) { *err = "bad protobuf"; return false; }
  if (model_type != 2) { *err = "not a BPE model"; return false; }
  if (!identity || add_dummy_prefix || remove_extra_ws || !escape_ws) {
    *err = "unsupported normalizer (expected identity, no dummy prefix, keep whitespace, escape whitespace)";
    return false;
  }
  for (int i = 0; i < 256; ++i) byte_ids_[i] = -1;
  std::vector<int> lens;
  for (int i = 0; i < (int)pieces_.size(); ++i) {
    const Piece& pc = pieces_[i];
    bool normal = pc.type == NORMAL || pc.type == USER_DEFINED || pc.type == UNUSED;
    auto& m = normal ? normal_ : reserved_;
    if (!m.emplace(pc.piece, i).second) { *err = "duplicate piece " + pc.piece; return false; }
    if (pc.type == USER_DEFINED) { user_.emplace(pc.piece, i); user_first_[(unsigned char)pc.piece[0]] = true; lens.push_back((int)pc.piece.size()); }
    if (pc.type == BYTE) {
      unsigned v;
      if (sscanf(pc.piece.c_str(), "<0x%02X>", &v) == 1 && v < 256) byte_ids_[v] = i;
    }
  }
  std::sort(lens.begin(), lens.end(), std::greater<int>());
  lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
  user_lens_ = lens;
  if (byte_fallback_)
    for (int i = 0; i < 256; ++i)
      if (byte_ids_[i] < 0) { *err = "missing byte piece"; return false; }
  return true;
}

int Tokenizer::piece_to_id(std::string_view s) const {
  auto it = reserved_.find(s);
  if (it != reserved_.end()) return it->second;
  auto it2 = normal_.find(s);
  if (it2 != normal_.end()) return it2->second;
  return unk_id_;
}

std::vector<int> Tokenizer::encode(std::string_view text, bool add_bos) const {
  // Normalize: identity + escape whitespace (' ' -> U+2581 "\xE2\x96\x81").
  std::string norm;
  norm.reserve(text.size() * 3);
  for (char c : text) {
    if (c == ' ') norm += "\xE2\x96\x81";
    else norm += c;
  }

  struct Symbol { int prev, next; bool freeze; std::string_view piece; };
  struct Pair { int left, right; float score; size_t size; };
  std::vector<Symbol> sym;
  std::string_view rest(norm);
  const char* base = norm.data();
  while (!rest.empty()) {
    size_t n = 0;
    bool freeze = false;
    if (user_first_[(unsigned char)rest[0]])
    for (int L : user_lens_) {  // longest user-defined prefix match
      if ((size_t)L <= rest.size() && user_.find(rest.substr(0, L)) != user_.end()) { n = L; freeze = true; break; }
    }
    if (!n) n = std::min(utf8_len((unsigned char)rest[0]), rest.size());
    int idx = (int)sym.size();
    sym.push_back({idx - 1, rest.size() == n ? -1 : idx + 1, freeze, rest.substr(0, n)});
    rest.remove_prefix(n);
  }

  auto cmp = [](const Pair& a, const Pair& b) {  // max-heap on score, ties -> smaller left first
    if (a.score != b.score) return a.score < b.score;
    return a.left > b.left;
  };
  std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> agenda(cmp);
  SvMap<std::pair<std::string_view, std::string_view>> rev_merge;
  auto add_pair = [&](int l, int r) {
    if (l == -1 || r == -1 || sym[l].freeze || sym[r].freeze) return;
    std::string_view piece(sym[l].piece.data(), sym[l].piece.size() + sym[r].piece.size());
    auto it = normal_.find(piece);
    if (it == normal_.end()) return;
    agenda.push({l, r, pieces_[it->second].score, piece.size()});
    if (is_unused(it->second)) rev_merge[std::string(piece)] = {sym[l].piece, sym[r].piece};
  };
  for (size_t i = 1; i < sym.size(); ++i) add_pair(int(i) - 1, int(i));
  while (!agenda.empty()) {
    Pair top = agenda.top();
    agenda.pop();
    Symbol& L = sym[top.left];
    Symbol& R = sym[top.right];
    if (L.piece.empty() || R.piece.empty() || L.piece.size() + R.piece.size() != top.size) continue;
    L.piece = std::string_view(L.piece.data(), L.piece.size() + R.piece.size());
    L.next = R.next;
    if (R.next >= 0) sym[R.next].prev = top.left;
    R.piece = std::string_view();
    add_pair(L.prev, top.left);
    add_pair(top.left, L.next);
  }
  (void)base;

  std::vector<int> out;
  if (add_bos) out.push_back(bos_id_);
  // resegment unused pieces, then byte-fallback unknown pieces
  std::vector<std::string_view> stack;
  auto emit = [&](std::string_view w) {
    int id = piece_to_id(w);
    if (id == unk_id_ && byte_fallback_) {
      for (unsigned char b : w) out.push_back(byte_ids_[b]);
    } else {
      out.push_back(id);
    }
  };
  for (int i = sym.empty() ? -1 : 0; i != -1; i = sym[i].next) {
    stack.push_back(sym[i].piece);
    while (!stack.empty()) {
      std::string_view w = stack.back();
      stack.pop_back();
      int id = piece_to_id(w);
      if (is_unused(id)) {
        auto it = rev_merge.find(w);
        if (it != rev_merge.end()) {
          stack.push_back(it->second.second);
          stack.push_back(it->second.first);
          continue;
        }
      }
      emit(w);
    }
  }
  return out;
}

}  // namespace pi05
