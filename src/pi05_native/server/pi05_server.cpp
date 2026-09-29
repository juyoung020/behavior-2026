// pi05_server: native openpi-protocol websocket policy server (Linux), the submission-side twin of the in-process
// evaluator glue. Protocol = openpi websocket_b1k_server (msgpack + numpy "__ndarray__" maps, see
// BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/network_utils.py):
//   connect  -> server sends msgpack(metadata = {})
//   request  -> obs map (camera rgb [B?,H,W,C] uint8, "<robot>::proprio" [B?,P] f32, ...), optional
//               "__action_chunk_size__": K
//   response -> {"action": [B?, 23] f32, "server_timing": {...}} (+ "action_chunk": [B?, K, 23] when asked)
//   {"reset": true} -> reset, no response.   GET /healthz -> 200 OK.
// Hand-written RFC 6455 websocket, SHA-1/base64 and msgpack (same approach as the Rust relay src/agent/src/ws.rs:
// TCP_NODELAY, TCP_QUICKACK re-armed after every read, one writev per response). No third-party libraries.
//
//   pi05_server --weights W.pi05w [--port 8000] [--prompt "..."] [--replan 16] [--device 0]
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../include/pi05_native.h"

namespace {

// ---------------------------------------------------------------- SHA-1 + base64 (handshake only)
std::string sha1(const std::string& msg) {
  uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
  std::string m = msg;
  uint64_t bits = (uint64_t)msg.size() * 8;
  m += (char)0x80;
  while (m.size() % 64 != 56) m += (char)0;
  for (int i = 7; i >= 0; --i) m += (char)((bits >> (i * 8)) & 0xff);
  auto rol = [](uint32_t x, int n) { return (x << n) | (x >> (32 - n)); };
  for (size_t off = 0; off < m.size(); off += 64) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint8_t)m[off + 4 * i] << 24 | (uint8_t)m[off + 4 * i + 1] << 16 | (uint8_t)m[off + 4 * i + 2] << 8 |
             (uint8_t)m[off + 4 * i + 3];
    for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20) f = (b & c) | (~b & d), k = 0x5A827999;
      else if (i < 40) f = b ^ c ^ d, k = 0x6ED9EBA1;
      else if (i < 60) f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDC;
      else f = b ^ c ^ d, k = 0xCA62C1D6;
      uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d, d = c, c = rol(b, 30), b = a, a = t;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e;
  }
  std::string out;
  for (uint32_t v : h)
    for (int i = 3; i >= 0; --i) out += (char)((v >> (i * 8)) & 0xff);
  return out;
}

std::string b64(const std::string& s) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  size_t i = 0;
  for (; i + 2 < s.size(); i += 3) {
    uint32_t v = (uint8_t)s[i] << 16 | (uint8_t)s[i + 1] << 8 | (uint8_t)s[i + 2];
    o += T[v >> 18], o += T[(v >> 12) & 63], o += T[(v >> 6) & 63], o += T[v & 63];
  }
  if (i + 1 == s.size()) {
    uint32_t v = (uint8_t)s[i] << 16;
    o += T[v >> 18], o += T[(v >> 12) & 63], o += "==";
  } else if (i + 2 == s.size()) {
    uint32_t v = (uint8_t)s[i] << 16 | (uint8_t)s[i + 1] << 8;
    o += T[v >> 18], o += T[(v >> 12) & 63], o += T[(v >> 6) & 63], o += '=';
  }
  return o;
}

// ---------------------------------------------------------------- socket helpers
void quickack(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one);
}

struct Conn {
  int fd;
  std::vector<uint8_t> rb;
  size_t rp = 0, re = 0;
  bool fill() {
    if (rp == re) rp = re = 0;
    if (rb.size() - re < 65536) rb.resize(re + (1 << 20));
    ssize_t n = ::read(fd, rb.data() + re, rb.size() - re);
    quickack(fd);  // the kernel turns delayed ACK back on by itself: re-arm after every read
    if (n <= 0) return false;
    re += (size_t)n;
    return true;
  }
  bool read_exact(uint8_t* dst, size_t n) {
    while (n) {
      if (rp == re && !fill()) return false;
      size_t k = std::min(n, re - rp);
      memcpy(dst, rb.data() + rp, k);
      rp += k, dst += k, n -= k;
    }
    return true;
  }
  bool write_all(const iovec* iov, int cnt) {
    std::vector<iovec> v(iov, iov + cnt);
    size_t i = 0;
    while (i < v.size()) {
      ssize_t n = ::writev(fd, v.data() + i, (int)(v.size() - i));
      if (n < 0) return false;
      while (n > 0 && i < v.size()) {
        if ((size_t)n >= v[i].iov_len) n -= v[i].iov_len, ++i;
        else v[i].iov_base = (char*)v[i].iov_base + n, v[i].iov_len -= n, n = 0;
      }
    }
    return true;
  }
  // Reads one complete message (reassembles fragments, answers ping). Returns opcode (1 text, 2 binary, 8 close).
  int read_message(std::vector<uint8_t>& msg) {
    msg.clear();
    int op0 = -1;
    for (;;) {
      uint8_t h[2];
      if (!read_exact(h, 2)) return -1;
      const bool fin = h[0] & 0x80;
      const int op = h[0] & 0x0f;
      uint64_t len = h[1] & 0x7f;
      if (len == 126) { uint8_t b[2]; if (!read_exact(b, 2)) return -1; len = (b[0] << 8) | b[1]; }
      else if (len == 127) {
        uint8_t b[8];
        if (!read_exact(b, 8)) return -1;
        len = 0;
        for (int i = 0; i < 8; ++i) len = (len << 8) | b[i];
      }
      uint8_t key[4] = {0, 0, 0, 0};
      const bool masked = h[1] & 0x80;
      if (masked && !read_exact(key, 4)) return -1;
      size_t at = msg.size();
      if (op >= 8) {  // control frame
        std::vector<uint8_t> p(len);
        if (!read_exact(p.data(), len)) return -1;
        for (size_t i = 0; i < len; ++i) p[i] ^= key[i & 3];
        if (op == 9) send(10, p.data(), p.size());
        if (op == 8) return 8;
        continue;
      }
      msg.resize(at + len);
      if (!read_exact(msg.data() + at, len)) return -1;
      if (masked)
        for (size_t i = 0; i < len; ++i) msg[at + i] ^= key[i & 3];
      if (op0 < 0) op0 = op;
      if (fin) return op0;
    }
  }
  bool send(int op, const void* data, size_t n) {
    uint8_t h[10];
    size_t hl = 2;
    h[0] = 0x80 | op;
    if (n < 126) h[1] = (uint8_t)n;
    else if (n < 65536) h[1] = 126, h[2] = n >> 8, h[3] = n & 0xff, hl = 4;
    else {
      h[1] = 127;
      for (int i = 0; i < 8; ++i) h[2 + i] = (uint8_t)(n >> (56 - 8 * i));
      hl = 10;
    }
    iovec iov[2] = {{h, hl}, {(void*)data, n}};
    return write_all(iov, 2);
  }
};

// ---------------------------------------------------------------- msgpack
struct MP {  // reader over one buffer
  const uint8_t* p;
  const uint8_t* e;
  bool ok = true;
  uint8_t u8() { if (p >= e) { ok = false; return 0; } return *p++; }
  uint64_t be(int n) { uint64_t v = 0; for (int i = 0; i < n; ++i) v = (v << 8) | u8(); return v; }
  // Value view: type 's' str, 'b' bin, 'i' int, 'f' float, 'm' map, 'a' array, 't' bool, 'n' nil
  struct V { char t = 'n'; const uint8_t* data = nullptr; uint64_t n = 0; int64_t i = 0; double f = 0; const uint8_t* at = nullptr; };
  V next() {
    V v;
    v.at = p;
    uint8_t c = u8();
    if (c <= 0x7f) { v.t = 'i'; v.i = c; }
    else if (c >= 0xe0) { v.t = 'i'; v.i = (int8_t)c; }
    else if ((c & 0xf0) == 0x80) { v.t = 'm'; v.n = c & 0x0f; }
    else if ((c & 0xf0) == 0x90) { v.t = 'a'; v.n = c & 0x0f; }
    else if ((c & 0xe0) == 0xa0) { v.t = 's'; v.n = c & 0x1f; }
    else switch (c) {
      case 0xc0: v.t = 'n'; break;
      case 0xc2: v.t = 't'; v.i = 0; break;
      case 0xc3: v.t = 't'; v.i = 1; break;
      case 0xc4: v.t = 'b'; v.n = be(1); break;
      case 0xc5: v.t = 'b'; v.n = be(2); break;
      case 0xc6: v.t = 'b'; v.n = be(4); break;
      case 0xca: { v.t = 'f'; uint32_t u = (uint32_t)be(4); float f; memcpy(&f, &u, 4); v.f = f; break; }
      case 0xcb: { v.t = 'f'; uint64_t u = be(8); memcpy(&v.f, &u, 8); break; }
      case 0xcc: v.t = 'i'; v.i = (int64_t)be(1); break;
      case 0xcd: v.t = 'i'; v.i = (int64_t)be(2); break;
      case 0xce: v.t = 'i'; v.i = (int64_t)be(4); break;
      case 0xcf: v.t = 'i'; v.i = (int64_t)be(8); break;
      case 0xd0: v.t = 'i'; v.i = (int8_t)be(1); break;
      case 0xd1: v.t = 'i'; v.i = (int16_t)be(2); break;
      case 0xd2: v.t = 'i'; v.i = (int32_t)be(4); break;
      case 0xd3: v.t = 'i'; v.i = (int64_t)be(8); break;
      case 0xd9: v.t = 's'; v.n = be(1); break;
      case 0xda: v.t = 's'; v.n = be(2); break;
      case 0xdb: v.t = 's'; v.n = be(4); break;
      case 0xdc: v.t = 'a'; v.n = be(2); break;
      case 0xdd: v.t = 'a'; v.n = be(4); break;
      case 0xde: v.t = 'm'; v.n = be(2); break;
      case 0xdf: v.t = 'm'; v.n = be(4); break;
      default: ok = false;
    }
    if (v.t == 's' || v.t == 'b') {
      if ((uint64_t)(e - p) < v.n) { ok = false; return v; }
      v.data = p;
      p += v.n;
    }
    return v;
  }
  void skip(const V& v) {  // skip the children of a container
    if (v.t == 'm') for (uint64_t i = 0; i < 2 * v.n && ok; ++i) skip(next());
    else if (v.t == 'a') for (uint64_t i = 0; i < v.n && ok; ++i) skip(next());
  }
};

struct NdArr {
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  std::string dtype;
  std::vector<int64_t> shape;
};

// Parses {b"__ndarray__": True, b"data": bin, b"dtype": str, b"shape": [..]} (keys may be str or bin).
bool parse_nd(MP& r, const MP::V& m, NdArr* out) {
  bool is_nd = false;
  for (uint64_t i = 0; i < m.n && r.ok; ++i) {
    MP::V k = r.next();
    std::string key = (k.t == 's' || k.t == 'b') ? std::string((const char*)k.data, k.n) : "";
    MP::V v = r.next();
    if (key == "__ndarray__") is_nd = v.t == 't' && v.i;
    else if (key == "data" && (v.t == 'b' || v.t == 's')) out->data = v.data, out->nbytes = v.n;
    else if (key == "dtype" && (v.t == 's' || v.t == 'b')) out->dtype.assign((const char*)v.data, v.n);
    else if (key == "shape" && v.t == 'a') {
      for (uint64_t j = 0; j < v.n; ++j) out->shape.push_back(r.next().i);
    } else r.skip(v);
  }
  return is_nd && r.ok;
}

struct W {  // msgpack writer
  std::vector<uint8_t> b;
  void u8(uint8_t v) { b.push_back(v); }
  void be(uint64_t v, int n) { for (int i = n - 1; i >= 0; --i) b.push_back((uint8_t)(v >> (8 * i))); }
  void map(uint32_t n) { if (n < 16) u8(0x80 | n); else u8(0xde), be(n, 2); }
  void arr(uint32_t n) { if (n < 16) u8(0x90 | n); else u8(0xdc), be(n, 2); }
  void str(const std::string& s) {
    if (s.size() < 32) u8(0xa0 | s.size());
    else if (s.size() < 256) u8(0xd9), be(s.size(), 1);
    else u8(0xda), be(s.size(), 2);
    b.insert(b.end(), s.begin(), s.end());
  }
  void bin_head(uint64_t n) { if (n < 256) u8(0xc4), be(n, 1); else if (n < 65536) u8(0xc5), be(n, 2); else u8(0xc6), be(n, 4); }
  void bin(const std::string& s) { bin_head(s.size()); b.insert(b.end(), s.begin(), s.end()); }
  void uint(uint64_t v) { if (v < 128) u8((uint8_t)v); else u8(0xcf), be(v, 8); }
  void f64(double d) { uint64_t u; memcpy(&u, &d, 8); u8(0xcb); be(u, 8); }
  void boolean(bool v) { u8(v ? 0xc3 : 0xc2); }
  // numpy float32 array in the openpi encoding; payload bytes appended by the caller as a separate iovec
  void nd_f32_head(const std::vector<int64_t>& shape, size_t nbytes) {
    map(4);
    bin("__ndarray__"); boolean(true);
    bin("data"); bin_head(nbytes);
  }
  void nd_f32_tail(const std::vector<int64_t>& shape) {
    bin("dtype"); str("<f4");
    bin("shape"); arr((uint32_t)shape.size());
    for (auto d : shape) uint((uint64_t)d);
  }
};

// ---------------------------------------------------------------- server
struct Server {
  Pi05Engine* eng = nullptr;
  Pi05Info info{};
  std::mutex mu;
  std::string prompt;
  int replan = 16;
  std::atomic<int> next_conn{0};
};

bool http_head(Conn& c, std::string* head) {
  for (;;) {
    std::string s((const char*)c.rb.data() + c.rp, c.re - c.rp);
    size_t p = s.find("\r\n\r\n");
    if (p != std::string::npos) {
      *head = s.substr(0, p);
      c.rp += p + 4;
      return true;
    }
    if (c.re - c.rp > 65536 || !c.fill()) return false;
  }
}

std::string header(const std::string& head, const std::string& name) {
  size_t pos = 0;
  while ((pos = head.find("\r\n", pos)) != std::string::npos) {
    pos += 2;
    size_t colon = head.find(':', pos), eol = head.find("\r\n", pos);
    if (colon == std::string::npos || (eol != std::string::npos && colon > eol)) continue;
    std::string k = head.substr(pos, colon - pos);
    bool eq = k.size() == name.size();
    for (size_t i = 0; eq && i < k.size(); ++i) eq = tolower(k[i]) == tolower(name[i]);
    if (eq) {
      std::string v = head.substr(colon + 1, (eol == std::string::npos ? head.size() : eol) - colon - 1);
      size_t a = v.find_first_not_of(' '), b = v.find_last_not_of(" \r");
      return a == std::string::npos ? "" : v.substr(a, b - a + 1);
    }
  }
  return "";
}

void serve(Server* S, int fd) {
  Conn c{fd};
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  quickack(fd);
  std::string head;
  if (!http_head(c, &head)) { close(fd); return; }
  const std::string path = head.substr(head.find(' ') + 1, head.find(' ', head.find(' ') + 1) - head.find(' ') - 1);
  std::string up = header(head, "upgrade");
  for (auto& ch : up) ch = (char)tolower(ch);
  if (up != "websocket") {
    std::string body = path == "/healthz" ? "OK\n" : "not found\n";
    std::string resp = std::string("HTTP/1.1 ") + (path == "/healthz" ? "200 OK" : "404 Not Found") +
                       "\r\nContent-Type: text/plain\r\nContent-Length: " + std::to_string(body.size()) +
                       "\r\nConnection: close\r\n\r\n" + body;
    ::write(fd, resp.data(), resp.size());
    close(fd);
    return;
  }
  std::string resp = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " +
                     b64(sha1(header(head, "sec-websocket-key") + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")) + "\r\n\r\n";
  ::write(fd, resp.data(), resp.size());
  const int conn_id = S->next_conn++;
  {  // metadata
    W w;
    w.map(0);
    c.send(2, w.b.data(), w.b.size());
  }
  const std::string prop_key = std::string(S->info.robot_name) + "::proprio";
  std::vector<uint8_t> msg;
  std::vector<float> acts, chunk;
  for (;;) {
    int op = c.read_message(msg);
    if (op < 0 || op == 8) break;
    auto t0 = std::chrono::steady_clock::now();
    MP r{msg.data(), msg.data() + msg.size()};
    MP::V top = r.next();
    if (top.t != 'm') break;
    NdArr cams[3], prop;
    bool reset = false;
    int64_t chunk_k = 0;
    for (uint64_t i = 0; i < top.n && r.ok; ++i) {
      MP::V k = r.next();
      std::string key = (k.t == 's' || k.t == 'b') ? std::string((const char*)k.data, k.n) : "";
      MP::V v = r.next();
      int cam = -1;
      for (int j = 0; j < 3; ++j)
        if (key == S->info.cam_keys[j]) cam = j;
      if (key == "reset") reset = true, r.skip(v);
      else if (key == "__action_chunk_size__") chunk_k = v.i;
      else if (cam >= 0 && v.t == 'm') parse_nd(r, v, &cams[cam]);
      else if (key == prop_key && v.t == 'm') parse_nd(r, v, &prop);
      else r.skip(v);
    }
    if (!r.ok) { fprintf(stderr, "pi05_server: bad msgpack\n"); break; }
    if (reset) {
      std::lock_guard<std::mutex> g(S->mu);
      for (int b = 0; b < 64; ++b) pi05_reset(S->eng, conn_id * 64 + b);
      continue;
    }
    if (prop.dtype != "<f4" || !cams[0].data || !cams[1].data || !cams[2].data) {
      std::string e = "pi05_server: request needs float32 " + prop_key + " and the three camera images";
      c.send(1, e.data(), e.size());
      break;
    }
    const bool batched = prop.shape.size() == 2;
    const int B = batched ? (int)prop.shape[0] : 1;
    const int P = (int)prop.shape.back();
    const int ad = S->info.action_dim;
    acts.assign((size_t)B * ad, 0.f);
    const int K = chunk_k > 1 ? (int)chunk_k : 0;
    chunk.assign((size_t)B * (K ? K : 1) * ad, 0.f);
    float infer_ms = 0;
    bool fail = false;
    {
      std::lock_guard<std::mutex> g(S->mu);
      for (int b = 0; b < B && !fail; ++b) {
        Pi05Image im[3];
        for (int j = 0; j < 3; ++j) {
          const auto& s = cams[j].shape;
          const int nd = (int)s.size();
          const int H = (int)s[nd - 3], Wd = (int)s[nd - 2], C = (int)s[nd - 1];
          im[j] = {cams[j].data + (size_t)b * (batched ? (size_t)H * Wd * C : 0), H, Wd, (int64_t)Wd * C, C, 0};
        }
        const float* pp = reinterpret_cast<const float*>(prop.data) + (size_t)b * P;
        Pi05Timing t{};
        // one action per step; with a chunk request, the next K actions of the same receding-horizon chunk
        for (int k = 0; k < (K ? K : 1); ++k) {
          float* dst = K ? &chunk[((size_t)b * K + k) * ad] : &acts[(size_t)b * ad];
          int rc = pi05_act(S->eng, conn_id * 64 + b, im, pp, P, S->prompt.c_str(), S->replan, dst, &t);
          if (rc < 0) { fail = true; break; }
          if (k > 0 && rc == 1) {
            fprintf(stderr, "pi05_server: chunk of %d crosses the replanning boundary (replan %d)\n", K, S->replan);
          }
          infer_ms += t.total_ms;
        }
        if (K) memcpy(&acts[(size_t)b * ad], &chunk[(size_t)b * K * ad], ad * 4);
      }
    }
    if (fail) {
      std::string e = std::string("pi05_server: ") + pi05_last_error(S->eng);
      c.send(1, e.data(), e.size());
      break;
    }
    // response: {"action": nd, ["action_chunk": nd,] "server_timing": {"infer_ms": x}}
    std::vector<int64_t> ashape = batched ? std::vector<int64_t>{B, ad} : std::vector<int64_t>{ad};
    std::vector<int64_t> cshape = batched ? std::vector<int64_t>{B, K, ad} : std::vector<int64_t>{K, ad};
    W h1, h2, h3;
    h1.map(K ? 3 : 2);
    h1.str("action");
    h1.nd_f32_head(ashape, acts.size() * 4);
    h2.nd_f32_tail(ashape);
    if (K) {
      h2.str("action_chunk");
      h2.nd_f32_head(cshape, chunk.size() * 4);
    }
    const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (K) h3.nd_f32_tail(cshape);
    h3.str("server_timing");
    h3.map(2);
    h3.str("infer_ms"); h3.f64(infer_ms);
    h3.str("total_ms"); h3.f64(total);
    const size_t n = h1.b.size() + acts.size() * 4 + h2.b.size() + (K ? chunk.size() * 4 : 0) + h3.b.size();
    uint8_t fh[10];
    size_t fl = 2;
    fh[0] = 0x82;
    if (n < 126) fh[1] = (uint8_t)n;
    else if (n < 65536) fh[1] = 126, fh[2] = n >> 8, fh[3] = n & 0xff, fl = 4;
    else { fh[1] = 127; for (int i = 0; i < 8; ++i) fh[2 + i] = (uint8_t)(n >> (56 - 8 * i)); fl = 10; }
    iovec iov[6] = {{fh, fl}, {h1.b.data(), h1.b.size()}, {acts.data(), acts.size() * 4}, {h2.b.data(), h2.b.size()},
                    {chunk.data(), K ? chunk.size() * 4 : 0}, {h3.b.data(), h3.b.size()}};
    if (!c.write_all(iov, 6)) break;
  }
  {
    std::lock_guard<std::mutex> g(S->mu);
    for (int b = 0; b < 64; ++b) pi05_reset(S->eng, conn_id * 64 + b);
  }
  close(fd);
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights, prompt = "Turn on the radio receiver that's on the table in the living room.";
  int port = 8000, device = 0, replan = 16;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--weights") weights = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "--prompt") prompt = next();
    else if (a == "--replan") replan = std::stoi(next());
    else if (a == "--device") device = std::stoi(next());
  }
  if (weights.empty()) { fprintf(stderr, "usage: pi05_server --weights W.pi05w [--port 8000] [--prompt ...] [--replan 16]\n"); return 2; }
  Server S;
  char err[512];
  auto t0 = std::chrono::steady_clock::now();
  S.eng = pi05_create(weights.c_str(), device, err, sizeof err);
  if (!S.eng) { fprintf(stderr, "pi05_server: %s\n", err); return 1; }
  pi05_info(S.eng, &S.info);
  S.prompt = prompt;
  S.replan = replan;
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;
  if (bind(ls, (sockaddr*)&addr, sizeof addr) != 0 || listen(ls, 16) != 0) { perror("bind/listen"); return 1; }
  fprintf(stderr, "pi05_server: ready on :%d in %.1f s (weights %.2f GiB), prompt=\"%s\", replan %d\n", port,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), S.info.weight_bytes / 1073741824.0,
          prompt.c_str(), replan);
  for (;;) {
    int fd = accept(ls, nullptr, nullptr);
    if (fd < 0) continue;
    std::thread(serve, &S, fd).detach();
  }
}
