//! 손으로 짠 웹소켓 서버 쪽(RFC 6455, 확장 없음) + /healthz HTTP 응답.
//! 평가기 클라이언트(websockets.sync.client, compression=None)와 requests.get(/healthz) 를 받는다.

use crate::hash::{b64, sha1};
use std::io::{self, Read, Write};
use std::net::TcpStream;

const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// 지연 ACK 끄기(리눅스 TCP_QUICKACK). 커널이 한 번 쓰고 되돌리므로 읽을 때마다 다시 건다.
#[inline]
pub fn quickack(s: &TcpStream) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let one: libc::c_int = 1;
        unsafe {
            libc::setsockopt(
                s.as_raw_fd(),
                libc::IPPROTO_TCP,
                libc::TCP_QUICKACK,
                &one as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = s;
}

fn bad(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s.into())
}

pub struct WsConn {
    s: TcpStream,
    buf: Vec<u8>, // 읽어 두었으나 아직 안 쓴 바이트
    quick: bool,
}

pub enum Accepted {
    Ws(WsConn),
    Http,
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().skip(1).find_map(|l| {
        let (k, v) = l.split_once(':')?;
        if k.trim().eq_ignore_ascii_case(name) {
            Some(v.trim())
        } else {
            None
        }
    })
}

/// 연결 하나를 받아 HTTP 머리를 읽고: /healthz 면 200 OK 로 답하고 닫음, 웹소켓 업그레이드면 101 로 넘김.
pub fn accept(mut s: TcpStream, nodelay: bool, quick: bool) -> io::Result<Accepted> {
    if nodelay {
        s.set_nodelay(true)?;
    }
    if quick {
        quickack(&s);
    }
    let mut buf = Vec::with_capacity(2048);
    let mut tmp = [0u8; 4096];
    let (head, rest) = loop {
        let n = s.read(&mut tmp)?;
        if quick {
            quickack(&s);
        }
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "HTTP 머리 전에 끊김"));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break (String::from_utf8_lossy(&buf[..p]).into_owned(), buf[p + 4..].to_vec());
        }
        if buf.len() > 64 * 1024 {
            return Err(bad("HTTP 머리가 너무 김"));
        }
    };
    let path = head.lines().next().and_then(|l| l.split_whitespace().nth(1)).unwrap_or("/").to_string();
    let upgrade = header(&head, "Upgrade").map(|v| v.eq_ignore_ascii_case("websocket")).unwrap_or(false);
    if path == "/healthz" || !upgrade {
        let (code, body) = if path == "/healthz" { ("200 OK", "OK\n") } else { ("426 Upgrade Required", "Failed to open a WebSocket connection.\n") };
        let resp = format!(
            "HTTP/1.1 {code}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        s.write_all(resp.as_bytes())?;
        let _ = s.shutdown(std::net::Shutdown::Both);
        return Ok(Accepted::Http);
    }
    let key = header(&head, "Sec-WebSocket-Key").ok_or_else(|| bad("Sec-WebSocket-Key 없음"))?;
    let accept = b64(&sha1(format!("{}{}", key.trim(), GUID).as_bytes()));
    let resp = format!("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n");
    s.write_all(resp.as_bytes())?;
    Ok(Accepted::Ws(WsConn { s, buf: rest, quick }))
}

impl WsConn {
    fn fill(&mut self, n: usize) -> io::Result<()> {
        let mut tmp = vec![0u8; 256 * 1024];
        while self.buf.len() < n {
            let k = self.s.read(&mut tmp)?;
            if self.quick {
                quickack(&self.s);
            }
            if k == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "연결 끊김"));
            }
            self.buf.extend_from_slice(&tmp[..k]);
        }
        Ok(())
    }
    fn take(&mut self, n: usize) -> io::Result<Vec<u8>> {
        self.fill(n)?;
        let rest = self.buf.split_off(n);
        Ok(std::mem::replace(&mut self.buf, rest))
    }

    fn send_frame(&mut self, op: u8, data: &[u8]) -> io::Result<()> {
        let mut h = Vec::with_capacity(10 + data.len());
        h.push(0x80 | op);
        let n = data.len();
        if n < 126 {
            h.push(n as u8);
        } else if n < 65536 {
            h.push(126);
            h.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            h.push(127);
            h.extend_from_slice(&(n as u64).to_be_bytes());
        }
        h.extend_from_slice(data);
        self.s.write_all(&h)
    }

    pub fn send_bin(&mut self, data: &[u8]) -> io::Result<()> {
        self.send_frame(0x2, data)
    }

    /// 메시지 하나(바이너리·텍스트, 조각 이어 붙임). 닫힘이면 None. ping 에는 pong 으로 답한다.
    pub fn recv(&mut self) -> io::Result<Option<Vec<u8>>> {
        let mut msg: Vec<u8> = Vec::new();
        loop {
            let h = self.take(2)?;
            let fin = h[0] & 0x80 != 0;
            let op = h[0] & 0x0f;
            let masked = h[1] & 0x80 != 0;
            let mut n = (h[1] & 0x7f) as u64;
            if n == 126 {
                let b = self.take(2)?;
                n = u16::from_be_bytes([b[0], b[1]]) as u64;
            } else if n == 127 {
                let b = self.take(8)?;
                n = u64::from_be_bytes(b.try_into().unwrap());
            }
            let mask = if masked { Some(self.take(4)?) } else { None };
            let mut payload = self.take(n as usize)?;
            if let Some(m) = mask {
                for (i, x) in payload.iter_mut().enumerate() {
                    *x ^= m[i & 3];
                }
            }
            match op {
                0x0 | 0x1 | 0x2 => {
                    msg.extend_from_slice(&payload);
                    if fin {
                        return Ok(Some(msg));
                    }
                }
                0x8 => {
                    // 닫기: 받은 상태 코드를 그대로 돌려주고 끝
                    let _ = self.send_frame(0x8, &payload[..payload.len().min(2)]);
                    let _ = self.s.shutdown(std::net::Shutdown::Both);
                    return Ok(None);
                }
                0x9 => self.send_frame(0xA, &payload)?,
                0xA => {}
                _ => return Err(bad(format!("모르는 opcode {op}"))),
            }
        }
    }
}
