//! replaysrv — 기록해 둔 행동열을 그대로 돌려주는 정책 서버(평가기 일치 검증·실행기 재생 판용). 받은 관측은 해시·값으로 남긴다.
//! tools/replay_policy_server.py 의 Rust 판: 같은 프로토콜, 같은 행동 바이트, 같은 관측 기록(server_log.npz 내용).
//!
//!     replaysrv --actions <결과폴더>/actions.npz --port 8010 [--log <폴더>/server_log.npz] [--once]
//!               [--perturb STEP:DIM:DELTA] [--quickack] [--host 127.0.0.1]
//!
//! - 프로토콜: omnigibson/eval/utils/network_utils.py 의 WebsocketPolicyServer 와 같다(msgpack + 넘파이 확장,
//!   접속하면 metadata {} 먼저, {"reset": True} 는 응답 없음, GET /healthz -> 200 OK).
//! - 기록: 요청마다 배열 값 k 가 uint8 이거나 한 행 원소가 4096 넘으면 행마다 blake2b-128 해시("hash::k"), 아니면
//!   float64 값("val::k"). reset 또는 연결 끝에 server_log.npz 로(파이썬판처럼 같은 경로에 덮어씀).
//! - --quickack: TCP_NODELAY + 읽을 때마다 TCP_QUICKACK(리눅스). 응답 바이트는 그대로.
//! - 기록된 환경 수와 요청의 환경 수가 다르면 0 번 환경의 행동을 모든 환경에 준다. 기록이 끝나면 0 행동.

mod hash;
mod mp;
mod ws;

use mp::{as_nd, Mp};
use npz::{Array, Data};
use std::collections::{BTreeMap, BTreeSet};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn log(msg: &str) {
    eprintln!("{} {}", chrono_like_now(), msg);
}

fn chrono_like_now() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:03}", t.as_secs(), t.subsec_millis())
}

enum Rec {
    Hash(Vec<String>),
    Val(Vec<usize>, Vec<f64>), // shape (n, d), 값
}

struct Replay {
    actions: Vec<f32>, // (T, N, A)
    t_len: usize,
    n_rec: usize,
    a_dim: usize,
    log_path: String,
    perturb: Option<(usize, usize, f32)>,
    t: usize,
    rows: Vec<BTreeMap<String, Rec>>,
    episode: usize,
}

/// 넘파이 dtype 문자열 -> (원소 바이트 수, f64 로 바꾸는 함수)
fn elem(dtype: &str) -> Option<(usize, fn(&[u8]) -> f64)> {
    Some(match dtype {
        "<f8" => (8, |b| f64::from_le_bytes(b.try_into().unwrap())),
        "<f4" => (4, |b| f32::from_le_bytes(b.try_into().unwrap()) as f64),
        "<i8" => (8, |b| i64::from_le_bytes(b.try_into().unwrap()) as f64),
        "<i4" => (4, |b| i32::from_le_bytes(b.try_into().unwrap()) as f64),
        "<i2" => (2, |b| i16::from_le_bytes(b.try_into().unwrap()) as f64),
        "|i1" => (1, |b| b[0] as i8 as f64),
        "<u8" => (8, |b| u64::from_le_bytes(b.try_into().unwrap()) as f64),
        "<u4" => (4, |b| u32::from_le_bytes(b.try_into().unwrap()) as f64),
        "<u2" => (2, |b| u16::from_le_bytes(b.try_into().unwrap()) as f64),
        "|u1" => (1, |b| b[0] as f64),
        "|b1" => (1, |b| (b[0] != 0) as u8 as f64),
        _ => return None,
    })
}

impl Replay {
    fn record(&mut self, obs: &[(Mp, Mp)]) {
        let mut row = BTreeMap::new();
        for (k, v) in obs {
            let key = match k {
                Mp::Str(s) => s.clone(),
                _ => continue,
            };
            let Some(nd) = as_nd(v) else { continue }; // 배열이 아닌 값(np.asarray 가 0 차원)은 건너뜀 — 파이썬판과 같음
            if nd.shape.is_empty() {
                continue;
            }
            let Some((isz, conv)) = elem(nd.dtype) else { continue };
            let n = nd.shape[0];
            let per: usize = nd.shape[1..].iter().product();
            if nd.dtype == "|u1" || per > 4096 {
                let rb = per * isz;
                let hs = (0..n).map(|i| hash::hex(&hash::blake2b(&nd.data[i * rb..(i + 1) * rb], 16))).collect();
                row.insert(format!("hash::{key}"), Rec::Hash(hs));
            } else {
                let vals = nd.data.chunks_exact(isz).take(n * per).map(conv).collect();
                row.insert(format!("val::{key}"), Rec::Val(vec![n, per], vals));
            }
        }
        self.rows.push(row);
    }

    fn save(&self) {
        if self.log_path.is_empty() || self.rows.is_empty() {
            return;
        }
        let keys: BTreeSet<&String> = self.rows.iter().flat_map(|r| r.keys()).collect();
        let t = self.rows.len();
        let mut out: Vec<(String, Array)> = Vec::new();
        for k in keys {
            if k.starts_with("hash::") {
                let n = self.rows.iter().filter_map(|r| match r.get(k) { Some(Rec::Hash(h)) => Some(h.len()), _ => None }).max().unwrap_or(0);
                let mut v = Vec::with_capacity(t * n);
                for r in &self.rows {
                    match r.get(k) {
                        Some(Rec::Hash(h)) => v.extend(h.iter().cloned()),
                        _ => v.extend(std::iter::repeat(String::new()).take(n)),
                    }
                }
                out.push((k.clone(), Array { shape: vec![t, n], kind: 'U', data: Data::S(v) }));
            } else {
                let shape = self.rows.iter().find_map(|r| match r.get(k) { Some(Rec::Val(s, _)) => Some(s.clone()), _ => None }).unwrap();
                let sz: usize = shape.iter().product();
                let mut v = Vec::with_capacity(t * sz);
                for r in &self.rows {
                    match r.get(k) {
                        Some(Rec::Val(_, x)) => v.extend_from_slice(x),
                        _ => v.extend(std::iter::repeat(f64::NAN).take(sz)),
                    }
                }
                let mut sh = vec![t];
                sh.extend(shape);
                out.push((k.clone(), Array { shape: sh, kind: 'f', data: Data::F(v) }));
            }
        }
        match npz::save_npz(&self.log_path, &out) {
            Ok(()) => log(&format!("관측 기록 저장: {} ({} 스텝)", self.log_path, t)),
            Err(e) => log(&format!("관측 기록 저장 실패: {e}")),
        }
    }

    /// 파이썬판 act() 와 같다. 반환: (행동 f32 바이트, shape)
    fn act(&mut self, obs: &[(Mp, Mp)]) -> Result<(Vec<u8>, Vec<usize>), String> {
        self.record(obs);
        let prop = obs
            .iter()
            .find(|(k, _)| matches!(k, Mp::Str(s) if s.ends_with("::proprio")))
            .and_then(|(_, v)| as_nd(v))
            .ok_or("요청에 ::proprio 가 없다")?;
        let batched = prop.shape.len() > 1;
        let n = if batched { prop.shape[0] } else { 1 };
        let ad = self.a_dim;
        let mut a = vec![0f32; n * ad];
        if self.t < self.t_len {
            let base = self.t * self.n_rec * ad;
            for e in 0..n {
                let src = if self.n_rec != n { 0 } else { e };
                a[e * ad..(e + 1) * ad].copy_from_slice(&self.actions[base + src * ad..base + (src + 1) * ad]);
            }
        }
        if let Some((s, d, delta)) = self.perturb {
            if self.t == s {
                for e in 0..n {
                    a[e * ad + d] += delta;
                }
                log(&format!("음성 대조: 스텝 {} 행동[{}] += {}", self.t, d, delta));
            }
        }
        self.t += 1;
        let (data, shape) = if batched { (a, vec![n, ad]) } else { (a[..ad].to_vec(), vec![ad]) };
        Ok((data.iter().flat_map(|x| x.to_le_bytes()).collect(), shape))
    }
}

fn serve(conn: &mut ws::WsConn, st: &Arc<Mutex<Replay>>) -> std::io::Result<()> {
    let mut o = Vec::new();
    mp::w_map(&mut o, 0); // metadata {}
    conn.send_bin(&o)?;
    while let Some(msg) = conn.recv()? {
        let v = mp::Rd::new(&msg).read().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let Mp::Map(kv) = v else { continue };
        let mut r = st.lock().unwrap();
        if kv.iter().any(|(k, _)| matches!(k, Mp::Str(s) if s == "reset")) {
            if !r.rows.is_empty() {
                r.save();
            }
            r.t = 0;
            r.rows.clear();
            r.episode += 1;
            continue;
        }
        let t0 = Instant::now();
        let (data, shape) = r.act(&kv).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let infer_ms = t0.elapsed().as_secs_f64() * 1e3;
        drop(r);
        let mut o = Vec::with_capacity(data.len() + 128);
        mp::w_map(&mut o, 2);
        mp::w_str(&mut o, "action");
        mp::w_nd(&mut o, "<f4", &shape, &data);
        mp::w_str(&mut o, "server_timing");
        mp::w_map(&mut o, 1);
        mp::w_str(&mut o, "infer_ms");
        mp::w_f64(&mut o, infer_ms);
        conn.send_bin(&o)?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
            .or_else(|| args.iter().find_map(|a| a.strip_prefix(&format!("{name}=")).map(|s| s.to_string())))
    };
    let flag = |name: &str| args.iter().any(|a| a == name);
    let Some(actions_path) = get("--actions") else {
        eprintln!("쓰는 법: replaysrv --actions A.npz [--port 8010] [--log L.npz] [--once] [--perturb S:D:V] [--quickack] [--host 127.0.0.1]");
        std::process::exit(2);
    };
    let port: u16 = get("--port").and_then(|p| p.parse().ok()).unwrap_or(8010);
    let host = get("--host").unwrap_or_else(|| "127.0.0.1".into());
    let once = flag("--once");
    let quick = flag("--quickack");
    let perturb = get("--perturb").filter(|s| !s.is_empty()).map(|s| {
        let p: Vec<&str> = s.split(':').collect();
        (p[0].parse::<usize>().unwrap(), p[1].parse::<usize>().unwrap(), p[2].parse::<f64>().unwrap() as f32)
    });
    let arrs = npz::load_npz(&actions_path).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) });
    let a = arrs.into_iter().find(|(k, _)| k == "actions").map(|x| x.1).unwrap_or_else(|| { eprintln!("actions 키 없음"); std::process::exit(2) });
    let (t_len, n_rec, a_dim) = match a.shape.len() {
        2 => (a.shape[0], 1, a.shape[1]),
        3 => (a.shape[0], a.shape[1], a.shape[2]),
        _ => { eprintln!("actions 모양이 (T,A) 나 (T,N,A) 가 아님"); std::process::exit(2) }
    };
    let Data::F(v) = a.data else { eprintln!("actions 가 숫자가 아님"); std::process::exit(2) };
    let actions: Vec<f32> = v.iter().map(|&x| x as f32).collect(); // 파이썬판 astype(np.float32)
    let st = Arc::new(Mutex::new(Replay {
        actions, t_len, n_rec, a_dim, log_path: get("--log").unwrap_or_default(), perturb, t: 0, rows: Vec::new(), episode: 0,
    }));
    let lis = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|e| { eprintln!("{host}:{port}: {e}"); std::process::exit(2) });
    log(&format!("재생 서버: 행동 ({t_len}, {n_rec}, {a_dim}) 포트 {port} quickack={quick}"));
    for s in lis.incoming() {
        let Ok(s) = s else { continue };
        let st2 = st.clone();
        // 연결마다 스레드(파이썬판 asyncio 처럼 health check 와 웹소켓이 겹쳐도 됨). --once 면 첫 웹소켓이 끝날 때 끝낸다.
        std::thread::spawn(move || match ws::accept(s, true, quick) { // asyncio 처럼 TCP_NODELAY 는 늘 켬
            Ok(ws::Accepted::Http) => {}
            Ok(ws::Accepted::Ws(mut c)) => {
                log("접속");
                if let Err(e) = serve(&mut c, &st2) {
                    log(&format!("연결 오류: {e}"));
                }
                log("접속 종료");
                st2.lock().unwrap().save();
                if once {
                    std::process::exit(0);
                }
            }
            Err(e) => log(&format!("받기 실패: {e}")),
        });
    }
}
