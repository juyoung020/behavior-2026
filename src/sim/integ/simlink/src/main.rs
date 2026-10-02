//! simlink — 평가기(Windows) ↔ WSL 통합 노드. ROS 없음.
//!
//! 한 프로세스에서 두 일을 한다(docs/통합_실시간.md):
//! 1. **계획기 호스트**: `bagent::link`(전송층) + Core·Decider(그대로). 평가기 안 파이썬 접착부가 TCP 하나로 붙는다.
//! 2. **관측 → scenemap**(2D SLAM + YOLOE 물체 지도, docs/scenemap_설계.md): 같은 프로세스 C ABI(`sm.rs`).
//!    - 매 스텝 `sm_push_proprio`(proprio 61, stamp = 그 스텝 시뮬 시각)
//!    - keyframe 영상은 `sm_push_image`(평가기 원 텐서 RGBA u8·깊이 f32 m 그대로, stamp = 장면 시각 k-1, 검출은 scenemap 이
//!      YOLOE 로). 카메라 외부 자세·팔 끝·그리퍼는 scenemap 이 proprio 로 계산한다(k-1 짝은 stamp 로).
//!    - 계획기 질의는 `--graph scenemap` → `bagent::setup::register_scene` 로 등록한 [`sm::SmQuery`](스냅숏).
//!    - 위치: `--pose-source scenemap`(기본)이면 scenemap `pose()` 를 계획기 추정기 고정점으로(`external_fix`).
//!
//! scenemap 입력은 따로 스레드(링크 스레드를 막지 않음). 경계 스텝 패킷은 절대 버리지 않고, 경계 전 `settle` 은 그 패킷이
//! scenemap 에 반영될 때까지 기다린다. 평소 패킷은 줄이 차면 영상만 버린다(proprio 는 남김).
//!
//!   simlink --listen 0.0.0.0:7801 --llm kau --graph scenemap --trace-dir … [--pose-source scenemap|simlink] [bagent link 인자]

mod sm;

use bagent::link::{self, Hello, ObsPacket, ObsSink, KIND_DEPTH_F32, KIND_RGBA8};
use bagent::odom::Pose;
use bagent::util::Args;
use serde_json::{json, Value};
use sm::Scenemap;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

enum Msg {
    Hello(Hello),
    Reset(u32),
    Pkt(ObsPacket),
}

#[derive(Default)]
struct State {
    /// 반영된 마지막 (환경, 스텝) 번호(경계 settle 용)
    done_seq: u64,
    /// scenemap 자세 (시뮬 시각, 자세), 새 값인가
    fix: Option<(f64, Pose)>,
    fix_new: bool,
    push_proprio_us: Vec<f64>,
    push_image_ms: Vec<f64>,
    /// link 수신 → scenemap 반영 끝 [ms](영상 있는 스텝)
    recv_to_map_ms: Vec<f64>,
    images: u64,
    black_rgb: u64,
    dropped_frames: u64,
    errors: u64,
    last_error: String,
    /// (link 시작 뒤 s, 물체 수)
    objects: Vec<(f64, i32)>,
    objects_max: i32,
}

struct Shared {
    g: Mutex<State>,
    cv: Condvar,
    t0: Instant,
}

fn pct(v: &[f64]) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
    let r = |x: f64| (x * 1000.0).round() / 1000.0;
    json!({"n": s.len(), "p50": r(p(0.5)), "p90": r(p(0.9)), "p99": r(p(0.99)), "max": r(s[s.len() - 1])})
}

struct ScenemapSink {
    tx: SyncSender<Msg>,
    sh: Arc<Shared>,
    seq: u64,
    boundary_seq: std::collections::HashMap<usize, u64>,
    use_fix: bool,
}

impl ObsSink for ScenemapSink {
    fn hello(&mut self, h: &Hello) {
        let _ = self.tx.send(Msg::Hello(h.clone()));
    }
    fn reset(&mut self, ep: u32) {
        let _ = self.tx.send(Msg::Reset(ep));
    }
    fn push(&mut self, pkt: ObsPacket) {
        self.seq += 1;
        if pkt.boundary {
            self.boundary_seq.insert(pkt.env, self.seq);
            let _ = self.tx.send(Msg::Pkt(pkt)); // 경계는 기다려서라도 보낸다
            return;
        }
        match self.tx.try_send(Msg::Pkt(pkt)) {
            Ok(()) => {}
            Err(TrySendError::Full(Msg::Pkt(mut p))) => {
                // 줄이 찼다: 영상만 버리고 proprio 는 보낸다(자세 적분이 비지 않게)
                let n = p.frames.len() as u64;
                p.frames.clear();
                self.sh.g.lock().unwrap().dropped_frames += n;
                let _ = self.tx.send(Msg::Pkt(p));
            }
            Err(_) => {}
        }
    }
    fn settle(&mut self, env: usize, _step: u64, timeout: std::time::Duration) -> Option<f64> {
        let want = *self.boundary_seq.get(&env)?;
        let t0 = Instant::now();
        let mut g = self.sh.g.lock().unwrap();
        while g.done_seq < want {
            let Some(left) = timeout.checked_sub(t0.elapsed()) else { break };
            let (g2, to) = self.sh.cv.wait_timeout(g, left).unwrap();
            g = g2;
            if to.timed_out() {
                break;
            }
        }
        Some(t0.elapsed().as_secs_f64() * 1e3)
    }
    fn external_fix(&mut self, env: usize) -> Option<(f64, Pose)> {
        if env != 0 || !self.use_fix {
            return None;
        }
        let mut g = self.sh.g.lock().unwrap();
        if g.fix_new {
            g.fix_new = false;
            return g.fix;
        }
        None
    }
    fn stats(&mut self) -> Value {
        let g = self.sh.g.lock().unwrap();
        json!({"scenemap": if Scenemap::is_stub() { "stub" } else { "libscenemap" }, "images": g.images,
               "push_proprio_us": pct(&g.push_proprio_us), "push_image_ms": pct(&g.push_image_ms), "recv_to_map_ms": pct(&g.recv_to_map_ms),
               "dropped_frames": g.dropped_frames, "black_rgb_frames": g.black_rgb, "errors": g.errors, "last_error": g.last_error,
               "objects_max": g.objects_max, "objects_timeline": g.objects.iter().map(|(t, n)| json!([(t * 10.0).round() / 10.0, n])).collect::<Vec<_>>()})
    }
}

/// scenemap 입력 스레드: 패킷 하나 = proprio 한 번 + 영상(카메라마다 RGBA·깊이를 한 장으로).
fn worker(rx: Receiver<Msg>, sm: Arc<Scenemap>, sh: Arc<Shared>) {
    let mut ks: [[f64; 4]; 3] = [[0.0; 4]; 3];
    let mut seq = 0u64;
    let err = |sh: &Shared, e: String| {
        let mut g = sh.g.lock().unwrap();
        g.errors += 1;
        g.last_error = e;
    };
    while let Ok(m) = rx.recv() {
        match m {
            Msg::Hello(h) => {
                seq = 0; // 연결마다 sink 의 번호도 0 부터
                sh.g.lock().unwrap().done_seq = 0;
                for c in &h.cams {
                    let id = match c.name.as_str() {
                        "head" => 0,
                        "left_wrist" => 1,
                        "right_wrist" => 2,
                        _ => continue,
                    };
                    ks[id] = c.k;
                }
            }
            Msg::Reset(ep) => {
                eprintln!("[simlink] 새 판 {ep}: scenemap reset");
                if let Err(e) = sm.reset() {
                    err(&sh, e);
                }
            }
            Msg::Pkt(p) => {
                seq += 1;
                let t0 = Instant::now();
                if let Err(e) = sm.push_proprio(p.t, &p.proprio) {
                    err(&sh, e);
                }
                let tp = t0.elapsed().as_secs_f64() * 1e6;
                let mut img_ms = None;
                if !p.frames.is_empty() {
                    let ti = Instant::now();
                    for cam in 0..3u8 {
                        let rgba = p.frames.iter().find(|f| f.cam == cam && f.kind == KIND_RGBA8);
                        let depth = p.frames.iter().find(|f| f.cam == cam && f.kind == KIND_DEPTH_F32);
                        let Some(any) = rgba.or(depth) else { continue };
                        // 영상 k = 장면 k-1: stamp 는 img_t(직전 스텝 시각). scenemap 이 그 stamp 의 proprio 와 짝짓는다.
                        if let Err(e) = sm.push_image(p.img_t, cam, any.w, any.h, rgba.map(|f| f.data.as_slice()), depth.map(|f| f.data.as_slice()), ks[cam as usize]) {
                            err(&sh, e);
                        }
                    }
                    img_ms = Some(ti.elapsed().as_secs_f64() * 1e3);
                }
                let snap = sm.snapshot();
                let mut g = sh.g.lock().unwrap();
                g.push_proprio_us.push(tp);
                if let Some(ms) = img_ms {
                    g.push_image_ms.push(ms);
                    g.images += 1;
                    g.recv_to_map_ms.push(p.recv.elapsed().as_secs_f64() * 1e3);
                    g.black_rgb += p.frames.iter().filter(|f| f.is_rgb() && f.is_black()).count() as u64;
                }
                if let Ok(s) = snap {
                    let ps = s.pose();
                    g.fix = Some((ps.stamp, Pose { x: ps.x, y: ps.y, yaw: ps.yaw }));
                    g.fix_new = true;
                    let n = s.status().n_objects;
                    if g.objects.last().map(|x| x.1 != n).unwrap_or(true) {
                        let t = sh.t0.elapsed().as_secs_f64();
                        g.objects.push((t, n));
                    }
                    g.objects_max = g.objects_max.max(n);
                }
                g.done_seq = seq;
                drop(g);
                sh.cv.notify_all();
            }
        }
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|s| s == "--help" || s == "-h") {
        println!("simlink — 평가기 ↔ 계획기 + scenemap(같은 프로세스). 인자는 `bagent link` 와 같고 더해서 --graph scenemap --pose-source scenemap|simlink --scenemap-config JSON --queue 6");
        return;
    }
    let a = Args::parse(raw);
    let sm = match Scenemap::new(&a.str_or("scenemap-config", "")) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("scenemap: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("[simlink] scenemap: {}", if Scenemap::is_stub() { "가짜 구현(src/sim/integ/scenemap_stub) — 물체 지도 없음, 자세 = base_qvel 적분" } else { "libscenemap" });
    let _ = bagent::setup::register_scene(Arc::new(sm::SmQuery(sm.clone())));
    let catalog = match bagent::setup::load_catalog(&a) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("과제 카드: {e}");
            std::process::exit(1);
        }
    };
    let use_fix = a.str_or("pose-source", "scenemap") == "scenemap";
    let cfg = match link::cfg_from_args(&a, catalog) {
        Ok(mut c) => {
            if use_fix {
                c.pose = "corrected".into(); // scenemap 자세를 고정점으로 받는 추정기
            }
            c
        }
        Err(e) => {
            eprintln!("설정: {e}");
            std::process::exit(1);
        }
    };
    if let Some(d) = &cfg.trace_dir {
        eprintln!("[simlink] 기록: {}", d.display());
    }
    let sh = Arc::new(Shared { g: Mutex::new(State::default()), cv: Condvar::new(), t0: Instant::now() });
    let (tx, rx) = sync_channel::<Msg>(a.num("queue", 6usize));
    let (sm2, sh2) = (sm.clone(), sh.clone());
    let w = std::thread::spawn(move || worker(rx, sm2, sh2));
    let make = move || -> Box<dyn ObsSink> {
        Box::new(ScenemapSink { tx: tx.clone(), sh: sh.clone(), seq: 0, boundary_seq: Default::default(), use_fix })
    };
    let r = link::run(cfg, &make);
    drop(make);
    let _ = w.join();
    if let Err(e) = r {
        eprintln!("[simlink] {e}");
        std::process::exit(1);
    }
}
