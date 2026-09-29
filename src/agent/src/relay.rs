//! 중계기(다리): 평가기 ↔ [중계기] ↔ π0.5 서버.
//!
//! 매 스텝 경로(병목 제로 목표):
//! - 평가기 프레임(클라이언트 → 서버, XOR 마스크됨)을 재사용 버퍼로 받고 **마스크를 풀지 않는다**.
//! - msgpack 최상위 맵을 마스크된 채로 훑어 base_qvel·그리퍼·task_id 만 그 자리에서 읽는다(수십 바이트).
//! - 보낼 때는 [맵 머리(개수+주입 수) | 원래 항목 바이트 그대로 | 주입 꼬리] 를 writev 한 번으로.
//!   맵 머리 길이가 바뀐 만큼 마스크 키를 회전시켜([`ws::rotate_key`]) 원래 바이트를 다시 마스크하지 않는다.
//! - π0.5 응답(서버 → 클라이언트, 마스크 없음)은 그대로 평가기에 넘긴다.
//! - LLM 은 이 경로에 없다. 경계(판 시작·예산 소진·정기 확인·이동 멈춤·그리퍼 변화)에서만 계획기 스레드로 넘긴다.
//!   `pause` 면 결정이 올 때까지 이번 관측을 붙잡는다(시뮬레이터 시간 정지, 결정 재현 가능). 그동안 두 소켓의
//!   ping 에 답한다(평가기 ping_timeout 300 s, π0.5 서버 기본 20 s). `pause` 가 아니면 지시를 바꾸지 않고 계속 넘기다가
//!   결정이 오는 스텝에 바꾼다(벽시계 지연 0, 대신 재현성은 기록으로만).
//!
//! 관측·행동 값은 한 바이트도 바꾸지 않는다: 원래 항목 바이트는 그대로 가고, 주입 키(`__agent_prompt__`,
//! `__agent_flush__`)는 π0.5 서버 쪽 훅(`tools/serve_b1k_agent.py`)이 꺼내 쓰고 지운다.

use crate::catalog::{Catalog, TaskCard};
use crate::monitor::{MonitorCfg, Trigger};
use crate::msgpack::{self, AnySrc, Masked, Plain};
use crate::planner::{Agent, Decision};
use crate::session::EnvSession;
use crate::trace::Tracer;
use crate::wire::{self, Cam};
use crate::ws::{self, Accepted, Buf, Conn, OP_BIN, OP_CLOSE};
use serde_json::json;
use std::io;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub enum Mode {
    /// 아무것도 주입하지 않고 그대로(오버헤드 기준선)
    Passthrough,
    /// 고정 문장 주입
    Fixed(String),
    /// 스텝별 문장 표(시연 구간대로 바꾸기 등): (시작 스텝, 문장)
    Schedule(Vec<(u64, String)>),
    /// 에이전트가 경계마다 정한다
    Agent,
}

pub type AgentFactory = Arc<dyn Fn(usize, &TaskCard) -> Agent + Send + Sync>;

#[derive(Clone)]
pub struct RelayCfg {
    pub listen: String,
    pub upstream: String,
    pub mode: Mode,
    pub catalog: Arc<Catalog>,
    pub task_override: Option<String>,
    pub max_steps_override: Option<u64>,
    pub monitor: MonitorCfg,
    pub pause: bool,
    pub images: bool,
    pub image_side: usize,
    pub jpeg_quality: u8,
    pub trace_dir: Option<PathBuf>,
    pub factory: Option<AgentFactory>,
    /// 연결 하나만 처리하고 끝(시험용)
    pub once: bool,
    /// 이 스텝마다 지연 통계를 기록
    pub latency_every: u64,
    /// pause 에서 결정을 기다리는 최대 시간(넘으면 지금 문장으로 계속, 결정은 도착 시 반영)
    pub hard_wait_s: f64,
    pub hz: f64,
}

impl RelayCfg {
    pub fn new(listen: &str, upstream: &str, mode: Mode, catalog: Arc<Catalog>) -> RelayCfg {
        RelayCfg {
            listen: listen.into(),
            upstream: upstream.into(),
            mode,
            catalog,
            task_override: None,
            max_steps_override: None,
            monitor: MonitorCfg::default(),
            pause: true,
            images: true,
            image_side: 448,
            jpeg_quality: 80,
            trace_dir: None,
            factory: None,
            once: false,
            latency_every: 1000,
            hard_wait_s: 240.0,
            hz: 30.0,
        }
    }
}

enum PMsg {
    Decision(usize, Decision),
    Returned(usize, Box<Agent>),
}

struct Slot {
    sess: EnvSession,
    agent: Option<Box<Agent>>,
    /// 에이전트를 가진 환경인가(과제를 모르면 없다)
    owns: bool,
    /// 결정을 기다리는 중
    waiting: bool,
}

fn idle(s: &Slot) -> bool {
    !s.owns || (s.agent.is_some() && !s.waiting)
}

/// 스텝마다 잰 시간(µs)
#[derive(Default)]
pub struct Lat {
    pub scan: Vec<u32>,
    pub fwd: Vec<u32>,
    pub upstream: Vec<u32>,
    pub back: Vec<u32>,
    pub relay_total: Vec<u32>,
    pub plan_wait: Vec<u32>,
}

fn pct(v: &[u32]) -> serde_json::Value {
    if v.is_empty() {
        return json!(null);
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let p = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
    json!({"n": s.len(), "p50": p(0.5), "p90": p(0.9), "p99": p(0.99), "max": s[s.len() - 1]})
}

impl Lat {
    pub fn summary(&self) -> serde_json::Value {
        json!({"unit": "us", "scan": pct(&self.scan), "forward": pct(&self.fwd), "upstream_rtt": pct(&self.upstream),
               "back": pct(&self.back), "relay_total": pct(&self.relay_total), "plan_wait": pct(&self.plan_wait)})
    }
}

pub fn run(cfg: RelayCfg) -> io::Result<()> {
    let l = TcpListener::bind(&cfg.listen)?;
    eprintln!("[relay] {} ← 평가기, → π0.5 {} (모드 {:?}, pause {})", cfg.listen, cfg.upstream, mode_name(&cfg.mode), cfg.pause);
    let tracer = match &cfg.trace_dir {
        Some(d) => Some(Tracer::create(d)?),
        None => None,
    };
    for s in l.incoming() {
        let s = match s {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[relay] accept 오류: {e}");
                continue;
            }
        };
        let cfg2 = cfg.clone();
        let tr = tracer.clone();
        let up = cfg.upstream.clone();
        let health = move || crate::http::get(&format!("http://{up}/healthz"), Duration::from_secs(2)).map(|r| r.status == 200).unwrap_or(false);
        let conn = match Conn::accept_with(s, health) {
            Ok(Accepted::Ws(c)) => c,
            Ok(Accepted::Http(_)) => continue,
            Err(e) => {
                eprintln!("[relay] 핸드셰이크 오류: {e}");
                continue;
            }
        };
        if cfg.once {
            let r = serve(cfg2, conn, tr.clone());
            if let Some(t) = &tr {
                t.flush();
            }
            return r.map(|_| ());
        }
        std::thread::spawn(move || {
            if let Err(e) = serve(cfg2, conn, tr.clone()) {
                eprintln!("[relay] 연결 끝: {e}");
            }
            if let Some(t) = &tr {
                t.flush();
            }
        });
    }
    Ok(())
}

fn mode_name(m: &Mode) -> &'static str {
    match m {
        Mode::Passthrough => "passthrough",
        Mode::Fixed(_) => "fixed",
        Mode::Schedule(_) => "schedule",
        Mode::Agent => "agent",
    }
}

/// 평가기 연결 하나를 처리한다. 반환: 지연 통계.
pub fn serve(cfg: RelayCfg, mut ev: Conn, tracer: Option<Tracer>) -> io::Result<Lat> {
    let mut up = Conn::connect(&cfg.upstream)?;
    let mut ebuf = Buf::default();
    let mut ubuf = Buf::default();
    let mut cbuf = Buf::default();
    let mut suffix: Vec<u8> = Vec::with_capacity(1024);
    let mut scratch: Vec<u8> = Vec::with_capacity(1024);
    let mut lat = Lat::default();

    // 서버가 먼저 보내는 metadata 를 그대로 넘긴다
    let (op, _) = up.read_message(&mut ubuf)?;
    ev.write_frame_raw(true, op, None, &[ubuf.data()])?;

    let (tx, rx) = mpsc::channel::<PMsg>();
    let mut slots: Vec<Slot> = Vec::new();
    let mut task: Option<TaskCard> = cfg.task_override.as_ref().and_then(|n| cfg.catalog.task_by_name(n).cloned());
    let mut episode: u32 = 0;
    let mut steps: u64 = 0;
    let trace = |env: usize, kind: &str, v: serde_json::Value| {
        if let Some(t) = &tracer {
            t.event(env, kind, v);
        }
    };
    trace(0, "relay_start", json!({"mode": mode_name(&cfg.mode), "pause": cfg.pause, "upstream": cfg.upstream}));

    loop {
        let (op, mask) = ev.read_message(&mut ebuf)?;
        let t_in = Instant::now();
        if op == OP_CLOSE {
            let _ = up.send_close(1000);
            break;
        }
        if op != OP_BIN || matches!(cfg.mode, Mode::Passthrough) {
            // 그대로 넘기기(바이트 동일)
            forward_raw(&mut up, op, mask, ebuf.data())?;
            if op == OP_BIN && !is_reset_quick(ebuf.data(), mask) {
                let t_f = Instant::now();
                let (op2, _) = up.read_message(&mut ubuf)?;
                let t_r = Instant::now();
                ev.write_frame_raw(true, op2, None, &[ubuf.data()])?;
                let t_o = Instant::now();
                lat.fwd.push(us(t_f - t_in));
                lat.upstream.push(us(t_r - t_f));
                lat.back.push(us(t_o - t_r));
                lat.relay_total.push(us((t_f - t_in) + (t_o - t_r)));
                steps += 1;
            }
            continue;
        }

        let src = match mask {
            Some(k) => AnySrc::M(Masked { buf: ebuf.data(), key: k }),
            None => AnySrc::P(Plain(ebuf.data())),
        };
        let view = wire::view(&src).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let t_scan = Instant::now();

        if view.is_reset {
            // 새 판: 계획기가 돌아올 때까지 기다린 뒤 모두 초기화
            wait_all_returned(&mut slots, &rx, &tx, &mut ev, &mut up, &mut cbuf, &cfg)?;
            episode += 1;
            for s in slots.iter_mut() {
                s.sess.reset(episode);
                if let Some(a) = s.agent.as_mut() {
                    a.reset_episode(episode);
                }
            }
            trace(0, "episode_reset", json!({"episode": episode, "steps_total": steps}));
            forward_raw(&mut up, op, mask, ebuf.data())?; // 응답 없음
            continue;
        }

        // 과제·환경 수 확인(첫 관측)
        if task.is_none() {
            task = view.task_id.and_then(|i| cfg.catalog.task_by_index(i)).cloned();
            trace(0, "task", json!({"task_id": view.task_id, "task": task.as_ref().map(|t| t.name.clone())}));
        }
        if slots.len() != view.batch {
            let (tp, ms) = task.as_ref().map(|t| (t.prompt.clone(), t.max_steps)).unwrap_or_default();
            let ms = cfg.max_steps_override.unwrap_or(if ms == 0 { 100_000 } else { ms });
            slots = (0..view.batch)
                .map(|e| {
                    let mut agent = match (&cfg.mode, &cfg.factory, &task) {
                        (Mode::Agent, Some(f), Some(t)) => Some(Box::new(f(e, t))),
                        _ => None,
                    };
                    if let Some(a) = agent.as_mut() {
                        a.set_tracer(tracer.clone());
                        a.reset_episode(episode);
                    }
                    Slot { sess: EnvSession::new(e, cfg.monitor.clone(), cfg.hz, ms, &tp), owns: agent.is_some(), agent, waiting: false }
                })
                .collect();
        }

        // 매 스텝: 오도메트리·감시(산술만)
        let mut triggered: Vec<(usize, Trigger, [f64; 2])> = Vec::new();
        for (e, s) in slots.iter_mut().enumerate() {
            let g = view.grippers(&src, e);
            if let Some(t) = s.sess.on_obs(view.base_qvel(&src, e), g) {
                triggered.push((e, t, g));
            }
        }

        // 비동기로 도착한 결정 반영
        drain(&rx, &mut slots, &tracer);

        let mut plan_us = 0u32;
        match &cfg.mode {
            Mode::Agent if !triggered.is_empty() => {
                let tp = Instant::now();
                for (e, t, g) in triggered {
                    if slots[e].waiting {
                        continue;
                    }
                    if slots[e].agent.is_none() {
                        if cfg.pause {
                            wait_returned(e, &mut slots, &rx, &mut ev, &mut up, &mut cbuf, &cfg)?;
                        } else {
                            continue;
                        }
                    }
                    let Some(mut agent) = slots[e].agent.take() else { continue };
                    let images = if cfg.images { snapshot(&view, &src, e, cfg.image_side, cfg.jpeg_quality) } else { vec![] };
                    let mut bev = slots[e].sess.event(t, g, images);
                    if let Some(tr) = &tracer {
                        bev.image_files = bev
                            .images
                            .iter()
                            .map(|(c, j)| tr.save_image(&format!("ep{}_env{}_s{}_{}.jpg", bev.episode, e, bev.step, c), j))
                            .collect();
                    }
                    slots[e].waiting = true;
                    slots[e].sess.mon.busy = true;
                    let txc = tx.clone();
                    std::thread::spawn(move || {
                        let d = agent.decide(&bev);
                        let _ = txc.send(PMsg::Decision(e, d));
                        agent.maintain();
                        agent.release();
                        let _ = txc.send(PMsg::Returned(e, agent));
                    });
                }
                if cfg.pause {
                    wait_decisions(&mut slots, &rx, &mut ev, &mut up, &mut cbuf, &cfg, &tracer)?;
                }
                plan_us = us(tp.elapsed());
            }
            Mode::Fixed(p) => {
                for s in slots.iter_mut() {
                    s.sess.prompt = p.clone();
                }
            }
            Mode::Schedule(tab) => {
                for s in slots.iter_mut() {
                    if let Some((_, p)) = tab.iter().rev().find(|(st, _)| *st <= s.sess.mon.step) {
                        if *p != s.sess.prompt {
                            s.sess.prompt = p.clone();
                            s.sess.pending_flush = true;
                        }
                    }
                }
            }
            _ => {}
        }

        // 주입 꼬리 만들기 → 복사 없이 보내기
        let t_fwd0 = Instant::now();
        let inject = !matches!(cfg.mode, Mode::Agent) || task.is_some();
        let extra = if inject {
            let prompts: Vec<String> = slots.iter().map(|s| s.sess.prompt.clone()).collect();
            let any_flush = slots.iter().any(|s| s.sess.pending_flush);
            let flush: Vec<bool> = slots.iter().map(|s| s.sess.pending_flush).collect();
            for s in slots.iter_mut() {
                if s.sess.pending_flush {
                    trace(s.sess.env, "prompt_applied", json!({"step": s.sess.mon.step, "prompt": s.sess.prompt}));
                }
                s.sess.pending_flush = false;
            }
            wire::build_suffix(&prompts, if any_flush { Some(&flush) } else { None }, view.batched, &mut suffix)
        } else {
            suffix.clear();
            0
        };
        forward_obs(&mut up, ebuf.data(), mask, &view.top, &suffix, extra, &mut scratch)?;
        let t_f = Instant::now();
        let (op2, _) = up.read_message(&mut ubuf)?;
        let t_r = Instant::now();
        ev.write_frame_raw(true, op2, None, &[ubuf.data()])?;
        let t_o = Instant::now();
        for s in slots.iter_mut() {
            s.sess.mon.advance();
        }
        steps += 1;
        let pre = (t_scan - t_in) + (t_f - t_fwd0);
        lat.scan.push(us(t_scan - t_in));
        lat.fwd.push(us(t_f - t_fwd0));
        lat.upstream.push(us(t_r - t_f));
        lat.back.push(us(t_o - t_r));
        lat.relay_total.push(us(pre + (t_o - t_r)));
        if plan_us > 0 {
            lat.plan_wait.push(plan_us);
        }
        if cfg.latency_every > 0 && steps % cfg.latency_every == 0 {
            trace(0, "latency", json!({"steps": steps, "stats": lat.summary()}));
        }
    }
    // 남은 계획기 정리
    let _ = wait_all_returned(&mut slots, &rx, &tx, &mut ev, &mut up, &mut cbuf, &cfg);
    trace(0, "relay_end", json!({"steps": steps, "stats": lat.summary()}));
    for s in &slots {
        if let Some(a) = &s.agent {
            trace(s.sess.env, "agent_stats", json!(a.core.stats));
        }
    }
    Ok(lat)
}

#[inline]
fn us(d: Duration) -> u32 {
    d.as_micros().min(u32::MAX as u128) as u32
}

fn is_reset_quick(data: &[u8], mask: Option<[u8; 4]>) -> bool {
    let src = match mask {
        Some(k) => AnySrc::M(Masked { buf: data, key: k }),
        None => AnySrc::P(Plain(data)),
    };
    msgpack::scan_top(&src).map(|t| t.find(|k| k == "reset").is_some()).unwrap_or(false)
}

/// 프레임을 바이트 그대로 넘긴다(같은 마스크 키).
fn forward_raw(up: &mut Conn, op: u8, mask: Option<[u8; 4]>, data: &[u8]) -> io::Result<()> {
    match mask {
        Some(k) => up.write_frame_raw(true, op, Some(k), &[data]),
        None => up.send(op, &[data]),
    }
}

/// 관측 + 주입 꼬리를 복사 없이 보낸다.
pub fn forward_obs(up: &mut Conn, data: &[u8], mask: Option<[u8; 4]>, top: &msgpack::TopMap, suffix: &[u8], extra: usize, scratch: &mut Vec<u8>) -> io::Result<()> {
    let (nh, nhl) = msgpack::map_hdr(top.count + extra);
    let old = top.hdr_len;
    let body = &data[old..];
    match mask {
        Some(k) if extra == 0 => up.write_frame_raw(true, OP_BIN, Some(k), &[data]),
        Some(k) => {
            let k2 = ws::rotate_key(k, nhl as isize - old as isize);
            let mut h = [0u8; 5];
            h[..nhl].copy_from_slice(&nh[..nhl]);
            ws::mask_at(&mut h[..nhl], k2, 0);
            scratch.clear();
            scratch.extend_from_slice(suffix);
            ws::mask_at(scratch, k2, nhl + body.len());
            up.write_frame_raw(true, OP_BIN, Some(k2), &[&h[..nhl], body, scratch])
        }
        None => {
            // 조각 모음 등 평문: 새 키로 전부 마스크(드문 경로)
            let k2 = up.new_mask_key();
            scratch.clear();
            scratch.extend_from_slice(&nh[..nhl]);
            scratch.extend_from_slice(body);
            scratch.extend_from_slice(suffix);
            ws::mask_at(scratch, k2, 0);
            up.write_frame_raw(true, OP_BIN, Some(k2), &[scratch])
        }
    }
}

/// 경계 순간의 카메라 영상 → 축소 → JPEG.
fn snapshot(view: &wire::ObsView, src: &AnySrc, env: usize, side: usize, q: u8) -> Vec<(String, Vec<u8>)> {
    [Cam::Head, Cam::LeftWrist, Cam::RightWrist]
        .iter()
        .filter_map(|&c| view.image(src, c, env).map(|img| (c.name().to_string(), img.downscale(if c == Cam::Head { side } else { side / 2 }).jpeg(q))))
        .collect()
}

fn drain(rx: &mpsc::Receiver<PMsg>, slots: &mut [Slot], tracer: &Option<Tracer>) {
    while let Ok(m) = rx.try_recv() {
        handle(m, slots, tracer);
    }
}

fn handle(m: PMsg, slots: &mut [Slot], tracer: &Option<Tracer>) {
    match m {
        PMsg::Decision(e, d) => {
            if let Some(s) = slots.get_mut(e) {
                s.sess.apply(&d);
                s.waiting = false;
                s.sess.mon.busy = false;
                if let Some(t) = tracer {
                    t.event(e, "decision_applied", json!({"step": s.sess.mon.step, "kind": d.key()}));
                }
            }
        }
        PMsg::Returned(e, a) => {
            if let Some(s) = slots.get_mut(e) {
                s.agent = Some(a);
            }
        }
    }
}

/// 결정이 모두 올 때까지 두 소켓의 제어 프레임(ping)에 답하며 기다린다.
fn pump_until(
    done: &dyn Fn(&[Slot]) -> bool,
    slots: &mut [Slot],
    rx: &mpsc::Receiver<PMsg>,
    ev: &mut Conn,
    up: &mut Conn,
    cbuf: &mut Buf,
    cfg: &RelayCfg,
    tracer: &Option<Tracer>,
) -> io::Result<bool> {
    let t0 = Instant::now();
    loop {
        drain(rx, slots, tracer);
        if done(slots) {
            return Ok(true);
        }
        if t0.elapsed().as_secs_f64() > cfg.hard_wait_s {
            return Ok(false);
        }
        match rx.recv_timeout(Duration::from_millis(0)) {
            Ok(m) => {
                handle(m, slots, tracer);
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(false),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        let ready = ws::poll_readable(&[&*ev, &*up], 20)?;
        if ready[0] && !ev.service_control(cbuf)? {
            return Err(io::Error::new(io::ErrorKind::ConnectionAborted, "평가기가 계획 중에 연결을 닫음"));
        }
        if ready[1] && !up.service_control(cbuf)? {
            return Err(io::Error::new(io::ErrorKind::ConnectionAborted, "π0.5 서버가 계획 중에 연결을 닫음"));
        }
    }
}

fn wait_decisions(slots: &mut [Slot], rx: &mpsc::Receiver<PMsg>, ev: &mut Conn, up: &mut Conn, cbuf: &mut Buf, cfg: &RelayCfg, tracer: &Option<Tracer>) -> io::Result<()> {
    let ok = pump_until(&|s: &[Slot]| s.iter().all(|x| !x.waiting), slots, rx, ev, up, cbuf, cfg, tracer)?;
    if !ok {
        eprintln!("[relay] 결정이 {} s 안에 안 옴 — 지금 문장으로 계속, 결정은 도착하면 반영", cfg.hard_wait_s);
    }
    Ok(())
}

fn wait_returned(e: usize, slots: &mut [Slot], rx: &mpsc::Receiver<PMsg>, ev: &mut Conn, up: &mut Conn, cbuf: &mut Buf, cfg: &RelayCfg) -> io::Result<()> {
    pump_until(&|s: &[Slot]| s[e].agent.is_some(), slots, rx, ev, up, cbuf, cfg, &None).map(|_| ())
}

fn wait_all_returned(slots: &mut [Slot], rx: &mpsc::Receiver<PMsg>, _tx: &mpsc::Sender<PMsg>, ev: &mut Conn, up: &mut Conn, cbuf: &mut Buf, cfg: &RelayCfg) -> io::Result<()> {
    if slots.iter().all(idle) {
        return Ok(());
    }
    pump_until(&|s: &[Slot]| s.iter().all(idle), slots, rx, ev, up, cbuf, cfg, &None).map(|_| ())
}
