//! simlink — 평가기(Windows) ↔ WSL 통합 노드.
//!
//! 한 프로세스에서 두 일을 한다(docs/통합_실시간.md):
//! 1. **계획기 호스트**: `bagent::link`(전송층) + Core·Decider(그대로). 평가기 안 파이썬 접착부가 TCP 하나로 붙는다.
//! 2. **관측 → meridian**: link 가 넘기는 [`ObsPacket`] 을 ROS 2 토픽으로 낸다(r2r, Fast DDS 공유메모리 프로필).
//!    - `/camera/rgb` rgb8 640x480, `/camera/depth` 16UC1 mm, `/camera/info` — frontend 입력 계약(meridian 문서 2.1, obs_player 와
//!      같은 변환: 가운데 4:3 자르기·면적 축소·깊이 mm·K 옮기기)
//!    - 매 스텝 `/base_qvel`(TwistStamped, 로봇 기준 원값)·`/camera/extrinsic`(PoseStamped, 베이스 기준 머리 카메라 = robot2cam,
//!      proprio 순기구학) — 영상이 있는 스텝이면 영상과 같은 stamp. meridian_odom(깊이 보정 위치 추정기)의 입력
//!    - 매 스텝 `/robot/eef`(PoseArray [왼, 오른] 베이스 기준 팔 끝, proprio 17:24·42:49)·`/robot/gripper`(JointState, 손가락 합
//!      24:26·49:51) — 같은 stamp 규칙. meridian 옮겨진 물체 처리(잡기·놓기) 입력
//!    - `--pose-source simlink`(지금 기본): link 의 추정기(적분)로 `/camera/pose`(광학 프레임 map 자세)·`/base_pose` 도 낸다.
//!      `--pose-source meridian`: 그 둘은 meridian_odom 이 내고, simlink 는 `/base_pose` 를 받아 계획기 추정기에 고정점으로 넣는다
//!    - 손목: `/camera_left/*`, `/camera_right/*`(원래 크기 480x480) — 영상이 온 스텝만
//!    - 받기: `/graph_update_event`(graphcore) → 관측 → 그래프 반영 지연, 경계 settle
//!    - scene_server `status` 를 주기적으로 물어 노드 수 기록
//!
//! ROS 발행은 따로 스레드(링크 스레드를 막지 않음). 경계 스텝 패킷은 절대 버리지 않고, 평소 패킷은 줄이 차면 영상만 버린다.
//!
//!   simlink --listen 0.0.0.0:7801 --llm kau --graph meridian --trace-dir … [--scene 127.0.0.1:7791] [bagent link 인자]

mod convert;

use bagent::link::{self, Frame, Hello, ObsPacket, ObsSink, CAM_HEAD, KIND_DEPTH_F32, KIND_RGB8, KIND_RGBA8};
use bagent::odom::Pose;
use bagent::pose;
use bagent::util::Args;
use convert::CamConv;
use futures::stream::StreamExt;
use r2r::builtin_interfaces::msg::Time;
use r2r::geometry_msgs::msg::{Point, Pose as RPose, PoseArray, PoseStamped, Quaternion, Twist, TwistStamped, Vector3};
use r2r::sensor_msgs::msg::JointState;
use r2r::sensor_msgs::msg::{CameraInfo, Image};
use r2r::std_msgs::msg::Header;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

enum Msg {
    Hello(Hello),
    Reset(u32),
    Pkt(ObsPacket),
}

#[derive(Default)]
struct GraphState {
    /// 가장 최근 그래프 이벤트(하트비트 아님)의 stamp(유닉스 ns)
    last_event_stamp: i128,
    events: u64,
    heartbeats: u64,
    /// 발행한 머리 영상 stamp → (발행 끝 순간, link 가 그 스텝을 받은 순간)
    published: HashMap<i128, (Instant, Instant)>,
    pub_order: VecDeque<i128>,
    /// 관측(발행) → graphcore commit 이벤트 수신 [ms]
    lat_ms: Vec<f64>,
    /// link 가 그 영상을 받은 순간 → 이벤트 [ms]
    age_ms: Vec<f64>,
    /// link 수신 → 머리 영상 발행 끝 [ms] (변환 포함)
    pub_ms: Vec<f64>,
    conv_ms: Vec<f64>,
    /// meridian_odom `/base_pose` 최신값 (stamp, 자세), 새 값인가
    ext_fix: Option<(i128, Pose)>,
    ext_new: bool,
    ext_count: u64,
    /// (link 시작 뒤 s, 노드 수, events_applied)
    nodes: Vec<(f64, u64, u64)>,
    dropped_frames: u64,
    head_published: u64,
    wrist_published: u64,
    last_nodes: Option<u64>,
    max_nodes: u64,
    scene_ok: bool,
}

struct Shared {
    g: Mutex<GraphState>,
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
    json!({"n": s.len(), "p50": (p(0.5) * 100.0).round() / 100.0, "p90": (p(0.9) * 100.0).round() / 100.0, "max": (s[s.len() - 1] * 100.0).round() / 100.0})
}

struct RosSink {
    tx: SyncSender<Msg>,
    sh: Arc<Shared>,
    /// 환경별 마지막 경계 패킷 stamp
    boundary_stamp: HashMap<usize, i128>,
}

impl ObsSink for RosSink {
    fn hello(&mut self, h: &Hello) {
        let _ = self.tx.send(Msg::Hello(h.clone()));
    }
    fn reset(&mut self, ep: u32) {
        let _ = self.tx.send(Msg::Reset(ep));
    }
    fn push(&mut self, pkt: ObsPacket) {
        if pkt.boundary {
            // 그래프 이벤트 stamp = 영상 stamp(last_seen) → 경계 영상의 stamp 로 기다린다
            self.boundary_stamp.insert(pkt.env, pkt.img_stamp_ns);
            let _ = self.tx.send(Msg::Pkt(pkt)); // 경계는 기다려서라도 보낸다
            return;
        }
        match self.tx.try_send(Msg::Pkt(pkt)) {
            Ok(()) => {}
            Err(TrySendError::Full(Msg::Pkt(mut p))) => {
                // 줄이 찼다: 영상만 버리고 자세는 보낸다(frontend 자세 기록이 비지 않게)
                let n = p.frames.len() as u64;
                p.frames.clear();
                if self.tx.try_send(Msg::Pkt(p)).is_err() {
                    // 그래도 차면 이번 스텝은 통째로 버림
                }
                self.sh.g.lock().unwrap().dropped_frames += n;
            }
            Err(_) => {}
        }
    }
    fn settle(&mut self, env: usize, _step: u64, timeout: Duration) -> Option<f64> {
        let want = *self.boundary_stamp.get(&env)?;
        let t0 = Instant::now();
        let mut g = self.sh.g.lock().unwrap();
        while g.last_event_stamp < want {
            let Some(left) = timeout.checked_sub(t0.elapsed()) else { break };
            let (g2, to) = self.sh.cv.wait_timeout(g, left).unwrap();
            g = g2;
            if to.timed_out() {
                break;
            }
        }
        Some(t0.elapsed().as_secs_f64() * 1e3)
    }
    fn external_fix(&mut self, env: usize) -> Option<(i128, Pose)> {
        if env != 0 {
            return None;
        }
        let mut g = self.sh.g.lock().unwrap();
        if g.ext_new {
            g.ext_new = false;
            return g.ext_fix;
        }
        None
    }
    fn stats(&mut self) -> Value {
        let g = self.sh.g.lock().unwrap();
        json!({"graph_events": g.events, "heartbeats": g.heartbeats, "obs_to_graph_ms": pct(&g.lat_ms), "recv_to_graph_ms": pct(&g.age_ms),
               "recv_to_publish_ms": pct(&g.pub_ms), "convert_ms": pct(&g.conv_ms), "head_published": g.head_published,
               "wrist_published": g.wrist_published, "dropped_frames": g.dropped_frames, "external_poses": g.ext_count, "nodes_last": g.last_nodes, "nodes_max": g.max_nodes,
               "scene_ok": g.scene_ok, "nodes_timeline": g.nodes.iter().map(|(t, n, e)| json!([(t * 10.0).round() / 10.0, n, e])).collect::<Vec<_>>()})
    }
}

fn stamp(ns: i128) -> Time {
    Time { sec: ns.div_euclid(1_000_000_000) as i32, nanosec: ns.rem_euclid(1_000_000_000) as u32 }
}

fn stamp_ns(t: &Time) -> i128 {
    t.sec as i128 * 1_000_000_000 + t.nanosec as i128
}

fn pose_msg(t: &Time, frame: &str, r: &pose::Mat3, p: &[f64; 3]) -> PoseStamped {
    let q = pose::mat_to_quat(r);
    PoseStamped {
        header: Header { stamp: t.clone(), frame_id: frame.into() },
        pose: RPose { position: Point { x: p[0], y: p[1], z: p[2] }, orientation: Quaternion { x: q[0], y: q[1], z: q[2], w: q[3] } },
    }
}

fn info_msg(t: &Time, frame: &str, c: &CamConv) -> CameraInfo {
    let [fx, fy, cx, cy] = c.k_out;
    CameraInfo {
        header: Header { stamp: t.clone(), frame_id: frame.into() },
        width: c.out_w as u32,
        height: c.out_h as u32,
        distortion_model: "plumb_bob".into(),
        d: vec![0.0; 5],
        k: vec![fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        r: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p: vec![fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        ..Default::default()
    }
}

struct CamPub {
    rgb: r2r::Publisher<Image>,
    depth: r2r::Publisher<Image>,
    info: r2r::Publisher<CameraInfo>,
    pose: r2r::Publisher<PoseStamped>,
    frame: String,
    conv: Option<CamConv>,
}

fn ros_thread(rx: Receiver<Msg>, sh: Arc<Shared>, stop: Arc<AtomicBool>, a: Args) -> Result<(), String> {
    let e = |x: r2r::Error| x.to_string();
    let ctx = r2r::Context::create().map_err(e)?;
    let mut node = r2r::Node::create(ctx, "behavior_simlink", "").map_err(e)?;
    let qos = r2r::QosProfile::default();
    let mk = |node: &mut r2r::Node, pfx: &str, frame: &str| -> Result<CamPub, String> {
        Ok(CamPub {
            rgb: node.create_publisher::<Image>(&format!("{pfx}/rgb"), qos.clone()).map_err(e)?,
            depth: node.create_publisher::<Image>(&format!("{pfx}/depth"), qos.clone()).map_err(e)?,
            info: node.create_publisher::<CameraInfo>(&format!("{pfx}/info"), qos.clone()).map_err(e)?,
            pose: node.create_publisher::<PoseStamped>(&format!("{pfx}/pose"), qos.clone().keep_last(100)).map_err(e)?,
            frame: frame.into(),
            conv: None,
        })
    };
    let mut cams = [
        mk(&mut node, &a.str_or("head-prefix", "/camera"), "head_camera_optical")?,
        mk(&mut node, "/camera_left", "left_wrist_camera_optical")?,
        mk(&mut node, "/camera_right", "right_wrist_camera_optical")?,
    ];
    let own_pose = a.str_or("pose-source", "simlink") == "simlink";
    // simlink 가 자세를 낼 때만 /base_pose 발행(meridian 이면 meridian_odom 이 낸다 — 발행자 둘이 안 되게)
    let p_base = if own_pose { Some(node.create_publisher::<PoseStamped>("/base_pose", qos.clone().keep_last(100)).map_err(e)?) } else { None };
    let p_qvel = node.create_publisher::<TwistStamped>("/base_qvel", qos.clone().keep_last(100)).map_err(e)?;
    let p_ext = node.create_publisher::<PoseStamped>("/camera/extrinsic", qos.clone().keep_last(100)).map_err(e)?;
    let p_eef = node.create_publisher::<PoseArray>("/robot/eef", qos.clone().keep_last(100)).map_err(e)?;
    let p_grip = node.create_publisher::<JointState>("/robot/gripper", qos.clone().keep_last(100)).map_err(e)?;
    let ev_sub = node
        .subscribe::<r2r::meridian_msgs::msg::GraphUpdateEventDev>("/graph_update_event", r2r::QosProfile::default().keep_last(64))
        .map_err(e)?;
    let ext_sub = if own_pose { None } else { Some(node.subscribe::<PoseStamped>("/base_pose", r2r::QosProfile::default().keep_last(100)).map_err(e)?) };
    let mut pool = futures::executor::LocalPool::new();
    use futures::task::LocalSpawnExt;
    let sp = pool.spawner();
    let sh1 = sh.clone();
    sp.spawn_local(ev_sub.for_each(move |m| {
        let now = Instant::now();
        let mut g = sh1.g.lock().unwrap();
        if m.is_heartbeat {
            g.heartbeats += 1;
        } else {
            g.events += 1;
            let s = stamp_ns(&m.header.stamp);
            if let Some((tp, tr)) = g.published.get(&s).copied() {
                g.lat_ms.push((now - tp).as_secs_f64() * 1e3);
                g.age_ms.push((now - tr).as_secs_f64() * 1e3);
            }
            if s > g.last_event_stamp {
                g.last_event_stamp = s;
            }
            drop(g);
            sh1.cv.notify_all();
        }
        futures::future::ready(())
    }))
    .map_err(|x| x.to_string())?;
    if let Some(sub) = ext_sub {
        let sh2 = sh.clone();
        sp.spawn_local(sub.for_each(move |m| {
            let q = &m.pose.orientation;
            let yaw = (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z));
            let mut g = sh2.g.lock().unwrap();
            g.ext_fix = Some((stamp_ns(&m.header.stamp), Pose { x: m.pose.position.x, y: m.pose.position.y, yaw }));
            g.ext_new = true;
            g.ext_count += 1;
            futures::future::ready(())
        }))
        .map_err(|x| x.to_string())?;
    }
    eprintln!(
        "[simlink] ROS 노드 준비: /camera/* /camera_left/* /camera_right/* /base_qvel /camera/extrinsic{}, 받기 /graph_update_event{}",
        if own_pose { " /camera/pose /base_pose" } else { "" },
        if own_pose { "" } else { " /base_pose(meridian_odom)" }
    );
    let pubs = Pubs { base: p_base, qvel: p_qvel, ext: p_ext, eef: p_eef, grip: p_grip, own_pose };

    while !stop.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(2)) {
            Ok(Msg::Hello(h)) => {
                for c in h.cams.iter() {
                    let id = match c.name.as_str() {
                        "head" => 0,
                        "left_wrist" => 1,
                        "right_wrist" => 2,
                        _ => continue,
                    };
                    cams[id].conv = Some(CamConv::new(c.w as usize, c.h as usize, c.k, id == 0));
                }
            }
            Ok(Msg::Reset(ep)) => eprintln!("[simlink] 새 판 {ep}"),
            Ok(Msg::Pkt(p)) => publish(&mut cams, &pubs, &p, &sh).map_err(e)?,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        node.spin_once(Duration::ZERO);
        pool.run_until_stalled();
    }
    Ok(())
}

fn image(t: &Time, frame: &str, w: usize, h: usize, enc: &str, bpp: usize, data: Vec<u8>) -> Image {
    Image { header: Header { stamp: t.clone(), frame_id: frame.into() }, height: h as u32, width: w as u32, encoding: enc.into(), is_bigendian: 0, step: (w * bpp) as u32, data }
}

struct Pubs {
    base: Option<r2r::Publisher<PoseStamped>>,
    qvel: r2r::Publisher<TwistStamped>,
    ext: r2r::Publisher<PoseStamped>,
    eef: r2r::Publisher<PoseArray>,
    grip: r2r::Publisher<JointState>,
    own_pose: bool,
}

fn publish(cams: &mut [CamPub; 3], pb: &Pubs, p: &ObsPacket, sh: &Shared) -> Result<(), r2r::Error> {
    let t = stamp(p.stamp_ns);
    // meridian_odom 입력(매 스텝, 영상과 같은 stamp): base_qvel 원값, 머리 카메라 robot2cam(proprio 순기구학)
    pb.qvel.publish(&TwistStamped {
        header: Header { stamp: t.clone(), frame_id: "base_link".into() },
        twist: Twist { linear: Vector3 { x: p.qvel[0], y: p.qvel[1], z: 0.0 }, angular: Vector3 { x: 0.0, y: 0.0, z: p.qvel[2] } },
    })?;
    if let Some(rel) = &p.cam_rel[CAM_HEAD as usize] {
        pb.ext.publish(&PoseStamped {
            header: Header { stamp: t.clone(), frame_id: "base_link".into() },
            pose: RPose { position: Point { x: rel[0], y: rel[1], z: rel[2] }, orientation: Quaternion { x: rel[3], y: rel[4], z: rel[5], w: rel[6] } },
        })?;
    }
    // meridian 옮겨진 물체 처리 입력(매 스텝 이 스텝 stamp — 영상 k(stamp k-1)와는 k-1 값이 짝): 팔 끝 [왼, 오른], 그리퍼 합
    pb.eef.publish(&PoseArray {
        header: Header { stamp: t.clone(), frame_id: "base_link".into() },
        poses: p
            .eef
            .iter()
            .map(|e| RPose { position: Point { x: e[0], y: e[1], z: e[2] }, orientation: Quaternion { x: e[3], y: e[4], z: e[5], w: e[6] } })
            .collect(),
    })?;
    pb.grip.publish(&JointState {
        header: Header { stamp: t.clone(), frame_id: "base_link".into() },
        name: vec!["left_gripper".into(), "right_gripper".into()],
        position: p.grip.to_vec(),
        ..Default::default()
    })?;
    let has = |cam: u8| p.frames.iter().any(|f| f.cam == cam);
    if let Some(b) = &pb.base {
        let (rwb, twb) = pose::base_rt(&p.base);
        b.publish(&pose_msg(&t, "map", &rwb, &twb))?;
    }
    // 머리 /camera/pose 는 매 스텝 이 스텝 상태의 stamp 로(frontend 는 영상 stamp 이하 최신 자세를 쓴다 → 영상(k)의 stamp 는
    // k-1 이므로 k-1 에 낸 자세와 맞물린다). meridian_odom 이 낼 때(pose-source meridian)는 내지 않는다.
    if pb.own_pose {
        if let Some(rel) = &p.cam_rel[CAM_HEAD as usize] {
            let (r, tr) = pose::cam_optical(&p.base, rel);
            cams[CAM_HEAD as usize].pose.publish(&pose_msg(&t, "map", &r, &tr))?;
        }
    }
    // 영상 stamp = 직전 스텝(장면 시각 k-1). 손목 자세도 그 시각 값으로 영상과 같은 stamp.
    let ti = stamp(p.img_stamp_ns);
    for cam in [1u8, 2u8] {
        if !has(cam) {
            continue;
        }
        if let Some(rel) = &p.img_cam_rel[cam as usize] {
            let (r, tr) = pose::cam_optical(&p.img_base, rel);
            cams[cam as usize].pose.publish(&pose_msg(&ti, "map", &r, &tr))?;
        }
    }
    if p.frames.is_empty() {
        return Ok(());
    }
    let mut conv_ms = 0.0;
    let mut head = false;
    for cam in 0..3u8 {
        let fr: Vec<&Frame> = p.frames.iter().filter(|f| f.cam == cam).collect();
        if fr.is_empty() {
            continue;
        }
        let cp = &mut cams[cam as usize];
        if cp.conv.is_none() {
            let f = fr[0];
            cp.conv = Some(CamConv::new(f.w as usize, f.h as usize, [f.w as f64 / 2.0, f.w as f64 / 2.0, f.w as f64 / 2.0, f.h as f64 / 2.0], cam == 0));
        }
        let c = cp.conv.as_ref().unwrap();
        let mut msgs = Vec::new();
        for f in fr {
            if f.w as usize != c.in_w || f.h as usize != c.in_h {
                continue;
            }
            let t1 = Instant::now();
            let m = match f.kind {
                KIND_RGBA8 | KIND_RGB8 => {
                    let ch = if f.kind == KIND_RGBA8 { 4 } else { 3 };
                    (true, image(&ti, &cp.frame, c.out_w, c.out_h, "rgb8", 3, c.rgb(&f.data, ch)))
                }
                KIND_DEPTH_F32 => (false, image(&ti, &cp.frame, c.out_w, c.out_h, "16UC1", 2, c.depth_mm(&f.data))),
                _ => continue,
            };
            conv_ms += t1.elapsed().as_secs_f64() * 1e3;
            msgs.push(m);
        }
        cp.info.publish(&info_msg(&ti, &cp.frame, c))?;
        // RGB 먼저, 깊이 다음(obs_player 와 같은 순서)
        for (_, m) in msgs.iter().filter(|m| m.0) {
            cp.rgb.publish(m)?;
        }
        for (_, m) in msgs.iter().filter(|m| !m.0) {
            cp.depth.publish(m)?;
        }
        if cam == CAM_HEAD {
            head = true;
        }
    }
    let mut g = sh.g.lock().unwrap();
    if head {
        g.head_published += 1;
        g.published.insert(p.img_stamp_ns, (Instant::now(), p.recv));
        g.pub_order.push_back(p.img_stamp_ns);
        while g.pub_order.len() > 4000 {
            let k = g.pub_order.pop_front().unwrap();
            g.published.remove(&k);
        }
        g.pub_ms.push(p.recv.elapsed().as_secs_f64() * 1e3);
    }
    if p.frames.iter().any(|f| f.cam != CAM_HEAD) {
        g.wrist_published += 1;
    }
    g.conv_ms.push(conv_ms);
    Ok(())
}

/// scene_server `status` 를 주기적으로 물어 노드 수를 기록한다(끊겨도 계속 다시 시도).
fn scene_poller(addr: String, every: Duration, sh: Arc<Shared>, stop: Arc<AtomicBool>) {
    let mut conn: Option<BufReader<std::net::TcpStream>> = None;
    while !stop.load(Ordering::Relaxed) {
        if conn.is_none() {
            conn = std::net::TcpStream::connect(&addr).ok().map(|s| {
                let _ = s.set_nodelay(true);
                let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
                BufReader::new(s)
            });
        }
        let mut ok = false;
        if let Some(c) = conn.as_mut() {
            let mut line = String::new();
            if c.get_mut().write_all(b"{\"op\":\"status\"}\n").is_ok() && c.read_line(&mut line).map(|n| n > 0).unwrap_or(false) {
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    let n = v.get("num_objects").and_then(|x| x.as_u64()).unwrap_or(0);
                    let ev = v.get("events_applied").and_then(|x| x.as_u64()).unwrap_or(0);
                    let mut g = sh.g.lock().unwrap();
                    let t = sh.t0.elapsed().as_secs_f64();
                    if g.last_nodes != Some(n) || g.nodes.last().map(|x| t - x.0 > 10.0).unwrap_or(true) {
                        g.nodes.push((t, n, ev));
                    }
                    g.last_nodes = Some(n);
                    g.max_nodes = g.max_nodes.max(n);
                    g.scene_ok = true;
                    ok = true;
                }
            }
        }
        if !ok {
            conn = None;
        }
        std::thread::sleep(every);
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|s| s == "--help" || s == "-h") {
        println!("simlink — 평가기 ↔ 계획기 + meridian 관측 발행. 인자는 `bagent link` 와 같고 더해서 --scene 127.0.0.1:7791 --scene-every-ms 500 --queue 6 --head-prefix /camera --pose-source simlink|meridian");
        return;
    }
    let a = Args::parse(raw);
    let catalog = match bagent::setup::load_catalog(&a) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("과제 카드: {e}");
            std::process::exit(1);
        }
    };
    let cfg = match link::cfg_from_args(&a, catalog) {
        Ok(mut c) => {
            if a.str_or("pose-source", "simlink") == "meridian" {
                c.pose = "corrected".into(); // meridian_odom /base_pose 를 고정점으로 받는 추정기
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
    let sh = Arc::new(Shared { g: Mutex::new(GraphState::default()), cv: Condvar::new(), t0: Instant::now() });
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = sync_channel::<Msg>(a.num("queue", 6usize));
    let (sh2, st2, a2) = (sh.clone(), stop.clone(), a.clone());
    let ros = std::thread::spawn(move || {
        if let Err(e) = ros_thread(rx, sh2, st2, a2) {
            eprintln!("[simlink] ROS 스레드 오류: {e}");
        }
    });
    if a.str_or("scene", "127.0.0.1:7791") != "none" {
        let (sh3, st3) = (sh.clone(), stop.clone());
        let addr = a.str_or("scene", "127.0.0.1:7791");
        let every = Duration::from_millis(a.num("scene-every-ms", 500));
        std::thread::spawn(move || scene_poller(addr, every, sh3, st3));
    }
    let make = move || -> Box<dyn ObsSink> { Box::new(RosSink { tx: tx.clone(), sh: sh.clone(), boundary_stamp: HashMap::new() }) };
    let r = link::run(cfg, &make);
    stop.store(true, Ordering::Relaxed);
    let _ = ros.join();
    if let Err(e) = r {
        eprintln!("[simlink] {e}");
        std::process::exit(1);
    }
}
