//! sgview — scene-graph viewer server (Rust). Replaces the Python/viser viewer.
//!
//! Reads the folder scenemap writes (`view.json`, `map.pgm` + `map.yaml`, `objects/O<id>_*`) and serves it to a three.js page.
//! No Python, no Spark-DSG, no crates besides std: everything the page needs comes from `view.json`, which already carries
//! objects, rooms, the compact graph, robot pose and grid info. Walls as 2D lines / numbers come from scenemap's C++ (`walls.cpp`).
//!
//!   sgview <memory_dir> [--port 8080] [--bind 0.0.0.0]
//!
//! GET /                      the page
//! GET /api/view?v=<ver>      {"v": ver, "map_v": ver, "view": <view.json>} or {"unchanged": true} (poll; ver = mtime+size)
//! GET /api/map               PGM pixels (row 0 = max y), size in X-W / X-H headers
//! GET /api/walls?x=&y=&yaw=  {"segments": [[ax,ay,bx,by]..], "state": [56 floats], "pose": [x,y,yaw]}
//! GET /file/<relative path>  any file below the memory dir (PNG crops, PLY points)
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const INDEX_HTML: &str = include_str!("../assets/index.html");
const THREE_JS: &[u8] = include_bytes!("../assets/three.min.js");
const ORBIT_JS: &[u8] = include_bytes!("../assets/OrbitControls.js");

const WALL_STATE_LEN: usize = 56;

extern "C" {
    fn sgv_wall_segments(cells: *const i8, w: i32, h: i32, res: f64, ox: f64, oy: f64, out: *mut f64, cap: i32) -> i32;
    fn sgv_wall_state(cells: *const i8, w: i32, h: i32, res: f64, ox: f64, oy: f64, segs: *const f64, n: i32, pose: *const f64, out: *mut f32);
}

/// Occupancy grid in scenemap layout (row 0 = min y; −1 unknown, 0..100 occupied %), plus its wall segments.
struct WallMap {
    ver: String,
    w: i32,
    h: i32,
    res: f64,
    ox: f64,
    oy: f64,
    cells: Vec<i8>,
    segs: Vec<f64>, // ax ay bx by …
}

struct State {
    dir: PathBuf,
    walls: Mutex<Option<Arc<WallMap>>>,
}

fn file_ver(p: &Path) -> Option<String> {
    let m = fs::metadata(p).ok()?;
    let t = m.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos();
    Some(format!("{}-{}", t, m.len()))
}

struct Pgm {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

fn read_pgm(p: &Path) -> Option<Pgm> {
    let b = fs::read(p).ok()?;
    let mut i = 0;
    let mut tok = Vec::new();
    while tok.len() < 4 {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < b.len() && b[i] == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while i < b.len() && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        if s == i {
            return None;
        }
        tok.push(String::from_utf8_lossy(&b[s..i]).to_string());
    }
    i += 1; // one whitespace after maxval
    if tok[0] != "P5" || tok[3] != "255" {
        return None;
    }
    let (w, h): (usize, usize) = (tok[1].parse().ok()?, tok[2].parse().ok()?);
    if b.len() < i + w * h {
        return None;
    }
    Some(Pgm { w, h, px: b[i..i + w * h].to_vec() })
}

/// resolution and origin (x, y) of a ROS map.yaml
fn read_yaml(p: &Path) -> (f64, f64, f64) {
    let (mut res, mut ox, mut oy) = (0.05, 0.0, 0.0);
    if let Ok(s) = fs::read_to_string(p) {
        for line in s.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if let Some(v) = line.strip_prefix("resolution:") {
                res = v.trim().parse().unwrap_or(res);
            } else if let Some(v) = line.strip_prefix("origin:") {
                let n: Vec<f64> = v.trim().trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if n.len() >= 2 {
                    ox = n[0];
                    oy = n[1];
                }
            }
        }
    }
    (res, ox, oy)
}

/// The wall map of the current `map.pgm` (cached by mtime+size; segments are recomputed only when the map file changes).
fn wall_map(st: &State) -> Option<Arc<WallMap>> {
    let pgm_path = st.dir.join("map.pgm");
    let ver = file_ver(&pgm_path)?;
    let mut g = st.walls.lock().unwrap();
    if let Some(w) = g.as_ref() {
        if w.ver == ver {
            return Some(w.clone());
        }
    }
    let pgm = read_pgm(&pgm_path)?;
    let (res, ox, oy) = read_yaml(&st.dir.join("map.yaml"));
    let (w, h) = (pgm.w, pgm.h);
    // PGM: row 0 = max y, grey ≤ 90 occupied, 205 unknown, else free  →  scenemap layout (row 0 = min y)
    let mut cells = vec![0i8; w * h];
    for y in 0..h {
        let src = &pgm.px[(h - 1 - y) * w..(h - y) * w];
        let dst = &mut cells[y * w..(y + 1) * w];
        for x in 0..w {
            dst[x] = if src[x] <= 90 { 100 } else if src[x] == 205 { -1 } else { 0 };
        }
    }
    let mut buf = vec![0f64; 4 * 512];
    let mut n = unsafe { sgv_wall_segments(cells.as_ptr(), w as i32, h as i32, res, ox, oy, buf.as_mut_ptr(), 512) } as usize;
    if n > 512 {
        buf = vec![0f64; 4 * n];
        n = unsafe { sgv_wall_segments(cells.as_ptr(), w as i32, h as i32, res, ox, oy, buf.as_mut_ptr(), n as i32) } as usize;
    }
    buf.truncate(4 * n);
    let wm = Arc::new(WallMap { ver, w: w as i32, h: h as i32, res, ox, oy, cells, segs: buf });
    *g = Some(wm.clone());
    Some(wm)
}

fn json_f64s(v: &[f64]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{:.4}", x));
    }
    s.push(']');
    s
}

fn walls_json(st: &State, q: &HashMap<String, String>) -> Option<String> {
    let wm = wall_map(st)?;
    let get = |k: &str| q.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    let pose = [get("x"), get("y"), get("yaw")];
    let mut out = [0f32; WALL_STATE_LEN];
    unsafe {
        sgv_wall_state(wm.cells.as_ptr(), wm.w, wm.h, wm.res, wm.ox, wm.oy, wm.segs.as_ptr(), (wm.segs.len() / 4) as i32, pose.as_ptr(), out.as_mut_ptr());
    }
    let segs: Vec<String> = wm.segs.chunks(4).map(|c| json_f64s(c)).collect();
    let state: Vec<String> = out.iter().map(|x| format!("{:.5}", x)).collect();
    Some(format!(
        "{{\"segments\":[{}],\"state\":[{}],\"pose\":{},\"map_v\":\"{}\",\"n_sectors\":16,\"k_segments\":8,\"max_range\":4.0}}",
        segs.join(","),
        state.join(","),
        json_f64s(&pose),
        wm.ver
    ))
}

fn content_type(p: &str) -> &'static str {
    match p.rsplit('.').next().unwrap_or("") {
        "png" => "image/png",
        "json" => "application/json",
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript",
        "ply" | "pgm" => "application/octet-stream",
        _ => "application/octet-stream",
    }
}

fn respond(s: &mut TcpStream, code: u16, ctype: &str, extra: &str, body: &[u8]) {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n{}\r\n",
        code,
        reason,
        ctype,
        body.len(),
        extra
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
}

fn parse_query(q: &str) -> HashMap<String, String> {
    q.split('&')
        .filter_map(|kv| {
            let mut it = kv.splitn(2, '=');
            Some((it.next()?.to_string(), it.next().unwrap_or("").to_string()))
        })
        .collect()
}

/// `rel` below `root`, no `..`, no absolute parts
fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    let mut p = root.to_path_buf();
    for c in Path::new(rel).components() {
        match c {
            Component::Normal(x) => p.push(x),
            _ => return None,
        }
    }
    Some(p)
}

fn handle(mut s: TcpStream, st: Arc<State>) {
    let mut rd = BufReader::new(s.try_clone().unwrap());
    let mut line = String::new();
    if rd.read_line(&mut line).is_err() {
        return;
    }
    loop {
        let mut h = String::new();
        if rd.read_line(&mut h).unwrap_or(0) <= 2 {
            break;
        }
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    if method != "GET" {
        return respond(&mut s, 400, "text/plain", "", b"GET only");
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let q = parse_query(query);
    match path {
        "/" | "/index.html" => respond(&mut s, 200, "text/html; charset=utf-8", "", INDEX_HTML.as_bytes()),
        "/three.min.js" => respond(&mut s, 200, "application/javascript", "", THREE_JS),
        "/OrbitControls.js" => respond(&mut s, 200, "application/javascript", "", ORBIT_JS),
        "/api/view" => {
            let vp = st.dir.join("view.json");
            let ver = file_ver(&vp).unwrap_or_default();
            if ver.is_empty() {
                return respond(&mut s, 200, "application/json", "", b"{\"waiting\":true}");
            }
            if q.get("v").map(|v| v == &ver).unwrap_or(false) {
                return respond(&mut s, 200, "application/json", "", b"{\"unchanged\":true}");
            }
            match fs::read_to_string(&vp) {
                Ok(t) if t.trim_start().starts_with('{') && t.trim_end().ends_with('}') => {
                    let map_v = file_ver(&st.dir.join("map.pgm")).unwrap_or_default();
                    let body = format!("{{\"v\":\"{}\",\"map_v\":\"{}\",\"view\":{}}}", ver, map_v, t);
                    respond(&mut s, 200, "application/json", "", body.as_bytes())
                }
                _ => respond(&mut s, 200, "application/json", "", b"{\"waiting\":true}"), // mid-write: next poll
            }
        }
        "/api/map" => match read_pgm(&st.dir.join("map.pgm")) {
            Some(p) => respond(&mut s, 200, "application/octet-stream", &format!("X-W: {}\r\nX-H: {}\r\n", p.w, p.h), &p.px),
            None => respond(&mut s, 404, "text/plain", "", b"no map"),
        },
        "/api/walls" => match walls_json(&st, &q) {
            Some(j) => respond(&mut s, 200, "application/json", "", j.as_bytes()),
            None => respond(&mut s, 404, "text/plain", "", b"no map"),
        },
        _ if path.starts_with("/file/") => {
            let rel = &path["/file/".len()..];
            match safe_join(&st.dir, rel).and_then(|p| fs::read(p).ok()) {
                Some(b) => respond(&mut s, 200, content_type(rel), "", &b),
                None => respond(&mut s, 404, "text/plain", "", b"not found"),
            }
        }
        _ => respond(&mut s, 404, "text/plain", "", b"not found"),
    }
}

fn main() {
    let mut dir: Option<String> = None;
    let (mut port, mut bind) = (8080u16, "0.0.0.0".to_string());
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--bind" => bind = it.next().unwrap_or(bind),
            _ => dir = Some(a),
        }
    }
    let dir = match dir {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!("usage: sgview <memory_dir> [--port 8080] [--bind 0.0.0.0]");
            std::process::exit(2);
        }
    };
    let st = Arc::new(State { dir, walls: Mutex::new(None) });
    let l = TcpListener::bind((bind.as_str(), port)).unwrap_or_else(|e| {
        eprintln!("cannot listen on {}:{}: {}", bind, port, e);
        std::process::exit(1);
    });
    let _ = SystemTime::now();
    eprintln!("sgview: http://localhost:{}  (memory dir {})", port, st.dir.display());
    for c in l.incoming().flatten() {
        let st = st.clone();
        std::thread::spawn(move || handle(c, st));
    }
}
