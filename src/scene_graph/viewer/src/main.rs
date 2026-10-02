//! sgview — 로봇 물체 기억(① scene_graph) 2D 뷰어.
//!   sgview <dir>     dir = sm_save_dsg 출력(view.json · map.pgm · map.yaml · scene.json)
//! view.json 이 바뀌면 다시 읽어 그린다: 회색 = 2D 점유 격자, 파란 화살표 = 로봇, 점 = 물체 노드(색 = 상태),
//! 큰 가구(structural) = 상자. 휠 = 확대, 끌기 = 이동, F = 맞추기, Q/Esc = 끝.
use font8x8::{UnicodeFonts, BASIC_FONTS};
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use serde::Deserialize;
use std::{fs, path::PathBuf, time::{Duration, Instant, SystemTime}};

#[derive(Deserialize, Default, Clone)]
struct Grid { resolution: f64, origin: [f64; 2], width: usize, height: usize }
#[derive(Deserialize, Clone)]
struct Obj { id: u32, name: String, state: String, pos: [f64; 3], extent: [f64; 3], n_obs: u32, structural: bool, first_pos: [f64; 3] }
#[derive(Deserialize, Clone)]
struct Ev { t: f64, id: u32, kind: String }
#[derive(Deserialize, Default, Clone)]
struct View { stamp: f64, pose: [f64; 3], grid: Grid, objects: Vec<Obj>, events: Vec<Ev> }

const W: usize = 1000;
const H: usize = 800;

fn read_pgm(p: &PathBuf) -> Option<(usize, usize, Vec<u8>)> {
    let b = fs::read(p).ok()?;
    // P5\n<w> <h>\n255\n<data>
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 && i < b.len() {
        while i < b.len() && b[i].is_ascii_whitespace() { i += 1; }
        let s = i;
        while i < b.len() && !b[i].is_ascii_whitespace() { i += 1; }
        fields.push(String::from_utf8_lossy(&b[s..i]).to_string());
    }
    i += 1;
    let (w, h): (usize, usize) = (fields.get(1)?.parse().ok()?, fields.get(2)?.parse().ok()?);
    (b.len() >= i + w * h).then(|| (w, h, b[i..i + w * h].to_vec()))
}

fn color(state: &str) -> u32 {
    match state { "seen" => 0x2ecc71, "moved" => 0xf39c12, "held" => 0xe74c3c, "gone" => 0x7f8c8d, _ => 0xffffff }
}

struct Canvas { buf: Vec<u32> }
impl Canvas {
    fn px(&mut self, x: i64, y: i64, c: u32) {
        if x >= 0 && y >= 0 && (x as usize) < W && (y as usize) < H { self.buf[y as usize * W + x as usize] = c; }
    }
    fn disc(&mut self, cx: i64, cy: i64, r: i64, c: u32) {
        for dy in -r..=r { for dx in -r..=r { if dx * dx + dy * dy <= r * r { self.px(cx + dx, cy + dy, c); } } }
    }
    fn line(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, c: u32) {
        let n = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
        for k in 0..=n { self.px(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, c); }
    }
    fn rect(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, c: u32) {
        self.line(x0, y0, x1, y0, c); self.line(x1, y0, x1, y1, c); self.line(x1, y1, x0, y1, c); self.line(x0, y1, x0, y0, c);
    }
    fn text(&mut self, x: i64, y: i64, s: &str, c: u32) {
        let mut cx = x;
        for ch in s.chars() {
            if let Some(g) = BASIC_FONTS.get(ch) {
                for (row, bits) in g.iter().enumerate() {
                    for col in 0..8 { if bits & (1 << col) != 0 { self.px(cx + col, y + row as i64, c); } }
                }
            }
            cx += 8;
        }
    }
}

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    let mut win = Window::new("robot memory (scenemap / Spark-DSG)", W, H, WindowOptions::default()).expect("window");
    win.set_target_fps(30);
    let (mut view, mut grid_img) = (View::default(), None::<(usize, usize, Vec<u8>)>);
    let mut last_mtime = SystemTime::UNIX_EPOCH;
    let mut last_poll = Instant::now() - Duration::from_secs(1);
    let (mut scale, mut off, mut fit) = (40.0f64, [W as f64 / 2.0, H as f64 / 2.0], true);   // px/m, 화면 원점
    let mut drag: Option<(f32, f32)> = None;
    let mut reloads = 0u64;
    while win.is_open() && !win.is_key_down(Key::Escape) && !win.is_key_down(Key::Q) {
        if last_poll.elapsed() > Duration::from_millis(200) {
            last_poll = Instant::now();
            if let Ok(m) = fs::metadata(dir.join("view.json")).and_then(|m| m.modified()) {
                if m != last_mtime {
                    if let Ok(v) = fs::read(dir.join("view.json")).map_err(|_| ()).and_then(|b| serde_json::from_slice::<View>(&b).map_err(|_| ())) {
                        view = v; last_mtime = m; reloads += 1;
                        grid_img = read_pgm(&dir.join("map.pgm"));
                    }
                }
            }
        }
        if win.is_key_down(Key::F) { fit = true; }
        let g = view.grid.clone();
        if fit && g.width > 0 {
            let (wm, hm) = (g.width as f64 * g.resolution, g.height as f64 * g.resolution);
            scale = ((W as f64 - 40.0) / wm).min((H as f64 - 80.0) / hm);
            off = [20.0 - g.origin[0] * scale, H as f64 - 20.0 + g.origin[1] * scale];
            fit = false;
        }
        if let Some((_, sy)) = win.get_scroll_wheel() {
            if let Some((mx, my)) = win.get_mouse_pos(MouseMode::Clamp) {
                let f = if sy > 0.0 { 1.15 } else { 1.0 / 1.15 };
                off = [mx as f64 - (mx as f64 - off[0]) * f, my as f64 - (my as f64 - off[1]) * f];
                scale *= f;
            }
        }
        if win.get_mouse_down(MouseButton::Left) {
            if let Some((mx, my)) = win.get_mouse_pos(MouseMode::Clamp) {
                if let Some((px, py)) = drag { off[0] += (mx - px) as f64; off[1] += (my - py) as f64; }
                drag = Some((mx, my));
            }
        } else { drag = None; }
        let to = |x: f64, y: f64| ((off[0] + x * scale) as i64, (off[1] - y * scale) as i64);
        let mut cv = Canvas { buf: vec![0x1b1f24; W * H] };
        // 격자 (map.pgm: 위가 +y)
        if let Some((gw, gh, ref px)) = grid_img {
            if gw == g.width && gh == g.height && g.resolution > 0.0 {
                for sy in 0..H {
                    for sx in 0..W {
                        let x = (sx as f64 - off[0]) / scale; let y = (off[1] - sy as f64) / scale;
                        let (cx, cy) = (((x - g.origin[0]) / g.resolution) as i64, ((y - g.origin[1]) / g.resolution) as i64);
                        if cx < 0 || cy < 0 || cx >= gw as i64 || cy >= gh as i64 { continue; }
                        let v = px[(gh - 1 - cy as usize) * gw + cx as usize];
                        cv.buf[sy * W + sx] = match v { 205 => 0x2a2f36, 0 => 0xd0d0d0, _ => 0x3b4350 };
                    }
                }
            }
        }
        // 물체
        for o in &view.objects {
            let (x, y) = to(o.pos[0], o.pos[1]);
            let c = color(&o.state);
            if o.structural {
                let (a, b) = to(o.pos[0] - o.extent[0] / 2.0, o.pos[1] - o.extent[1] / 2.0);
                let (d, e) = to(o.pos[0] + o.extent[0] / 2.0, o.pos[1] + o.extent[1] / 2.0);
                cv.rect(a, e, d, b, 0x5dade2);
                cv.text(a + 2, e + 2, &format!("{}#{}", o.name, o.id), 0x5dade2);
            } else {
                if o.state == "moved" {   // 처음 본 자리 → 지금 자리
                    let (fx, fy) = to(o.first_pos[0], o.first_pos[1]);
                    cv.line(fx, fy, x, y, 0x7f6a3a);
                    cv.disc(fx, fy, 2, 0x7f6a3a);
                }
                cv.disc(x, y, 5, c);
                cv.text(x + 8, y - 4, &format!("{}#{} ({})", o.name, o.id, o.n_obs), c);
            }
        }
        // 로봇
        let (rx, ry) = to(view.pose[0], view.pose[1]);
        let (hx, hy) = to(view.pose[0] + 0.5 * view.pose[2].cos(), view.pose[1] + 0.5 * view.pose[2].sin());
        cv.disc(rx, ry, 7, 0x3498db);
        cv.line(rx, ry, hx, hy, 0xffffff);
        // 머리글·최근 사건
        let n_small = view.objects.iter().filter(|o| !o.structural).count();
        cv.text(10, 8, &format!("t={:.1}s  objects {} (furniture {})  reload {}  [wheel zoom, drag pan, F fit]",
            view.stamp, n_small, view.objects.len() - n_small, reloads), 0xecf0f1);
        cv.text(10, 20, "green seen  orange moved  red held  gray gone  blue box furniture", 0x95a5a6);
        for (i, e) in view.events.iter().rev().take(6).enumerate() {
            let name = view.objects.iter().find(|o| o.id == e.id).map(|o| o.name.as_str()).unwrap_or("?");
            cv.text(W as i64 - 330, 8 + 12 * i as i64, &format!("{:6.1}s {:<10} {}#{}", e.t, e.kind, name, e.id), 0xbdc3c7);
        }
        win.update_with_buffer(&cv.buf, W, H).ok();
    }
}
