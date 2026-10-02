//! tracecmp — 두 평가 실행의 스텝별 기록을 비교해 항목마다 '비트 동일 / 허용오차 안 / 다름' 으로 판정한다.
//! tools/trace_compare.py 의 Rust 판: 같은 입력이면 같은 판정·같은 출력(글자 단위)·같은 종료 코드.
//!
//!     tracecmp <기준 결과폴더> <비교 결과폴더> [--env-a 0 --env-b 0] [--check] [--strict] [--negative]
//!              [--show k1,k2] [--pixels-report-only]
//!
//! 읽는 것(결과폴더 안): trace.npz(평가기 쪽 기록), server_log.npz(서버가 받은 관측, 키 앞에 "srv|"), json/*.json(공식 결과),
//! trace_images.npz(몇 스텝의 영상 원본 — 있으면 픽셀 차이 표).

mod pyfmt;

use npz as npy;
use npy::{Array, Data};
use pyfmt::{fmt_e, fmt_f, num, py_eq, py_str};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

// 임계 (파이썬 TOL 과 같은 순서 — 앞에서부터 처음 맞는 것)
const TOL: &[(&str, f64)] = &[
    ("robot_pose", 1e-6),
    ("robot_qpos", 1e-6),
    ("robot_qvel", 1e-5),
    ("obj::", 1e-6),
    ("obs::", 1e-6),
    ("val::", 1e-6),
    ("obs_mean::", 1e-3),
    ("agent_delta", 1e-7),
    ("action", 0.0),
];
const EXACT_PREFIX: &[&str] = &["obs_hash::", "hash::", "goal_satisfied", "terminated", "truncated", "active"];
const V_SAME: &str = "비트 동일";
const V_TOL: &str = "허용오차 안";
const V_DIFF: &str = "다름";
const V_NA: &str = "비교 불가";

fn tol(k: &str) -> f64 {
    for (p, v) in TOL {
        if k == *p || (p.ends_with("::") && k.starts_with(p)) {
            return *v;
        }
    }
    1e-6
}

struct Run {
    keys: Vec<String>,
    arr: HashMap<String, Array>,
    json: HashMap<String, Value>,
}

impl Run {
    fn get(&self, k: &str) -> Option<&Array> {
        self.arr.get(k)
    }
}

fn load_run(d: &str) -> Result<Run, String> {
    let mut keys = Vec::new();
    let mut arr = HashMap::new();
    for (name, pre) in [("trace.npz", ""), ("server_log.npz", "srv|")] {
        let p = format!("{}/{}", d.trim_end_matches(['/', '\\']), name);
        if std::path::Path::new(&p).exists() {
            for (k, a) in npy::load_npz(&p)? {
                let key = format!("{pre}{k}");
                if !arr.contains_key(&key) {
                    keys.push(key.clone());
                }
                arr.insert(key, a);
            }
        }
    }
    let mut json = HashMap::new();
    let jd = format!("{}/json", d.trim_end_matches(['/', '\\']));
    if let Ok(rd) = std::fs::read_dir(&jd) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".json") {
                let txt = std::fs::read_to_string(e.path()).map_err(|er| format!("{name}: {er}"))?;
                let v: Value = serde_json::from_str(&pyfmt::json_lenient(&txt)).map_err(|er| format!("{name}: {er}"))?;
                json.insert(name, v);
            }
        }
    }
    Ok(Run { keys, arr, json })
}

/// 스텝 i, 환경 e 칸의 원소 범위
fn cell(a: &Array, i: usize, e: usize) -> std::ops::Range<usize> {
    let n = a.shape.get(1).copied().unwrap_or(1);
    let c = a.cell_size();
    let s = (i * n + e) * c;
    s..s + c
}

fn f64s(a: &Array) -> Option<&Vec<f64>> {
    if let Data::F(v) = &a.data {
        Some(v)
    } else {
        None
    }
}

/// 반환: (판정, 최대|차이|, 처음 비트가 다른 스텝, 처음 임계 넘은 스텝, 비교 스텝 수)
fn compare_key(k: &str, a: &Array, b: &Array, ea: usize, eb: usize, off: Option<(&[f64], &[f64])>)
               -> (&'static str, f64, Option<usize>, Option<usize>, usize) {
    let t = a.len0().min(b.len0());
    if t == 0 {
        return (V_NA, f64::NAN, None, None, 0);
    }
    let base = k.split_once('|').map(|x| x.1).unwrap_or(k);
    if a.shape.get(2..).unwrap_or(&[]) != b.shape.get(2..).unwrap_or(&[]) {
        return (V_DIFF, f64::INFINITY, Some(0), Some(0), t);
    }
    let exact = a.kind == 'U' || a.kind == 'S' || a.kind == 'b' || EXACT_PREFIX.iter().any(|p| base.starts_with(p));
    if exact {
        let mut neq = Vec::with_capacity(t);
        for i in 0..t {
            let (ra, rb) = (cell(a, i, ea), cell(b, i, eb));
            let eq = match (&a.data, &b.data) {
                (Data::S(x), Data::S(y)) => x[ra] == y[rb],
                (Data::B(x), Data::B(y)) => x[ra] == y[rb],
                (Data::U8(x), Data::U8(y)) => x[ra] == y[rb],
                (Data::F(x), Data::F(y)) => x[ra].iter().zip(&y[rb]).all(|(p, q)| p == q),
                _ => false,
            };
            neq.push(!eq);
        }
        let first = neq.iter().position(|&x| x);
        let cnt = neq.iter().filter(|&&x| x).count() as f64;
        return (if first.is_none() { V_SAME } else { V_DIFF }, cnt, first, first, t);
    }
    let (x, y) = match (f64s(a), f64s(b)) {
        (Some(x), Some(y)) => (x, y),
        _ => {
            // 숫자로 못 바꾸는 것(파이썬이면 astype 에서 죽는 경우) — 다름으로
            return (V_DIFF, f64::INFINITY, Some(0), Some(0), t);
        }
    };
    let last = a.shape.last().copied().unwrap_or(1).max(1);
    let use_off = off.filter(|_| base.starts_with("obj::") || base.starts_with("robot_pose"));
    let tl = tol(base);
    let mut per_step = vec![0.0f64; t];
    let mut bitdiff = vec![false; t];
    for i in 0..t {
        let (ra, rb) = (cell(a, i, ea), cell(b, i, eb));
        let mut mx = 0.0f64;
        let mut any_diff = false;
        let mut first_el = true;
        for (j, (&xv0, &yv0)) in x[ra].iter().zip(&y[rb]).enumerate() {
            let (mut xv, mut yv) = (xv0, yv0);
            if let Some((oa, ob)) = use_off {
                let c = j % last;
                if c < 3 {
                    xv -= oa.get(c).copied().unwrap_or(0.0);
                    yv -= ob.get(c).copied().unwrap_or(0.0);
                }
            }
            let both_nan = xv.is_nan() && yv.is_nan();
            let mut d = if both_nan { 0.0 } else { (xv - yv).abs() };
            if d.is_nan() {
                d = f64::INFINITY;
            }
            mx = if first_el { d } else { mx.max(d) };
            first_el = false;
            let (xe, ye) = if both_nan { (0.0, 0.0) } else { (xv, yv) };
            if xe != ye {
                any_diff = true;
            }
        }
        per_step[i] = mx;
        bitdiff[i] = any_diff;
    }
    let first_bit = bitdiff.iter().position(|&v| v);
    let first_over = per_step.iter().position(|&v| v > tl);
    let m = per_step.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let verdict = if first_bit.is_none() { V_SAME } else if first_over.is_none() { V_TOL } else { V_DIFF };
    (verdict, m, first_bit, first_over, t)
}

fn img_report(da: &str, db: &str, ea: usize, eb: usize) -> Result<(), String> {
    let pa = format!("{}/trace_images.npz", da.trim_end_matches(['/', '\\']));
    let pb = format!("{}/trace_images.npz", db.trim_end_matches(['/', '\\']));
    if !(std::path::Path::new(&pa).exists() && std::path::Path::new(&pb).exists()) {
        return Ok(());
    }
    let za: HashMap<String, Array> = npy::load_npz(&pa)?.into_iter().collect();
    let zb: HashMap<String, Array> = npy::load_npz(&pb)?.into_iter().collect();
    let index = |z: &HashMap<String, Array>, e: usize| -> HashMap<String, String> {
        let mut m = HashMap::new();
        for k in z.keys() {
            let p: Vec<&str> = k.splitn(3, '|').collect();
            if p.len() == 3 && p[1] == e.to_string() {
                m.insert(format!("{}|{}", p[0], p[2]), k.clone());
            }
        }
        m
    };
    let (ka, kb) = (index(&za, ea), index(&zb, eb));
    println!("\n영상 원본 비교 (스텝|카메라: 검은 화면 여부 A/B, 다른 픽셀 비율, 평균|차|, 최대|차|, 0~255)");
    let mut common: Vec<&String> = ka.keys().filter(|k| kb.contains_key(*k)).collect();
    common.sort_by(|p, q| {
        let sp: i64 = p.split('|').next().unwrap().parse().unwrap_or(0);
        let sq: i64 = q.split('|').next().unwrap().parse().unwrap_or(0);
        (sp, p.as_str()).cmp(&(sq, q.as_str()))
    });
    for key in common {
        let (x, y) = (&za[&ka[key]], &zb[&kb[key]]);
        let (xs, ys) = match (&x.data, &y.data) {
            (Data::U8(p), Data::U8(q)) => (p, q),
            _ => continue,
        };
        let ch = x.shape.last().copied().unwrap_or(1);
        let use_c = ch.min(3);
        let npix = xs.len() / ch.max(1);
        let (mut bx, mut by) = (true, true);
        let (mut sum, mut mx, mut ndiff) = (0u64, 0u8, 0usize);
        for p in 0..npix {
            let mut pd = 0u8;
            for c in 0..use_c {
                let (u, v) = (xs[p * ch + c], ys[p * ch + c]);
                if u != 0 {
                    bx = false;
                }
                if v != 0 {
                    by = false;
                }
                let d = u.abs_diff(v);
                sum += d as u64;
                pd = pd.max(d);
            }
            mx = mx.max(pd);
            if pd > 0 {
                ndiff += 1;
            }
        }
        let step = key.split('|').next().unwrap();
        let rest = key.split('|').nth(1).unwrap_or("");
        let cam = if key.contains("::") {
            rest.split("::").nth(1).and_then(|s| s.split(':').nth(1)).unwrap_or(rest).to_string()
        } else {
            key.to_string()
        };
        let pct = ndiff as f64 / npix.max(1) as f64 * 100.0;
        let mean = sum as f64 / (npix * use_c).max(1) as f64;
        println!("  {:>4}|{:<22} 검정 {}/{}  다른 픽셀 {:>5}%  평균 {:>6}  최대 {:>3}",
                 step, cam, if bx { 'O' } else { '-' }, if by { 'O' } else { '-' }, fmt_f(pct, 1), fmt_f(mean, 3), mx);
    }
    Ok(())
}

fn compare_json(ja: &HashMap<String, Value>, jb: &HashMap<String, Value>) -> (Vec<(String, String, String, String, &'static str)>, usize) {
    let names: BTreeSet<&String> = ja.keys().chain(jb.keys()).collect();
    let mut out = Vec::new();
    let mut bad = 0;
    for name in names {
        let (a, b) = (ja.get(name), jb.get(name));
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                out.push((name.clone(), "한쪽에 없음".into(), String::new(), String::new(), V_DIFF));
                bad += 1;
                continue;
            }
        };
        let mut rows: Vec<(String, Value, Value)> = Vec::new();
        for key in ["success", "steps"] {
            rows.push((key.into(), a.get(key).cloned().unwrap_or(Value::Null), b.get(key).cloned().unwrap_or(Value::Null)));
        }
        let q = |v: &Value| v.get("q_score").and_then(|x| x.get("final")).cloned().unwrap_or(Value::Null);
        rows.push(("q_score".into(), q(a), q(b)));
        if let Some(Value::Object(ad)) = a.get("agent_distance") {
            for (part, va) in ad {
                let vb = b.get("agent_distance").and_then(|x| x.get(part)).cloned().unwrap_or(Value::Null);
                rows.push((format!("agent_distance.{part}"), va.clone(), vb));
            }
        }
        for (key, va, vb) in rows {
            let same = py_eq(&va, &vb);
            let ok = if key.starts_with("agent_distance") {
                match (num(&va), num(&vb)) {
                    (Some(x), Some(y)) => (x - y).abs() <= 1e-6 * 1.0f64.max(x.abs()),
                    _ => false,
                }
            } else {
                same
            };
            let verdict = if same { V_SAME } else if ok { V_TOL } else { V_DIFF };
            if verdict == V_DIFF {
                bad += 1;
            }
            out.push((name.clone(), key, py_str(&va), py_str(&vb), verdict));
        }
    }
    (out, bad)
}

fn pad_left(s: &str, w: usize) -> String {
    format!("{:<w$}", s, w = w)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let (mut env_a, mut env_b) = (0usize, 0usize);
    let (mut check, mut strict, mut negative, mut pix_only) = (false, false, false, false);
    let mut show = String::new();
    let mut i = 0;
    while i < args.len() {
        let s = args[i].as_str();
        let val = |i: usize| -> String { args.get(i + 1).cloned().unwrap_or_default() };
        match s {
            "--env-a" => { env_a = val(i).parse().unwrap_or(0); i += 1; }
            "--env-b" => { env_b = val(i).parse().unwrap_or(0); i += 1; }
            "--show" => { show = val(i); i += 1; }
            "--check" => check = true,
            "--strict" => strict = true,
            "--negative" => negative = true,
            "--pixels-report-only" => pix_only = true,
            _ if s.starts_with("--env-a=") => env_a = s[8..].parse().unwrap_or(0),
            _ if s.starts_with("--env-b=") => env_b = s[8..].parse().unwrap_or(0),
            _ if s.starts_with("--show=") => show = s[7..].to_string(),
            _ => pos.push(s.to_string()),
        }
        i += 1;
    }
    if pos.len() != 2 {
        eprintln!("쓰는 법: tracecmp <A 결과폴더> <B 결과폴더> [--env-a N] [--env-b N] [--check] [--strict] [--negative] [--show k1,k2] [--pixels-report-only]");
        std::process::exit(2);
    }
    let (da, db) = (pos[0].clone(), pos[1].clone());
    let ra = load_run(&da).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) });
    let mut rb = if negative { Run { keys: vec![], arr: HashMap::new(), json: HashMap::new() } } else {
        load_run(&db).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) })
    };
    if negative {
        rb = Run { keys: ra.keys.clone(), arr: ra.arr.clone(), json: ra.json.clone() };
        let mut mid: Option<usize> = None;
        for k in &ra.keys {
            let v = rb.arr.get_mut(k).unwrap();
            if mid.is_none() {
                mid = Some(v.len0() / 2);
            }
            let base = k.split_once('|').map(|x| x.1).unwrap_or(k);
            if base.starts_with("obj::") && v.kind == 'f' {
                let m = mid.unwrap();
                let n = v.shape.get(1).copied().unwrap_or(1);
                let c = v.cell_size();
                let t = v.len0();
                if let Data::F(d) = &mut v.data {
                    for s in m..t {
                        d[(s * n + env_b) * c] += 1e-5; // 과제 물체 한 개 x 를 10 um 옮김
                    }
                }
                break;
            }
        }
        for k in &ra.keys {
            let base = k.split_once('|').map(|x| x.1).unwrap_or(k);
            if base.starts_with("obs_hash::") || base.starts_with("hash::") {
                let v = rb.arr.get_mut(k).unwrap();
                let n = v.shape.get(1).copied().unwrap_or(1);
                if let Data::S(d) = &mut v.data {
                    d[mid.unwrap() * n + env_b] = "0".repeat(32); // 영상 한 장 바꿈
                }
                break;
            }
        }
        println!("[음성 대조] B = A 복사본에 스텝 {} 부터 과제 물체 x +1e-5 m, 영상 해시 1 개 변경", mid.map(|m| m.to_string()).unwrap_or("None".into()));
    }

    let off_row = |r: &Run, e: usize| -> Option<Vec<f64>> {
        r.get("static::scene_offset").and_then(|a| f64s(a).map(|v| {
            let c = a.step_size();
            v[e * c..(e + 1) * c].to_vec()
        }))
    };
    let (oa, ob) = (off_row(&ra, env_a), off_row(&rb, env_b));
    let mut keys: Vec<&String> = ra.keys.iter().filter(|k| rb.arr.contains_key(*k) && !k.starts_with("static::")).collect();
    keys.sort();
    println!("A = {} (env {})\nB = {} (env {})", da, env_a, db, env_b);
    println!("{}{}{:>12}{:>9}{:>8}{:>6}", pad_left("항목", 58), pad_left("판정", 10), "최대|차이|", "첫 비트차", "첫 초과", "스텝");
    let order = [V_SAME, V_TOL, V_DIFF, V_NA];
    let mut counts: HashMap<&str, usize> = order.iter().map(|&k| (k, 0)).collect();
    let mut pix: HashMap<&str, usize> = order.iter().map(|&k| (k, 0)).collect();
    for k in keys {
        let off = match (&oa, &ob) {
            (Some(x), Some(y)) => Some((x.as_slice(), y.as_slice())),
            (Some(x), None) => Some((x.as_slice(), &[][..])),
            _ => None,
        };
        let (v, m, fb, fo, t) = compare_key(k, &ra.arr[k], &rb.arr[k], env_a, env_b, off);
        let base = k.split_once('|').map(|x| x.1).unwrap_or(k);
        let shown = if pix_only && (base.starts_with("obs_hash::") || base.starts_with("hash::") || base.starts_with("obs_mean::")) {
            *pix.get_mut(v).unwrap() += 1;
            format!("{v}(픽셀)")
        } else {
            *counts.get_mut(v).unwrap() += 1;
            v.to_string()
        };
        let fb_s = fb.map(|x| x.to_string()).unwrap_or("-".into());
        let fo_s = fo.map(|x| x.to_string()).unwrap_or("-".into());
        let chars: Vec<char> = k.chars().collect();
        let name = if chars.len() <= 57 { k.clone() } else { format!("…{}", chars[chars.len() - 56..].iter().collect::<String>()) };
        println!("{}{}{:>12}{:>9}{:>8}{:>6}", pad_left(&name, 58), pad_left(&shown, 10), fmt_e(m, 3), fb_s, fo_s, t);
    }
    for k in show.split(',').filter(|s| !s.is_empty()) {
        if let (Some(a), Some(b)) = (ra.get(k), rb.get(k)) {
            if let (Some(x), Some(y)) = (f64s(a), f64s(b)) {
                let t = a.len0().min(b.len0());
                let d: Vec<f64> = (0..t).map(|i| {
                    let (p, q) = (cell(a, i, env_a), cell(b, i, env_b));
                    x[p].iter().zip(&y[q]).map(|(u, v)| (u - v).abs()).fold(f64::NEG_INFINITY, |acc, z| {
                        if acc.is_nan() || z.is_nan() { f64::NAN } else { acc.max(z) }
                    })
                }).collect();
                let want: BTreeSet<usize> = [0usize, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, t.wrapping_sub(1)]
                    .into_iter().filter(|&i| i < t).collect();
                let parts: Vec<String> = want.iter().map(|&i| format!("스텝{} {}", i, fmt_e(d[i], 2))).collect();
                println!("  {}: {}", k, parts.join(", "));
            }
        }
    }
    if let Err(e) = img_report(&da, &db, env_a, env_b) {
        eprintln!("{e}");
    }
    let (jrows, jbad) = compare_json(&ra.json, &rb.json);
    println!("\n결과 JSON");
    for (name, key, va, vb, v) in &jrows {
        println!("  {}{}{:>22}{:>22}  {}", pad_left(name, 34), pad_left(key, 24), va, vb, v);
    }
    println!("\n요약: 비트 동일 {}, 허용오차 안 {}, 다름 {}, 비교 불가 {}, JSON 다름 {}",
             counts[V_SAME], counts[V_TOL], counts[V_DIFF], counts[V_NA], jbad);
    if pix_only {
        println!("픽셀(판정에서 뺌): 비트 동일 {}, 허용오차 안 {}, 다름 {}", pix[V_SAME], pix[V_TOL], pix[V_DIFF]);
    }
    let bad = counts[V_DIFF] + jbad + if strict { counts[V_TOL] } else { 0 };
    if negative {
        println!("음성 대조: {}", if bad > 0 { "잡았다 (정상)" } else { "못 잡았다 -- 비교가 무디다" });
        std::process::exit(if bad > 0 { 0 } else { 1 });
    }
    if check {
        if bad == 0 { println!("통과") } else { println!("실패 -- {} 개 항목", bad) }
        std::process::exit(if bad > 0 { 1 } else { 0 });
    }
}
