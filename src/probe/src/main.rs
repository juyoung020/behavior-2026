//! 지시 형식 오프라인 실험 (open loop).
//!
//! 시연의 단계 구간 안 프레임마다 π0.5 에 같은 관측(카메라 3장·상태)과 형식만 다른 prompt 를 넣고,
//! 예측 행동 묶음을 시연 행동과 비교한다. 이 바이너리는 실험 로직만 맡는다:
//!   probe prompts <samples.json> <prompts.jsonl>                     형식별 prompt 줄 만들기
//!   probe report  <samples.json> <prompts.jsonl> <preds 접두사> <out.md> [제목]
//!                                                                    예측(<접두사>.f32, <접두사>_gt.f32, <접두사>_meta.json) 채점 → 표
//! 표본(samples.json + .npz)은 openpi-comet scripts/probe_prep.py, 추론은 scripts/probe_infer.py (정책이 파이썬/JAX 라 그쪽).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;

#[derive(Deserialize, Clone)]
struct Base {
    dx_m: f64,
    dy_m: f64,
    dyaw_deg: f64,
}

#[derive(Deserialize, Clone)]
struct Sample {
    id: usize,
    episode_index: i64,
    frame: i64,
    stage_idx: usize,
    skill: String,
    objects: Vec<String>,
    stage_frames: Vec<i64>,
    remaining_base: Base,
    all_skills: Vec<String>,
    all_objects: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct Samples {
    horizon: usize,
    samples: Vec<Sample>,
}

#[derive(Serialize, Deserialize, Clone)]
struct PromptRow {
    row: usize,
    sample: usize,
    format: String,
    text: String,
}

/// 2026 데모로 openpi 를 학습할 때 π0.5 가 받는 문장(과제 이름, "_"→" ")
const TASK_NAME: &str = "turning on radio";
/// 서빙 때 문장 (openpi configs/tasks/b1k.py, Comet scripts/task_mapping.json 과 같음)
const TASK_SENTENCE: &str = "Turn on the radio receiver that's on the table in the living room.";
const PURPOSE: &str = "to turn on the radio receiver on the table";

const FORMATS: [(&str, &str); 7] = [
    ("f0_task_name", "⓪ 과제 이름 `turning on radio` (2026 데모 학습 문장)"),
    ("f1_task", "① 과제 문장 (서빙 문장)"),
    ("f1r_task_repeat", "①' 과제 문장 한 번 더 (잡음 기준)"),
    ("f2_skill", "② 기술 + 물체 (Comet 형식)"),
    ("f3_purpose", "③ 목적 / 예상 행동"),
    ("f4_numeric", "④ 숫자 명령"),
    ("f5_wrong_skill", "⑤ 다른 단계의 기술 (대조)"),
];

/// "coffee_table_koagbh_0" -> "coffee table", "radio_89" -> "radio"
fn object_name(id: &str) -> String {
    let mut t: Vec<&str> = id.split('_').collect();
    if t.len() > 1 && t.last().map_or(false, |s| s.chars().all(|c| c.is_ascii_digit())) {
        t.pop();
    }
    if t.len() > 1 && t.last().map_or(false, |s| s.len() == 6 && s.chars().all(|c| c.is_ascii_lowercase())) {
        t.pop();
    }
    t.join(" ")
}

fn skill_phrase(skill: &str, objs: &[String]) -> String {
    let o: Vec<String> = objs.iter().map(|s| object_name(s)).collect();
    let first = o.first().cloned().unwrap_or_default();
    match (skill, o.len()) {
        ("move to", _) => format!("move to the {first}"),
        ("pick up from", n) if n >= 2 => format!("pick up the {first} from the {}", o[1]),
        ("place on", n) if n >= 2 => format!("place the {first} on the {}", o[1]),
        ("place in", n) if n >= 2 => format!("place the {first} in the {}", o[1]),
        (s, 0) => s.to_string(),
        (s, _) => format!("{s} the {}", o.join(" and the ")),
    }
}

fn numeric(phrase: &str, b: &Base) -> String {
    let fwd = if b.dx_m >= 0.0 { "forward" } else { "backward" };
    let side = if b.dy_m >= 0.0 { "left" } else { "right" };
    let turn = if b.dyaw_deg >= 0.0 { "left" } else { "right" };
    format!(
        "{phrase}: go {fwd} {:.1} m, {:.1} m to the {side}, turn {turn} {:.0} degrees",
        b.dx_m.abs(),
        b.dy_m.abs(),
        b.dyaw_deg.abs()
    )
}

fn prompt_text(fmt: &str, s: &Sample) -> String {
    let phrase = skill_phrase(&s.skill, &s.objects);
    match fmt {
        "f0_task_name" => TASK_NAME.into(),
        "f1_task" | "f1r_task_repeat" => TASK_SENTENCE.into(),
        "f2_skill" => phrase,
        "f3_purpose" => format!("Purpose: {PURPOSE}. Expected action: {phrase}."),
        "f4_numeric" => numeric(&phrase, &s.remaining_base),
        "f5_wrong_skill" => {
            let n = s.all_skills.len();
            (1..n)
                .map(|k| (s.stage_idx + k) % n)
                .map(|j| skill_phrase(&s.all_skills[j], &s.all_objects[j]))
                .find(|p| *p != phrase)
                .unwrap_or_else(|| "open the door".into())
        }
        _ => unreachable!(),
    }
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

const GROUPS: [(&str, &[usize]); 4] = [
    ("이동 3", &[0, 1, 2]),
    ("몸통 4", &[3, 4, 5, 6]),
    ("팔 14", &[7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]),
    ("그리퍼 2", &[14, 22]),
];

/// 묶음 [H][23] 두 개의 차원 묶음별 평균 절대 차
fn group_l1(a: &[f32], b: &[f32], h: usize, dims: &[usize]) -> f64 {
    let mut s = 0.0;
    for t in 0..h {
        for &d in dims {
            s += (a[t * 23 + d] - b[t * 23 + d]).abs() as f64;
        }
    }
    s / (h * dims.len()) as f64
}

fn mean_dim(a: &[f32], h: usize, d: usize) -> f64 {
    (0..h).map(|t| a[t * 23 + d] as f64).sum::<f64>() / h as f64
}

fn cos2(a: (f64, f64), b: (f64, f64)) -> Option<f64> {
    let na = (a.0 * a.0 + a.1 * a.1).sqrt();
    let nb = (b.0 * b.0 + b.1 * b.1).sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return None;
    }
    Some((a.0 * b.0 + a.1 * b.1) / (na * nb))
}

#[derive(Default, Clone)]
struct Acc {
    n: usize,
    sum: f64,
}
impl Acc {
    fn add(&mut self, v: f64) {
        self.n += 1;
        self.sum += v;
    }
    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }
}
fn f(v: Option<f64>, digits: usize) -> String {
    v.map_or("–".into(), |x| format!("{x:.digits$}"))
}

/// 시연 이동 속도 평균이 이보다 작으면 방향 비교에서 뺀다(정지 구간)
const MOVE_THR: f64 = 0.05;
/// 지시 이동량이 이보다 작으면 "숫자 명령 방향" 비교에서 뺀다
const INSTR_THR_M: f64 = 0.2;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("prompts") if args.len() == 4 => prompts(&args[2], &args[3]),
        Some("report") if args.len() >= 6 => report(&args[2], &args[3], &args[4], &args[5], args.get(6).map(String::as_str)),
        _ => {
            eprintln!("사용법: probe prompts <samples.json> <prompts.jsonl> | probe report <samples.json> <prompts.jsonl> <preds 접두사> <out.md> [제목]");
            std::process::exit(2);
        }
    }
}

fn load_samples(p: &str) -> Samples {
    serde_json::from_str(&fs::read_to_string(p).expect("samples.json")).expect("samples.json 형식")
}

fn prompts(samples: &str, out: &str) {
    let s = load_samples(samples);
    let mut buf = String::new();
    let mut row = 0;
    // 표본 바깥, 형식 안쪽: 같은 관측에 형식만 바꿔 연달아 부른다
    for smp in &s.samples {
        for (fmt, _) in FORMATS {
            let r = PromptRow { row, sample: smp.id, format: fmt.into(), text: prompt_text(fmt, smp) };
            buf.push_str(&serde_json::to_string(&r).unwrap());
            buf.push('\n');
            row += 1;
        }
    }
    fs::write(out, buf).expect("prompts 쓰기");
    println!("prompt {row}줄 ({}표본 × {}형식) -> {out}", s.samples.len(), FORMATS.len());
}

fn report(samples: &str, prompts_p: &str, preds: &str, out: &str, title: Option<&str>) {
    let s = load_samples(samples);
    let rows: Vec<PromptRow> = fs::read_to_string(prompts_p)
        .expect("prompts.jsonl")
        .lines()
        .map(|l| serde_json::from_str(l).expect("prompt 줄"))
        .collect();
    let h = s.horizon;
    let pred = read_f32(&format!("{preds}.f32"));
    let gt = read_f32(&format!("{preds}_gt.f32"));
    assert_eq!(pred.len(), rows.len() * h * 23, "예측 크기");
    assert_eq!(gt.len(), s.samples.len() * h * 23, "정답 크기");
    let meta: serde_json::Value =
        fs::read_to_string(format!("{preds}_meta.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let p_of = |r: usize| &pred[r * h * 23..(r + 1) * h * 23];
    let g_of = |i: usize| &gt[i * h * 23..(i + 1) * h * 23];
    // 표본별 형식 -> 줄 번호
    let mut idx: BTreeMap<(usize, String), usize> = BTreeMap::new();
    for r in &rows {
        idx.insert((r.sample, r.format.clone()), r.row);
    }
    let skills: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        for smp in &s.samples {
            if !v.contains(&smp.skill) {
                v.push(smp.skill.clone());
            }
        }
        v
    };

    // (형식, 기술 또는 "전체") -> 지표 누적
    #[derive(Default, Clone)]
    struct M {
        err: [Acc; 4],
        sens: [Acc; 4],
        dir: Acc,
        yaw: Acc,
        instr: Acc,
    }
    let mut m: BTreeMap<(String, String), M> = BTreeMap::new();
    for smp in &s.samples {
        let g = g_of(smp.id);
        let Some(&r1) = idx.get(&(smp.id, "f1_task".into())) else { continue };
        let p1 = p_of(r1);
        let gv = (mean_dim(g, h, 0), mean_dim(g, h, 1));
        let gw = mean_dim(g, h, 2);
        let rb = &smp.remaining_base;
        for (fmt, _) in FORMATS {
            let Some(&r) = idx.get(&(smp.id, fmt.to_string())) else { continue };
            let p = p_of(r);
            let pv = (mean_dim(p, h, 0), mean_dim(p, h, 1));
            let pw = mean_dim(p, h, 2);
            for key in [smp.skill.clone(), "전체".to_string()] {
                let e = m.entry((fmt.to_string(), key)).or_default();
                for (gi, (_, dims)) in GROUPS.iter().enumerate() {
                    e.err[gi].add(group_l1(p, g, h, dims));
                    if fmt != "f1_task" {
                        e.sens[gi].add(group_l1(p, p1, h, dims));
                    }
                }
                if (gv.0 * gv.0 + gv.1 * gv.1).sqrt() > MOVE_THR {
                    if let Some(c) = cos2(pv, gv) {
                        e.dir.add(c);
                    }
                }
                if gw.abs() > MOVE_THR {
                    e.yaw.add(if pw.signum() == gw.signum() { 1.0 } else { 0.0 });
                }
                if (rb.dx_m * rb.dx_m + rb.dy_m * rb.dy_m).sqrt() > INSTR_THR_M {
                    if let Some(c) = cos2(pv, (rb.dx_m, rb.dy_m)) {
                        e.instr.add(c);
                    }
                }
            }
        }
    }

    let mut md = String::new();
    let _ = writeln!(md, "### {}\n", title.unwrap_or("결과"));
    let _ = writeln!(
        md,
        "- 모델 `{}` (`{}`), 표본 {} (에피소드 {}개 × 단계 × 프레임), prompt 줄 {}, 묶음 길이 H = {} 스텝",
        meta["policy_config"].as_str().unwrap_or("?"),
        meta["policy_dir"].as_str().unwrap_or("?"),
        s.samples.len(),
        s.samples.iter().map(|x| x.episode_index).collect::<std::collections::BTreeSet<_>>().len(),
        rows.len(),
        h
    );
    let _ = writeln!(
        md,
        "- 추론: 로딩 {} s, 첫 호출 {} s, 이후 중앙 {} ms (p90 {} ms), JAX GPU 메모리 현재 {} GiB / 최대 {} GiB\n",
        meta["load_s"], meta["infer_first_s"], meta["infer_ms_median"], meta["infer_ms_p90"], meta["jax_bytes_in_use_gib"], meta["jax_peak_bytes_gib"]
    );

    let _ = writeln!(md, "**표 1. 형식별 — 시연 행동과의 차이(평균 절대 차, 낮을수록 시연에 가까움)와 ① 대비 변화량**\n");
    let _ = writeln!(
        md,
        "| 형식 | 이동 3 | 몸통 4 | 팔 14 | 그리퍼 2 | 이동 방향 cos (n) | 회전 부호 일치 (n) | ① 대비 변화: 이동 | 팔 | 그리퍼 |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|---|");
    for (fmt, name) in FORMATS {
        let Some(e) = m.get(&(fmt.to_string(), "전체".into())) else { continue };
        let _ = writeln!(
            md,
            "| {name} | {} | {} | {} | {} | {} ({}) | {} ({}) | {} | {} | {} |",
            f(e.err[0].mean(), 4),
            f(e.err[1].mean(), 4),
            f(e.err[2].mean(), 4),
            f(e.err[3].mean(), 4),
            f(e.dir.mean(), 2),
            e.dir.n,
            f(e.yaw.mean(), 2),
            e.yaw.n,
            f(e.sens[0].mean(), 4),
            f(e.sens[2].mean(), 4),
            f(e.sens[3].mean(), 4),
        );
    }
    let _ = writeln!(
        md,
        "\n- \"① 대비 변화\" = 같은 관측에서 ① 과제 문장 때의 예측과 얼마나 다른가. ①' 줄이 같은 문장을 한 번 더 넣었을 때의 차이(표본 잡음 = 기준선)다. 이보다 뚜렷이 커야 \"문장이 행동을 바꾼다\".\n- 이동 방향 cos: 묶음 평균 (vx, vy) 가 시연과 같은 쪽이면 1, 반대면 -1. 시연 평균 속도 {MOVE_THR} 미만(정지)은 뺐다. 회전 부호 일치: 시연 평균 wz 가 {MOVE_THR} 이상일 때 부호가 같은 비율.\n"
    );

    let _ = writeln!(md, "**표 2. 단계(기술)별 — 이동·팔 오차와 ① 대비 변화(이동 / 팔)**\n");
    let mut head = String::from("| 형식 |");
    let mut sep = String::from("|---|");
    for sk in &skills {
        let _ = write!(head, " {sk}: 이동 오차 · 팔 오차 · 변화(이동/팔) |");
        sep.push_str("---|");
    }
    let _ = writeln!(md, "{head}\n{sep}");
    for (fmt, name) in FORMATS {
        let mut line = format!("| {name} |");
        for sk in &skills {
            match m.get(&(fmt.to_string(), sk.clone())) {
                Some(e) => {
                    let _ = write!(
                        line,
                        " {} · {} · {}/{} |",
                        f(e.err[0].mean(), 3),
                        f(e.err[2].mean(), 3),
                        f(e.sens[0].mean(), 3),
                        f(e.sens[2].mean(), 3)
                    );
                }
                None => line.push_str(" – |"),
            }
        }
        let _ = writeln!(md, "{line}");
    }

    let _ = writeln!(
        md,
        "\n**표 3. 이동 방향이 \"지시한 이동\"과 맞나** — 예측 묶음 평균 (vx, vy) 와, 그 프레임부터 단계 끝까지 시연이 실제로 간 (dx, dy)(= ④ 숫자 명령에 넣은 값)의 cos. 이동량 {INSTR_THR_M} m 이상인 표본만.\n"
    );
    let _ = writeln!(md, "| 형식 | 전체 cos (n) | move to 단계 cos (n) |");
    let _ = writeln!(md, "|---|---|---|");
    for (fmt, name) in FORMATS {
        let all = m.get(&(fmt.to_string(), "전체".into())).map(|e| e.instr.clone()).unwrap_or_default();
        let mv = m.get(&(fmt.to_string(), "move to".into())).map(|e| e.instr.clone()).unwrap_or_default();
        let _ = writeln!(md, "| {name} | {} ({}) | {} ({}) |", f(all.mean(), 2), all.n, f(mv.mean(), 2), mv.n);
    }

    let _ = writeln!(md, "\n**prompt 예** (표본마다 형식만 바뀐다)\n");
    let mut shown = std::collections::BTreeSet::new();
    for smp in &s.samples {
        if !shown.insert(smp.skill.clone()) {
            continue;
        }
        let _ = writeln!(md, "- 단계 `{}` (에피소드 {}, 프레임 {}, 구간 {:?}):", smp.skill, smp.episode_index, smp.frame, smp.stage_frames);
        for (fmt, name) in FORMATS {
            if fmt == "f1r_task_repeat" || fmt == "f1_task" || fmt == "f0_task_name" {
                continue;
            }
            let _ = writeln!(md, "  - {name}: `{}`", prompt_text(fmt, smp));
        }
    }
    fs::write(out, &md).expect("결과 쓰기");
    println!("{md}");
}
