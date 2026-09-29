//! 프레임별 표 만들기 — 원래 openpi 변환 중 "프레임 하나로 정해지는" 부분을 한 번 계산해 둔다.
//!
//! 원래 (openpi pi05_b1k, 샘플마다 파이썬):                         여기 (프레임마다 한 번, Rust):
//!   B1KInputs: 상태 61 → 23 (그리퍼 두 손가락 합, float32)          state_ex.f32   [n, S]
//!   Normalize: (x - mean) / (std + 1e-6)  — mean·std 가 float64     state_out.f32  [n, 32] (float64 로 계산 → float32)
//!   TokenizePrompt: 정규화 상태를 256칸으로 나눠 문장에 넣고 SentencePiece  tok_off.u64 [n+1], tok_ids.i32
//!   PadStatesAndActions: 32 차원으로 0 채움
//!   영상: 에피소드 시작 시각 + 프레임 시각 → round(× fps) 번째 프레임  req.i32 [n, cams, 2] (파일 번호, 프레임)
//!   행동 창 32개·델타·정규화는 샘플마다 달라서 C++ 로더가 한다 →   action.f32 [n, A], ep_last.i32 [n]
//!
//! 각 줄의 출처: openpi policies/b1k_policy.py:22-41, transforms.py:115-139·294-312·374-383, models/tokenizer.py:22-48,
//! lerobot datasets/dataset_reader.py:187-223, video_utils.py:299-322. 원래와 같은지는 tools/ft_verify.py 가 전수 대조한다.
//!
//! 스펙(spec.json)은 파이썬 접착부(src/fasttrain/native.py)가 openpi 설정에서 뽑아 쓴다(로봇 인덱스·델타 대응·정규화 통계
//! 원본 float64 바이트·과제 문장·토크나이저 경로). 데이터 파일 하나씩 읽고 쓰므로 100과제 규모에서도 메모리가 일정하다.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int64Type};
use arrow_array::{Array, RecordBatch};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use serde::Deserialize;

use crate::mp4::{self, R};

#[derive(Deserialize)]
struct Proprio {
    indices: Vec<usize>,
    eef: bool,
}

#[derive(Deserialize)]
struct Delta {
    action: Vec<usize>,
    state: Vec<usize>,
}

#[derive(Deserialize)]
struct Spec {
    root: String,
    out: String,
    idx_dir: String,
    episodes: Option<Vec<i64>>,
    cams: Vec<String>,
    state_key: String,
    action_key: String,
    proprio: Vec<Proprio>,
    delta: Vec<Delta>,
    action_dim: usize,
    model_dim: usize,
    horizon: usize,
    norm_file: String, // float64 LE: state_mean[S], state_std[S], action_mean[A], action_std[A]
    prompts: BTreeMap<String, String>,
    tokenizer: String,
    max_token_len: usize,
    discrete_state: bool,
    tolerance_s: f64,
    threads: Option<usize>,
}

fn io<E: std::fmt::Display>(what: &str) -> impl Fn(E) -> String + '_ {
    move |e| format!("{what}: {e}")
}

fn sorted_parquets(dir: &Path) -> R<Vec<PathBuf>> {
    // lerobot io_utils.py:73 — sorted(pq_dir.glob("*/*.parquet"))
    let mut out = vec![];
    let mut subs: Vec<PathBuf> = fs::read_dir(dir).map_err(io("read_dir"))?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    subs.sort();
    for s in subs.iter().filter(|p| p.is_dir()) {
        let mut fsv: Vec<PathBuf> = fs::read_dir(s)
            .map_err(io("read_dir"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "parquet").unwrap_or(false))
            .collect();
        fsv.sort();
        out.extend(fsv);
    }
    Ok(out)
}

fn read_parquet(path: &Path, cols: &[&str]) -> R<Vec<RecordBatch>> {
    let f = File::open(path).map_err(io("parquet open"))?;
    let b = ParquetRecordBatchReaderBuilder::try_new(f).map_err(io("parquet"))?;
    let schema = b.schema().clone();
    let mut roots = vec![];
    for c in cols {
        roots.push(schema.index_of(c).map_err(|_| format!("{}: 열 없음 {c}", path.display()))?);
    }
    let mask = ProjectionMask::roots(b.parquet_schema(), roots);
    let rd = b.with_projection(mask).with_batch_size(1 << 16).build().map_err(io("parquet"))?;
    rd.map(|x| x.map_err(|e| format!("parquet 읽기: {e}"))).collect()
}

fn col_i64<'a>(b: &'a RecordBatch, name: &str) -> R<&'a arrow_array::PrimitiveArray<Int64Type>> {
    Ok(b.column_by_name(name).ok_or(format!("열 없음 {name}"))?.as_primitive::<Int64Type>())
}

struct EpVid {
    file: usize,
    from_ts: f64,
}
struct Ep {
    from: i64,
    to: i64,
    vids: Vec<EpVid>,
}

struct VFile {
    path: String,
    idx: String,
    timescale: f64,
    fps: f64,
    pts: Vec<i64>,
}

fn index_path(idx_dir: &str, root: &str, video: &str) -> String {
    // src/fasttrain/native.py 의 ensure_index 와 같은 이름: videos/ 아래 상대 경로의 / 를 . 으로
    let rel = video.strip_prefix(&format!("{root}/videos/")).unwrap_or(video);
    format!("{idx_dir}/{}.ftidx", rel.replace('/', "."))
}

fn mtime(p: &str) -> Option<std::time::SystemTime> {
    fs::metadata(p).and_then(|m| m.modified()).ok()
}

fn open_video(path: &str, idx: &str) -> R<VFile> {
    if mtime(idx).zip(mtime(path)).map(|(a, b)| a < b).unwrap_or(true) {
        mp4::cmd_index(path, idx)?;
    }
    let t = mp4::load_track(path)?;
    let step = t.pts.windows(2).map(|w| w[1] - w[0]).collect::<HashSet<_>>();
    if step.len() != 1 {
        return Err(format!("{path}: 프레임 간격이 일정하지 않다 {step:?} — average_fps 를 다시 맞춰야 한다"));
    }
    let d = *step.iter().next().unwrap();
    // torchcodec(approximate) average_fps = 스트림 avg_frame_rate = timescale / 표본 길이 (이 데이터: 15360/512 = 30)
    Ok(VFile { path: path.into(), idx: idx.into(), timescale: t.timescale as f64, fps: t.timescale as f64 / d as f64, pts: t.pts })
}

struct RowIn {
    abs: i64,
    ep: i64,
    task: i64,
    ts: f32,
    state: Vec<f32>,
    action: Vec<f32>,
}

struct RowOut {
    state_ex: Vec<f32>,
    state_out: Vec<f32>,
    tokens: Vec<i32>,
}

fn digitize(x: f64) -> i64 {
    // np.digitize(x, np.linspace(-1, 1, 257)[:-1]) - 1 : bins[k] = -1 + k/128 (정확히 표현되는 값), 개수 = bins <= x 인 수
    let mut lo = 0usize;
    let mut hi = 256usize;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let b = -1.0 + (mid as f64) * 0.0078125;
        if b <= x {
            lo = mid + 1
        } else {
            hi = mid
        }
    }
    lo as i64 - 1
}

fn compute(sp: &Spec, norm: &[f64], s_dim: usize, rows: &[RowIn], spp: &sentencepiece::SentencePieceProcessor, bos: i32) -> R<Vec<RowOut>> {
    let (smean, sstd) = (&norm[..s_dim], &norm[s_dim..2 * s_dim]);
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        // B1KInputs.extract_state_from_proprio (float32, 그리퍼 두 값 합)
        let mut ex = Vec::with_capacity(s_dim);
        for p in &sp.proprio {
            if p.eef {
                if p.indices.len() != 2 {
                    return Err("그리퍼 합은 두 값만 지원 (torch 합 순서를 따로 맞춰야 한다)".into());
                }
                ex.push(r.state[p.indices[0]] + r.state[p.indices[1]]);
            } else {
                ex.extend(p.indices.iter().map(|&i| r.state[i]));
            }
        }
        // Normalize: float32 상태 - float64 평균 → float64 (numpy 형 승격)
        let n64: Vec<f64> = (0..s_dim).map(|d| ((ex[d] as f64) - smean[d]) / (sstd[d] + 1e-6)).collect();
        let mut st = vec![0f32; sp.model_dim];
        for d in 0..s_dim {
            st[d] = n64[d] as f32; // 원래는 float64 로 넘어가 JAX 가 float32 로 바꾼다 (x64 꺼짐)
        }
        // PaligemmaTokenizer.tokenize (tokenizer.py:22-48)
        let prompt = sp.prompts.get(&r.task.to_string()).ok_or(format!("과제 {} 문장 없음", r.task))?;
        let cleaned = prompt.trim().replace('_', " ").replace('\n', " ");
        let text = if sp.discrete_state {
            let s: Vec<String> = n64.iter().map(|&x| digitize(x).to_string()).collect();
            format!("Task: {cleaned}, State: {};\nAction: ", s.join(" "))
        } else {
            return Err("discrete_state=false (pi0 형식) 는 아직 지원 안 함".into());
        };
        let pieces = spp.encode(&text).map_err(io("sentencepiece"))?;
        let mut tok = Vec::with_capacity(pieces.len() + 1);
        tok.push(bos);
        tok.extend(pieces.iter().map(|p| p.id as i32));
        tok.truncate(sp.max_token_len);
        out.push(RowOut { state_ex: ex, state_out: st, tokens: tok });
    }
    Ok(out)
}

struct Writers {
    abs: BufWriter<File>,
    task: BufWriter<File>,
    ep_last: BufWriter<File>,
    state_ex: BufWriter<File>,
    action: BufWriter<File>,
    state_out: BufWriter<File>,
    tok_off: BufWriter<File>,
    tok_ids: BufWriter<File>,
    req: BufWriter<File>,
}

fn w(dir: &str, name: &str) -> R<BufWriter<File>> {
    Ok(BufWriter::with_capacity(1 << 20, File::create(format!("{dir}/{name}")).map_err(io(name))?))
}

fn put<T: Copy>(wr: &mut BufWriter<File>, v: &[T]) -> R<()> {
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    wr.write_all(b).map_err(io("쓰기"))
}

pub fn run(spec_path: &str) -> R<()> {
    let t0 = Instant::now();
    let sp: Spec = serde_json::from_str(&fs::read_to_string(spec_path).map_err(io("spec"))?).map_err(io("spec json"))?;
    let s_dim: usize = sp.proprio.iter().map(|p| if p.eef { 1 } else { p.indices.len() }).sum();
    let a_dim = sp.action_dim;
    let nb = fs::read(&sp.norm_file).map_err(io("norm"))?;
    let norm: Vec<f64> = nb.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    if norm.len() != 2 * s_dim + 2 * a_dim {
        return Err(format!("정규화 통계 길이 {} != 2*{s_dim} + 2*{a_dim}", norm.len()));
    }
    fs::create_dir_all(&sp.out).map_err(io("out"))?;
    fs::create_dir_all(&sp.idx_dir).map_err(io("idx_dir"))?;
    let root = sp.root.trim_end_matches('/').to_string();
    let want: Option<HashSet<i64>> = sp.episodes.as_ref().map(|v| v.iter().copied().collect());

    // --- 에피소드 메타 (lerobot meta/episodes)
    let mut eps: HashMap<i64, Ep> = HashMap::new();
    let mut files: Vec<VFile> = vec![];
    let mut fid: HashMap<String, usize> = HashMap::new();
    let mut cols: Vec<String> = vec!["episode_index".into(), "dataset_from_index".into(), "dataset_to_index".into()];
    for c in &sp.cams {
        for k in ["chunk_index", "file_index", "from_timestamp"] {
            cols.push(format!("videos/{c}/{k}"));
        }
    }
    let colr: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
    for p in sorted_parquets(&Path::new(&root).join("meta/episodes"))? {
        for b in read_parquet(&p, &colr)? {
            let e = col_i64(&b, "episode_index")?;
            let fr = col_i64(&b, "dataset_from_index")?;
            let to = col_i64(&b, "dataset_to_index")?;
            for i in 0..b.num_rows() {
                let ei = e.value(i);
                if want.as_ref().map(|s| !s.contains(&ei)).unwrap_or(false) {
                    continue;
                }
                let mut vids = vec![];
                for c in &sp.cams {
                    let ch = col_i64(&b, &format!("videos/{c}/chunk_index"))?.value(i);
                    let fi = col_i64(&b, &format!("videos/{c}/file_index"))?.value(i);
                    let ft = b
                        .column_by_name(&format!("videos/{c}/from_timestamp"))
                        .ok_or("from_timestamp 없음")?
                        .as_primitive::<Float64Type>()
                        .value(i);
                    let path = format!("{root}/videos/{c}/chunk-{ch:03}/file-{fi:03}.mp4");
                    let id = match fid.get(&path) {
                        Some(&id) => id,
                        None => {
                            let idx = index_path(&sp.idx_dir, &root, &path);
                            files.push(open_video(&path, &idx)?);
                            fid.insert(path.clone(), files.len() - 1);
                            files.len() - 1
                        }
                    };
                    vids.push(EpVid { file: id, from_ts: ft });
                }
                eps.insert(ei, Ep { from: fr.value(i), to: to.value(i), vids });
            }
        }
    }
    if let Some(wset) = &want {
        for e in wset {
            if !eps.contains_key(e) {
                return Err(format!("에피소드 {e} 가 meta/episodes 에 없다"));
            }
        }
    }
    eprintln!("에피소드 {} 개, 영상 파일 {} 개 ({:.1}s)", eps.len(), files.len(), t0.elapsed().as_secs_f64());

    // --- 프레임 표
    let mut wr = Writers {
        abs: w(&sp.out, "abs_index.i64")?,
        task: w(&sp.out, "task.i32")?,
        ep_last: w(&sp.out, "ep_last.i32")?,
        state_ex: w(&sp.out, "state_ex.f32")?,
        action: w(&sp.out, "action.f32")?,
        state_out: w(&sp.out, "state_out.f32")?,
        tok_off: w(&sp.out, "tok_off.u64")?,
        tok_ids: w(&sp.out, "tok_ids.i32")?,
        req: w(&sp.out, "req.i32")?,
    };
    let threads = sp.threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)).max(1);
    let mut n_rows: i64 = 0;
    let mut tok_total: u64 = 0;
    put(&mut wr.tok_off, &[0u64])?;
    let (mut prev_ep, mut prev_abs) = (i64::MIN, 0i64);
    let mut max_tok = 0usize;
    let mut n_trunc = 0usize;
    let dcols = ["index", "episode_index", "task_index", "timestamp", sp.state_key.as_str(), sp.action_key.as_str()];
    for p in sorted_parquets(&Path::new(&root).join("data"))? {
        let mut rows: Vec<RowIn> = vec![];
        for b in read_parquet(&p, &dcols)? {
            let ix = col_i64(&b, "index")?;
            let ep = col_i64(&b, "episode_index")?;
            let tk = col_i64(&b, "task_index")?;
            let ts = b.column_by_name("timestamp").ok_or("timestamp 없음")?.as_primitive::<Float32Type>();
            let st = b.column_by_name(&sp.state_key).ok_or("state 없음")?.as_list::<i32>();
            let ac = b.column_by_name(&sp.action_key).ok_or("action 없음")?.as_list::<i32>();
            let stv = st.values().as_primitive::<Float32Type>();
            let acv = ac.values().as_primitive::<Float32Type>();
            for i in 0..b.num_rows() {
                let e = ep.value(i);
                if want.as_ref().map(|s| !s.contains(&e)).unwrap_or(false) {
                    continue;
                }
                let (so, se) = (st.value_offsets()[i] as usize, st.value_offsets()[i + 1] as usize);
                let (ao, ae) = (ac.value_offsets()[i] as usize, ac.value_offsets()[i + 1] as usize);
                if ae - ao != a_dim {
                    return Err(format!("행동 길이 {} != {a_dim}", ae - ao));
                }
                rows.push(RowIn {
                    abs: ix.value(i),
                    ep: e,
                    task: tk.value(i),
                    ts: ts.value(i),
                    state: stv.values()[so..se].to_vec(),
                    action: acv.values()[ao..ae].to_vec(),
                });
            }
        }
        if rows.is_empty() {
            continue;
        }
        // 병렬 계산 (스레드마다 토크나이저 하나)
        let chunk = (rows.len() + threads - 1) / threads;
        let results: Vec<R<Vec<RowOut>>> = std::thread::scope(|s| {
            let hs: Vec<_> = rows
                .chunks(chunk)
                .map(|part| {
                    let sp = &sp;
                    let norm = &norm;
                    s.spawn(move || -> R<Vec<RowOut>> {
                        let spp = sentencepiece::SentencePieceProcessor::open(&sp.tokenizer).map_err(io("토크나이저"))?;
                        let bos = spp.bos_id().ok_or("bos 없음")? as i32;
                        compute(sp, norm, s_dim, part, &spp, bos)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("스레드 실패".into()))).collect()
        });
        let mut outs = vec![];
        for r in results {
            outs.extend(r?);
        }
        for (r, o) in rows.iter().zip(outs.iter()) {
            let ep = eps.get(&r.ep).ok_or(format!("에피소드 {} 메타 없음", r.ep))?;
            // 연속성: 에피소드 안에서 절대 번호가 1씩, 에피소드는 처음부터 끝까지 (행동 창 계산이 여기에 기댄다)
            if r.ep == prev_ep {
                if r.abs != prev_abs + 1 {
                    return Err(format!("에피소드 {} 안에서 번호가 끊긴다 {} → {}", r.ep, prev_abs, r.abs));
                }
            } else {
                if prev_ep != i64::MIN {
                    let pe = &eps[&prev_ep];
                    if prev_abs != pe.to - 1 {
                        return Err(format!("에피소드 {prev_ep} 가 끝까지 안 왔다"));
                    }
                }
                if r.abs != ep.from {
                    return Err(format!("에피소드 {} 가 처음부터 시작하지 않는다", r.ep));
                }
            }
            prev_ep = r.ep;
            prev_abs = r.abs;
            let ep_last_rel = n_rows + (ep.to - 1 - r.abs);
            // 영상 요청 (lerobot dataset_reader.py:212-221, 250-251; video_utils.py:302-322)
            let mut req = [0i32; 16];
            for (c, v) in ep.vids.iter().enumerate() {
                let f = &files[v.file];
                let shifted = v.from_ts + r.ts as f64;
                let fr = (shifted * f.fps).round_ties_even() as i64;
                if fr < 0 || fr as usize >= f.pts.len() {
                    return Err(format!("{}: 프레임 {fr} 범위 밖", f.path));
                }
                let loaded = f.pts[fr as usize] as f64 / f.timescale;
                // lerobot: float32 텐서로 |질의 - 불러온 시각| < tolerance
                if !(((shifted as f32) - (loaded as f32)).abs() < sp.tolerance_s as f32) {
                    return Err(format!("{}: 시각 허용오차 초과 {shifted} vs {loaded}", f.path));
                }
                req[2 * c] = v.file as i32;
                req[2 * c + 1] = fr as i32;
            }
            put(&mut wr.abs, &[r.abs])?;
            put(&mut wr.task, &[r.task as i32])?;
            put(&mut wr.ep_last, &[ep_last_rel as i32])?;
            put(&mut wr.state_ex, &o.state_ex)?;
            put(&mut wr.action, &r.action)?;
            put(&mut wr.state_out, &o.state_out)?;
            put(&mut wr.tok_ids, &o.tokens)?;
            tok_total += o.tokens.len() as u64;
            put(&mut wr.tok_off, &[tok_total])?;
            put(&mut wr.req, &req[..2 * ep.vids.len()])?;
            max_tok = max_tok.max(o.tokens.len());
            n_trunc += (o.tokens.len() == sp.max_token_len) as usize;
            n_rows += 1;
        }
        eprintln!("  {} → 누적 {} 프레임 ({:.1}s)", p.display(), n_rows, t0.elapsed().as_secs_f64());
    }
    if prev_ep != i64::MIN && prev_abs != eps[&prev_ep].to - 1 {
        return Err(format!("에피소드 {prev_ep} 가 끝까지 안 왔다"));
    }
    for x in [&mut wr.abs, &mut wr.task, &mut wr.ep_last, &mut wr.state_ex, &mut wr.action, &mut wr.state_out, &mut wr.tok_off, &mut wr.tok_ids, &mut wr.req] {
        x.flush().map_err(io("flush"))?;
    }
    // 행동 정규화 통계 (C++ 로더가 읽는다)
    let mut nw = w(&sp.out, "norm_action.f64")?;
    put(&mut nw, &norm[2 * s_dim..])?;
    nw.flush().map_err(io("flush"))?;
    // C++ 가 읽는 단순 텍스트 머리 (키 값...)
    let mut m = String::new();
    m += "ftable 1\n";
    m += &format!("n {n_rows}\nstate_dim {s_dim}\naction_dim {a_dim}\nmodel_dim {}\nhorizon {}\nmax_token_len {}\ncams {}\n", sp.model_dim, sp.horizon, sp.max_token_len, sp.cams.len());
    let mut dl = String::new();
    let mut nd = 0;
    for d in &sp.delta {
        if d.action.len() != d.state.len() {
            return Err("델타 대응 길이가 다르다".into());
        }
        for (a, s) in d.action.iter().zip(&d.state) {
            dl += &format!(" {a} {s}");
            nd += 1;
        }
    }
    m += &format!("delta {nd}{dl}\n");
    m += &format!("files {}\n", files.len());
    for f in &files {
        m += &format!("file\t{}\t{}\n", f.path, f.idx);
    }
    fs::write(format!("{}/table.txt", sp.out), m).map_err(io("table.txt"))?;
    println!(
        "표 {}: 프레임 {n_rows}, 영상 파일 {}, 토큰 최대 {max_tok} (한도 {}에 걸린 프레임 {n_trunc}), {:.1}s",
        sp.out,
        files.len(),
        sp.max_token_len,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
