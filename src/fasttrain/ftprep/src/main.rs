//! ftprep — 학습 데이터 가속용 준비 도구 (GPU 를 안 쓰는 부분).
//!
//! ftprep index <in.mp4> <out.ftidx>
//!     mp4 의 첫 영상 트랙에서 표본(패킷)마다 파일 위치·크기·키프레임 여부·표시 시각을 뽑고,
//!     hvcC 의 매개변수 집합(VPS/SPS/PPS)을 Annex-B 로 붙여 저장한다. C++ NVDEC 디코더가 이것만 보고
//!     파일에서 필요한 바이트만 pread 해서 디코더에 넣는다(libavformat 불필요).
//! ftprep yuvgrid <W> <H> <out.yuv>
//!     yuv420p 원시 프레임들. 2x2 블록마다 (U,V) 한 쌍과 Y 4개를 넣어, 모든 (Y,U,V) 2^24 조합을 한 번씩 담는다.
//!     블록 번호 c (프레임을 넘어 이어짐): U = (c >> 14) & 255, V = (c >> 6) & 255, Y = (c & 63) * 4 + (dy * 2 + dx).
//!     이 영상을 무손실로 인코딩해 원래 디코더(torchcodec)로 풀면 색 변환 표가 된다 (src/fasttrain/lut.py).
//!
//! 외부 크레이트 없이 표준 라이브러리만 쓴다.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};

type R<T> = Result<T, String>;

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}
fn be64(b: &[u8]) -> u64 {
    u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// 상자(box) 하나: 종류, 본문 시작(파일 위치), 본문 길이.
struct BoxHdr {
    kind: [u8; 4],
    body: u64,
    len: u64,
}

fn read_hdr(f: &mut File, pos: u64, end: u64) -> R<Option<BoxHdr>> {
    if pos + 8 > end {
        return Ok(None);
    }
    let mut h = [0u8; 16];
    f.seek(SeekFrom::Start(pos)).map_err(|e| e.to_string())?;
    f.read_exact(&mut h[..8]).map_err(|e| e.to_string())?;
    let size32 = be32(&h[0..4]) as u64;
    let kind = [h[4], h[5], h[6], h[7]];
    let (hdr, size) = match size32 {
        1 => {
            f.read_exact(&mut h[8..16]).map_err(|e| e.to_string())?;
            (16, be64(&h[8..16]))
        }
        0 => (8, end - pos),
        s => (8, s),
    };
    if size < hdr || pos + size > end {
        return Err(format!("깨진 상자 {:?} @{}", String::from_utf8_lossy(&kind), pos));
    }
    Ok(Some(BoxHdr { kind, body: pos + hdr, len: size - hdr }))
}

fn children(f: &mut File, start: u64, len: u64) -> R<Vec<BoxHdr>> {
    let mut out = vec![];
    let end = start + len;
    let mut pos = start;
    while let Some(b) = read_hdr(f, pos, end)? {
        pos = b.body + b.len;
        out.push(b);
    }
    Ok(out)
}

fn find<'a>(v: &'a [BoxHdr], k: &[u8; 4]) -> Option<&'a BoxHdr> {
    v.iter().find(|b| &b.kind == k)
}

fn body(f: &mut File, b: &BoxHdr) -> R<Vec<u8>> {
    let mut buf = vec![0u8; b.len as usize];
    f.seek(SeekFrom::Start(b.body)).map_err(|e| e.to_string())?;
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

struct Track {
    width: u32,
    height: u32,
    timescale: u32,
    nal_len_size: u32,
    params: Vec<u8>, // Annex-B
    offsets: Vec<u64>,
    sizes: Vec<u32>,
    key: Vec<bool>,
    pts: Vec<i64>,
}

fn parse_hvcc(c: &[u8]) -> R<(u32, Vec<u8>)> {
    if c.len() < 23 {
        return Err("hvcC 가 짧다".into());
    }
    let nal_len_size = (c[21] & 3) as u32 + 1;
    let narr = c[22] as usize;
    let mut p = 23;
    let mut params = vec![];
    for _ in 0..narr {
        let n = u16::from_be_bytes([c[p + 1], c[p + 2]]) as usize;
        p += 3;
        for _ in 0..n {
            let l = u16::from_be_bytes([c[p], c[p + 1]]) as usize;
            p += 2;
            params.extend_from_slice(&[0, 0, 0, 1]);
            params.extend_from_slice(&c[p..p + l]);
            p += l;
        }
    }
    Ok((nal_len_size, params))
}

fn parse_track(f: &mut File, trak: &BoxHdr) -> R<Option<Track>> {
    let tk = children(f, trak.body, trak.len)?;
    let mdia = find(&tk, b"mdia").ok_or("mdia 없음")?;
    let md = children(f, mdia.body, mdia.len)?;
    let hdlr = body(f, find(&md, b"hdlr").ok_or("hdlr 없음")?)?;
    if &hdlr[8..12] != b"vide" {
        return Ok(None);
    }
    let mdhd = body(f, find(&md, b"mdhd").ok_or("mdhd 없음")?)?;
    let timescale = if mdhd[0] == 1 { be32(&mdhd[20..24]) } else { be32(&mdhd[12..16]) };
    let minf = find(&md, b"minf").ok_or("minf 없음")?;
    let mi = children(f, minf.body, minf.len)?;
    let stbl = find(&mi, b"stbl").ok_or("stbl 없음")?;
    let st = children(f, stbl.body, stbl.len)?;

    // stsd → hvc1/hev1 → hvcC
    let stsd = find(&st, b"stsd").ok_or("stsd 없음")?;
    let entry = read_hdr(f, stsd.body + 8, stsd.body + stsd.len)?.ok_or("stsd 비었음")?;
    if &entry.kind != b"hvc1" && &entry.kind != b"hev1" {
        return Err(format!("HEVC 가 아니다: {}", String::from_utf8_lossy(&entry.kind)));
    }
    let eb = body(f, &entry)?;
    let width = u16::from_be_bytes([eb[24], eb[25]]) as u32;
    let height = u16::from_be_bytes([eb[26], eb[27]]) as u32;
    let sub = children(f, entry.body + 78, entry.len - 78)?;
    let hvcc = body(f, find(&sub, b"hvcC").ok_or("hvcC 없음")?)?;
    let (nal_len_size, params) = parse_hvcc(&hvcc)?;

    // 표본 크기
    let stsz = body(f, find(&st, b"stsz").ok_or("stsz 없음")?)?;
    let fixed = be32(&stsz[4..8]);
    let n = be32(&stsz[8..12]) as usize;
    let sizes: Vec<u32> = if fixed != 0 { vec![fixed; n] } else { (0..n).map(|i| be32(&stsz[12 + 4 * i..])).collect() };
    // 덩어리 위치
    let chunk_off: Vec<u64> = if let Some(b) = find(&st, b"stco") {
        let d = body(f, b)?;
        let c = be32(&d[4..8]) as usize;
        (0..c).map(|i| be32(&d[8 + 4 * i..]) as u64).collect()
    } else {
        let d = body(f, find(&st, b"co64").ok_or("stco/co64 없음")?)?;
        let c = be32(&d[4..8]) as usize;
        (0..c).map(|i| be64(&d[8 + 8 * i..])).collect()
    };
    // 표본 → 덩어리
    let stsc = body(f, find(&st, b"stsc").ok_or("stsc 없음")?)?;
    let ne = be32(&stsc[4..8]) as usize;
    let ent: Vec<(usize, usize)> =
        (0..ne).map(|i| (be32(&stsc[8 + 12 * i..]) as usize, be32(&stsc[12 + 12 * i..]) as usize)).collect();
    let mut offsets = Vec::with_capacity(n);
    let mut s = 0usize;
    for ci in 0..chunk_off.len() {
        let chunk_no = ci + 1;
        let per = ent.iter().rev().find(|e| e.0 <= chunk_no).map(|e| e.1).unwrap_or(0);
        let mut off = chunk_off[ci];
        for _ in 0..per {
            if s >= n {
                break;
            }
            offsets.push(off);
            off += sizes[s] as u64;
            s += 1;
        }
    }
    if offsets.len() != n {
        return Err(format!("표본 위치 수 {} != 표본 수 {}", offsets.len(), n));
    }
    // 키프레임
    let key: Vec<bool> = if let Some(b) = find(&st, b"stss") {
        let d = body(f, b)?;
        let c = be32(&d[4..8]) as usize;
        let mut k = vec![false; n];
        for i in 0..c {
            k[be32(&d[8 + 4 * i..]) as usize - 1] = true;
        }
        k
    } else {
        vec![true; n]
    };
    // 디코딩 시각(stts) + 표시 오프셋(ctts)
    let stts = body(f, find(&st, b"stts").ok_or("stts 없음")?)?;
    let ne = be32(&stts[4..8]) as usize;
    let mut dts = Vec::with_capacity(n);
    let mut t: i64 = 0;
    for i in 0..ne {
        let cnt = be32(&stts[8 + 8 * i..]);
        let dur = be32(&stts[12 + 8 * i..]) as i64;
        for _ in 0..cnt {
            dts.push(t);
            t += dur;
        }
    }
    let mut pts = dts.clone();
    if let Some(b) = find(&st, b"ctts") {
        let d = body(f, b)?;
        let ver = d[0];
        let ne = be32(&d[4..8]) as usize;
        let mut s = 0;
        for i in 0..ne {
            let cnt = be32(&d[8 + 8 * i..]) as usize;
            let raw = be32(&d[12 + 8 * i..]);
            let off = if ver == 1 { raw as i32 as i64 } else { raw as i64 };
            for _ in 0..cnt {
                if s < n {
                    pts[s] += off;
                }
                s += 1;
            }
        }
    }
    // edit list(elst) 의 시작 이동: 디코더(ffmpeg)는 첫 표시 시각을 elst 만큼 당긴다.
    if let Some(edts) = find(&tk, b"edts") {
        let ed = children(f, edts.body, edts.len)?;
        if let Some(elst) = find(&ed, b"elst") {
            let d = body(f, elst)?;
            let ver = d[0];
            let ne = be32(&d[4..8]);
            if ne >= 1 {
                let media_time = if ver == 1 { be64(&d[16..24]) as i64 } else { be32(&d[12..16]) as i32 as i64 };
                if media_time > 0 {
                    for p in pts.iter_mut() {
                        *p -= media_time;
                    }
                }
            }
        }
    }
    Ok(Some(Track { width, height, timescale, nal_len_size, params, offsets, sizes, key, pts }))
}

fn cmd_index(inp: &str, out: &str) -> R<()> {
    let mut f = File::open(inp).map_err(|e| format!("{inp}: {e}"))?;
    let flen = f.metadata().map_err(|e| e.to_string())?.len();
    let top = children(&mut f, 0, flen)?;
    let moov = find(&top, b"moov").ok_or("moov 없음")?;
    let mv = children(&mut f, moov.body, moov.len)?;
    let mut track = None;
    for b in mv.iter().filter(|b| &b.kind == b"trak") {
        if let Some(t) = parse_track(&mut f, b)? {
            track = Some(t);
            break;
        }
    }
    let t = track.ok_or("영상 트랙 없음")?;
    let n = t.sizes.len();
    // 표시 순서 = 디코딩 순서인지 (bframes=0 이면 그렇다) — C++ 디코더가 이것을 가정한다.
    let monotonic = t.pts.windows(2).all(|w| w[1] > w[0]);
    let keys: Vec<usize> = (0..n).filter(|&i| t.key[i]).collect();
    let max_gop = keys.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(n).max(n - keys.last().copied().unwrap_or(0));
    let mut w = BufWriter::new(File::create(out).map_err(|e| format!("{out}: {e}"))?);
    let mut put = |b: &[u8]| w.write_all(b).map_err(|e| e.to_string());
    put(b"FTIDX1\0\0")?;
    for v in [t.width, t.height, t.timescale, t.nal_len_size, t.params.len() as u32, monotonic as u32] {
        put(&v.to_le_bytes())?;
    }
    put(&t.params)?;
    put(&(n as u64).to_le_bytes())?;
    for i in 0..n {
        put(&t.offsets[i].to_le_bytes())?;
        put(&t.sizes[i].to_le_bytes())?;
        put(&(t.key[i] as u32).to_le_bytes())?;
        put(&t.pts[i].to_le_bytes())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    println!(
        "{inp}: {}x{} 표본 {n} 키프레임 {} 최대 GOP {max_gop} timescale {} 첫 pts {} 표시순=디코딩순 {monotonic} 매개변수 {} B",
        t.width,
        t.height,
        keys.len(),
        t.timescale,
        t.pts.first().copied().unwrap_or(0),
        t.params.len()
    );
    Ok(())
}

fn cmd_yuvgrid(w: usize, h: usize, out: &str) -> R<()> {
    if w % 2 != 0 || h % 2 != 0 {
        return Err("W, H 는 짝수".into());
    }
    let (bw, bh) = (w / 2, h / 2);
    let per_frame = bw * bh;
    let total = 1usize << 22; // (U,V) 65536 쌍 × Y 64 묶음(4개씩)
    let frames = (total + per_frame - 1) / per_frame;
    let mut o = BufWriter::new(File::create(out).map_err(|e| e.to_string())?);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; per_frame];
    let mut v = vec![0u8; per_frame];
    for fi in 0..frames {
        for by in 0..bh {
            for bx in 0..bw {
                let mut c = fi * per_frame + by * bw + bx;
                if c >= total {
                    c = 0; // 마지막 프레임 나머지는 0번 블록 반복
                }
                u[by * bw + bx] = ((c >> 14) & 255) as u8;
                v[by * bw + bx] = ((c >> 6) & 255) as u8;
                let yb = ((c & 63) * 4) as u8;
                for dy in 0..2 {
                    for dx in 0..2 {
                        y[(2 * by + dy) * w + 2 * bx + dx] = yb + (dy * 2 + dx) as u8;
                    }
                }
            }
        }
        o.write_all(&y).map_err(|e| e.to_string())?;
        o.write_all(&u).map_err(|e| e.to_string())?;
        o.write_all(&v).map_err(|e| e.to_string())?;
    }
    o.flush().map_err(|e| e.to_string())?;
    println!("{out}: {w}x{h} yuv420p {frames} 프레임 (블록 {total} 개)");
    Ok(())
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let r = match a.get(1).map(|s| s.as_str()) {
        Some("index") if a.len() >= 4 => {
            // 여러 파일: index in1 out1 in2 out2 ...
            let mut r = Ok(());
            for p in a[2..].chunks(2) {
                if p.len() == 2 {
                    if let Err(e) = cmd_index(&p[0], &p[1]) {
                        r = Err(e);
                        break;
                    }
                }
            }
            r
        }
        Some("yuvgrid") if a.len() == 5 => match (a[2].parse(), a[3].parse()) {
            (Ok(w), Ok(h)) => cmd_yuvgrid(w, h, &a[4]),
            _ => Err("W H 는 정수".into()),
        },
        _ => Err("사용법: ftprep index <in.mp4> <out.ftidx> [...] | ftprep yuvgrid <W> <H> <out.yuv>".into()),
    };
    if let Err(e) = r {
        eprintln!("오류: {e}");
        std::process::exit(1);
    }
}
