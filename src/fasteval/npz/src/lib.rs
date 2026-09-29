//! npz(zip 안의 .npy 들) 읽기·쓰기 — numpy np.savez_compressed 와 같은 모양(C 순서, 리틀엔디언). src/fasteval 의 Rust 도구들이 같이 쓴다.
//! 읽는 dtype: <f8 <f4 <f2 |b1 |u1 <i8 <i4 <i2 |i1 <u8 <u4 <u2 <U{n}(UTF-32LE 고정폭 문자열).
//! 쓰는 dtype: F -> <f8, B -> |b1, U8 -> |u1, S -> <U{가장 긴 글자 수}(np.array(...).astype(str) 와 같은 폭).

use std::fs::File;
use std::io::{Read, Write};

#[derive(Clone, Debug)]
pub enum Data {
    F(Vec<f64>),   // 실수·정수(정수는 f64 로 올린다 — 파이썬 쪽도 비교 전에 float64 로 바꾼다)
    B(Vec<bool>),
    S(Vec<String>),
    U8(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct Array {
    pub shape: Vec<usize>,
    /// numpy dtype.kind: 'f' 'i' 'u' 'b' 'U'
    pub kind: char,
    pub data: Data,
}

impl Array {
    pub fn len0(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }
    /// 축 0 한 칸(스텝 하나)의 원소 수
    pub fn step_size(&self) -> usize {
        self.shape.iter().skip(1).product()
    }
    /// 축 0·1 한 칸(스텝 하나·환경 하나)의 원소 수 = shape[2..] 곱
    pub fn cell_size(&self) -> usize {
        self.shape.iter().skip(2).product()
    }
}

fn parse_header(h: &str) -> Result<(String, bool, Vec<usize>), String> {
    let descr = {
        let i = h.find("'descr'").ok_or("npy 머리에 descr 없음")?;
        let rest = &h[i + 7..];
        let q1 = rest.find('\'').ok_or("descr 따옴표")?;
        let rest2 = &rest[q1 + 1..];
        let q2 = rest2.find('\'').ok_or("descr 따옴표")?;
        rest2[..q2].to_string()
    };
    let fortran = h.contains("'fortran_order': True");
    let shape = {
        let i = h.find("'shape'").ok_or("npy 머리에 shape 없음")?;
        let rest = &h[i..];
        let p1 = rest.find('(').ok_or("shape 괄호")?;
        let p2 = rest.find(')').ok_or("shape 괄호")?;
        rest[p1 + 1..p2]
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().map_err(|e| format!("shape 숫자: {e}")))
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok((descr, fortran, shape))
}

pub fn parse_npy(b: &[u8]) -> Result<Array, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("npy 마법 수가 아님".into());
    }
    let major = b[6];
    let (hlen, off) = if major == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[off..off + hlen]).map_err(|e| format!("npy 머리 UTF-8: {e}"))?;
    let (descr, fortran, shape) = parse_header(header)?;
    if fortran {
        return Err("fortran_order 배열은 안 받는다".into());
    }
    let n: usize = shape.iter().product();
    let d = &b[off + hlen..];
    let need = |size: usize| -> Result<(), String> {
        if d.len() < n * size {
            Err(format!("npy 데이터가 짧다 ({descr})"))
        } else {
            Ok(())
        }
    };
    let (kind, data) = match descr.as_str() {
        "<f8" => {
            need(8)?;
            ('f', Data::F(d.chunks_exact(8).take(n).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()))
        }
        "<f4" => {
            need(4)?;
            ('f', Data::F(d.chunks_exact(4).take(n).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect()))
        }
        "<f2" => {
            need(2)?;
            ('f', Data::F(d.chunks_exact(2).take(n).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect()))
        }
        "|b1" => {
            need(1)?;
            ('b', Data::B(d[..n].iter().map(|&x| x != 0).collect()))
        }
        "|u1" => {
            need(1)?;
            ('u', Data::U8(d[..n].to_vec()))
        }
        "|i1" => {
            need(1)?;
            ('i', Data::F(d[..n].iter().map(|&x| x as i8 as f64).collect()))
        }
        "<i8" => {
            need(8)?;
            ('i', Data::F(d.chunks_exact(8).take(n).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect()))
        }
        "<i4" => {
            need(4)?;
            ('i', Data::F(d.chunks_exact(4).take(n).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64).collect()))
        }
        "<i2" => {
            need(2)?;
            ('i', Data::F(d.chunks_exact(2).take(n).map(|c| i16::from_le_bytes([c[0], c[1]]) as f64).collect()))
        }
        "<u8" => {
            need(8)?;
            ('u', Data::F(d.chunks_exact(8).take(n).map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f64).collect()))
        }
        "<u4" => {
            need(4)?;
            ('u', Data::F(d.chunks_exact(4).take(n).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f64).collect()))
        }
        "<u2" => {
            need(2)?;
            ('u', Data::F(d.chunks_exact(2).take(n).map(|c| u16::from_le_bytes([c[0], c[1]]) as f64).collect()))
        }
        s if s.starts_with("<U") => {
            let w: usize = s[2..].parse().map_err(|e| format!("문자열 폭: {e}"))?;
            need(4 * w)?;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let cell = &d[i * 4 * w..(i + 1) * 4 * w];
                let mut st = String::new();
                for c in cell.chunks_exact(4) {
                    let cp = u32::from_le_bytes(c.try_into().unwrap());
                    if cp == 0 {
                        break; // numpy 는 뒤쪽 NUL 을 떼고 돌려준다
                    }
                    st.push(char::from_u32(cp).ok_or("잘못된 코드 포인트")?);
                }
                v.push(st);
            }
            ('U', Data::S(v))
        }
        other => return Err(format!("다루지 않는 dtype {other}")),
    };
    Ok(Array { shape, kind, data })
}

fn f16_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    if e == 0 {
        s * m * 2f64.powi(-24)
    } else if e == 31 {
        if m == 0.0 {
            s * f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        s * (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

/// npz 를 파일 안 순서 그대로 (이름, 배열) 로 — np.load(...).files 순서와 같다
pub fn load_npz(path: &str) -> Result<Vec<(String, Array)>, String> {
    let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::with_capacity(z.len());
    for i in 0..z.len() {
        let mut e = z.by_index(i).map_err(|e| format!("{path}: {e}"))?;
        let name = e.name().to_string();
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        let mut buf = Vec::with_capacity(e.size() as usize);
        e.read_to_end(&mut buf).map_err(|e| format!("{path}:{name}: {e}"))?;
        out.push((key, parse_npy(&buf).map_err(|e| format!("{path}:{name}: {e}"))?));
    }
    Ok(out)
}

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let sh = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {sh}, }}");
    // numpy 처럼 전체(10 + 머리) 길이를 64 의 배수로, 끝은 줄바꿈
    let total = 10 + h.len() + 1;
    let pad = (64 - total % 64) % 64;
    h.push_str(&" ".repeat(pad));
    h.push('\n');
    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out
}

/// 배열 하나를 .npy 바이트로
pub fn to_npy(a: &Array) -> Vec<u8> {
    match &a.data {
        Data::F(v) => {
            let mut b = npy_header("<f8", &a.shape);
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        }
        Data::B(v) => {
            let mut b = npy_header("|b1", &a.shape);
            b.extend(v.iter().map(|&x| x as u8));
            b
        }
        Data::U8(v) => {
            let mut b = npy_header("|u1", &a.shape);
            b.extend_from_slice(v);
            b
        }
        Data::S(v) => {
            let w = v.iter().map(|s| s.chars().count()).max().unwrap_or(0).max(1);
            let mut b = npy_header(&format!("<U{w}"), &a.shape);
            for s in v {
                let mut k = 0;
                for c in s.chars() {
                    b.extend_from_slice(&(c as u32).to_le_bytes());
                    k += 1;
                }
                for _ in k..w {
                    b.extend_from_slice(&0u32.to_le_bytes());
                }
            }
            b
        }
    }
}

/// np.savez_compressed 처럼 (이름, 배열) 들을 zip(deflate) 안의 <이름>.npy 로
pub fn save_npz(path: &str, entries: &[(String, Array)]) -> Result<(), String> {
    let f = File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut z = zip::ZipWriter::new(f);
    let opt = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (k, a) in entries {
        z.start_file(format!("{k}.npy"), opt).map_err(|e| format!("{path}:{k}: {e}"))?;
        z.write_all(&to_npy(a)).map_err(|e| format!("{path}:{k}: {e}"))?;
    }
    z.finish().map_err(|e| format!("{path}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let arrs = vec![
            ("f".to_string(), Array { shape: vec![2, 1, 3], kind: 'f', data: Data::F(vec![1.0, -2.5, f64::NAN, 4.0, 5.0, 6.0]) }),
            ("s".to_string(), Array { shape: vec![2, 1], kind: 'U', data: Data::S(vec!["abc".into(), "".into()]) }),
            ("b".to_string(), Array { shape: vec![3], kind: 'b', data: Data::B(vec![true, false, true]) }),
        ];
        let p = std::env::temp_dir().join("npz_roundtrip_test.npz");
        let ps = p.to_str().unwrap();
        save_npz(ps, &arrs).unwrap();
        let back = load_npz(ps).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(back.len(), 3);
        assert_eq!(back[0].1.shape, vec![2, 1, 3]);
        if let Data::F(v) = &back[0].1.data { assert!(v[2].is_nan()); assert_eq!(v[1], -2.5); } else { panic!() }
        if let Data::S(v) = &back[1].1.data { assert_eq!(v, &vec!["abc".to_string(), "".to_string()]); } else { panic!() }
        if let Data::B(v) = &back[2].1.data { assert_eq!(v, &vec![true, false, true]); } else { panic!() }
    }
}
