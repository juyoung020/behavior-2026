//! 손으로 짠 msgpack 읽기·쓰기 — openpi / BEHAVIOR 평가기의 msgpack_numpy 형식
//! (넘파이 배열 = {b"__ndarray__": True, b"data": bin, b"dtype": "<f4", b"shape": [..]}) 을 다룬다.

#[derive(Clone, Debug, PartialEq)]
pub enum Mp {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    F32(f32),
    F64(f64),
    Str(String),
    Bin(Vec<u8>),
    Arr(Vec<Mp>),
    Map(Vec<(Mp, Mp)>),
    Ext(i8, Vec<u8>),
}

pub struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Rd { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err("msgpack: 데이터가 짧다".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn be(&mut self, n: usize) -> Result<u64, String> {
        let s = self.take(n)?;
        Ok(s.iter().fold(0u64, |a, &x| (a << 8) | x as u64))
    }
    fn items(&mut self, n: usize) -> Result<Vec<Mp>, String> {
        (0..n).map(|_| self.read()).collect()
    }
    fn pairs(&mut self, n: usize) -> Result<Vec<(Mp, Mp)>, String> {
        (0..n).map(|_| Ok((self.read()?, self.read()?))).collect()
    }
    fn string(&mut self, n: usize) -> Result<Mp, String> {
        Ok(Mp::Str(String::from_utf8(self.take(n)?.to_vec()).map_err(|e| format!("msgpack 문자열: {e}"))?))
    }
    pub fn read(&mut self) -> Result<Mp, String> {
        let c = self.u8()?;
        Ok(match c {
            0x00..=0x7f => Mp::UInt(c as u64),
            0x80..=0x8f => Mp::Map(self.pairs((c & 0x0f) as usize)?),
            0x90..=0x9f => Mp::Arr(self.items((c & 0x0f) as usize)?),
            0xa0..=0xbf => self.string((c & 0x1f) as usize)?,
            0xc0 => Mp::Nil,
            0xc2 => Mp::Bool(false),
            0xc3 => Mp::Bool(true),
            0xc4 | 0xc5 | 0xc6 => {
                let n = self.be(1 << (c - 0xc4))? as usize;
                Mp::Bin(self.take(n)?.to_vec())
            }
            0xc7 | 0xc8 | 0xc9 => {
                let n = self.be(1 << (c - 0xc7))? as usize;
                let t = self.u8()? as i8;
                Mp::Ext(t, self.take(n)?.to_vec())
            }
            0xca => Mp::F32(f32::from_bits(self.be(4)? as u32)),
            0xcb => Mp::F64(f64::from_bits(self.be(8)?)),
            0xcc..=0xcf => Mp::UInt(self.be(1 << (c - 0xcc))?),
            0xd0 => Mp::Int(self.u8()? as i8 as i64),
            0xd1 => Mp::Int(self.be(2)? as u16 as i16 as i64),
            0xd2 => Mp::Int(self.be(4)? as u32 as i32 as i64),
            0xd3 => Mp::Int(self.be(8)? as i64),
            0xd4..=0xd8 => {
                let t = self.u8()? as i8;
                Mp::Ext(t, self.take(1 << (c - 0xd4))?.to_vec())
            }
            0xd9 | 0xda | 0xdb => {
                let n = self.be(1 << (c - 0xd9))? as usize;
                self.string(n)?
            }
            0xdc | 0xdd => {
                let n = self.be(2 << (c - 0xdc))? as usize;
                Mp::Arr(self.items(n)?)
            }
            0xde | 0xdf => {
                let n = self.be(2 << (c - 0xde))? as usize;
                Mp::Map(self.pairs(n)?)
            }
            0xe0..=0xff => Mp::Int(c as i8 as i64),
            0xc1 => return Err("msgpack: 쓰지 않는 바이트 0xc1".into()),
        })
    }
}

impl Mp {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Mp::UInt(u) => Some(*u),
            Mp::Int(i) if *i >= 0 => Some(*i as u64),
            _ => None,
        }
    }
    /// 맵에서 키(문자열 또는 bin 바이트) 찾기
    pub fn get(&self, key: &str) -> Option<&Mp> {
        if let Mp::Map(kv) = self {
            for (k, v) in kv {
                match k {
                    Mp::Str(s) if s == key => return Some(v),
                    Mp::Bin(b) if b.as_slice() == key.as_bytes() => return Some(v),
                    _ => {}
                }
            }
        }
        None
    }
}

/// msgpack_numpy 배열: (dtype 문자열, shape, 원 바이트)
pub struct Nd<'a> {
    pub dtype: &'a str,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

pub fn as_nd(v: &Mp) -> Option<Nd<'_>> {
    if !matches!(v.get("__ndarray__"), Some(Mp::Bool(true))) {
        return None;
    }
    let data = match v.get("data")? {
        Mp::Bin(b) => b.as_slice(),
        _ => return None,
    };
    let dtype = match v.get("dtype")? {
        Mp::Str(s) => s.as_str(),
        _ => return None,
    };
    let shape = match v.get("shape")? {
        Mp::Arr(a) => a.iter().map(|x| x.as_u64().map(|u| u as usize)).collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    Some(Nd { dtype, shape, data })
}

// ---- 쓰기 (파이썬 msgpack.Packer(use_bin_type=True) 와 같은 바이트 형식) ----
pub fn w_map(o: &mut Vec<u8>, n: usize) {
    if n < 16 {
        o.push(0x80 | n as u8);
    } else if n < 65536 {
        o.push(0xde);
        o.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        o.push(0xdf);
        o.extend_from_slice(&(n as u32).to_be_bytes());
    }
}
pub fn w_arr(o: &mut Vec<u8>, n: usize) {
    if n < 16 {
        o.push(0x90 | n as u8);
    } else if n < 65536 {
        o.push(0xdc);
        o.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        o.push(0xdd);
        o.extend_from_slice(&(n as u32).to_be_bytes());
    }
}
pub fn w_str(o: &mut Vec<u8>, s: &str) {
    let n = s.len();
    if n < 32 {
        o.push(0xa0 | n as u8);
    } else if n < 256 {
        o.push(0xd9);
        o.push(n as u8);
    } else if n < 65536 {
        o.push(0xda);
        o.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        o.push(0xdb);
        o.extend_from_slice(&(n as u32).to_be_bytes());
    }
    o.extend_from_slice(s.as_bytes());
}
pub fn w_bin(o: &mut Vec<u8>, b: &[u8]) {
    let n = b.len();
    if n < 256 {
        o.push(0xc4);
        o.push(n as u8);
    } else if n < 65536 {
        o.push(0xc5);
        o.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        o.push(0xc6);
        o.extend_from_slice(&(n as u32).to_be_bytes());
    }
    o.extend_from_slice(b);
}
pub fn w_uint(o: &mut Vec<u8>, v: u64) {
    if v < 128 {
        o.push(v as u8);
    } else if v < 256 {
        o.push(0xcc);
        o.push(v as u8);
    } else if v < 65536 {
        o.push(0xcd);
        o.extend_from_slice(&(v as u16).to_be_bytes());
    } else if v < (1 << 32) {
        o.push(0xce);
        o.extend_from_slice(&(v as u32).to_be_bytes());
    } else {
        o.push(0xcf);
        o.extend_from_slice(&v.to_be_bytes());
    }
}
pub fn w_f64(o: &mut Vec<u8>, v: f64) {
    o.push(0xcb);
    o.extend_from_slice(&v.to_bits().to_be_bytes());
}
pub fn w_bool(o: &mut Vec<u8>, v: bool) {
    o.push(if v { 0xc3 } else { 0xc2 });
}
/// msgpack_numpy 배열: {b"__ndarray__": True, b"data": bin, b"dtype": str, b"shape": [..]} (키 순서도 파이썬 pack_data 와 같게)
pub fn w_nd(o: &mut Vec<u8>, dtype: &str, shape: &[usize], data: &[u8]) {
    w_map(o, 4);
    w_bin(o, b"__ndarray__");
    w_bool(o, true);
    w_bin(o, b"data");
    w_bin(o, data);
    w_bin(o, b"dtype");
    w_str(o, dtype);
    w_bin(o, b"shape");
    w_arr(o, shape.len());
    for &s in shape {
        w_uint(o, s as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nd_roundtrip() {
        let mut o = Vec::new();
        w_map(&mut o, 2);
        w_str(&mut o, "action");
        let data: Vec<u8> = [1.0f32, -2.0, 3.5].iter().flat_map(|x| x.to_le_bytes()).collect();
        w_nd(&mut o, "<f4", &[1, 3], &data);
        w_str(&mut o, "server_timing");
        w_map(&mut o, 1);
        w_str(&mut o, "infer_ms");
        w_f64(&mut o, 0.25);
        let v = Rd::new(&o).read().unwrap();
        let nd = as_nd(v.get("action").unwrap()).unwrap();
        assert_eq!(nd.dtype, "<f4");
        assert_eq!(nd.shape, vec![1, 3]);
        assert_eq!(nd.data, data.as_slice());
        assert_eq!(v.get("server_timing").unwrap().get("infer_ms"), Some(&Mp::F64(0.25)));
    }
}
