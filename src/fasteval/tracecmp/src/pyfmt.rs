//! 파이썬과 같은 글자로 숫자를 찍는다(출력이 tools/trace_compare.py 와 한 글자도 안 달라야 해서).

/// 파이썬 format(x, ".{p}e"): 2.306e-03, 0.000e+00, inf, nan
pub fn fmt_e(x: f64, p: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{:.*e}", p, x);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    format!("{}e{}{:02}", mant, if e < 0 { '-' } else { '+' }, e.abs())
}

/// 파이썬 format(x, ".{p}f")
pub fn fmt_f(x: f64, p: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    format!("{:.*}", p, x)
}

/// 파이썬 repr(float) / str(float): 가장 짧은 되돌림 자릿수, 지수 -4 <= e < 16 이면 고정 소수점
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let s = format!("{:e}", x); // 가장 짧은 되돌림 자릿수: "-6.470457566507548e-3"
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&e) {
        if e >= 0 {
            let ip = (e + 1) as usize;
            let (int_part, frac) = if digits.len() > ip {
                (digits[..ip].to_string(), digits[ip..].to_string())
            } else {
                (format!("{}{}", digits, "0".repeat(ip - digits.len())), "0".to_string())
            };
            format!("{sign}{int_part}.{frac}")
        } else {
            format!("{sign}0.{}{}", "0".repeat((-e - 1) as usize), digits)
        }
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{sign}{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    }
}

// 파이썬 json.load 는 NaN·Infinity·-Infinity 낱말을 받는다(serde_json 은 안 받음).
// 문자열 밖의 이 낱말을 표식 문자열로 바꿔 읽고, 값을 쓸 때 다시 숫자로 본다.
pub const NAN_TAG: &str = "\u{1}NaN";
pub const INF_TAG: &str = "\u{1}Infinity";
pub const NINF_TAG: &str = "\u{1}-Infinity";

pub fn json_lenient(txt: &str) -> String {
    let mut out = String::with_capacity(txt.len() + 16);
    let mut in_str = false;
    let mut esc = false;
    let mut it = txt.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
            out.push(c);
            continue;
        }
        let rest = &txt[i..];
        let hit = [("-Infinity", "\"\\u0001-Infinity\""), ("Infinity", "\"\\u0001Infinity\""), ("NaN", "\"\\u0001NaN\"")]
            .iter()
            .find(|(w, _)| rest.starts_with(*w))
            .copied();
        if let Some((w, rep)) = hit {
            out.push_str(rep);
            for _ in 1..w.chars().count() {
                it.next();
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// json 값의 숫자(파이썬 float/int 로 읽혔을 값). NaN·Infinity 표식도 숫자로.
pub fn num(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) if s == NAN_TAG => Some(f64::NAN),
        serde_json::Value::String(s) if s == INF_TAG => Some(f64::INFINITY),
        serde_json::Value::String(s) if s == NINF_TAG => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// 파이썬 str(json.load 로 읽은 값)
pub fn py_str(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                py_float(n.as_f64().unwrap())
            }
        }
        Value::String(s) if s == NAN_TAG || s == INF_TAG || s == NINF_TAG => py_float(num(v).unwrap()),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// 파이썬 == (json 값끼리): 숫자는 값으로(1 == 1.0, NaN != NaN), 나머지는 같은 종류끼리
pub fn py_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    if let (Value::Number(x), Value::Number(y)) = (a, b) {
        if let (Some(i), Some(j)) = (x.as_i64(), y.as_i64()) {
            return i == j;
        }
    }
    match (num(a), num(b)) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repr() {
        assert_eq!(py_float(0.006470457566507548), "0.006470457566507548");
        assert_eq!(py_float(880.2314119918054), "880.2314119918054");
        assert_eq!(py_float(1e-5), "1e-05");
        assert_eq!(py_float(0.0001), "0.0001");
        assert_eq!(py_float(16.7), "16.7");
        assert_eq!(py_float(1e16), "1e+16");
        assert_eq!(py_float(123456789012345678.0), "1.2345678901234568e+17");
        assert_eq!(py_float(2.0), "2.0");
        assert_eq!(fmt_e(0.0023064, 3), "2.306e-03");
        assert_eq!(fmt_e(0.0, 3), "0.000e+00");
        assert_eq!(fmt_e(9.0, 3), "9.000e+00");
        assert_eq!(fmt_e(1.617e10, 2), "1.62e+10");
    }
    #[test]
    fn lenient() {
        let v: serde_json::Value =
            serde_json::from_str(&json_lenient(r#"{"a": Infinity, "b": -Infinity, "c": NaN, "d": "NaN Infinity", "e": 1.5}"#)).unwrap();
        assert_eq!(py_str(&v["a"]), "inf");
        assert_eq!(py_str(&v["b"]), "-inf");
        assert_eq!(py_str(&v["c"]), "nan");
        assert_eq!(py_str(&v["d"]), "NaN Infinity");
        assert_eq!(py_str(&v["e"]), "1.5");
        assert!(!py_eq(&v["c"], &v["c"]));
        assert!(py_eq(&v["a"], &v["a"]));
    }
}
