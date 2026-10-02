//! ftprep — 네이티브 학습 데이터 로더용 준비 도구 (GPU 를 안 쓰는 부분, 한 번 돌린다).
//!
//! ftprep index <in.mp4> <out.ftidx> [...]
//!     mp4 의 첫 영상 트랙에서 표본(패킷)마다 파일 위치·크기·키프레임 여부·표시 시각을 뽑고,
//!     hvcC 의 매개변수 집합(VPS/SPS/PPS)을 Annex-B 로 붙여 저장한다. C++ NVDEC 디코더가 이것만 보고
//!     파일에서 필요한 바이트만 pread 해서 디코더에 넣는다(libavformat 불필요).
//! ftprep yuvgrid <W> <H> <out.yuv>
//!     yuv420p 원시 프레임들. 2x2 블록마다 (U,V) 한 쌍과 Y 4개를 넣어, 모든 (Y,U,V) 2^24 조합을 한 번씩 담는다.
//!     블록 번호 c (프레임을 넘어 이어짐): U = (c >> 14) & 255, V = (c >> 6) & 255, Y = (c & 63) * 4 + (dy * 2 + dx).
//!     이 영상을 무손실로 인코딩해 원래 디코더(torchcodec)로 풀면 색 변환 표가 된다 (src/vla/fasttrain/lut.py).
//! ftprep table <spec.json>
//!     프레임별 표 (상태 추출값·원 행동·최종 상태·토큰·영상 요청). 원래 openpi 변환 중 프레임마다 정해지는 부분을
//!     여기서 한 번 계산해 두고, C++ 로더가 샘플마다 행동 창(32개)의 델타·정규화만 한다. table.rs 참고.

mod mp4;
mod table;

use std::env;

fn main() {
    let a: Vec<String> = env::args().collect();
    let r = match a.get(1).map(|s| s.as_str()) {
        Some("index") if a.len() >= 4 => {
            // 여러 파일: index in1 out1 in2 out2 ...
            let mut r = Ok(());
            for p in a[2..].chunks(2) {
                if p.len() == 2 {
                    if let Err(e) = mp4::cmd_index(&p[0], &p[1]) {
                        r = Err(e);
                        break;
                    }
                }
            }
            r
        }
        Some("yuvgrid") if a.len() == 5 => match (a[2].parse(), a[3].parse()) {
            (Ok(w), Ok(h)) => mp4::cmd_yuvgrid(w, h, &a[4]),
            _ => Err("W H 는 정수".into()),
        },
        Some("table") if a.len() == 3 => table::run(&a[2]),
        _ => Err("사용법: ftprep index <in.mp4> <out.ftidx> [...] | ftprep yuvgrid <W> <H> <out.yuv> | ftprep table <spec.json>".into()),
    };
    if let Err(e) = r {
        eprintln!("오류: {e}");
        std::process::exit(1);
    }
}
