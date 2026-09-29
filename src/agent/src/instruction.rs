//! 지시 계약: 에이전트가 정한 단계(구조체) → π0.5 가 받는 문장 한두 줄.
//!
//! π0.5 문장 입력은 토큰 200개이고 그 안에 로봇 상태가 들어간다. 실측(PaliGemma 토크나이저, 상태 25차원):
//! 빈 지시일 때 "Task: , State: …;\nAction: " 가 101토큰 → 지시에 남는 것 99토큰. 여유를 두고 기본 90토큰
//! (openpi `models/tokenizer.py` tokenize, `models/pi0_config.py` max_token_len=200, `configs/robots/b1k.py` proprio 25차원).
//!
//! 형식 4가지(plan.md 4.1 비교용, 실행 중 바꿔 끼울 수 있음):
//! ① task    과제 문장 그대로(기본 체크포인트가 학습한 형태)
//! ② subtask 시연 주석 어휘 문장("pick up radio from coffee table")
//! ③ purpose "Purpose: … . Expected action: … ."(사용자 제안)
//! ④ metric  ② + 로봇 기준 이동량("go forward 2.1 m, 0.4 m to the left, turn left 30 degrees", 앞 +x, 왼쪽 +y, 반시계 +yaw)
//!           — 이동량은 LLM 이 아니라 그래프 좌표와 오도메트리로 계산한다.

use crate::util::tokens_with_margin;
#[cfg(test)]
use crate::util::estimate_tokens;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_TOKENS: usize = 90;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Task,
    Subtask,
    Purpose,
    Metric,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        match s.trim().to_lowercase().as_str() {
            "task" | "1" => Some(Format::Task),
            "subtask" | "2" => Some(Format::Subtask),
            "purpose" | "3" => Some(Format::Purpose),
            "metric" | "4" => Some(Format::Metric),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Task => "task",
            Format::Subtask => "subtask",
            Format::Purpose => "purpose",
            Format::Metric => "metric",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Metric {
    pub forward_m: f64,
    pub left_m: f64,
    pub turn_deg: f64,
}

impl Metric {
    pub fn text(&self) -> String {
        let mut parts = Vec::new();
        if self.forward_m.abs() >= 0.05 {
            parts.push(format!("go {} {:.1} m", if self.forward_m > 0.0 { "forward" } else { "backward" }, self.forward_m.abs()));
        }
        if self.left_m.abs() >= 0.05 {
            parts.push(format!("{:.1} m to the {}", self.left_m.abs(), if self.left_m > 0.0 { "left" } else { "right" }));
        }
        if self.turn_deg.abs() >= 5.0 {
            parts.push(format!("turn {} {:.0} degrees", if self.turn_deg > 0.0 { "left" } else { "right" }, self.turn_deg.abs()));
        }
        parts.join(", ")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Instruction {
    /// 어휘 35종 중 하나
    pub skill: String,
    /// 물체 참조(그래프 id 또는 이름), 주석 칸 순서
    pub objects: Vec<String>,
    /// 표시 이름(참조를 풀어 둔 것)
    pub names: Vec<String>,
    #[serde(default)]
    pub spatial: Vec<String>,
    #[serde(default)]
    pub memory: Option<String>,
    #[serde(default)]
    pub purpose: String,
    #[serde(default)]
    pub expected: String,
    #[serde(default)]
    pub metric: Option<Metric>,
    pub budget_steps: u64,
}

fn clean(s: &str) -> String {
    let s: String = s.chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c } else { ' ' }).collect();
    s.split_whitespace().collect::<Vec<_>>().join(" ").trim_end_matches(['.', ' ']).to_string()
}

impl Instruction {
    pub fn subtask_text(&self) -> String {
        match crate::vocab::lookup(&self.skill) {
            Some(k) => k.render(&self.names, &self.spatial, self.memory.as_deref()),
            None => clean(&format!("{} {}", self.skill, self.names.join(" "))),
        }
    }

    /// 형식에 맞춰 문장을 만들고 토큰 예산 안으로 줄인다.
    pub fn render(&self, fmt: Format, task_prompt: &str, max_tokens: usize) -> String {
        let sub = self.subtask_text();
        let candidates: Vec<String> = match fmt {
            Format::Task => vec![clean(task_prompt) + "."],
            Format::Subtask => vec![sub.clone()],
            Format::Purpose => {
                let p = clean(&self.purpose);
                let e = clean(if self.expected.trim().is_empty() { &sub } else { &self.expected });
                let mut v = Vec::new();
                if !p.is_empty() {
                    v.push(format!("Purpose: {p}. Expected action: {e}."));
                }
                v.push(format!("Purpose: {}. Expected action: {sub}.", if p.is_empty() { clean(task_prompt) } else { p }));
                v.push(format!("Expected action: {sub}."));
                v
            }
            Format::Metric => {
                let m = self.metric.map(|m| m.text()).unwrap_or_default();
                if m.is_empty() {
                    vec![sub.clone()]
                } else {
                    vec![format!("{sub}: {m}"), sub.clone()]
                }
            }
        };
        for c in &candidates {
            if tokens_with_margin(c) <= max_tokens {
                return c.clone();
            }
        }
        // 마지막: 단어 단위로 자른다
        let mut out = String::new();
        for w in candidates.last().unwrap().split_whitespace() {
            let t = if out.is_empty() { w.to_string() } else { format!("{out} {w}") };
            if tokens_with_margin(&t) > max_tokens {
                break;
            }
            out = t;
        }
        out
    }

    pub fn same_step(&self, other: &Instruction) -> bool {
        self.skill == other.skill && self.objects == other.objects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radio_pick() -> Instruction {
        Instruction {
            skill: "pick up from".into(),
            objects: vec!["radio_89".into(), "coffee_table_koagbh_0".into()],
            names: vec!["radio".into(), "coffee table".into()],
            purpose: "hold the radio so its button can be pressed".into(),
            expected: "grasp the radio on the coffee table with one hand and lift it".into(),
            metric: Some(Metric { forward_m: 0.6, left_m: -0.2, turn_deg: 12.0 }),
            budget_steps: 800,
            ..Default::default()
        }
    }

    #[test]
    fn four_formats() {
        let i = radio_pick();
        let task = "Turn on the radio receiver that's on the table in the living room.";
        assert_eq!(i.render(Format::Task, task, 90), task);
        assert_eq!(i.render(Format::Subtask, task, 90), "pick up radio from coffee table");
        assert_eq!(
            i.render(Format::Purpose, task, 90),
            "Purpose: hold the radio so its button can be pressed. Expected action: grasp the radio on the coffee table with one hand and lift it."
        );
        assert_eq!(i.render(Format::Metric, task, 90), "pick up radio from coffee table: go forward 0.6 m, 0.2 m to the right, turn left 12 degrees");
    }

    #[test]
    fn budget_shrinks() {
        let mut i = radio_pick();
        i.purpose = "word ".repeat(200);
        let s = i.render(Format::Purpose, "t", 30);
        assert!(estimate_tokens(&s) <= 30, "{s}");
        assert!(s.contains("Expected action"));
    }

    /// 실제 PaliGemma 토크나이저 토큰 수(openpi venv 에서 sentencepiece 로 잰 값)와 비교:
    /// 여유를 더한 추정은 실제 이상, 그리고 너무 크지 않아야 한다.
    #[test]
    fn estimate_is_conservative() {
        for (real, s) in crate::instruction::CALIBRATION {
            let e = tokens_with_margin(s);
            assert!(e >= *real, "여유 추정 {e} < 실제 {real}: {s}");
            assert!(e <= real + real / 3 + 4, "여유 추정 {e} 이 너무 큼(실제 {real}): {s}");
            assert!(estimate_tokens(s) + 3 >= *real);
        }
    }
}

/// (실제 토큰 수, 문장). 잰 방법: docs/에이전트_설계.md 의 토큰 예산 절(556문장 중 추정이 모자랐던 것 포함).
pub const CALIBRATION: &[(usize, &str)] = &[
    (16, "Turn on the radio receiver that's on the table in the living room."),
    (6, "pick up radio from coffee table"),
    (23, "Purpose: turn on the radio so it plays. Expected action: walk to the coffee table and face the radio."),
    (27, "move to radio: go forward 2.1 m, 0.4 m to the left, turn left 30 degrees"),
    (53, "Take the four mousetraps from the cabinet in the bathroom and place them on the bathroom floor. Make sure all four end up on the same floor surface, and ensure that at least two of them are either under or directly next to the same bathroom sink."),
    (5, "move to mousetrap"),
    (21, "Purpose: progress the task 'setting mousetraps'. Expected action: move to mousetrap."),
    (29, "move to mousetrap: go forward 1.5 m, 0.3 m to the right, turn left 20 degrees"),
    (8, "pick up mousetrap from bottom cabinet"),
    (32, "pick up mousetrap from bottom cabinet: go forward 1.5 m, 0.3 m to the right, turn left 20 degrees"),
];
