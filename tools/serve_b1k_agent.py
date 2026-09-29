"""π0.5 서버 훅: 중계기(`bagent relay`, src/agent)가 관측 맵에 덧붙인 지시 문장을 π0.5 입력 prompt 로 쓴다.

평가기 관측에는 prompt 가 없고, π0.5 문장은 서버 시작 때 고정된다(openpi src/openpi/shared/eval_b1k_wrapper.py
B1KPolicyWrapper.text_prompt → process_input 의 "prompt"). 그래서 단계마다 문장을 바꾸려면 서버가 받아 줘야 한다.
openpi 코드는 고치지 않고 scripts/b1k/serve_b1k.py 를 그대로 부르되 B1KPolicyWrapper 만 아래 것으로 바꿔 끼운다.

- 관측의 "__agent_prompt__" (문자열, 또는 배치 평가면 환경별 문자열 목록) → 그 환경의 prompt
- "__agent_flush__" (불, 또는 목록) → 그 환경의 남은 행동 묶음을 버리고 이번 스텝에 바로 새로 추론(receding_horizon 만)
- 두 키는 꺼낸 뒤 지운다 → 나머지 관측은 원래와 같다. 키가 없으면 원래 동작(시작 때 정한 과제 문장).

실행(WSL, openpi):  cd ~/openpi && uv run /mnt/c/behavior-2026/tools/serve_b1k_agent.py <serve_b1k.py 인자 그대로>
실행 스크립트: tools/run_pi05_server_agent.sh (포트 8100, 평가기는 중계기 8000 으로)
"""
import os
import sys

import numpy as np

from openpi.shared import eval_b1k_wrapper as _w

PROMPT_KEY = "__agent_prompt__"
FLUSH_KEY = "__agent_flush__"


class AgentPromptWrapper(_w.B1KPolicyWrapper):
    """B1KPolicyWrapper + 관측에 실려 온 단계 지시."""

    _agent_prompts = None
    agent_log = None  # 시험용: [(prompt 목록, flush 목록)]

    def act(self, input_obs):
        prompts = input_obs.pop(PROMPT_KEY, None)
        flush = input_obs.pop(FLUSH_KEY, None)
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        self._agent_prompts = prompts
        if flush is not None and self.control_mode == "receding_horizon" and self.sequence_indices is not None:
            for i, f in enumerate(np.atleast_1d(np.asarray(flush, dtype=bool))):
                if f and i < self.sequence_indices.shape[0]:
                    # 다음 needs_inference 계산에서 이 환경은 묶음이 끝난 것으로 보인다 → 이번 스텝에 새로 추론
                    self.sequence_indices[i, 0] = self.sequence_lengths[i, 0]
        if self.agent_log is not None:
            self.agent_log.append((prompts, flush))
        return super().act(input_obs)

    def process_input(self, obs):
        out = super().process_input(obs)
        p = self._agent_prompts
        if p is not None:
            for i, d in enumerate(out):
                d["prompt"] = p[i] if isinstance(p, (list, tuple)) else p
        return out


def main():
    openpi = os.path.expanduser(os.environ.get("OPENPI_DIR", "~/openpi"))
    sys.path.insert(0, os.path.join(openpi, "scripts", "b1k"))
    import logging

    import serve_b1k
    import tyro

    serve_b1k.B1KPolicyWrapper = AgentPromptWrapper
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("에이전트 지시 훅 사용: %s / %s", PROMPT_KEY, FLUSH_KEY)
    serve_b1k.main(tyro.cli(serve_b1k.Args))


if __name__ == "__main__":
    main()
