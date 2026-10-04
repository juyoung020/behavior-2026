"""GPU 없이 훅(tools/serve_b1k_agent.py)과 중계기를 파이썬 쪽 실제 코드로 시험한다.

  (openpi venv) python check_agent_hook.py --port 8111 [--n 300]

- π0.5 대신 추론할 때마다 받은 prompt 를 적는 가짜 정책을 AgentPromptWrapper(=진짜 B1KPolicyWrapper 상속)에 끼우고,
  openpi 의 B1K 웹소켓 서버(src/openpi/serving/websocket_b1k_server.py)로 띄운다 → 평가기와 같은 파이썬 websockets·msgpack.
- 관측 n 개를 받으면 요약을 찍고 끝낸다: 추론 횟수, 추론에 들어간 prompt 들, flush 로 앞당겨진 추론 수, 주입 키가 지워졌는지.
"""
import argparse
import collections
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_b1k_agent import FLUSH_KEY, PROMPT_KEY, AgentPromptWrapper  # noqa: E402


class StubPolicy:
    """openpi Policy 흉내: infer(dict) → {"actions": (32, 23)}"""

    def __init__(self):
        self.calls = []

    def infer(self, x):
        assert PROMPT_KEY not in x and FLUSH_KEY not in x, "주입 키가 모델 입력까지 새어 들어감"
        self.calls.append(x["prompt"])
        return {"actions": np.zeros((32, 23), dtype=np.float32)}

    def reset(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    from openpi.serving import websocket_b1k_server as srv

    stub = StubPolicy()
    w = AgentPromptWrapper(policy=stub, robot="b1k/R1Pro", text_prompt="(task prompt at start)",
                           control_mode="receding_horizon", action_horizon=16, max_len=32)
    w.agent_log = []
    orig_act = w.act
    seen = {"obs": 0}
    done = threading.Event()

    def act(obs):
        seen["obs"] += 1
        r = orig_act(obs)
        if seen["obs"] >= a.n:
            done.set()
        return r

    w.act = act
    server = srv.WebsocketPolicyServer(policy=w, host=a.host, port=a.port, metadata={"stub": True})
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[hook-check] {a.host}:{a.port} 대기 (관측 {a.n}개)", flush=True)
    done.wait()
    flushes = sum(1 for _, f in w.agent_log if f is not None and np.any(f))
    print(f"[hook-check] 관측 {seen['obs']}개, 추론 {len(stub.calls)}회, flush 달린 스텝 {flushes}")
    print(f"[hook-check] 추론 prompt 빈도: {dict(collections.Counter(stub.calls))}")
    print(f"[hook-check] 처음 추론 prompt: {stub.calls[:3]}")


if __name__ == "__main__":
    main()
