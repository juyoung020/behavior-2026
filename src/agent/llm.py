"""로컬 LLM 서버 호출 (OpenAI 호환 /v1/chat/completions, 예: llama.cpp CUDA 서버).

Qwen3.5-9B 양자화판을 GPU 에서 서빙한다(CPU 는 안 씀). 외부 패키지 없이 표준 라이브러리만.
"""
import base64
import json
import urllib.request


class LLM:
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", model: str = "qwen3.5-9b", timeout: float = 240):
        # timeout 은 평가기 웹소켓 ping_timeout(300 s)보다 짧게 — docs/π05_인지연결_설계.md 2.7
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]


def image_part(jpeg_bytes: bytes) -> dict:
    """카메라 영상(JPEG)을 메시지에 넣는 형식."""
    b64 = base64.b64encode(jpeg_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
