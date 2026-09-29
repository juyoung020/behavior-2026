"""기록해 둔 행동열을 그대로 돌려주는 정책 서버 (평가기 일치 검증용). 받은 관측은 전부 해시로 남긴다.

    python C:\\behavior-2026\\tools\\replay_policy_server.py --actions <결과폴더>\\actions.npz --port 8010 \\
           --log <새 결과폴더>\\server_log.npz --once [--perturb 100:7:0.01]

- 평가기는 공식 명령 그대로(--port 8010) 돌린다. 행동이 관측과 무관하게 고정되므로 정책 쪽 잡음(서버 추론의
  난수·GPU 비결정성)이 빠지고 시뮬레이터만 남는다.
- 관측 기록은 서버 쪽에서만 한다 -> 평가기 프로세스에 아무것도 붙이지 않은 실행에서도 '정책이 받은 것'을 비교할 수 있다.
- 프로토콜: omnigibson/eval/utils/network_utils.py 의 WebsocketPolicyServer 와 같다
  (msgpack + 넘파이 확장, 접속하면 metadata 먼저, {"reset": True} 는 응답 없음, /healthz).
- --perturb STEP:DIM:DELTA  STEP 번째 요청의 행동 DIM 에 DELTA 를 더한다(음성 대조: 비교 도구가 이 차이를 잡는지).
- 기록된 환경 수와 요청의 환경 수가 다르면 0 번 환경의 행동을 모든 환경에 준다. 기록이 끝나면 0 행동.
- --quickack : 서버 소켓에 TCP_NODELAY 를 켜고, 받을 때마다 TCP_QUICKACK 을 다시 켠다(리눅스). 큰 요청(0.6 MB)의 마지막 조각에
  리눅스 지연 ACK(약 40 ms)가 걸려 보내는 쪽이 멈추는 것을 없애는 시험. 바이트는 그대로다(응답 내용 불변).
"""
import argparse
import asyncio
import functools
import hashlib
import http
import logging
import socket
import time

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as _server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("replay")


def pack_data(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)


TCP_QUICKACK = getattr(socket, "TCP_QUICKACK", None)  # 리눅스만


class QuickAckConnection(_server.ServerConnection):
    """받을 때마다 TCP_QUICKACK 을 다시 켜는 연결(리눅스는 이 표시를 한 번 쓰고 되돌린다, man 7 tcp)."""

    def connection_made(self, transport):
        super().connection_made(transport)
        self._qa_sock = transport.get_extra_info("socket")
        if self._qa_sock is not None:
            self._qa_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if TCP_QUICKACK is not None:
                self._qa_sock.setsockopt(socket.IPPROTO_TCP, TCP_QUICKACK, 1)

    def data_received(self, data):
        super().data_received(data)
        if self._qa_sock is not None and TCP_QUICKACK is not None:
            self._qa_sock.setsockopt(socket.IPPROTO_TCP, TCP_QUICKACK, 1)


def _health_check(connection, request):
    if hasattr(request, "path") and request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class Replay:
    def __init__(self, actions, log_path, perturb, once):
        self.actions = actions  # (T, N, A)
        self.log_path = log_path
        self.perturb = perturb
        self.once = once
        self.t = 0
        self.rows = []
        self.episode = 0
        self.done = asyncio.Event()

    def _record(self, obs):
        row = {}
        for k, v in obs.items():
            a = np.asarray(v)
            if a.ndim == 0:
                continue
            n = a.shape[0]
            if a.dtype == np.uint8 or a[0].size > 4096:
                row[f"hash::{k}"] = [hashlib.blake2b(np.ascontiguousarray(a[i]).tobytes(), digest_size=16).hexdigest()
                                     for i in range(n)]
            else:
                row[f"val::{k}"] = a.reshape(n, -1).astype(np.float64)
        self.rows.append(row)

    def save(self):
        if not self.log_path or not self.rows:
            return
        out = {}
        keys = sorted({k for r in self.rows for k in r})
        for k in keys:
            vals = [r.get(k) for r in self.rows]
            if k.startswith("hash::"):
                n = max(len(v) for v in vals if v is not None)
                out[k] = np.array([v if v is not None else [""] * n for v in vals]).astype(str)
            else:
                shape = next(v.shape for v in vals if v is not None)
                out[k] = np.stack([v if v is not None else np.full(shape, np.nan) for v in vals])
        np.savez_compressed(self.log_path, **out)
        log.info(f"관측 기록 저장: {self.log_path} ({len(self.rows)} 스텝)")

    def act(self, obs):
        self._record(obs)
        prop = next(v for k, v in obs.items() if k.endswith("::proprio"))
        n = prop.shape[0] if prop.ndim > 1 else 1
        if self.t < len(self.actions):
            a = self.actions[self.t]
            a = np.broadcast_to(a[:1], (n, a.shape[-1])).copy() if a.shape[0] != n else a.copy()
        else:
            a = np.zeros((n, self.actions.shape[-1]), dtype=np.float32)
        if self.perturb and self.t == self.perturb[0]:
            a[:, self.perturb[1]] += self.perturb[2]
            log.info(f"음성 대조: 스텝 {self.t} 행동[{self.perturb[1]}] += {self.perturb[2]}")
        self.t += 1
        return a.astype(np.float32) if prop.ndim > 1 else a[0].astype(np.float32)

    async def handler(self, ws):
        log.info(f"접속 {ws.remote_address}")
        packer = Packer()
        await ws.send(packer.pack({}))
        try:
            while True:
                msg = unpackb(await ws.recv(), strict_map_key=False)
                if "reset" in msg:
                    if self.rows:
                        self.save()
                    self.t, self.rows = 0, []
                    self.episode += 1
                    continue
                t0 = time.monotonic()
                a = self.act(msg)
                await ws.send(packer.pack({"action": a, "server_timing": {"infer_ms": (time.monotonic() - t0) * 1e3}}))
        except websockets.ConnectionClosed:
            log.info("접속 종료")
        finally:
            self.save()
            if self.once:
                self.done.set()


async def main_async(a):
    actions = np.load(a.actions)["actions"].astype(np.float32)
    if actions.ndim == 2:
        actions = actions[:, None]
    perturb = None
    if a.perturb:
        s, d, v = a.perturb.split(":")
        perturb = (int(s), int(d), float(v))
    r = Replay(actions, a.log, perturb, a.once)
    log.info(f"재생 서버: 행동 {actions.shape} 포트 {a.port} quickack={a.quickack}")
    extra = {"create_connection": QuickAckConnection} if a.quickack else {}
    async with _server.serve(r.handler, "127.0.0.1", a.port, compression=None, max_size=None,
                             process_request=_health_check, **extra):
        if a.once:
            await r.done.wait()
        else:
            await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--log", default="")
    ap.add_argument("--once", action="store_true", help="첫 접속이 끝나면 종료")
    ap.add_argument("--perturb", default="", help="STEP:DIM:DELTA")
    ap.add_argument("--quickack", action="store_true", help="TCP_NODELAY + 받을 때마다 TCP_QUICKACK(리눅스)")
    asyncio.run(main_async(ap.parse_args()))
