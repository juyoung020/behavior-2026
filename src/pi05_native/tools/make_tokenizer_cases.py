"""Offline: write tokenizer test cases (text -> PaliGemma token ids) from the Python sentencepiece reference.

Output lines: <utf8 text as hex>\t<ids separated by spaces>   (encode(text, add_bos=True))
Cases: every BEHAVIOR task instruction in the pi05 prompt format with random discretized states
(openpi src/openpi/models/tokenizer.py:22-33), plus random ASCII/Unicode noise to exercise
byte fallback, user-defined symbols and whitespace runs.
"""
import json
import pathlib
import random
import sys

import numpy as np
import sentencepiece

tok_path = pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"
tasks_jsonl = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
n_noise = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

sp = sentencepiece.SentencePieceProcessor(model_proto=tok_path.read_bytes())
rng = random.Random(0)
nrng = np.random.default_rng(0)
tasks = [json.loads(l)["task"] for l in tasks_jsonl.read_text().splitlines() if l.strip()]
tasks += ["turn on radio", "pick up the radio_receiver", "  move to\nthe table  ", "Task_with_underscores"]

cases = []
for t in tasks:
    for _ in range(10):
        state = nrng.uniform(-1.3, 1.3, size=23)
        disc = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        cleaned = t.strip().replace("_", " ").replace("\n", " ")
        cases.append(f"Task: {cleaned}, State: {' '.join(map(str, disc))};\nAction: ")
alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()[]{}<>-_=+*/\\|@#$%^&~`\t\n") \
    + ["  ", "   ", "\n\n", "é", "ü", "ß", "日", "本", "語", "한", "국", "🙂", "▁", "<mask>", "<unused3>", "<start_of_turn>", "\x00", "\x7f"]
for _ in range(n_noise):
    cases.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))))

with out.open("w") as f:
    for c in cases:
        ids = sp.encode(c, add_bos=True)
        f.write(c.encode().hex() + "\t" + " ".join(map(str, ids)) + "\n")
print(f"{len(cases)} cases -> {out}")
