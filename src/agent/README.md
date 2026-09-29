# agent — high-level planner agent + evaluator↔π0.5 relay (Rust)

The "judgement" layer of `plan.md` §1–2. It keeps the long plan, memory and step tracking, and gives π0.5 only the
instruction for the current step. Design, decisions, measurements and how to run: **[docs/에이전트_설계.md](../../docs/에이전트_설계.md)** (Korean).

## Two paths

| Path | What | Files |
|---|---|---|
| Every step (relay) | Evaluator ↔ relay ↔ π0.5 server. Observation and action bytes pass through unchanged; the relay appends `__agent_prompt__` (and `__agent_flush__`) to the observation map. Masked WebSocket payloads are never unmasked (mask-key rotation), frames are cut-through streamed, sockets use `TCP_QUICKACK`. Odometry and boundary detection are plain arithmetic. **No LLM here.** | `relay.rs` `ws.rs` `msgpack.rs` `wire.rs` `session.rs` `odom.rs` `monitor.rs` |
| Step boundaries (agent) | Called only at episode start, budget exhaustion, periodic checks, base settling, gripper changes. OpenAI-compatible Chat Completions with tool calling (`tools` / `tool_calls` / `role:"tool"`) decides the next step. | `planner.rs` `tools.rs` `context.rs` `memory.rs` `plan.rs` `graph.rs` `llm.rs` |

The π0.5 server side is a 70-line glue file, `tools/serve_b1k_agent.py`: it swaps openpi's `B1KPolicyWrapper` for a subclass
that reads `__agent_prompt__` as the prompt and removes the injected keys (openpi itself is not modified).

## Decision interface

`Agent = Core (state) + Decider`. `Core` owns the stage log, plan checklist, object memory (`back` / `the other`),
reference resolution, step budgets and instruction rendering. Deciders: `LlmDecider` (tool-calling loop) and
`PriorDecider` (no LLM, follows the demonstration prior). An RL decider plugs into the same trait;
`Core::decision_input()` is the structured state and is written to every boundary record.

## Files

| File | Role |
|---|---|
| `main.rs` | CLI `bagent`: `relay`, `sim`, `replay`, `build-assets`, `render`, `schedule`, `mock-llm`, `fake-pi`, `bench`, `bench-local`, `bench-image`, `llm-check` |
| `relay.rs` | Relay: cut-through vs. hold, planner threads, ping handling while planning, latency stats |
| `ws.rs` | Hand-written RFC 6455: handshake (`/healthz`), frames, mask rotation, vectored writes, `poll(2)`, `TCP_QUICKACK` |
| `msgpack.rs` / `wire.rs` | Zero-copy scan of masked msgpack; observation keys, `base_qvel`, grippers, images, injected suffix |
| `session.rs` / `odom.rs` / `monitor.rs` | Per-environment odometry (same integration as `src/meridian/demo_player.py`), five boundary triggers |
| `planner.rs` | `Core`, `Decider`, `LlmDecider`, `PriorDecider`, automatic evidence, deterministic fallback |
| `tools.rs` | Tool schemas and execution: `issue_command`, `continue_current`, `finish`, `graph_query`, `resolve_reference`, `look`, `robot_state`, `goal_status`, `set_plan`, `remember` |
| `context.rs` / `memory.rs` / `plan.rs` | Single system message context, recent-turn window + summaries after the decision, checklist |
| `graph.rs` | meridian `scene_server` client (TCP JSON lines, `127.0.0.1:7791`), static/shared graphs, object memory |
| `catalog.rs` / `bddl.rs` / `vocab.rs` / `instruction.rs` | Task cards (`assets/tasks.json`: prompts, limits, BDDL, top demo step orders, step budgets), 35-skill vocabulary, 4 instruction formats, π0.5 token budget |
| `llm.rs` / `http.rs` / `codec.rs` | Chat Completions (explicit deterministic sampling), hand-written HTTP, `curl` for HTTPS, replay/fake LLMs, local-server up/down hooks; base64, SHA-1 |
| `trace.rs` / `replay.rs` | JSONL execution records, timeline, single-file HTML player, replay verification |
| `mockworld.rs` / `fakes.rs` | Fake world (fake executor), rule-based fake LLM (in-process and HTTP), fake π0.5 server, fake evaluator |
| `prompts/system.md` | System prompt (English) |
| `tests/e2e.rs` | End-to-end tests (relay byte identity, fake-world episodes, fallback, replay, HTTP LLM) |

## Quick start (WSL)

```bash
export PATH=$HOME/.cargo/bin:$PATH CARGO_TARGET_DIR=$HOME/cargo-target/agent
cd /mnt/c/behavior-2026/src/agent
cargo test --release                                  # 40 tests
cargo run --release -- sim --scenario trash --llm oracle
set -a; . ~/.config/behavior-2026/kau.env; set +a     # API key via environment only
cargo run --release -- sim --scenario radio --llm kau --no-images
cargo run --release -- relay --listen 0.0.0.0:8000 --upstream 127.0.0.1:8100 --mode agent --llm kau --graph meridian
```

Runs are written to `runs/` (git-ignored).
