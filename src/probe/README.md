# probe — offline instruction-format probe for π0.5 (Rust)

Does π0.5 follow the instruction text? For demo frames inside annotated stages, the same observation (3 cameras, state)
is fed with differently formatted prompts, and the predicted 32-step action chunk is compared with the demonstration (open loop).
No simulator. Results and conclusions: **[docs/실험_지시형식_오프라인.md](../../docs/실험_지시형식_오프라인.md)** (Korean).

```bash
# one command (WSL): prepare samples if missing -> prompts -> inference (GPU) -> report
bash /mnt/c/behavior-2026/tools/run_probe.sh pt50  20 2
bash /mnt/c/behavior-2026/tools/run_probe.sh radio 20 2
```

| Command | What |
|---|---|
| `probe prompts <samples.json> <prompts.jsonl>` | 7 formats per sample: task name, task sentence, task sentence again (noise baseline), skill + objects, purpose / expected action, numeric command (odometry from `base_qvel`), wrong-stage skill (control) |
| `probe report <samples.json> <prompts.jsonl> <preds prefix> <out.md> [title]` | Mean absolute error per group (base 3, torso 4, arms 14, grippers 2), base direction cosine, yaw sign agreement, change vs. the task-sentence prediction (prompt sensitivity), per-stage table |

Sample preparation (`tools/probe_prep.py`, reads LeRobot parquet + HEVC video) and inference (`tools/probe_infer.py`, calls the JAX
policy's `infer`) stay in Python because the data readers and the policy are Python; they are thin one-off glue.
