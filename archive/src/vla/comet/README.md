# comet — our changes to the 2025 2nd-place code (openpi-comet)

We keep only our commits as patch files, applied on top of upstream `mli0603/openpi-comet` commit `4bb2aa7`.
There is no separate fork repository.

| Patch | What |
|---|---|
| `0001` | Minimum changes to run on the 2026 evaluator (v3.9.x): module path, 61-dim proprio table, batch axis; planner LLM address from environment |
| `0002` | KAU API `extra_body`, "answer as a JSON list" planner prompt line, batched trace writes (also added a black-frame filler) |
| `0003` | Removes the black-frame filler again (root-cause fixes only); detection logs remain |

Setup: `tools/setup/setup_comet_wsl.sh` clones upstream, checks out `4bb2aa7` on `main` and runs `git am` on these patches.
Run notes (Korean): `docs/archive/실행기록_스펙_있는그대로.md`.
