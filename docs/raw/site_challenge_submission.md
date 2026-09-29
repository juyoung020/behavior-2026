# Submission Guidelines - BEHAVIOR

> 원본: https://behavior.stanford.edu/challenge/submission.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# Submission Guidelines

 [Open Submission Portal](https://behavior-1k-2026-challenge-leaderboard.hf.space/submit)

## Submission Overview

- **No formal registration is required** to participate in the challenge.
- **Full evaluation size:** 100 tasks x 10 instances x 1 rollout = 1,000 rollout outputs.
- **Partial submissions are allowed.** Missing rollout instances count as zero in the final score.
- **Multiple checkpoints** from the same team and model family are considered one entry.
- **No cherry-picking rollout results.** Simulation nondeterminism is expected, and different rollouts of the same policy may produce different results for a given instance. Submissions should report the prescribed rollout for each instance rather than selecting the best outcomes across repeated runs, instances, or tasks.
- **Robot configuration must be reproducible.** If you use a custom robot config with `--robot-config`, include the exact YAML/JSON file in your submission package.

## Evaluation Outputs

After running the evaluation script at `OmniGibson/omnigibson/eval/eval.py`, each rollout produces:

| Metrics JSON | Episode metrics, including task success score, normalized movement, and simulator time. |
|---|---|
| Rollout MP4 | Video recording of the evaluated trajectory. Run the evaluator with `--write-video`; rollout videos are required for challenge submissions. |

See [Evaluation and Rules](https://behavior.stanford.edu/challenge/evaluation.html) for the evaluation protocol, wrappers, metrics, and command-line options.

## Robot and Wrapper Configuration

Final evaluation must be reproducible from the files you submit. Include the exact evaluation wrapper and robot configuration used to generate your reported results.

- If you use the default robot setup, include the bundled `omnigibson/eval/r1pro.yaml` or clearly state that you used it unchanged.
- If you use a custom robot setup, include the complete robot config passed through `--robot-config`. The file must contain canonical OmniGibson robot fields such as `model`, `name`, `controller_config`, observation settings, sensor settings, and any `eval.camera_sensor_names` needed by the evaluation wrappers or video writer.
- The returned action array from your policy server must match the action space induced by the submitted robot config.
- Custom wrappers must still expose only RGB, depth, and proprioception to the policy for challenge-track evaluation.

> **Sample output JSON**
>
> ```
> {
>     "task": "turning_on_radio",
>     "instance_id": 0,
>     "rollout_id": 0,
>     "steps": 500,
>     "success": false,
>     "agent_distance": {
>         "base": 2.0,
>         "left": 1.5,
>         "right": 1.2
>     },
>     "normalized_agent_distance": {
>         "base": 1.5,
>         "left": 1.2,
>         "right": 1.1
>     },
>     "q_score": {
>         "final": 0.4
>     },
>     "time": {
>         "simulator_steps": 500,
>         "simulator_time": 16.6666666667,
>         "normalized_time": 1.6
>     }
> }
> ```

> **Do not edit evaluation outputs**
>
> Do not modify the output JSON files or rollout videos in any way.

## Final Model Evaluation

There are two supported ways to submit your model for final evaluation.

### Docker-based evaluation

**Recommended.** Submit a Docker image that serves your policy. We run OmniGibson outside the container and connect to your policy through the WebSocket policy client.

The submitted model should run on a single 24GB VRAM GPU. Final evaluation will use GPUs such as RTX 3090, A5000, and TitanRTX.

### IP address-based evaluation

Serve your policy yourself and provide an IP address that allows us to query it for evaluation. We only accept IP address-based submissions that expose at least 50 ports for parallel evaluation.

Common serving options include TorchServe, LitServe, vLLM, NVIDIA Triton, or an equivalent model-serving stack.

> **Submission confidentiality and open source**
>
> Submitted solutions will remain confidential unless participants explicitly grant permission for disclosure. We strongly encourage open-source submissions, as they help advance reproducible research and accelerate progress in embodied AI.

## Final Submission Package

Your final zip file should contain:

| Metrics JSON files | One JSON file for each rollout performed, up to 1,000 files. |
|---|---|
| Wrapper code | The `.py` wrapper used during evaluation. |
| Robot config | The exact robot `.yaml` or `.json` config used during evaluation, including any custom `controller_config`, sensor settings, observation settings, and `eval.camera_sensor_names`. |
| README | Instructions for evaluating your policy, including the full evaluator command, wrapper path, robot config path, Docker image details or IP address information. For IP address-based submissions, include at least 50 available ports. |

In addition, submit a link through the [submission portal](https://behavior-1k-2026-challenge-leaderboard.hf.space/submit) to all rollout MP4 videos, one for each rollout performed, up to 1,000 videos.
