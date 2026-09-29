# Evaluation and Rules - BEHAVIOR

> 원본: https://behavior.stanford.edu/challenge/evaluation.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# Evaluation and Rules

## Challenge Track

For the 2026 BEHAVIOR Challenge, there is a single evaluation track:

| Challenge track | Participants are restricted to robot onboard observations for policy inputs. |
|---|---|
| Allowed policy inputs | RGB + depth + proprioception |
| Robot embodiment | Not fixed. Participants may use the default R1Pro robot or provide their own OmniGibson-supported robot through a custom robot configuration file. |
| Not allowed during evaluation | No ground-truth segmentation, object state, target object pose, full-scene point cloud, robot global pose, or other simulator-only privileged information during evaluation. |

You are allowed to use privileged information during training (e.g. other observation modalities, task info, etc.), so long as you are not using it during challenge-track evaluation. BDDL task definitions can be used and are identical during evaluation. You may also collect additional data yourself via teleoperation, RL, scripted policies, or other approaches.

There are no restrictions on the type of policy used. Methods such as IL, RL, or TAMP are all allowed. Additional components like SLAM or LLM-based querying are also permitted, provided the policy follows the challenge-track observation restrictions during evaluation. If a submission depends on external model-query APIs, participants must provide the credentials, quota, and serving configuration needed for evaluation; the organizers will not cover external API usage costs.

## Running Evaluations

> **Use the v3.9.3 tag**
>
> Run challenge evaluation from the `v3.9.3` tag of `BEHAVIOR-1K`, not the older `v3.9.0` tag. The challenge evaluator and dataset interfaces in `v3.9.3` include the latest compatibility fixes that the organizers will use for evaluation.

We provide `OmniGibson/omnigibson/eval/eval.py` as the command-line entry point for running websocket-based evaluations. Start your policy server first, then run the evaluator from the repository root:

```
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 \
  --port 8000 \
  --instance-indices 0 \
  --num-rollouts 1 \
  --output-dir outputs/b1k_eval \
  --write-video
```

The evaluator connects to the policy server at `--host` and `--port`; the policy server is responsible for receiving observations and returning robot actions. The websocket interface is implemented by the evaluation utilities adapted from [openpi](https://github.com/Physical-Intelligence/openpi), and baseline servers such as OpenPI or GR00T can expose compatible endpoints.

Key arguments:

| `--task-name` | BEHAVIOR task id, e.g. `turning_on_radio`. The 2026 task list is available in the [Demo Gallery](https://behavior.stanford.edu/challenge/tasks/index.html). |
|---|---|
| `--host` and `--port` | Address of the websocket policy server. The default port is `8000`. The evaluator waits for the server health check at `/healthz`, then opens the websocket connection. |
| `--instance-indices` | Indices into the task's test instance list. Use indices `0 1 2 3 4 5 6 7 8 9` for reported evaluation results. Indices `0-19` are public test instances; indices `20-39` are hidden instances reserved for final evaluation. Select exactly one instance per logical environment in the evaluation batch. |
| `--num-envs` | Number of logical environments processed together as one evaluation batch. This must equal the number of values passed to `--instance-indices`. |
| `--num-rollouts` | Number of rollouts to run for each selected instance. |
| `--max-steps` | Optional episode timeout in simulator steps. If omitted, the evaluator uses a task-specific default timeout equal to `1.5x` the mean human demonstration length. |
| `--env-wrapper` | Full target path of the evaluation wrapper. The default is `omnigibson.eval.wrappers.DefaultWrapper`; use `omnigibson.eval.wrappers.RGBDFullResWrapper` for official RGB + depth challenge-track evaluation. |
| `--output-dir` | Directory where rollout results are written. JSON metrics are written under `<output-dir>/json/`. |
| `--write-video` | Save rollout MP4 videos under `<output-dir>/videos/`. This is required for challenge evaluation outputs because rollout videos are part of the submission. |
| `--video-fps` | Frame rate for saved rollout videos. |
| `--headless` / `--no-headless` | Run OmniGibson headless or with rendering UI. |

The evaluator sends flattened observations to the policy server. The server should return a msgpack-encoded response containing an `action` array with the robot action for the current step. The helper server implementation is `WebsocketPolicyServer` in `OmniGibson/omnigibson/eval/utils/network_utils.py`, and the evaluator-side client is `omnigibson.eval.policies.WebsocketPolicy`.

### Optional action-chunk replay protocol

Action-chunk replay is disabled by default. Setting `--replay-action-chunk-size K` with `K > 1` asks a compatible policy server for `K` actions by adding the reserved `action_chunk_size` field to the observation request. The server response must contain:

```
{
    "action": ...,        # shape: (..., action_dim)
    "action_chunk": ...,  # shape: (..., K, action_dim)
}
```

The first action in `action_chunk` must exactly equal `action`. The client returns that first action immediately, then executes the remaining `K - 1` actions without another websocket request. The cached chunk is cleared on episode reset. If a server omits `action_chunk`, the client logs a warning and falls back to one request per simulator step. The generic `WebsocketPolicyServer` implements this fallback protocol; a chunk-capable policy server must implement the optional response field itself.

Only enable replay when the server returns an action sequence intended to be executed open-loop from the current observation, such as one receding-horizon policy chunk, and choose `K` so replay never crosses the policy's replanning boundary. Do not enable it for temporal ensembling or for a server that repeatedly re-runs or blends the policy on the same stale observation; those actions are not equivalent to normal closed-loop evaluation.

Each successful rollout produces a JSON result containing `q_score`, `time`, `agent_distance`, and normalized efficiency metrics. For challenge submissions, run evaluation with `--write-video`; this records the head and wrist camera videos that must be submitted with the rollout metrics.

Example wrappers live under `omnigibson.eval.wrappers`:

### `DefaultWrapper`

Low-resolution RGB observations at `224 x 224`, plus proprioception. This is useful for faster debugging but does not include depth.

### `RGBDFullResWrapper`

Official RGB + depth challenge observations, with a `720 x 720` head camera and `480 x 480` wrist cameras.

You are welcome to use the provided wrappers or implement a custom wrapper for your own policy. Submitted evaluation wrappers must expose only RGB, depth, and proprioception to the policy. Include the wrapper code in your submission; the organizers will manually inspect it to ensure the challenge-track observation restrictions are followed and that the environment is not manipulated directly, e.g. by teleporting the robot or changing object states.

## Custom Robot Configuration

By default, the evaluator loads the bundled R1Pro robot config at `OmniGibson/omnigibson/eval/r1pro.yaml`. The 2026 challenge does not restrict participants to a fixed robot embodiment: you may instead pass a custom OmniGibson robot configuration file with `--robot-config`, as long as the robot and controllers are supported by OmniGibson and the policy still follows the challenge-track observation restrictions.

```
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --robot-config path/to/my_robot.yaml \
  --host 127.0.0.1 \
  --port 8000
```

The config file should contain one complete robot dictionary using canonical OmniGibson fields. In particular:

| `model` | Required and should be the lowercase OmniGibson robot model id, e.g. `r1pro`. Use `model`, not the deprecated `type` key. |
|---|---|
| `name` | Required. The default name is `robot_r1`. |
| `controller_config` | Controls the action space. You may use any robot controllers supported by OmniGibson, such as joint, IK, base, or gripper controllers, as long as the resulting action array returned by your policy matches `robot.action_dim`. |
| Standard robot fields | `obs_modalities`, `proprio_obs`, `sensor_config`, `action_normalize`, `grasping_mode`, and other standard robot config fields may be customized as needed, subject to the challenge-track observation restrictions. |
| Runtime start pose | The evaluator overwrites the robot `position` and `orientation` at runtime with the task instance's prescribed start pose. |
| Submission requirement | Include the exact robot config file in your final submission. |

If your robot is not already available in OmniGibson, see the [custom robot import tutorial](https://behavior.stanford.edu/tutorials/custom_robot_import.html) for how to import a new robot model into BEHAVIOR / OmniGibson before referencing it from a custom robot config.

Minimal structure:

```
model: r1pro
name: robot_r1
eval:
  camera_sensor_names:
    left_wrist: robot_r1:left_realsense_link:Camera:0
    right_wrist: robot_r1:right_realsense_link:Camera:0
    head: robot_r1:zed_link:Camera:0
obs_modalities:
  - proprio
  - rgb
controller_config:
  base:
    name: HolonomicBaseJointController
    motor_type: velocity
  arm_left:
    name: JointController
    motor_type: position
  arm_right:
    name: JointController
    motor_type: position
```

The optional `eval.camera_sensor_names` block maps evaluation camera roles to robot sensor names. It is used by the provided wrappers and video writer to identify the head and wrist cameras. `--write-video` requires the `head`, `left_wrist`, and `right_wrist` roles. The official `RGBDFullResWrapper` uses these roles to set the head camera to `720 x 720` and wrist cameras to `480 x 480`; all other vision sensors are treated as wrist-resolution sensors.

For controller syntax and supported controller types, see the [OmniGibson controller documentation](https://behavior.stanford.edu/omnigibson/controllers.html). For the recommended R1Pro baseline config, start from the bundled `OmniGibson/omnigibson/eval/r1pro.yaml` and modify `controller_config` or other robot fields as needed.

## Metrics and Results

We will calculate the following metric during policy rollout:

### Primary Metric (Ranking)

**Task success score:** Averaged across 100 tasks.

**Calculation:** Partial successes = (Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates).

### Secondary Metrics (Efficiency)

**Simulated time:** Total simulation time (hardware-independent).

**Distance navigated:** Accumulated distance traveled by the agent’s base body. This metric evaluates the efficiency of the agent in navigating the environment.

**Displacement of end effectors/hands:** Accumulated displacement of the agent’s end effectors/hands. This metric evaluates the efficiency of the agent in its interaction with the environment.

*Secondary metrics will be normalized using human averages from 200 demonstrations per task.*

The success score (**Q**) is the metric used for ranking submissions. If two submissions achieve the same score, secondary metrics will be used to break ties.

## Prizes

The 2026 BEHAVIOR Challenge has an $11,000 prize pool:

| 1st place | $5,000 |
|---|---|
| 2nd place | $3,000 |
| 3rd place | $2,000 |
| Outstanding open-source solution | $1,000 |

## Evaluation Protocol and Logistics

**Evaluation protocol:**

| Training | The training instances and human demonstrations (200 per task) are released to the public. |
|---|---|
| Self-evaluation and report | In addition to the 200 human-collected demonstrations, we provide 20 extra configuration instances for each task. Use the **first 10 public instances**, corresponding to instance indices `0-9` (`--instance-indices 0 1 2 3 4 5 6 7 8 9`), for evaluation results. Participants should report their performance on these 10 instances through the process described on the [submission page](https://behavior.stanford.edu/challenge/submission.html). You should evaluate your policy 1 time on each instance, using the default `1.5x` mean-human-length timeouts provided by our evaluation script. We will update the leaderboard once we sanity-check the performance. The **remaining 10 public instances**, indices `10-19`, are not used for leaderboard reporting and may serve as a test set before evaluating your final policy. |
| Simulation nondeterminism | Because the simulator can be nondeterministic, different rollouts of the same policy may produce different results for a given instance. This is expected. Participants should not cherry-pick rollout results for individual instances or assemble the best outcomes across runs, instances, or tasks to improve the reported success rate. The challenge uses many tasks and multiple instances per task to reduce the effect of rollout-level nondeterminism. |
| Final evaluation | We will hold out 10 more instances for final evaluation. After we freeze the leaderboard upon submission deadline, we will evaluate the top-5 solutions on the leaderboard using these instances. |
| Instance variation | Each instance differs in terms of initial object states and initial robot poses. |

## Performance Benchmarks

### System Spec

The following benchmarks were measured on:

| GPU | NVIDIA RTX 4090 (24GB VRAM) |
|---|---|
| CPU | AMD Ryzen 9 7950X 16-Core Processor (32 threads) |
| RAM | 128GB |
| OS | Ubuntu 22.04.5 LTS |

**Scene Load Time:** Approximately 150-300 seconds (one-time cost per trial, varies by scene complexity)

### Evaluation Frame Rate with Random Actions

The following table records the approximate frames per second (FPS) performance when running evaluation with random actions across different settings:

| Sensor Modality | Resolution (Head, Wrist) | FPS |
|---|---|---|
| RGB | 224x224, 224x224 | 24.55 |
| RGB | 720x720, 480x480 | 20.62 |
| RGB + depth | 224x224, 224x224 | 16.55 |
| RGB + depth | 720x720, 480x480 | 13.52 |
