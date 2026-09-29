# Dataset - BEHAVIOR

> 원본: https://behavior.stanford.edu/challenge/dataset.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# Dataset

## Data Format

For the 2026 challenge, we provide the following datasets hosted on HuggingFace:

**Raw HDF5 replay data.** [2026-challenge-rawdata](https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata) contains the original raw HDF5 data for the 20k teleoperation demos and is 1.44 TB in total. These files contain everything needed to replay exact trajectories in OmniGibson. Use them with `OmniGibson/scripts/learning/replay_obs.py` to replay trajectories and collect additional visual observations.

**LeRobot demo dataset.** [2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) contains 20,000 human-collected teleoperation demos across 100 tasks and is 3.27 TB in total. It follows the [LeRobot](https://github.com/huggingface/lerobot) V3 format with customizations for better data handling.

The demo dataset has the following structure:

| Folder | Description |
|---|---|
| `annotations` | Language annotations for each episode. |
| `data` | Low-dimensional data, including proprioceptions, actions, camera poses, and related episode data. |
| `meta` | Metadata folder containing episode-level information. |
| `videos` | Visual observations, including RGB and depth. |

The dataset includes 2 visual modalities: RGB (`rgb`) and Depth (`depth_linear`):

### RGB

RGB image of the scene from the camera perspective.

Shape

`(height, width, 4)`

Type

`numpy.uint8`

Resolution

720 x 720 head camera; 480 x 480 wrist cameras

Range

[0, 255]

 ![RGB observation example](https://behavior.stanford.edu/assets/challenge_2025/dataset_rgb.png)

### Depth Linear

Distance between the camera and scene geometry, with measurement linearly proportional to actual distance.

Shape

`(height, width)`

Type

`numpy.float32`

Encoding

Depth videos are log-quantized during replay and dequantized back to metric depth values.

Range

[0, 10] meters

 ![Depth observation example](https://behavior.stanford.edu/assets/challenge_2025/dataset_depth.png)

## Dataset Statistics

| Metric | Value |
|---|---|
| Total Trajectories | 20,000 |
| Total Tasks | 100 |
| Total Skills | 270,600 |
| Unique Skills | 31 |
| Avg. Skills per Trajectory | 27.06 |
| Avg. Trajectory Duration | 351.54 seconds / 5.9 minutes |

> **Show unique skills breakdown**
>
> - attach
> - chop
> - close door
> - close drawer
> - close lid
> - hand over
> - hang
> - hold
> - ignite
> - insert
> - move to
> - open door
> - open drawer
> - open lid
> - pick up from
> - place in
> - place in next to
> - place on
> - place on next to
> - place under
> - pour
> - press
> - push to
> - release
> - spray
> - sweep surface
> - tip over
> - turn off switch
> - turn on switch
> - turn to
> - wipe hard

Overall Demo Duration

![Overall Demo Duration](https://behavior.stanford.edu/assets/challenge_2026/overall_demo_duration.png)

Per Task Demo Duration

![Per Task Demo Duration](https://behavior.stanford.edu/assets/challenge_2026/per_task_demo_duration.png)
