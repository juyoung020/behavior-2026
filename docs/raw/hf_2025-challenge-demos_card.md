# HuggingFace behavior-1k/2025-challenge-demos 데이터셋 카드·파일 구성

> 원본: https://huggingface.co/datasets/behavior-1k/2025-challenge-demos
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
## 데이터셋 카드 (README.md 원문)

---
license: mit
task_categories:
- robotics
tags:
- LeRobot
- v
- '2'
- .
- '1'
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** mit

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.1",
    "robot_type": "R1Pro",
    "total_episodes": 10000,
    "total_frames": 119094660,
    "total_tasks": 50,
    "total_videos": 90000,
    "chunks_size": 10000,
    "fps": 30,
    "splits": {
        "train": "0:10000"
    },
    "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
    "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
    "metainfo_path": "meta/episodes/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
    "annotation_path": "annotations/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
    "features": {
        "observation.images.rgb.left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.rgb.right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.rgb.head": {
            "dtype": "video",
            "shape": [
                720,
                720,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 720,
                "video.width": 720,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.depth.left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "depth"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p16le",
                "video.is_depth_map": true,
                "has_audio": false
            }
        },
        "observation.images.depth.right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "depth"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p16le",
                "video.is_depth_map": true,
                "has_audio": false
            }
        },
        "observation.images.depth.head": {
            "dtype": "video",
            "shape": [
                720,
                720,
                3
            ],
            "names": [
                "height",
                "width",
                "depth"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 720,
                "video.width": 720,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p16le",
                "video.is_depth_map": true,
                "has_audio": false
            }
        },
        "observation.images.seg_instance_id.left_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.seg_instance_id.right_wrist": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 480,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.seg_instance_id.head": {
            "dtype": "video",
            "shape": [
                720,
                720,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 720,
                "video.width": 720,
                "video.channels": 3,
                "video.codec": "libx265",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                23
            ],
            "names": null
        },
        "timestamp": {
            "dtype": "float64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "observation.cam_rel_poses": {
            "dtype": "float32",
            "shape": [
                21
            ],
            "names": null
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                256
            ],
            "names": null
        },
        "observation.task_info": {
            "dtype": "float32",
            "shape": [
                null
            ],
            "names": null
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
@article{li2024behavior, 
  title={Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation}, 
  author={Li, Chengshu and Zhang, Ruohan and Wong, Josiah and Gokmen, Cem and Srivastava, Sanjana and Mart{'i}n-Mart{'i}n, Roberto and Wang, Chen and Levine, Gabrael and Ai, Wensi and Martinez, Benjamin and Yin, Hang and Lingelbach, Michael and Hwang, Minjune and Hiranaka, Ayano and Garlanka, Sujay and Aydin, Arman and Lee, Sharon and Sun, Jiankai and Anvari, Mona and Sharma, Manasi and Bansal, Dhruva and Hunter, Samuel and Kim, Kyu-Young and Lou, Alan and Matthews, Caleb R. and Villa-Renteria, Ivan and Tang, Jerry Huayang and Tang, Claire and Xia, Fei and Li, Yunzhu and Savarese, Silvio and Gweon, Hyowon and Liu, C. Karen and Wu, Jiajun and Fei-Fei, Li}, 
  journal={arXiv preprint arXiv:2403.09227}, 
  year={2024} 
}
```

## HuggingFace API 메타데이터

- lastModified: 2025-12-02T06:22:23.000Z
- gated: False · private: False
- 파일 수: 100000
- 총 크기(API usedStorage, 바이트): 2070933452683

| 파일 패턴 | 개수 | 합계 크기(GB) |
|---|---|---|
| `.gitattributes` | 1 | 0.00 |
| `README.md` | 1 | 0.00 |
| `annotations/task-N/episode_N.json` | 10000 | 0.13 |
| `data/task-N/episode_N.parquet` | 10000 | 164.53 |
| `meta/episodes.jsonl` | 1 | 0.00 |
| `meta/episodes/task-N/episode_N.json` | 10000 | 18.99 |
| `meta/episodes_stats.jsonl` | 1 | 0.37 |
| `meta/info.json` | 1 | 0.00 |
| `meta/tasks.jsonl` | 1 | 0.00 |
| `videos/task-N/observation.images.depth.head/episode_N.mpN` | 7800 | 576.37 |
| `videos/task-N/observation.images.depth.left_wrist/episode_N.mpN` | 7800 | 185.54 |
| `videos/task-N/observation.images.depth.right_wrist/episode_N.mpN` | 7800 | 183.52 |
| `videos/task-N/observation.images.rgb.head/episode_N.mpN` | 7800 | 145.61 |
| `videos/task-N/observation.images.rgb.left_wrist/episode_N.mpN` | 7800 | 68.65 |
| `videos/task-N/observation.images.rgb.right_wrist/episode_N.mpN` | 7800 | 69.50 |
| `videos/task-N/observation.images.seg_instance_id.head/episode_N.mpN` | 7800 | 91.30 |
| `videos/task-N/observation.images.seg_instance_id.left_wrist/episode_N.mpN` | 7794 | 34.53 |
| `videos/task-N/observation.images.seg_instance_id.right_wrist/episode_N.mpN` | 7600 | 34.16 |
