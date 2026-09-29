# behavior-1k/2026-challenge-demos meta/info.json 원문

> 원본: https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/resolve/main/meta/info.json
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
```json
{
    "codebase_version": "v3.0",
    "fps": 30,
    "features": {
        "action": {
            "dtype": "float32",
            "shape": [
                23
            ],
            "names": [
                "action"
            ]
        },
        "next.reward": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": [
                "reward"
            ]
        },
        "next.terminated": {
            "dtype": "bool",
            "shape": [
                1
            ],
            "names": [
                "done"
            ]
        },
        "next.truncated": {
            "dtype": "bool",
            "shape": [
                1
            ],
            "names": [
                "done"
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                61
            ],
            "names": [
                null
            ]
        },
        "observation.robot2cam_pose.left_realsense_link_camera_0": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": null
        },
        "observation.depth_linear.left_realsense_link_camera_0": {
            "dtype": "video",
            "shape": [
                480,
                480,
                1
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "video.height": 480,
                "video.width": 480,
                "video.codec": "hevc",
                "video.pix_fmt": "gray12le",
                "video.fps": 30,
                "video.channels": 1,
                "has_audio": false,
                "video.g": null,
                "video.crf": null,
                "video.preset": null,
                "video.fast_decode": null,
                "video.video_backend": "pyav",
                "video.extra_options": null,
                "video.depth_min": 0.01,
                "video.depth_max": 10.0,
                "video.shift": 3.5,
                "video.use_log": true,
                "is_depth_map": true,
                "depth_unit": "mm",
                "video.output_unit": "mm"
            }
        },
        "observation.rgb.left_realsense_link_camera_0": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "is_depth_map": false,
                "video.height": 480,
                "video.width": 480,
                "video.codec": "hevc",
                "video.pix_fmt": "yuv420p",
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false,
                "video.g": 8,
                "video.crf": 30,
                "video.preset": null,
                "video.fast_decode": 0,
                "video.video_backend": "pyav",
                "video.extra_options": {
                    "x265-params": "log-level=0:bframes=0"
                }
            }
        },
        "observation.robot2cam_pose.right_realsense_link_camera_0": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": null
        },
        "observation.depth_linear.right_realsense_link_camera_0": {
            "dtype": "video",
            "shape": [
                480,
                480,
                1
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "video.height": 480,
                "video.width": 480,
                "video.codec": "hevc",
                "video.pix_fmt": "gray12le",
                "video.fps": 30,
                "video.channels": 1,
                "has_audio": false,
                "video.g": null,
                "video.crf": null,
                "video.preset": null,
                "video.fast_decode": null,
                "video.video_backend": "pyav",
                "video.extra_options": null,
                "video.depth_min": 0.01,
                "video.depth_max": 10.0,
                "video.shift": 3.5,
                "video.use_log": true,
                "is_depth_map": true,
                "depth_unit": "mm",
                "video.output_unit": "mm"
            }
        },
        "observation.rgb.right_realsense_link_camera_0": {
            "dtype": "video",
            "shape": [
                480,
                480,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "is_depth_map": false,
                "video.height": 480,
                "video.width": 480,
                "video.codec": "hevc",
                "video.pix_fmt": "yuv420p",
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false,
                "video.g": 8,
                "video.crf": 30,
                "video.preset": null,
                "video.fast_decode": 0,
                "video.video_backend": "pyav",
                "video.extra_options": {
                    "x265-params": "log-level=0:bframes=0"
                }
            }
        },
        "observation.robot2cam_pose.zed_link_camera_0": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": null
        },
        "observation.depth_linear.zed_link_camera_0": {
            "dtype": "video",
            "shape": [
                720,
                720,
                1
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "video.height": 720,
                "video.width": 720,
                "video.codec": "hevc",
                "video.pix_fmt": "gray12le",
                "video.fps": 30,
                "video.channels": 1,
                "has_audio": false,
                "video.g": null,
                "video.crf": null,
                "video.preset": null,
                "video.fast_decode": null,
                "video.video_backend": "pyav",
                "video.extra_options": null,
                "video.depth_min": 0.01,
                "video.depth_max": 10.0,
                "video.shift": 3.5,
                "video.use_log": true,
                "is_depth_map": true,
                "depth_unit": "mm",
                "video.output_unit": "mm"
            }
        },
        "observation.rgb.zed_link_camera_0": {
            "dtype": "video",
            "shape": [
                720,
                720,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "is_depth_map": false,
                "video.height": 720,
                "video.width": 720,
                "video.codec": "hevc",
                "video.pix_fmt": "yuv420p",
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false,
                "video.g": 8,
                "video.crf": 30,
                "video.preset": null,
                "video.fast_decode": 0,
                "video.video_backend": "pyav",
                "video.extra_options": {
                    "x265-params": "log-level=0:bframes=0"
                }
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
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
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    },
    "total_episodes": 20000,
    "total_frames": 210916774,
    "total_tasks": 100,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "robot_type": "R1Pro",
    "splits": {
        "train": "0:20000"
    }
}
```
