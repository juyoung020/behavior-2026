# 리더보드 Space 파일 scripts/extract_self_reported_scores.py

> 원본: https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard/blob/main/scripts/extract_self_reported_scores.py
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
```python
#!/usr/bin/env python3
"""Extract aggregate self-reported scores from private submission artifacts."""

import argparse
import html
import io
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download


DRIVE_FOLDER_PATTERN = re.compile(r"drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)")
DRIVE_FILE_PATTERN = re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)")
DRIVE_ENTRY_PATTERN = re.compile(r'data-id="([^"]+)"[^>]+data-tooltip="([^"]+)"')
HF_DATASET_PATTERN = re.compile(r"huggingface\.co/datasets/([^/]+/[^/?#]+)")
TOTAL_CHALLENGE_TASKS = 100
IGNORED_SUBMISSION_IDS = {"20260924T054531Z-test-test"}


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "behavior-challenge-leaderboard/1.0"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def drive_folder_entries(folder_id: str) -> list[tuple[str, str]]:
    page = download(f"https://drive.google.com/drive/folders/{folder_id}").decode("utf-8")
    entries = []
    for file_id, name in DRIVE_ENTRY_PATTERN.findall(page):
        clean_name = re.sub(r"(?: Unknown)?(?: Shared folder)?$", "", html.unescape(name))
        entry = (file_id, clean_name)
        if entry not in entries:
            entries.append(entry)
    return entries


def drive_json_documents(folder_id: str) -> list[dict]:
    entries = drive_folder_entries(folder_id)
    json_entries = [(file_id, name) for file_id, name in entries if name.lower().endswith(".json")]
    if not json_entries:
        metric_folders = [file_id for file_id, name in entries if name.strip().casefold() == "metrics"]
        for metric_folder_id in metric_folders:
            json_entries.extend(
                (file_id, name)
                for file_id, name in drive_folder_entries(metric_folder_id)
                if name.lower().endswith(".json")
            )

    return [
        json.loads(download(f"https://drive.usercontent.google.com/download?id={file_id}&export=download"))
        for file_id, _ in json_entries
    ]


def zip_json_documents(content: bytes) -> list[dict]:
    documents = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".json"):
                continue
            document = json.loads(archive.read(name))
            if isinstance(document, dict) and "success" in document and "q_score" in document:
                documents.append(document)
    return documents


def drive_file_documents(file_id: str) -> list[dict]:
    content = download(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t")
    if content.startswith(b"PK"):
        return zip_json_documents(content)
    document = json.loads(content)
    return [document] if isinstance(document, dict) else document


def huggingface_dataset_documents(url: str) -> list[dict]:
    repo_match = HF_DATASET_PATTERN.search(url)
    if not repo_match:
        return []

    repo_id = repo_match.group(1)
    if "/resolve/" in urlparse(url).path:
        return zip_json_documents(download(url))

    api = HfApi()
    paths = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    metric_archives = [path for path in paths if path.lower().endswith(".zip") and "metric" in path.lower()]
    if metric_archives:
        archive = hf_hub_download(repo_id=repo_id, filename=metric_archives[0], repo_type="dataset")
        return zip_json_documents(Path(archive).read_bytes())

    documents = []
    for path in paths:
        if not path.lower().endswith(".json"):
            continue
        local_path = hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")
        document = json.loads(Path(local_path).read_text(encoding="utf-8"))
        if isinstance(document, dict) and "success" in document and "q_score" in document:
            documents.append(document)
    return documents


def artifact_documents(url: str) -> list[dict]:
    drive_match = DRIVE_FOLDER_PATTERN.search(url)
    if drive_match:
        return drive_json_documents(drive_match.group(1))
    drive_file_match = DRIVE_FILE_PATTERN.search(url)
    if drive_file_match:
        return drive_file_documents(drive_file_match.group(1))
    if HF_DATASET_PATTERN.search(url):
        return huggingface_dataset_documents(url)
    if urlparse(url).path.lower().endswith(".zip"):
        return zip_json_documents(download(url))
    if urlparse(url).path.lower().endswith(".json"):
        document = json.loads(download(url))
        return [document] if isinstance(document, dict) else document
    raise ValueError(f"Unsupported self-evaluation artifact URL: {url}")


def aggregate(documents: list[dict]) -> tuple[float, float, int, int]:
    episodes_by_task: dict[str, list[tuple[float, bool]]] = {}
    for document in documents:
        q_score = document.get("q_score", {}).get("final")
        success = document.get("success")
        task = document.get("task")
        if task and q_score is not None and success is not None:
            episodes_by_task.setdefault(str(task), []).append((float(q_score), bool(success)))
    if not episodes_by_task:
        raise ValueError("No rollout metrics were found")

    task_q_scores = []
    task_success_rates = []
    for episodes in episodes_by_task.values():
        task_q_scores.append(sum(q_score for q_score, _ in episodes) / len(episodes))
        task_success_rates.append(sum(success for _, success in episodes) / len(episodes))
    num_episodes = sum(len(episodes) for episodes in episodes_by_task.values())
    return (
        sum(task_q_scores) / TOTAL_CHALLENGE_TASKS,
        sum(task_success_rates) / TOTAL_CHALLENGE_TASKS,
        len(episodes_by_task),
        num_episodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="behavior-1k/challenge-submissions")
    parser.add_argument("--output", type=Path, default=Path("data/self_reported_results.jsonl"))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HF_WRITE_TOKEN to read the private submissions dataset")

    api = HfApi(token=token)
    existing_rows = {
        row["submission_id"]: row
        for row in (
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    } if args.output.exists() else {}
    rows = []
    paths = sorted(
        path
        for path in api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
        if path.startswith("2026/submissions/") and path.endswith(".json")
    )
    for path in paths:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=path,
            repo_type="dataset",
            token=token,
        )
        submission = json.loads(Path(local_path).read_text(encoding="utf-8"))
        if submission.get("submission_id") in IGNORED_SUBMISSION_IDS:
            continue
        artifact_url = submission.get("artifacts", {}).get("self_eval_results_url")
        if not artifact_url:
            raise ValueError(f"Submission {submission['submission_id']} has no self-evaluation artifact URL")
        try:
            q_score, success_rate, num_tasks, num_episodes = aggregate(artifact_documents(artifact_url))
        except Exception as exc:
            previous = existing_rows.get(submission["submission_id"])
            if previous:
                rows.append(previous)
            outcome = "kept existing score" if previous else "left score blank"
            print(f"{submission['team']}: {outcome}; could not read artifact ({exc})")
            continue
        rows.append(
            {
                "submission_id": submission["submission_id"],
                "q_score": q_score,
                "success_rate": success_rate,
                "num_tasks": num_tasks,
                "num_episodes": num_episodes,
            }
        )
        print(
            f"{submission['team']}: {num_tasks}/{TOTAL_CHALLENGE_TASKS} tasks, {num_episodes} episodes, "
            f"Q score {q_score:.3f}, full-task SR {success_rate:.1%}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
```
