# 공지 모음과 자주 묻는 질문(FAQ)

조사일: 2026-09-29

출처
- 2026 공지: https://behavior.stanford.edu/challenge/updates.html → [raw/site_challenge_updates.md](raw/site_challenge_updates.md) (로컬 원본 `BEHAVIOR-1K\docs\challenge\updates.md` 와 같은 내용)
- 코드 릴리스 노트: https://github.com/StanfordVL/BEHAVIOR-1K/releases → [raw/gh_StanfordVL_BEHAVIOR-1K_releases.md](raw/gh_StanfordVL_BEHAVIOR-1K_releases.md)
- HF 토론: [raw/hf_disc_behavior-1k_2026-challenge-demos_2.md](raw/hf_disc_behavior-1k_2026-challenge-demos_2.md), [raw/hf_disc_behavior-1k_2026-challenge-demos_1.md](raw/hf_disc_behavior-1k_2026-challenge-demos_1.md)
- 2025 공지(규칙 해석 참고): [raw/site_challenge_archive_2025_updates.md](raw/site_challenge_archive_2025_updates.md), [raw/site_challenge_archive_2025_index.md](raw/site_challenge_archive_2025_index.md)
- 사이트 FAQ·알려진 문제: https://behavior.stanford.edu/other/faq.html → [raw/site_other_faq.md](raw/site_other_faq.md), https://behavior.stanford.edu/other/known_issues.html → [raw/site_other_known_issues.md](raw/site_other_known_issues.md)
- 대회 공식 FAQ 페이지는 없다(`challenge/faq.html` 404). 질문은 Discord `#support`, 공지는 `#announcements` (채널 이름은 [가이드.md](가이드.md) 기준).

표기: "따옴표" = 원문 그대로. 나머지는 요약.

---

## 1. 2026 공지 (updates.html 전체)

### 08/24/2026

규칙 설명
- 원문: "Please use the `v3.9.2` tag of the `BEHAVIOR-1K` repository for challenge evaluation. It includes the fixes below."
  - 그 뒤 evaluation.html·baselines.html 은 `v3.9.3` 을 말하고, GitHub 에는 `v3.9.3-post1`(2026-09-29 게시)만 있다(`v3.9.3` 태그 없음). → 최신 권장은 v3.9.3 계열로 보이나 공지는 갱신되지 않았다.

버그 수정
- 원문: "Corrected the arm, gripper, and trunk velocity observations in the 2026 challenge demonstration dataset. These fields now use the raw simulator joint velocities from the original HDF5 demonstrations, and `meta/stats.json` has been recomputed accordingly." → 팔·그리퍼·몸통 관절 속도 5개 구간 수정. "Actions and all other dataset fields are unchanged."
- 원문: "Updated partial-scene evaluation to load the exact room instances specified for each task in `B100_task_misc.csv`." → 평가 때 불러오는 방이 과제 메타데이터와 정확히 일치하게.
- 원문: "Fixed observation loading with `RGBDFullResWrapper` by refreshing simulator handles after changing camera resolutions and before rebuilding the observation space."
- 원문: "Fixed bugs affecting the challenge leaderboard and submission form."

새 기능
- 참가 등록 폼 추가: https://forms.gle/Kf4ABLmDKbuK5Yhj6 → [제출지침.md](제출지침.md) 1절.

### 07/27/2026

규칙 설명
- 원문: "Please use the `v3.9.2` tag of the `BEHAVIOR-1K` repository for evaluation and replay workflows, rather than the older `v3.9.0` tag."
  (참고: v3.9.2 태그는 실제로 08/24 에 게시됐다 — 릴리스 노트 기준. 07/27 무렵 나온 것은 v3.9.1(07/28 게시).)

버그 수정
- 원문: "Updated the released demonstration dataset so `observation.state[0:3]` now records the R1Pro base velocity in the robot-local frame." — 베이스 속도를 로봇 기준 좌표로(행동의 베이스 명령과 같은 기준).
- 원문: "Fixed the released depth videos for the 2026 demonstration dataset." — 자세한 것은 HF 토론 #2. 토론 #2 는 실제로는 PR("Update episode metadata and dataset stats", 2026-07-15 생성, 07-17 병합)로, 깊이 인코딩 메타데이터와 `depth_unit` 을 `info.json` 에 넣는 커밋들이다. 병합 댓글 원문: "Merged after validation against the BEHAVIOR-1K local copy and wensi-ai/lerobot release/b1k compatibility checks."

새 기능
- 원문: "Added `meta/tasks.jsonl` with natural-language task descriptions for all 100 challenge tasks. The first 50 tasks follow the 2025 challenge descriptions with spelling/grammar fixes where needed; the remaining 50 were derived from the 2026 annotations and task definitions."
- 원문: "Uploaded per-episode language annotations for all 20,000 demonstrations."

### 공지 이후의 코드 변경 (릴리스 노트, 요약)

- v3.9.1 (07/28): HF 다운로드 방법 갱신, torch 스레드 수 옵션, R1Pro 베이스 속도 좌표계 수정.
- v3.9.2 (08/24): 평가가 `B100_task_misc.csv` 의 방을 쓰도록, RGBDFullResWrapper 핸들 갱신, 8월 공지.
- v3.9.3-post1 (09/29): "Merge vector into main"(여러 환경 배치 평가 = `--num-envs`), 입자 정리 뒤 핸들 갱신, Isaac Sim 휠 다운로드 재시도, 평가 조명 동기화의 ToggledOn 접근 수정 등. **공지 페이지엔 아직 반영되지 않았다.**
- 로컬 레포는 v3.9.3-post1 = 마지막 커밋 `bd049de` — 조명 동기화 수정 PR #2357 (2026-09-28, `git log` 확인).

## 2. 2025 공지 중 올해도 참고할 만한 것

2026 문서에 같은 문장이 없는 항목은 올해 규칙이라고 단정할 수 없다(Discord 확인 대상).

- 평가 때 무작위로 바뀌는 것(2025 10/08): 원문 "During evaluation, only the task-relevant object poses and the robot’s initial pose will be randomized." / "The object instances and the poses of background, scene-level objects will remain the same." — 2026 은 "Each instance differs in terms of initial object states and initial robot poses." 만 적혀 있다.
- 학습 때 오프라인 정보(2025 10/30): 원문 "For the Standard track, you are allowed to use any offline-stored information during training, as long as you don't query the simulator for privileged information during evaluation."
- Docker 호스팅(2025 10/30): 원문 "For Docker submissions, you can use any hosting (public or private), as long as we are able to evaluate your policy and you provide clear instructions on how to access and run evaluation with your submission. The simpler the access is, the better."
- 평가 인스턴스 데이터 수집 금지(2025 09/19): 원문 "However, you may not collect data on evaluation instances, as these are reserved for testing the generalization capability of your submitted policy." — 2026 문서에는 없음.
- 마감 연장(2025 11/13): 24시간 연장이 있었다. 원문 "Please plan accordingly as we will not be able to accept any late submissions!"
- 예상 제출 수 조사 폼(2025 11/07) — 2026 의 등록 폼과 같은 목적으로 보인다.

## 3. 자주 묻는 질문 (공식 문장으로 답할 수 있는 것)

**Q. 등록해야 참가할 수 있나?**
A. 아니다. 원문 "No formal registration is required to participate in the challenge." 등록 폼은 구속력 없는 규모 파악용("This form is non-binding.").

**Q. 팀 인원 제한이 있나? 팀원을 나중에 추가할 수 있나?**
A. 공식 문서에 인원 제한·변경 규정이 없다. 포털은 제출마다 팀원 이름을 받는다("Team member full names", 제출자 먼저). 확정하려면 Discord.

**Q. 일부 과제만 내도 되나?**
A. 된다. 원문 "Partial submissions are allowed. Missing rollout instances count as zero in the final score."

**Q. 여러 번 제출하면?**
A. 포털 원문 "The latest valid final submission per team will be used for official ranking." — 마지막 것이 공식.

**Q. 같은 팀이 모델을 여러 개 내도 되나?**
A. 원문 "Multiple checkpoints from the same team and model family are considered one entry." — 같은 계열 체크포인트 여러 개는 한 출품. 다른 계열을 별도 출품으로 인정하는지는 명시 없음.

**Q. 과제별로 다른 체크포인트를 써도 되나?**
A. 막는 문장은 없다. 평가기가 관측에 `task_id` 를 넣어 주고, 2025 1위는 과제 ID 로 체크포인트 4개를 바꿔 썼다. (해석)

**Q. 로봇을 바꿔도 되나? 행동 공간은?**
A. 된다. 원문 "Participants may use the default R1Pro robot or provide their own OmniGibson-supported robot through a custom robot configuration file." 컨트롤러도 OmniGibson 이 지원하는 것이면 자유. 로봇 설정 파일을 그대로 제출.

**Q. 평가 때 무엇을 입력으로 쓸 수 있나?**
A. "RGB + depth + proprioception". 정답 분할·물체 상태·목표 물체 pose·장면 점군·로봇 전역 pose 금지 → [평가규칙_원문.md](평가규칙_원문.md) 1절.

**Q. SLAM, LLM, 규칙(휴리스틱) 써도 되나?**
A. 원문 "Additional components like SLAM or LLM-based querying are also permitted, provided the policy follows the challenge-track observation restrictions during evaluation." 방법 제한 없음("There are no restrictions on the type of policy used.").

**Q. SAM·CLIP 같은 학습된 인식 모델은?**
A. 이름으로 언급한 문장은 없다. 금지는 "ground-truth segmentation"(시뮬레이터 정답)이고 정책 구성은 자유라 허용으로 해석되나, 확정은 Discord. (예전 문서에 원문처럼 적혔던 『External perception models (e.g., SAM, CLIP) are permitted』 는 페이지에 없는 문장이었다 — [대회규칙_팀_meridian.md](대회규칙_팀_meridian.md) 의 정정 참고)

**Q. 학습 때 정답 정보를 써도 되나?**
A. 된다. 원문 "You are allowed to use privileged information during training (e.g. other observation modalities, task info, etc.), so long as you are not using it during challenge-track evaluation."

**Q. 데이터를 더 모아도 되나?**
A. 된다. 원문 "You may also collect additional data yourself via teleoperation, RL, scripted policies, or other approaches." 포털에 추가 데이터 양을 적는 칸이 있다(선택).

**Q. 외부 API(예: GPT) 비용은?**
A. 참가자 부담. 원문 "the organizers will not cover external API usage costs."

**Q. 제출물을 공개해야 하나?**
A. 아니다. 원문 "Submitted solutions will remain confidential unless participants explicitly grant permission for disclosure." 공개하면 오픈소스상($1,000) 대상이 될 수 있다(선정 기준은 문서에 없음).

**Q. 같은 인스턴스를 여러 번 돌려 좋은 것만 내도 되나?**
A. 안 된다. 원문 "Participants should not cherry-pick rollout results for individual instances or assemble the best outcomes across runs, instances, or tasks to improve the reported success rate."

**Q. 제한시간은?**
A. 과제별 사람 시연 평균 × 1.5. 원문 "using the default `1.5x` mean-human-length timeouts provided by our evaluation script." 과제별 값 → [과제목록.md](과제목록.md).

**Q. 최종 순위는 어떻게 정하나?**
A. 원문 "After we freeze the leaderboard upon submission deadline, we will evaluate the top-5 solutions on the leaderboard using these instances." (비공개 10개 인스턴스)

**Q. 어떤 GPU 로 재평가하나?**
A. 원문 "The submitted model should run on a single 24GB VRAM GPU. Final evaluation will use GPUs such as RTX 3090, A5000, and TitanRTX."

**Q. 윈도우에서 돌아가나?**
A. 설치 문서 요구사항 원문: "Ubuntu 22.04+ / Windows 10+", "RAM: 32GB+", "GPU: NVIDIA RTX 2070+", "VRAM: 8GB+". 이 PC 는 Windows 에서 평가기, WSL 에서 π0.5 서버를 돌린다 → [설치기록.md](설치기록.md).

## 4. 사이트 FAQ·알려진 문제 중 관련 있는 것 (요약)

- FAQ: OmniGibson 은 NVIDIA Isaac Sim/Omniverse 위에 만든 시뮬레이터이고 BEHAVIOR-1K 는 그 위의 1,000개 활동 벤치마크. 빠르지 않지만 사실적 물리·렌더링이 장점이라는 설명.
- 알려진 문제: 여러 장면을 병렬로 돌리려면 OmniGibson 인스턴스를 따로 띄워야 한다는 항목(원문 "Currently, to run multiple scenes in parallel, you will need to launch separate instances of the OmniGibson environment.") — 2026 평가기의 `--num-envs` 는 같은 장면의 여러 인스턴스를 한 프로세스에서 묶는 기능이다.
- "HydraEngine rtx failed creating scene renderer" 가 나오면 지원 GPU 를 `OMNIGIBSON_GPU_ID` 로 지정.

## 5. 연락 채널

- Discord: https://discord.gg/bccR5vGFEx (공지·질문)
- 오피스아워: 원문 "Attend office hours every Monday, 5-6pm Pacific Time" → 한국시간 **화요일 09:00~10:00** (미국 서머타임 기간; 11/1 이후엔 10:00~11:00). Zoom 링크는 index.html.
- 주최: Stanford Vision and Learning Lab. 데이터 제공 Simovation, 후원 IMDA·Stanford HAI·Schmidt Family Foundation(index.html 스폰서 로고 기준).

## 6. Discord 에 물어볼 목록 (문서로 답이 안 나오는 것)

1. 최종 재평가에 쓰는 BEHAVIOR-1K 태그(v3.9.2 / v3.9.3 / v3.9.3-post1)
2. `DefaultWrapper`(RGB 224, 깊이 없음)로 만든 결과도 인정되는지
3. 공개 평가 인스턴스(0~19)에서 데이터 수집·학습 금지 여부(2025 에는 금지)
4. SAM·CLIP 등 학습된 인식 모델, 사전 제작 지도 사용 가능 여부
5. 관측에 들어오는 `task_id`, `cam_rel_poses` 사용 가능 여부(평가기가 넣어 줌)
6. 남이 공개한 2025 체크포인트에서 출발하는 것의 허용 여부·표기 방법
7. 팀원 추가·변경 규정
8. "Self-evaluation results URL" 에 final zip(JSON+래퍼+로봇 설정+README) 하나를 걸면 되는지
9. 최종 재평가 Docker 가 실제로 어느 GPU 에 배정되는지, GPU 를 지정·제외할 수 있는지. 원문 예시 "RTX 3090, A5000, and TitanRTX" 중
   TitanRTX 는 Turing(sm_75)이라 bf16 텐서코어가 없다. 우리 엔진은 sm_75 에서 fp16 텐서코어 대체 경로로 돈다
   ([π05_네이티브엔진.md](π05_네이티브엔진.md) 14절).
