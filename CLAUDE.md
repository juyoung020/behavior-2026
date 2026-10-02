# 작업 규칙

## 작업 단위마다 pull → 작업 → 커밋·푸시
- 작업을 시작하기 전에: `git checkout main && git pull --rebase --autostash` (서브모듈이 분리 HEAD 면 main 으로 먼저).
- 작업 하나가 끝나면 바로: 바뀐 경로만 지정해 커밋(`git add <경로>`, `git add -A` 금지) → `git push`.
- 이 저장소는 `juyoung020/robot-agent` 의 서브모듈(`src/behavior-2026`)이다. 여기서 푸시한 뒤 상위 저장소에서 서브모듈 포인터를 커밋·푸시한다.
- 커밋에 Claude 기여자 표기를 넣지 않는다(`.claude/settings.json`, `tools/git-hooks/commit-msg` — 클론마다 `git config core.hooksPath tools/git-hooks`).
- 데이터·에셋·키·가중치·영상은 커밋하지 않는다(README "Setup notes").

## 빌드
- CUDA 는 13.2 하나(`/usr/local/cuda-13.2`). 다른 버전을 깔지 않는다.
