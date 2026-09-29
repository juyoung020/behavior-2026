# arXiv 2512.06951 — Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge

> 원본: https://arxiv.org/abs/2512.06951
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
- 제목: Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge
- 저자: Ilia Larchenko, Gleb Zarin, Akash Karnatak
- 게시: 2025-12-07 · 갱신: 2025-12-21
- 분류: 2025 1위 기술 보고서

## Abstract

We present a vision-action policy that won 1st place in the 2025 BEHAVIOR Challenge - a large-scale benchmark featuring 50 diverse long-horizon household tasks in photo-realistic simulation, requiring bimanual manipulation, navigation, and context-aware decision making. Building on the Pi0.5 architecture, we introduce several innovations. Our primary contribution is correlated noise for flow matching, which improves training efficiency and enables correlation-aware inpainting for smooth action sequences. We also apply learnable mixed-layer attention and System 2 stage tracking for ambiguity resolution. Training employs multi-sample flow matching to reduce variance, while inference uses action compression and challenge-specific correction rules. Our approach achieves 26% q-score across all 50 tasks on both public and private leaderboards.
