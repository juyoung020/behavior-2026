# 비교: A = C:\behavior-2026\outputs\eval_turning_on_radio_20260929_195500_nf_a
#       B = C:\behavior-2026\outputs\exp_ported_dummy_radio_20260929_233051

물리·판정·지표·결과 JSON 은 비트 동일만 통과, 영상은 RTX 잡음이라 표에만(trace_compare --pixels-report-only).

| 짝 | 판정 | 요약 |
|---|---|---|
| . ~ replay\turning_on_radio\i0 | **실패** | 요약: 비트 동일 6, 허용오차 안 0, 다름 9, 비교 불가 0, JSON 다름 4 / 픽셀(판정에서 뺌): 비트 동일 0, 허용오차 안 0, 다름 9 |

짝 1 개 중 실패 1 개. 항목별 표: compare_*.txt
