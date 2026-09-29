# 리더보드 Space 파일 app.py

> 원본: https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard/blob/main/app.py
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
```python
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from html import escape
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUBMISSION_DATASET_REPO = os.environ.get("SUBMISSION_DATASET_REPO", "behavior-1k/challenge-submissions")
SUBMISSION_TOKEN = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
SUBMISSION_TYPE_OPTIONS = [
    "Docker image submission",
    "Policy server URL submission",
]
DATA_USAGE_OPTIONS = [
    "Not applicable",
    "<500h",
    "500-1000h",
    ">1000h",
]
SUBMISSION_DEADLINE_UTC = datetime(2026, 10, 17, 11, 59, tzinfo=timezone.utc)
RESULT_TABLE_MAX_HEIGHT = 10_000
PRIVATE_SUBMISSION_REFRESH_SECONDS = 60
IGNORED_SUBMISSION_IDS = {"20260924T054531Z-test-test"}

_private_submission_cache: dict[str, dict] = {}
_private_submission_cache_last_refresh = 0.0
_private_submission_refresh_in_progress = False
_private_submission_cache_lock = threading.Lock()

APP_CSS = """
:root {
  --b1k-border: #d8dee8;
  --b1k-muted: #3f4858;
  --b1k-ink: #172033;
  --b1k-surface: #ffffff;
  --b1k-soft: #f5f7fa;
  --b1k-accent: #0f766e;
}

.gradio-container {
  max-width: min(1720px, calc(100vw - 72px)) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}

#b1k-app {
  margin-left: auto;
  margin-right: auto;
}

.gradio-container .nav-holder {
  display: none !important;
}

#b1k-app .form,
#b1k-app .block,
#b1k-app .wrap,
#b1k-app .contain,
#b1k-app .tabitem,
#b1k-app .tabs {
  max-width: none !important;
  width: 100% !important;
}

#b1k-app label {
  color: #303847 !important;
  font-weight: 650 !important;
}

#b1k-app input::placeholder,
#b1k-app textarea::placeholder {
  color: #626c7c !important;
  opacity: 1 !important;
}

.b1k-hero {
  padding: 22px 24px;
  border: 1px solid #99d5cf;
  border-radius: 8px;
  background: linear-gradient(135deg, #eefaf7 0%, #f6fcfb 100%);
  margin-bottom: 14px;
}

.b1k-hero-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.b1k-title {
  color: var(--b1k-ink);
  font-size: 30px;
  line-height: 1.15;
  font-weight: 800;
  margin: 0 0 8px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.metrics-panel {
  border: 1px solid #d8cdfd;
  border-radius: 8px;
  background: linear-gradient(135deg, #fbfaff 0%, #f7f8ff 100%);
  padding: 14px;
  margin: 12px 0 16px;
}

.metric-card {
  border: 1px solid var(--b1k-border);
  border-radius: 8px;
  background: var(--b1k-surface);
  padding: 13px 14px;
}

.metric-card.total {
  border-top: 4px solid #2563eb;
}

.metric-card.verified {
  border-top: 4px solid #0f766e;
}

.metric-card.q-score {
  border-top: 4px solid #7c3aed;
}

.metric-card.full-success {
  border-top: 4px solid #c2410c;
}

.metric-label {
  color: var(--b1k-muted);
  font-size: 12px;
  font-weight: 650;
  text-transform: uppercase;
  letter-spacing: 0;
}

.metric-value {
  color: var(--b1k-ink);
  font-size: 24px;
  font-weight: 800;
  line-height: 1.15;
  margin-top: 4px;
}

.metric-card.total .metric-value {
  color: #2563eb;
}

.metric-card.verified .metric-value {
  color: #0f766e;
}

.metric-card.q-score .metric-value {
  color: #7c3aed;
}

.metric-card.full-success .metric-value {
  color: #c2410c;
}

.filter-panel {
  border: 1px solid var(--b1k-border);
  border-radius: 8px;
  background: var(--b1k-surface);
  padding: 12px 14px 8px;
  margin: 8px 0 14px;
  align-items: end;
}

.archive-note {
  border-left: 4px solid #7c3aed;
  background: #f5f3ff;
  color: var(--b1k-muted);
  padding: 11px 14px;
  margin: 8px 0 12px;
  font-size: 14px;
  line-height: 1.45;
}

.table-frame {
  border: 1px solid var(--b1k-border);
  border-radius: 8px;
  overflow: hidden;
  background: var(--b1k-surface);
  box-shadow: 0 8px 24px rgba(23, 32, 51, 0.06);
  min-height: 300px;
}

.compact-table,
.compact-table > div,
.compact-table .wrap,
.compact-table .table-wrap {
  width: 100% !important;
  max-width: none !important;
}

.compact-table table {
  width: 100% !important;
  border-collapse: collapse !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.compact-table th {
  background: var(--b1k-soft) !important;
  color: #2d3748 !important;
  font-size: 12px !important;
  font-weight: 750 !important;
  text-transform: uppercase !important;
  letter-spacing: 0 !important;
  border-bottom: 1px solid var(--b1k-border) !important;
}

.compact-table th:nth-child(even) {
  background: #eaf2f1 !important;
}

.compact-table td {
  background: #ffffff !important;
  color: var(--b1k-ink) !important;
  font-size: 13px !important;
  line-height: 1.35 !important;
  border-color: #edf1f6 !important;
}

.compact-table td:nth-child(even) {
  background: #f3f8f7 !important;
}

.leaderboard-2026 td:nth-child(9) {
  white-space: nowrap !important;
}

.compact-table td a {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 3px 8px;
  border: 1px solid #99d5cf;
  border-radius: 6px;
  background: #f0fdfa;
  color: #0f766e !important;
  font-size: 12px;
  font-weight: 750;
  line-height: 1;
  text-decoration: none !important;
  white-space: nowrap;
}

.compact-table td a + a {
  margin-left: 6px;
}

.compact-table td a:hover {
  border-color: #0f766e;
  background: #ccfbf1;
  color: #115e59 !important;
}

.compact-table textarea,
.compact-table input {
  font-size: 13px !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.refresh-button {
  flex: 0 0 122px !important;
  max-width: 122px !important;
}

.refresh-button button {
  min-width: 112px !important;
  height: 42px !important;
  border: 1px solid #0f766e !important;
  border-radius: 8px !important;
  background: #0f766e !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  box-shadow: 0 6px 14px rgba(15, 118, 110, 0.18) !important;
}

.refresh-button button:hover {
  background: #115e59 !important;
  border-color: #115e59 !important;
}

.submission-panel {
  border: 1px solid var(--b1k-border);
  border-radius: 8px;
  background: var(--b1k-surface);
  padding: 16px;
  margin-bottom: 14px;
}

.submission-panel:not(.submission-hero) {
  border-color: #d5dee8;
  background: #ffffff;
  box-shadow: none;
}

#b1k-app .submission-panel .form,
#b1k-app .submission-panel .wrap,
#b1k-app .submission-panel .contain,
#b1k-app .submission-panel .block {
  background: transparent !important;
}

#b1k-app .submission-panel .html-container,
#b1k-app .submission-panel .prose,
#b1k-app .submission-panel [data-testid="HTML"],
#b1k-app .submission-panel [data-testid="html"] {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.submission-hero {
  box-sizing: border-box;
  width: calc(100% + 24px);
  margin-left: -12px;
  margin-right: -12px;
  border-color: #99d5cf;
  background: linear-gradient(135deg, #eefaf7 0%, #f6fcfb 100%);
  padding: 14px 24px;
}

.submission-hero-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 28px;
}

.submission-hero-copy {
  flex: 1 1 auto;
  min-width: 0;
  max-width: 1120px;
}

.submission-deadline {
  flex: 0 0 350px;
  border: 1px solid #f2c9a2;
  border-radius: 8px;
  background: #fff7ed;
  padding: 12px 14px;
}

.submission-deadline-label {
  color: #9a3412;
  font-size: 12px;
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: 0;
  margin-bottom: 4px;
}

.submission-deadline-value {
  color: #172033;
  font-size: 18px;
  font-weight: 800;
  line-height: 1.25;
}

.submission-countdown {
  color: #9a3412;
  font-size: 13px;
  font-weight: 700;
  line-height: 1.35;
  margin-top: 6px;
}

.submission-deadline-widget {
  flex: 0 0 395px;
  width: 395px;
  height: 120px !important;
  min-height: 120px !important;
  aspect-ratio: auto !important;
  border: 0;
  display: block;
}

.leaderboard-deadline-row {
  margin-top: 14px;
}

.compact-deadline-widget {
  width: 100%;
  height: 36px !important;
  min-height: 36px !important;
  max-width: none;
  flex: 1 1 auto;
}

.submission-panel:not(.submission-hero)::before {
  display: block;
  border-left: 4px solid #0f766e;
  border-radius: 6px;
  background: #ecfdf5;
  padding: 10px 12px;
  color: var(--b1k-ink);
  font-size: 16px;
  font-weight: 800;
  line-height: 1.35;
  margin: 0 0 14px;
}

.team-panel {
  border-color: #2563eb !important;
}

.team-panel::before {
  content: "Team and Method";
  border-left-color: #2563eb !important;
  background: #eff6ff;
}

.policy-panel {
  border-color: #0f766e !important;
}

.policy-panel::before {
  content: "Policy Evaluation";
  border-left-color: #0f766e !important;
  background: #ecfdf5;
  margin-bottom: 14px;
}

.artifacts-panel {
  border-color: #7c3aed !important;
}

.artifacts-panel::before {
  content: "Artifacts and Releases";
  border-left-color: #7c3aed !important;
  background: #f5f3ff;
  margin-bottom: 14px;
}

.data-panel {
  border-color: #ea580c !important;
}

.data-panel::before {
  content: "Additional Data Usage";
  border-left-color: #ea580c !important;
  background: #fff7ed;
  margin-bottom: 14px;
}

.contacts-panel {
  border-color: #e11d48 !important;
}

.contacts-panel::before {
  content: "Contacts and Notes";
  border-left-color: #e11d48 !important;
  background: #fff1f2;
  margin-bottom: 14px;
}

.required-mark {
  color: #dc2626;
  font-weight: 800;
}

.submission-note {
  color: var(--b1k-muted);
  font-size: 14px;
  line-height: 1.45;
}

#submission-submit-button,
#submission-submit-button button,
#b1k-app #submission-submit-button,
#b1k-app #submission-submit-button button,
#b1k-app button.submit-button,
#b1k-app .submit-button button {
  min-width: 150px !important;
  height: 44px !important;
  border: 1px solid #5bb5a9 !important;
  border-radius: 8px !important;
  background: #5bb5a9 !important;
  color: #ffffff !important;
  font-weight: 750 !important;
  box-shadow: 0 6px 14px rgba(91, 181, 169, 0.18) !important;
  cursor: pointer !important;
  transition:
    background 160ms ease,
    border-color 160ms ease,
    box-shadow 160ms ease,
    transform 160ms ease !important;
}

#submission-submit-button:hover,
#submission-submit-button:hover button,
#submission-submit-button button:hover,
#b1k-app #submission-submit-button:hover,
#b1k-app #submission-submit-button:hover button,
#b1k-app #submission-submit-button button:hover,
#b1k-app button.submit-button:hover,
#b1k-app .submit-button:hover button,
#b1k-app .submit-button button:hover {
  background: #fb923c !important;
  border-color: #f97316 !important;
  box-shadow: 0 10px 22px rgba(249, 115, 22, 0.22) !important;
  transform: translateY(-1px) !important;
}

#submission-submit-button:active,
#submission-submit-button button:active,
#b1k-app #submission-submit-button:active,
#b1k-app #submission-submit-button button:active,
#b1k-app button.submit-button:active,
#b1k-app .submit-button button:active {
  background: #f97316 !important;
  border-color: #ea580c !important;
  box-shadow: 0 5px 12px rgba(249, 115, 22, 0.2) !important;
  transform: translateY(0) !important;
}

.submission-status {
  margin-top: 12px;
  padding: 12px 14px;
  border: 1px solid;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.45;
}

.submission-status strong {
  display: block;
  margin-bottom: 2px;
  color: inherit;
}

.submission-processing-note {
  margin-top: 0.45rem;
  color: #52606d;
  font-size: 0.9rem;
  text-align: center;
}

.submission-status-pending {
  border-color: #7dd3fc;
  background: #f0f9ff;
  color: #075985;
}

.submission-status-success {
  border-color: #86efac;
  background: #f0fdf4;
  color: #166534;
}

.submission-status-error {
  border-color: #fca5a5;
  background: #fef2f2;
  color: #991b1b;
}

#submission-status:has(.prose:empty) {
  display: none !important;
}

.submission-field-error {
  border-radius: 6px;
  background: #fff7f7 !important;
  box-shadow: 0 0 0 2px #dc2626 !important;
}

.submission-field-error label {
  color: #b91c1c !important;
}

.submission-field-error input,
.submission-field-error textarea,
.submission-field-error select {
  border-color: #dc2626 !important;
}

.portal-link-row {
  display: flex;
  justify-content: flex-end;
  margin: -4px 0 12px;
}

.hero-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 0 0 auto;
}

.portal-link,
.registration-link {
  display: inline-flex;
  align-items: center;
  min-height: 40px;
  padding: 0 14px;
  border: 1px solid #0f766e;
  border-radius: 8px;
  color: #ffffff !important;
  background: #0f766e;
  font-weight: 750;
  text-decoration: none !important;
  box-shadow: 0 6px 14px rgba(15, 118, 110, 0.18);
}

.portal-link:hover {
  background: #115e59;
  border-color: #115e59;
}

.registration-link {
  border-color: #0f766e;
  background: #ecfdf5;
  color: #0f766e !important;
  box-shadow: 0 4px 10px rgba(15, 118, 110, 0.1);
}

.registration-link:hover {
  background: #d1fae5;
  border-color: #115e59;
  color: #115e59 !important;
}

@media (max-width: 900px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .b1k-hero-header {
    flex-direction: column;
  }

  .hero-actions {
    flex-wrap: wrap;
  }

  .submission-hero-header {
    flex-direction: column;
  }

  .submission-deadline-widget:not(.compact-deadline-widget) {
    width: 100%;
    flex: 0 0 120px;
  }

  .submission-deadline {
    min-width: 0;
    width: 100%;
  }

  .b1k-title {
    font-size: 24px;
  }
}
"""

APP_JS = """
() => {
function b1kPatchSpaceLinks() {
  const directRoot = "https://behavior-1k-2026-challenge-leaderboard.hf.space/";
  const isEmbedded = window.self !== window.top;

  const patchLink = (link, directPath, fragment = "") => {
    const suffix = fragment ? `#${fragment}` : "";
    link.href = isEmbedded ? `${directRoot}${directPath}${suffix}` : `/${directPath}${suffix}`;
    link.target = isEmbedded ? "_blank" : "_self";
    link.rel = isEmbedded ? "noopener noreferrer" : "";
  };

  const patchAll = () => {
    document.querySelectorAll("a").forEach((link) => {
      const label = link.textContent.trim();
      if (label === "Home") {
        patchLink(link, "");
      } else if (label === "Submit Portal") {
        patchLink(link, "submit?v=validation-fields-2", "submission-portal-top");
      }
    });

    document.querySelectorAll(".compact-table td a").forEach((link) => {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    });
  };

  const ensureExternalReleaseLinks = () => {
    if (window.b1kExternalReleaseLinksPatched) {
      return;
    }

    document.addEventListener(
      "click",
      (event) => {
        const link = event.target.closest?.(".compact-table td a");
        if (!link) {
          return;
        }

        event.preventDefault();
        event.stopImmediatePropagation();
        window.open(link.href, "_blank", "noopener,noreferrer");
      },
      true,
    );
    window.b1kExternalReleaseLinksPatched = true;
  };

  const patchRequiredLabels = () => {
    document.querySelectorAll("label").forEach((label) => {
      if (label.dataset.b1kRequiredPatched === "true") {
        return;
      }

      const walker = document.createTreeWalker(label, NodeFilter.SHOW_TEXT);
      let textNode = walker.nextNode();
      while (textNode && !textNode.nodeValue.includes(" *")) {
        textNode = walker.nextNode();
      }
      if (!textNode) {
        return;
      }

      const markerIndex = textNode.nodeValue.indexOf(" *");
      const suffix = textNode.nodeValue.slice(markerIndex + 2);
      textNode.nodeValue = `${textNode.nodeValue.slice(0, markerIndex)} `;
      const marker = document.createElement("span");
      marker.className = "required-mark";
      marker.textContent = "*";
      textNode.parentNode.insertBefore(marker, textNode.nextSibling);
      if (suffix) {
        marker.parentNode.insertBefore(document.createTextNode(suffix), marker.nextSibling);
      }
      label.dataset.b1kRequiredPatched = "true";
    });
  };

  const enforceMethodDescriptionLimit = () => {
    const limit = 25;
    const field = document.getElementById("method-description-field");
    if (!field) {
      return;
    }

    const input = field.querySelector("textarea, input");
    if (!input) {
      return;
    }

    input.maxLength = limit;
    input.setAttribute("maxlength", String(limit));
    if (input.dataset.b1kMethodLimitPatched === "true") {
      return;
    }

    const trimValue = () => {
      if (input.value.length <= limit) {
        return;
      }

      input.value = input.value.slice(0, limit);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    };

    const selectedRange = () => {
      const start = input.selectionStart ?? input.value.length;
      const end = input.selectionEnd ?? input.value.length;
      return [start, end];
    };

    input.addEventListener("beforeinput", (event) => {
      if (!event.data) {
        return;
      }

      const [start, end] = selectedRange();
      const nextLength = input.value.length - (end - start) + event.data.length;
      if (nextLength > limit) {
        event.preventDefault();
      }
    });
    input.addEventListener("paste", (event) => {
      event.preventDefault();
      const pastedText = event.clipboardData?.getData("text") ?? "";
      const [start, end] = selectedRange();
      const remaining = limit - (input.value.length - (end - start));
      if (remaining <= 0) {
        return;
      }

      input.value = `${input.value.slice(0, start)}${pastedText.slice(0, remaining)}${input.value.slice(end)}`;
      input.selectionStart = input.selectionEnd = start + Math.min(pastedText.length, remaining);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    input.addEventListener("input", trimValue);
    trimValue();
    input.dataset.b1kMethodLimitPatched = "true";
  };

  const updateCountdown = () => {
    const countdown = document.getElementById("submission-countdown");
    if (!countdown) {
      return;
    }

    const deadline = new Date("2026-10-17T11:59:00Z");
    const remainingMs = deadline.getTime() - Date.now();
    if (remainingMs <= 0) {
      countdown.textContent = "Submission deadline has passed";
      return;
    }

    const totalSeconds = Math.floor(remainingMs / 1000);
    const days = Math.floor(totalSeconds / 86400);
    const hours = Math.floor((totalSeconds % 86400) / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    countdown.textContent = `${days} days ${hours} hours ${minutes} minutes ${seconds} seconds remaining`;
  };

  const ensureCountdownTimer = () => {
    updateCountdown();
    if (window.b1kCountdownTimer) {
      return;
    }

    window.b1kCountdownTimer = window.setInterval(updateCountdown, 1000);
  };

  const submissionValue = (fieldId, fallback = "") => {
    const field = document.getElementById(fieldId);
    if (!field) {
      return fallback;
    }
    const selectedChoice = field.querySelector('input[type="radio"]:checked');
    if (selectedChoice) {
      return selectedChoice.value;
    }
    const control = field.querySelector("textarea, input, select");
    return control ? control.value : fallback;
  };

  const submissionChecked = (fieldId) =>
    Boolean(document.getElementById(fieldId)?.querySelector('input[type="checkbox"]')?.checked);

  const setSubmissionStatus = (html) => {
    const prose = document.querySelector("#submission-status .prose");
    if (prose) {
      prose.innerHTML = html;
    }
  };

  const completedSubmissionData = (eventStream) => {
    for (const block of eventStream.split(/\\n\\n+/)) {
      const lines = block.split("\\n");
      if (lines[0] !== "event: complete") {
        continue;
      }
      const dataLine = lines.find((line) => line.startsWith("data: "));
      if (dataLine) {
        return JSON.parse(dataLine.slice(6));
      }
    }
    throw new Error("The submission service returned an incomplete response.");
  };

  const ensureSubmissionHandler = () => {
    if (window.b1kSubmissionHandlerBound) {
      return;
    }

    document.addEventListener(
      "click",
      async (event) => {
        const button = event.target.closest?.("#submission-submit-button");
        if (!button || !window.location.pathname.endsWith("/submit")) {
          return;
        }
        event.preventDefault();
        event.stopImmediatePropagation();
        button.disabled = true;
        button.textContent = "Submitting...";
        setSubmissionStatus(
          '<div class="submission-status submission-status-pending" role="status">' +
            "<strong>Submitting...</strong>Please keep this page open and do not click Submit again.</div>",
        );

        const data = [
          submissionValue("submit-team"),
          submissionValue("submit-team-members"),
          submissionValue("submit-affiliation"),
          submissionValue("method-description-field"),
          submissionValue("submit-policy-type"),
          submissionValue("submit-docker-image"),
          submissionValue("submit-docker-digest"),
          submissionValue("submit-policy-server"),
          submissionValue("submit-evaluation-readme"),
          submissionValue("submit-self-eval"),
          submissionValue("submit-video"),
          submissionValue("submit-release"),
          submissionValue("submit-primary-email"),
          submissionValue("submit-team-emails"),
          submissionValue("submit-human-data", "Not applicable"),
          submissionValue("submit-heuristic-data", "Not applicable"),
          submissionValue("submit-rl-data", "Not applicable"),
          submissionValue("submit-other-data", "Not applicable"),
          submissionValue("submit-other-data-description"),
          submissionValue("submit-comments"),
          submissionChecked("submit-rules"),
          submissionChecked("submit-public-release"),
        ];

        try {
          const callResponse = await fetch("/gradio_api/call/submit_entry", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data }),
          });
          if (!callResponse.ok) {
            throw new Error(`Submission request failed (${callResponse.status}).`);
          }
          const { event_id: eventId } = await callResponse.json();
          const resultResponse = await fetch(`/gradio_api/call/submit_entry/${eventId}`);
          if (!resultResponse.ok) {
            throw new Error(`Submission result failed (${resultResponse.status}).`);
          }
          const result = completedSubmissionData(await resultResponse.text());
          setSubmissionStatus(result[0]);
        } catch (error) {
          const message = String(error?.message || error).replace(/[&<>"']/g, (character) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;",
          })[character]);
          setSubmissionStatus(
            '<div class="submission-status submission-status-error" role="alert">' +
              `<strong>Submission not saved.</strong>${message}</div>`,
          );
        } finally {
          button.disabled = false;
          button.textContent = "Submit";
        }
      },
      true,
    );
    window.b1kSubmissionHandlerBound = true;
  };

  patchAll();
  ensureExternalReleaseLinks();
  patchRequiredLabels();
  enforceMethodDescriptionLimit();
  ensureCountdownTimer();
  window.requestAnimationFrame(() => {
    patchAll();
    ensureExternalReleaseLinks();
    patchRequiredLabels();
    enforceMethodDescriptionLimit();
    ensureCountdownTimer();
  });
  window.addEventListener("pageshow", () => {
    ensureCountdownTimer();
  });
  window.addEventListener("load", () => {
    ensureCountdownTimer();
  });

  new MutationObserver(() => {
    patchAll();
    patchRequiredLabels();
    enforceMethodDescriptionLimit();
    ensureCountdownTimer();
  }).observe(document.body, {
    childList: true,
    subtree: true,
  });
}

b1kPatchSpaceLinks();
}
"""

RESULT_COLUMNS = [
    "rank",
    "team",
    "method",
    "score",
    "success_rate",
    "q_score",
    "time_score",
    "efficiency_score",
    "num_tasks",
    "num_episodes",
    "evaluated_at",
    "verified",
    "submission_id",
    "report_url",
    "video_url",
    "logs_url",
]

LEADERBOARD_COLUMNS = [
    "rank",
    "team",
    "method",
    "status",
    "score",
    "success_rate",
    "q_score",
    "time_score",
    "efficiency_score",
    "num_tasks",
    "num_episodes",
    "submitted_at",
    "evaluated_at",
    "affiliation",
    "paper_url",
    "code_url",
    "model_url",
    "report_url",
    "video_url",
    "logs_url",
    "submission_id",
    "self_reported_q_score",
    "self_reported_success_rate",
]

SELF_REPORTED_RESULT_COLUMNS = [
    "submission_id",
    "q_score",
    "success_rate",
    "num_tasks",
    "num_episodes",
]

LEADERBOARD_DISPLAY_COLUMNS = [
    "Rank",
    "Team",
    "Affiliation",
    "Method",
    "Self-Reported Q Score",
    "Verified Q Score",
    "Self-Reported Full Task SR",
    "Verified Full Task SR",
    "Submitted",
    "Release",
]

LEADERBOARD_DATATYPES = ["str", "str", "str", "str", "str", "str", "str", "str", "str", "markdown"]

LEADERBOARD_DISPLAY_LABELS = {
    "rank": "Rank",
    "team": "Team",
    "affiliation": "Affiliation",
    "method": "Method",
    "submitted_at": "Submitted",
    "release": "Release",
}

SUBMISSION_COLUMNS = [
    "team",
    "method",
    "status",
    "submitted_at",
    "affiliation",
    "paper_url",
    "code_url",
    "model_url",
    "submission_id",
]

PER_TASK_COLUMNS = [
    "submission_id",
    "team",
    "affiliation",
    "method",
    "task",
    "score",
    "success_rate",
    "q_score",
    "time_score",
    "efficiency_score",
    "num_episodes",
    "mean_steps",
    "mean_time",
]

PER_TASK_DISPLAY_COLUMNS = [
    "Team",
    "Affiliation",
    "Method",
    "Task",
    "Q Score",
    "Full Task Success Rate",
    "Time Score",
    "Efficiency Score",
]

PER_TASK_DISPLAY_LABELS = {
    "team": "Team",
    "affiliation": "Affiliation",
    "method": "Method",
    "task": "Task",
    "q_score": "Q Score",
    "success_rate": "Full Task Success Rate",
    "time_score": "Time Score",
    "efficiency_score": "Efficiency Score",
}

LEADERBOARD_2025_COLUMNS = [
    "rank",
    "team",
    "affiliation",
    "track",
    "validation_success_rate",
    "test_success_rate",
    "validation_q_score",
    "test_q_score",
    "submitted_at",
    "code_url",
    "report_url",
]

LEADERBOARD_2025_DISPLAY_COLUMNS = [
    "Rank",
    "Team",
    "Affiliation",
    "Track",
    "Self-Reported Q Score",
    "Verified Q Score",
    "Self-Reported Full Task SR",
    "Verified Full Task SR",
    "Submitted",
    "Release",
]

LEADERBOARD_2025_DATATYPES = [
    "number",
    "str",
    "str",
    "str",
    "str",
    "str",
    "str",
    "str",
    "str",
    "html",
]

LEADERBOARD_2025_COLUMN_WIDTHS = [55, 155, 220, 90, 145, 130, 190, 145, 100, 105]
LEADERBOARD_COLUMN_WIDTHS = [55, 180, 220, 130, 125, 125, 145, 135, 145, 75]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}:{line_number}: {exc}") from exc
    return rows


def frame_from_jsonl(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(read_jsonl(path))
    if frame.empty:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in frame:
            frame[column] = None
    return frame


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def format_rate(value) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"{numeric * 100:.1f}%"


def format_date(value) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if pd.isna(timestamp) else timestamp.strftime("%Y-%m-%d")


def format_datetime(value) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if pd.isna(timestamp) else timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_release(row: pd.Series) -> str:
    links = []
    for label, column in (
        ("paper", "paper_url"),
        ("code", "code_url"),
        ("model", "model_url"),
        ("report", "report_url"),
        ("video", "video_url"),
        ("logs", "logs_url"),
    ):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            links.append(f"[{label.title()}]({str(value).strip()})")
    return " ".join(links)


def format_2025_release(row: pd.Series) -> str:
    value = row.get("report_url")
    if pd.isna(value) or not str(value).strip():
        return ""
    url = escape(str(value).strip(), quote=True)
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">Report</a>'


def load_results() -> pd.DataFrame:
    frame = frame_from_jsonl(DATA_DIR / "results.jsonl", RESULT_COLUMNS)
    if frame.empty:
        return frame

    frame = frame[frame["verified"].map(as_bool)].copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["success_rate"] = pd.to_numeric(frame["success_rate"], errors="coerce")
    frame["q_score"] = pd.to_numeric(frame["q_score"], errors="coerce")
    frame["time_score"] = pd.to_numeric(frame["time_score"], errors="coerce")
    frame["efficiency_score"] = pd.to_numeric(frame["efficiency_score"], errors="coerce")
    frame = frame.sort_values(
        by=["q_score", "success_rate", "score", "evaluated_at"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    return frame[RESULT_COLUMNS]


def load_self_reported_results() -> pd.DataFrame:
    frame = frame_from_jsonl(DATA_DIR / "self_reported_results.jsonl", SELF_REPORTED_RESULT_COLUMNS)
    if frame.empty:
        return frame

    frame["q_score"] = pd.to_numeric(frame["q_score"], errors="coerce")
    frame["success_rate"] = pd.to_numeric(frame["success_rate"], errors="coerce")
    return frame.drop_duplicates("submission_id", keep="last")


def normalized_team_name(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def refresh_private_submission_cache() -> None:
    global _private_submission_cache_last_refresh, _private_submission_refresh_in_progress

    refreshed_cache = None
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=SUBMISSION_TOKEN)
        paths = {
            path
            for path in api.list_repo_files(repo_id=SUBMISSION_DATASET_REPO, repo_type="dataset")
            if path.startswith("2026/submissions/") and path.endswith(".json")
        }
        refreshed_cache = {}
        for path in paths:
            local_path = hf_hub_download(
                repo_id=SUBMISSION_DATASET_REPO,
                filename=path,
                repo_type="dataset",
                token=SUBMISSION_TOKEN,
            )
            with Path(local_path).open("r", encoding="utf-8") as submission_file:
                response = json.load(submission_file)
            refreshed_cache[path] = {
                "team": clean_text(response.get("team")),
                "method": clean_text(response.get("method")),
                "status": "submitted",
                "submitted_at": clean_text(response.get("submitted_at")),
                "affiliation": clean_text(response.get("affiliation")),
                "paper_url": None,
                "code_url": None,
                "model_url": None,
                "submission_id": clean_text(response.get("submission_id")),
            }
    except Exception as exc:
        log.warning("Could not refresh private submissions: %s", exc)
    finally:
        with _private_submission_cache_lock:
            if refreshed_cache is not None:
                _private_submission_cache.clear()
                _private_submission_cache.update(refreshed_cache)
            _private_submission_cache_last_refresh = time.monotonic()
            _private_submission_refresh_in_progress = False


def start_private_submission_refresh() -> None:
    global _private_submission_refresh_in_progress

    if not SUBMISSION_TOKEN:
        return
    with _private_submission_cache_lock:
        if _private_submission_refresh_in_progress:
            return
        _private_submission_refresh_in_progress = True
    threading.Thread(target=refresh_private_submission_cache, daemon=True).start()


def private_submission_metadata() -> pd.DataFrame:
    if not SUBMISSION_TOKEN:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)

    with _private_submission_cache_lock:
        stale = time.monotonic() - _private_submission_cache_last_refresh >= PRIVATE_SUBMISSION_REFRESH_SECONDS
        snapshot = list(_private_submission_cache.values())
    if stale:
        start_private_submission_refresh()
    return pd.DataFrame(snapshot, columns=SUBMISSION_COLUMNS)


def invalidate_private_submission_cache() -> None:
    global _private_submission_cache_last_refresh

    with _private_submission_cache_lock:
        _private_submission_cache_last_refresh = 0.0
    start_private_submission_refresh()


def load_submissions() -> pd.DataFrame:
    public_frame = frame_from_jsonl(DATA_DIR / "submissions.jsonl", SUBMISSION_COLUMNS)
    private_frame = private_submission_metadata()
    frame = pd.concat([public_frame, private_frame], ignore_index=True)
    if frame.empty:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)

    frame = frame[~frame["submission_id"].isin(IGNORED_SUBMISSION_IDS)].copy()
    frame["team_key"] = frame["team"].map(normalized_team_name)
    frame["submitted_sort"] = pd.to_datetime(frame["submitted_at"], errors="coerce", utc=True)
    frame = frame.sort_values(
        by=["submitted_sort", "submitted_at", "team", "method"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    frame = frame.drop_duplicates("team_key", keep="first")
    return frame[SUBMISSION_COLUMNS].reset_index(drop=True)


def load_per_task() -> pd.DataFrame:
    frame = frame_from_jsonl(DATA_DIR / "per_task_results.jsonl", PER_TASK_COLUMNS)
    if frame.empty:
        return frame

    results = frame_from_jsonl(DATA_DIR / "results.jsonl", RESULT_COLUMNS)
    if not results.empty and "submission_id" in frame:
        metadata = results[["submission_id", "team", "method"]].drop_duplicates("submission_id")
        submissions = frame_from_jsonl(DATA_DIR / "submissions.jsonl", SUBMISSION_COLUMNS)
        if not submissions.empty:
            metadata = metadata.merge(
                submissions[["submission_id", "affiliation"]].drop_duplicates("submission_id"),
                on="submission_id",
                how="left",
            )
        else:
            metadata["affiliation"] = None
        frame = frame.drop(
            columns=[column for column in ("team", "method", "affiliation") if column in frame],
            errors="ignore",
        )
        frame = frame.merge(metadata, on="submission_id", how="left")

    for column in ("score", "success_rate", "q_score", "time_score", "efficiency_score", "mean_steps", "mean_time"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[PER_TASK_COLUMNS].sort_values(by=["task", "score"], ascending=[True, False])


def leaderboard_table() -> pd.DataFrame:
    results = load_results()
    submissions = load_submissions()
    self_reported = load_self_reported_results()
    if results.empty and submissions.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    if submissions.empty:
        frame = results.copy()
        frame["status"] = "verified"
        for column in SUBMISSION_COLUMNS:
            if column not in frame:
                frame[column] = None
    else:
        frame = submissions.merge(results, on="submission_id", how="outer", suffixes=("_submission", "_result"))
        for column in ("team", "method"):
            frame[column] = frame[f"{column}_submission"].combine_first(frame[f"{column}_result"])
        frame["status"] = frame["status"].fillna("verified")

    if not self_reported.empty:
        frame = frame.merge(
            self_reported[["submission_id", "q_score", "success_rate"]].rename(
                columns={
                    "q_score": "self_reported_q_score",
                    "success_rate": "self_reported_success_rate",
                }
            ),
            on="submission_id",
            how="left",
        )

    for column in LEADERBOARD_COLUMNS:
        if column not in frame:
            frame[column] = None

    verified_mask = frame.get("verified", pd.Series(False, index=frame.index)).map(as_bool)
    frame.loc[verified_mask, "status"] = "verified"
    frame["team_key"] = frame["team"].map(normalized_team_name)
    frame["submitted_sort"] = pd.to_datetime(frame["submitted_at"], errors="coerce", utc=True)
    frame["evaluated_sort"] = pd.to_datetime(frame["evaluated_at"], errors="coerce", utc=True)
    frame["latest_sort"] = frame["submitted_sort"].combine_first(frame["evaluated_sort"])
    frame = frame.sort_values("latest_sort", ascending=False, kind="mergesort")
    frame = frame.drop_duplicates("team_key", keep="first")

    frame["rank_sort"] = pd.to_numeric(frame["rank"], errors="coerce").fillna(1_000_000)
    frame["q_score_sort"] = pd.to_numeric(frame["q_score"], errors="coerce").fillna(-1.0)
    frame = frame.sort_values(
        by=["rank_sort", "q_score_sort", "submitted_at", "team", "method"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    return frame[LEADERBOARD_COLUMNS].reset_index(drop=True)


def filtered_leaderboard(query: str | None, status: str | None) -> pd.DataFrame:
    frame = leaderboard_table()
    if frame.empty:
        return frame

    if query:
        query = query.strip().lower()
        searchable = (
            frame["team"].fillna("").astype(str)
            + " "
            + frame["method"].fillna("").astype(str)
            + " "
            + frame["submission_id"].fillna("").astype(str)
        ).str.lower()
        frame = frame[searchable.str.contains(query, regex=False)]

    if status and status != "All":
        frame = frame[frame["status"].fillna("").astype(str) == status]

    return frame


def display_leaderboard(query: str | None, status: str | None) -> pd.DataFrame:
    frame = filtered_leaderboard(query, status)
    if frame.empty:
        return pd.DataFrame(columns=LEADERBOARD_DISPLAY_COLUMNS)

    frame = frame.copy()
    score_source = frame.copy()
    frame["release"] = frame.apply(format_release, axis=1)
    verified_mask = frame["status"].fillna("").astype(str).eq("verified")
    for column in LEADERBOARD_DISPLAY_LABELS:
        if column not in frame:
            frame[column] = None
    frame = frame[list(LEADERBOARD_DISPLAY_LABELS.keys())].rename(columns=LEADERBOARD_DISPLAY_LABELS)
    rank = pd.to_numeric(frame["Rank"], errors="coerce")
    frame["Rank"] = rank.map(lambda value: "" if pd.isna(value) else str(int(value)))
    frame["Submitted"] = frame["Submitted"].map(format_datetime)
    q_score = pd.to_numeric(score_source["q_score"], errors="coerce")
    success_rate = score_source["success_rate"]
    self_reported_q_score = pd.to_numeric(score_source["self_reported_q_score"], errors="coerce")
    self_reported_success_rate = score_source["self_reported_success_rate"]
    frame["Self-Reported Q Score"] = self_reported_q_score.map(
        lambda value: "" if pd.isna(value) else f"{value:.3f}"
    )
    frame["Verified Q Score"] = q_score.map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    frame["Self-Reported Full Task SR"] = self_reported_success_rate.map(format_rate)
    frame["Verified Full Task SR"] = success_rate.map(format_rate)
    verified_score_columns = [
        "Verified Q Score",
        "Verified Full Task SR",
    ]
    frame.loc[~verified_mask, verified_score_columns] = "Under verification"
    frame.loc[~verified_mask, "Rank"] = ""
    return frame[LEADERBOARD_DISPLAY_COLUMNS]


def per_task_table(submission_id: str | None) -> pd.DataFrame:
    frame = load_per_task()
    if submission_id:
        frame = frame[frame["submission_id"] == submission_id]
    return frame


def display_per_task(submission_id: str | None) -> pd.DataFrame:
    frame = per_task_table(submission_id)
    if frame.empty:
        return pd.DataFrame(columns=PER_TASK_DISPLAY_COLUMNS)

    frame = frame.copy()
    for column in PER_TASK_DISPLAY_LABELS:
        if column not in frame:
            frame[column] = None
    frame = frame[list(PER_TASK_DISPLAY_LABELS.keys())].rename(columns=PER_TASK_DISPLAY_LABELS)
    for column in ("Q Score", "Time Score", "Efficiency Score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round(3)
    frame["Full Task Success Rate"] = frame["Full Task Success Rate"].map(format_rate)
    return frame[PER_TASK_DISPLAY_COLUMNS]


def load_2025_results() -> pd.DataFrame:
    frame = frame_from_jsonl(DATA_DIR / "2025_results.jsonl", LEADERBOARD_2025_COLUMNS)
    if frame.empty:
        return frame

    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    return frame.sort_values("rank", kind="mergesort").reset_index(drop=True)


def display_2025_leaderboard(query: str | None, track: str | None) -> pd.DataFrame:
    frame = load_2025_results()
    if frame.empty:
        return pd.DataFrame(columns=LEADERBOARD_2025_DISPLAY_COLUMNS)

    if query:
        query = query.strip().lower()
        searchable = (
            frame["team"].fillna("").astype(str)
            + " "
            + frame["affiliation"].fillna("").astype(str)
        ).str.lower()
        frame = frame[searchable.str.contains(query, regex=False)]

    if track and track != "All":
        frame = frame[frame["track"] == track]

    frame = frame.copy()
    frame["release"] = frame.apply(format_2025_release, axis=1)
    frame = frame.rename(
        columns={
            "rank": "Rank",
            "team": "Team",
            "affiliation": "Affiliation",
            "track": "Track",
            "validation_success_rate": "Self-Reported Full Task SR",
            "test_success_rate": "Verified Full Task SR",
            "validation_q_score": "Self-Reported Q Score",
            "test_q_score": "Verified Q Score",
            "submitted_at": "Submitted",
            "release": "Release",
        }
    )
    for column in ("Self-Reported Full Task SR", "Verified Full Task SR"):
        frame[column] = frame[column].map(format_rate)
    for column in ("Self-Reported Q Score", "Verified Q Score"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric.map(lambda value: "" if pd.isna(value) else f"{value:.4f}".rstrip("0").rstrip("."))
    return frame[LEADERBOARD_2025_DISPLAY_COLUMNS]


def submission_choices() -> list[str]:
    frame = leaderboard_table()
    if frame.empty:
        return []
    return frame["submission_id"].dropna().astype(str).tolist()


def status_choices() -> list[str]:
    frame = leaderboard_table()
    if frame.empty or "status" not in frame:
        return ["All"]
    statuses = sorted(frame["status"].dropna().astype(str).unique().tolist())
    return ["All", *statuses]


def summary_html() -> str:
    leaderboard = leaderboard_table()
    verified = leaderboard[leaderboard["status"].astype(str) == "verified"] if not leaderboard.empty else leaderboard
    best_q_score = pd.to_numeric(verified.get("q_score"), errors="coerce").max() if not verified.empty else float("nan")
    best_q_score_text = "-" if pd.isna(best_q_score) else f"{best_q_score:.3f}"
    full_success = (
        pd.to_numeric(verified.get("success_rate"), errors="coerce").max() if not verified.empty else float("nan")
    )
    full_success_text = "-" if pd.isna(full_success) else f"{full_success * 100:.1f}%"

    cards = [
        ("total", "Total submissions", f"{len(leaderboard):,}", "#2563eb"),
        ("verified", "Verified entries", f"{len(verified):,}", "#0f766e"),
        ("q-score", "Best Q score", best_q_score_text, "#7c3aed"),
        ("full-success", "Best full task success rate", full_success_text, "#c2410c"),
    ]
    return '<div class="metrics-panel"><div class="metric-grid">' + "".join(
        f'<div class="metric-card {name}" style="border-top-color: {color};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color: {color};">{value}</div></div>'
        for name, label, value, color in cards
    ) + "</div></div>"


def clean_text(value: str | None) -> str:
    return str(value or "").strip()


def split_lines(value: str | None) -> list[str]:
    return [line.strip() for line in clean_text(value).splitlines() if line.strip()]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "submission"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def deadline_countdown_text() -> str:
    remaining = SUBMISSION_DEADLINE_UTC - datetime.now(timezone.utc)
    total_seconds = int(remaining.total_seconds())
    if total_seconds <= 0:
        return "Submission deadline has passed"

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{days} days {hours} hours {minutes} minutes {seconds} seconds remaining"


def deadline_countdown_widget_html(compact: bool = False) -> str:
    srcdoc = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html,
  body {
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .deadline-card {
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    border: 1px solid #f2c9a2;
    border-radius: 8px;
    background: #fff7ed;
    padding: 12px 14px;
  }
  .deadline-label {
    color: #9a3412;
    font-size: 12px;
    font-weight: 750;
    margin-bottom: 8px;
  }
  .deadline-value {
    color: #172033;
    font-size: 18px;
    font-weight: 800;
    line-height: 1.35;
  }
  .deadline-countdown {
    color: #9a3412;
    font-size: 13px;
    font-weight: 700;
    line-height: 1.35;
    margin-top: 9px;
  }
  .deadline-card.compact {
    display: flex;
    align-items: center;
    gap: 12px;
    border: 0;
    border-radius: 0;
    background: transparent;
    padding: 0;
  }
  .deadline-card.compact .deadline-label,
  .deadline-card.compact .deadline-value,
  .deadline-card.compact .deadline-countdown {
    margin: 0;
    color: #172033;
    font-size: 13px;
    font-weight: 700;
    line-height: 1.35;
    white-space: nowrap;
  }
  .deadline-card.compact .deadline-countdown {
    color: #9a3412;
  }
</style>
</head>
<body>
  <div class="deadline-card DEADLINE_LAYOUT_CLASS">
    <div class="deadline-label">Final submission deadline</div>
    <div class="deadline-value">Oct. 16, 2026, 11:59 PM AOE</div>
    <div id="countdown" class="deadline-countdown">Calculating time remaining...</div>
  </div>
<script>
  const deadline = new Date("2026-10-17T11:59:00Z");
  const countdown = document.getElementById("countdown");

  function updateCountdown() {
    const remainingMs = deadline.getTime() - Date.now();
    if (remainingMs <= 0) {
      countdown.textContent = "Submission deadline has passed";
      return;
    }

    const totalSeconds = Math.floor(remainingMs / 1000);
    const days = Math.floor(totalSeconds / 86400);
    const hours = Math.floor((totalSeconds % 86400) / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    const compact = document.querySelector(".deadline-card").classList.contains("compact");
    countdown.textContent = compact
      ? `${days}d ${hours}h ${minutes}m ${seconds}s remaining`
      : `${days} days ${hours} hours ${minutes} minutes ${seconds} seconds remaining`;
  }

  updateCountdown();
  window.setInterval(updateCountdown, 1000);

  function preventPortalAutofocus() {
    if (!window.parent.location.pathname.endsWith("/submit")) {
      return;
    }

    const parentDocument = window.parent.document;
    let corrected = false;
    parentDocument.querySelectorAll('[autofocus]').forEach((element) => {
      if (element === parentDocument.activeElement) {
        element.blur();
      }
      element.removeAttribute("autofocus");
      corrected = true;
    });
    if (corrected) {
      window.parent.scrollTo(0, 0);
    }
  }

  let lastInvalidFieldSignature = null;

  function setPortalFieldError(element, invalid) {
    if (!element) {
      return;
    }
    element.classList.toggle("submission-field-error", invalid);
    element.querySelectorAll("input, textarea, select").forEach((control) => {
      if (invalid) {
        control.setAttribute("aria-invalid", "true");
      } else {
        control.removeAttribute("aria-invalid");
      }
    });
  }

  function syncPortalFieldErrors() {
    if (!window.parent.location.pathname.endsWith("/submit")) {
      return;
    }

    const parentDocument = window.parent.document;
    const errorStatus = parentDocument.querySelector(".submission-status-error[data-invalid-fields]");
    const signature = errorStatus?.dataset.invalidFields || "";
    if (signature === lastInvalidFieldSignature) {
      return;
    }

    parentDocument.querySelectorAll(".submission-field-error").forEach((element) => {
      setPortalFieldError(element, false);
    });

    const invalidFields = signature
      .split(",")
      .map((fieldId) => fieldId.trim())
      .filter(Boolean)
      .map((fieldId) => parentDocument.getElementById(fieldId))
      .filter(Boolean);
    invalidFields.forEach((element) => setPortalFieldError(element, true));
    if (invalidFields.length > 0) {
      invalidFields[0].scrollIntoView({ behavior: "smooth", block: "center" });
    }
    lastInvalidFieldSignature = signature;
  }

  function portalValue(parentDocument, fieldId, fallback = "") {
    const field = parentDocument.getElementById(fieldId);
    if (!field) {
      return fallback;
    }
    const selectedChoice = field.querySelector('input[type="radio"]:checked');
    if (selectedChoice) {
      return selectedChoice.value;
    }
    const control = field.querySelector("textarea, input, select");
    return control ? control.value : fallback;
  }

  function portalChecked(parentDocument, fieldId) {
    return Boolean(
      parentDocument.getElementById(fieldId)?.querySelector('input[type="checkbox"]')?.checked
    );
  }

  function setPortalSubmissionStatus(parentDocument, html) {
    const prose = parentDocument.querySelector("#submission-status .prose");
    if (prose) {
      prose.innerHTML = html;
    }
  }

  function completedPortalSubmissionData(eventStream) {
    for (const block of eventStream.split(/\\n\\n+/)) {
      const lines = block.split("\\n");
      if (lines[0] !== "event: complete") {
        continue;
      }
      const dataLine = lines.find((line) => line.startsWith("data: "));
      if (dataLine) {
        return JSON.parse(dataLine.slice(6));
      }
    }
    throw new Error("The submission service returned an incomplete response.");
  }

  function bindPortalSubmission(parentDocument, parentWindow) {
    if (parentWindow.__b1kNativeSubmissionBound) {
      return;
    }
    parentDocument.addEventListener(
      "click",
      async (event) => {
        const button = event.target.closest?.("#submission-submit-button");
        if (!button) {
          return;
        }
        event.preventDefault();
        event.stopImmediatePropagation();
        button.disabled = true;
        button.textContent = "Submitting...";
        setPortalSubmissionStatus(
          parentDocument,
          '<div class="submission-status submission-status-pending" role="status">' +
            "<strong>Submitting...</strong>Please keep this page open and do not click Submit again.</div>"
        );

        const data = [
          portalValue(parentDocument, "submit-team"),
          portalValue(parentDocument, "submit-team-members"),
          portalValue(parentDocument, "submit-affiliation"),
          portalValue(parentDocument, "method-description-field"),
          portalValue(parentDocument, "submit-policy-type"),
          portalValue(parentDocument, "submit-docker-image"),
          portalValue(parentDocument, "submit-docker-digest"),
          portalValue(parentDocument, "submit-policy-server"),
          portalValue(parentDocument, "submit-evaluation-readme"),
          portalValue(parentDocument, "submit-self-eval"),
          portalValue(parentDocument, "submit-video"),
          portalValue(parentDocument, "submit-release"),
          portalValue(parentDocument, "submit-primary-email"),
          portalValue(parentDocument, "submit-team-emails"),
          portalValue(parentDocument, "submit-human-data", "Not applicable"),
          portalValue(parentDocument, "submit-heuristic-data", "Not applicable"),
          portalValue(parentDocument, "submit-rl-data", "Not applicable"),
          portalValue(parentDocument, "submit-other-data", "Not applicable"),
          portalValue(parentDocument, "submit-other-data-description"),
          portalValue(parentDocument, "submit-comments"),
          portalChecked(parentDocument, "submit-rules"),
          portalChecked(parentDocument, "submit-public-release"),
        ];

        try {
          const callResponse = await parentWindow.fetch("/gradio_api/call/submit_entry", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data }),
          });
          if (!callResponse.ok) {
            throw new Error("Submission request failed (" + callResponse.status + ").");
          }
          const call = await callResponse.json();
          const resultResponse = await parentWindow.fetch(
            "/gradio_api/call/submit_entry/" + call.event_id
          );
          if (!resultResponse.ok) {
            throw new Error("Submission result failed (" + resultResponse.status + ").");
          }
          const result = completedPortalSubmissionData(await resultResponse.text());
          setPortalSubmissionStatus(parentDocument, result[0]);
        } catch (error) {
          const message = String(error?.message || error).replace(/[&<>"']/g, (character) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;",
          })[character]);
          setPortalSubmissionStatus(
            parentDocument,
            '<div class="submission-status submission-status-error" role="alert">' +
              "<strong>Submission not saved.</strong>" + message + "</div>"
          );
        } finally {
          button.disabled = false;
          button.textContent = "Submit";
        }
      },
      true
    );
    parentWindow.__b1kNativeSubmissionBound = true;
  }

  function bindLeaderboardRefresh(parentWindow) {
    if (
      !document.querySelector(".deadline-card.compact") ||
      parentWindow.__b1kLeaderboardRefreshBound
    ) {
      return;
    }

    const refresh = () => {
      const refreshButton = parentWindow.document.querySelector("#leaderboard-refresh");
      refreshButton?.click();
    };

    refresh();
    parentWindow.setInterval(refresh, 60_000);
    parentWindow.__b1kLeaderboardRefreshBound = true;
  }

  if (window.parent.location.pathname.endsWith("/submit")) {
    preventPortalAutofocus();
    syncPortalFieldErrors();
    const parentDocument = window.parent.document;
    const parentWindow = window.parent;
    bindPortalSubmission(parentDocument, parentWindow);
    parentDocument.addEventListener("input", (event) => {
      setPortalFieldError(event.target.closest?.(".submission-field-error"), false);
    });
    parentDocument.addEventListener("change", (event) => {
      setPortalFieldError(event.target.closest?.(".submission-field-error"), false);
    });
    new window.parent.MutationObserver(() => {
      preventPortalAutofocus();
      syncPortalFieldErrors();
    }).observe(parentDocument.body, {
      childList: true,
      subtree: true,
    });
  } else {
    bindLeaderboardRefresh(window.parent);
  }
</script>
</body>
</html>
"""
    srcdoc = srcdoc.replace("DEADLINE_LAYOUT_CLASS", "compact" if compact else "")
    iframe_class = "submission-deadline-widget compact-deadline-widget" if compact else "submission-deadline-widget"
    return (
        f'<iframe class="{iframe_class}" title="Final submission deadline countdown" '
        f'srcdoc="{escape(srcdoc, quote=True)}"></iframe>'
    )


def submission_hero_html() -> str:
    countdown_widget = deadline_countdown_widget_html()
    return f"""
        <div id="submission-portal-top"></div>
        <section class="submission-panel submission-hero">
          <div class="submission-hero-header">
            <div class="submission-hero-copy">
              <h2 class="b1k-title">Official 2nd BEHAVIOR-1K Challenge Submission Portal</h2>
              <div class="submission-note">
                Submit one response per evaluation attempt. The latest valid final submission per team will be
                used for official ranking. Large artifacts should be provided as stable links; this portal stores
                the response JSON privately for organizer review.
              </div>
            </div>
            {countdown_widget}
          </div>
        </section>
        """


def make_submission_id(team: str, method: str, submitted_at: str) -> str:
    time_part = submitted_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    return f"{time_part}-{slugify(team)[:32]}-{slugify(method)[:32]}"


def submission_error(message: str, invalid_field_ids: list[str] | None = None) -> str:
    invalid_fields_attribute = ""
    validation_style = ""
    if invalid_field_ids:
        invalid_fields = escape(",".join(invalid_field_ids), quote=True)
        invalid_fields_attribute = f' data-invalid-fields="{invalid_fields}"'
        validation_style = """
            <style>
              .submission-field-error {
                border-radius: 6px;
                background: #fff7f7 !important;
                box-shadow: 0 0 0 2px #dc2626 !important;
              }
              .submission-field-error label {
                color: #b91c1c !important;
              }
            </style>
        """
    return (
        validation_style
        + f'<div class="submission-status submission-status-error" role="alert"{invalid_fields_attribute}>'
        f"<strong>Submission not saved.</strong>{escape(message)}"
        "</div>"
    )


def submission_success(submission_id: str) -> str:
    return (
        '<div class="submission-status submission-status-success" role="status">'
        "<strong>Submission saved.</strong>"
        f"Your submission ID is <code>{escape(submission_id)}</code>. "
        "You may now close this page."
        "</div>"
    )


def submission_result(status_html: str, invalid_field_ids: list[str] | None = None) -> str:
    return status_html


def submit_entry(
    team: str,
    team_members: str,
    affiliation: str,
    method: str,
    submission_type: str,
    docker_image_uri: str,
    docker_image_digest: str,
    policy_server_url: str,
    evaluation_readme_url: str,
    self_eval_results_url: str,
    video_url: str,
    release_url: str,
    primary_contact_email: str,
    team_contact_emails: str,
    human_teleop_data_usage: str,
    heuristic_data_usage: str,
    rl_data_usage: str,
    other_data_usage: str,
    other_data_description: str,
    additional_comments: str,
    rules_acknowledged: bool,
    public_release_consent: bool,
) -> str:
    team = clean_text(team)
    affiliation = clean_text(affiliation)
    method = clean_text(method)
    primary_contact_email = clean_text(primary_contact_email)
    docker_image_uri = clean_text(docker_image_uri)
    docker_image_digest = clean_text(docker_image_digest)
    policy_server_url = clean_text(policy_server_url)
    evaluation_readme_url = clean_text(evaluation_readme_url)
    self_eval_results_url = clean_text(self_eval_results_url)
    video_url = clean_text(video_url)

    missing = []
    invalid_field_ids = []
    for label, field_id, value in (
        ("team", "submit-team", team),
        ("team member names", "submit-team-members", split_lines(team_members)),
        ("affiliation", "submit-affiliation", affiliation),
        ("method description", "method-description-field", method),
        ("policy submission method", "submit-policy-type", submission_type),
        ("primary contact email", "submit-primary-email", primary_contact_email),
        ("evaluation README URL", "submit-evaluation-readme", evaluation_readme_url),
        ("self-evaluation results URL", "submit-self-eval", self_eval_results_url),
        ("video recordings URL", "submit-video", video_url),
    ):
        if not value:
            missing.append(label)
            invalid_field_ids.append(field_id)

    if submission_type == "Docker image submission" and not docker_image_uri:
        missing.append("Docker image URI")
        invalid_field_ids.append("submit-docker-image")
    if submission_type == "Policy server URL submission" and not policy_server_url:
        missing.append("policy server URL")
        invalid_field_ids.append("submit-policy-server")
    if not rules_acknowledged:
        missing.append("rules acknowledgement")
        invalid_field_ids.append("submit-rules")
    if missing:
        return submission_result(
            submission_error(
                "Missing required fields: " + ", ".join(missing) + ".",
                invalid_field_ids=invalid_field_ids,
            ),
            invalid_field_ids=invalid_field_ids,
        )
    if len(method) > 25:
        return submission_result(
            submission_error(
                "Method description must be 25 characters or fewer.",
                invalid_field_ids=["method-description-field"],
            ),
            invalid_field_ids=["method-description-field"],
        )

    if not SUBMISSION_TOKEN:
        return submission_result(
            submission_error(
                "The Space is missing `HF_WRITE_TOKEN`, so it cannot write to the private submissions dataset."
            )
        )

    submitted_at = utc_timestamp()
    submission_id = make_submission_id(team, method, submitted_at)
    response = {
        "schema_version": 1,
        "submission_id": submission_id,
        "submitted_at": submitted_at,
        "team": team,
        "team_members": split_lines(team_members),
        "affiliation": affiliation,
        "method": method,
        "track": "official",
        "submission_type": submission_type,
        "policy": {
            "docker_image_uri": docker_image_uri,
            "docker_image_digest": docker_image_digest,
            "policy_server_url": policy_server_url,
        },
        "artifacts": {
            "evaluation_readme_url": evaluation_readme_url,
            "self_eval_results_url": self_eval_results_url,
            "video_url": video_url,
            "release_url": clean_text(release_url),
        },
        "contacts": {
            "primary_contact_email": primary_contact_email,
            "team_contact_emails": split_lines(team_contact_emails),
        },
        "additional_data_usage": {
            "human_teleoperation": human_teleop_data_usage,
            "heuristics": heuristic_data_usage,
            "rl_data": rl_data_usage,
            "other": other_data_usage,
            "other_description": clean_text(other_data_description),
        },
        "additional_comments": clean_text(additional_comments),
        "rules_acknowledged": bool(rules_acknowledged),
        "public_release_consent": bool(public_release_consent),
    }

    try:
        from huggingface_hub import HfApi

        path_in_repo = f"2026/submissions/{submitted_at[:10]}/{submission_id}.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as temp_file:
            json.dump(response, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_path = temp_file.name
        try:
            HfApi(token=SUBMISSION_TOKEN).upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=path_in_repo,
                repo_id=SUBMISSION_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Add submission {submission_id}",
            )
            invalidate_private_submission_cache()
        finally:
            Path(temp_path).unlink(missing_ok=True)
    except Exception as exc:
        return submission_result(submission_error(f"Upload failed: {type(exc).__name__}: {exc}"))

    return submission_result(submission_success(submission_id))


def render_submission_portal(gr):
    gr.HTML(
        value=submission_hero_html,
        container=False,
    )

    with gr.Group(elem_classes=["submission-panel", "team-panel"]):
        with gr.Row():
            submit_team = gr.Textbox(label="Team *", placeholder="Team name", scale=1, elem_id="submit-team")
            submit_affiliation = gr.Textbox(
                label="Affiliation *",
                placeholder="Institution or organization",
                scale=1,
                elem_id="submit-affiliation",
            )
        submit_team_members = gr.Textbox(
            label="Team member full names *",
            placeholder="One name per line; put the submitter first",
            lines=3,
            elem_id="submit-team-members",
        )
        submit_method = gr.Textbox(
            label="Method description *",
            placeholder="Short description of the method, 25 characters max",
            max_length=25,
            elem_id="method-description-field",
        )

    with gr.Group(elem_classes=["submission-panel", "policy-panel"]):
        submit_type = gr.Radio(
            label="Policy submission method *",
            choices=SUBMISSION_TYPE_OPTIONS,
            elem_id="submit-policy-type",
        )
        with gr.Group():
            submit_docker_uri = gr.Textbox(
                label="Docker image URI * (required for Docker submissions)",
                placeholder="Required for Docker image submissions, e.g. ghcr.io/org/behavior-agent:latest",
                elem_id="submit-docker-image",
            )
            submit_docker_digest = gr.Textbox(
                label="Docker image digest (optional)",
                placeholder="Recommended for Docker submissions, e.g. sha256:...",
                elem_id="submit-docker-digest",
            )
        with gr.Group():
            submit_policy_server_url = gr.Textbox(
                label="Policy server URL * (required for policy server submissions)",
                placeholder="Required for policy server submissions, e.g. wss://example.org:5000",
                elem_id="submit-policy-server",
            )
        submit_eval_readme_url = gr.Textbox(
            label="Evaluation README URL *",
            placeholder="Required stable link with setup, authentication, capacity, and run instructions",
            elem_id="submit-evaluation-readme",
        )

    with gr.Group(elem_classes=["submission-panel", "artifacts-panel"]):
        with gr.Row():
            submit_self_eval_url = gr.Textbox(
                label="Self-evaluation results URL *",
                placeholder="Required link to rollout JSON bundle or archive",
                scale=1,
                elem_id="submit-self-eval",
            )
            submit_video_url = gr.Textbox(
                label="Video recordings URL *",
                placeholder="Required link to evaluation videos",
                scale=1,
                elem_id="submit-video",
            )
        submit_release_url = gr.Textbox(
            label="Open-source / release URL (optional)",
            placeholder="Optional paper, codebase, model, report, project website, or other public release link",
            elem_id="submit-release",
        )
        submit_public_release = gr.Checkbox(
            label=(
                "I acknowledge that the public release link, if provided above, will be shown on the leaderboard "
                "after verification."
            ),
            value=False,
            elem_id="submit-public-release",
        )

    with gr.Group(elem_classes=["submission-panel", "data-panel"]):
        with gr.Row():
            submit_human_data = gr.Dropdown(
                label="Human teleoperation (optional)",
                choices=DATA_USAGE_OPTIONS,
                value="Not applicable",
                scale=1,
                elem_id="submit-human-data",
            )
            submit_heuristic_data = gr.Dropdown(
                label="Heuristics (e.g. motion planning) (optional)",
                choices=DATA_USAGE_OPTIONS,
                value="Not applicable",
                scale=1,
                elem_id="submit-heuristic-data",
            )
        with gr.Row():
            submit_rl_data = gr.Dropdown(
                label="Data from RL (optional)",
                choices=DATA_USAGE_OPTIONS,
                value="Not applicable",
                scale=1,
                elem_id="submit-rl-data",
            )
            submit_other_data = gr.Dropdown(
                label="Other data (optional)",
                choices=DATA_USAGE_OPTIONS,
                value="Not applicable",
                scale=1,
                elem_id="submit-other-data",
            )
        submit_other_data_description = gr.Textbox(
            label="Other data description (optional)",
            placeholder="Describe any additional data source marked as Other",
            lines=2,
            elem_id="submit-other-data-description",
        )

    with gr.Group(elem_classes=["submission-panel", "contacts-panel"]):
        submit_primary_email = gr.Textbox(
            label="Primary contact email *",
            placeholder="Official correspondence will be sent here",
            elem_id="submit-primary-email",
        )
        submit_team_emails = gr.Textbox(
            label="Team member contact emails (optional)",
            placeholder="Optional; one email per line",
            lines=3,
            elem_id="submit-team-emails",
        )
        submit_comments = gr.Textbox(
            label="Additional comments (optional)",
            placeholder="Authentication, capacity, scheduling, or other evaluation notes",
            lines=4,
            elem_id="submit-comments",
        )
        submit_rules = gr.Checkbox(
            label="I confirm this submission follows the official rules and evaluation protocol. *",
            value=False,
            elem_id="submit-rules",
        )
        submit_button = gr.Button("Submit", elem_id="submission-submit-button", elem_classes=["submit-button"])
        gr.HTML(
            '<div class="submission-processing-note">Please allow up to 5 minutes for your entry to be reflected on the leaderboard. Please do not submit repeatedly while waiting for your entry to appear.</div>',
            container=False,
        )
        submit_status = gr.HTML(container=False, elem_id="submission-status")

    submit_button.click(
        fn=submit_entry,
        inputs=[
            submit_team,
            submit_team_members,
            submit_affiliation,
            submit_method,
            submit_type,
            submit_docker_uri,
            submit_docker_digest,
            submit_policy_server_url,
            submit_eval_readme_url,
            submit_self_eval_url,
            submit_video_url,
            submit_release_url,
            submit_primary_email,
            submit_team_emails,
            submit_human_data,
            submit_heuristic_data,
            submit_rl_data,
            submit_other_data,
            submit_other_data_description,
            submit_comments,
            submit_rules,
            submit_public_release,
        ],
        outputs=submit_status,
        queue=False,
        api_name="submit_entry",
    )


def build_app():
    import gradio as gr

    if SUBMISSION_TOKEN:
        refresh_private_submission_cache()

    def refresh_main(query: str | None, status: str | None):
        choices = status_choices()
        if status not in choices:
            status = "All"
        return summary_html(), gr.update(choices=choices, value=status), display_leaderboard(query, status)

    countdown_widget = deadline_countdown_widget_html(compact=True)

    with gr.Blocks(title="BEHAVIOR-1K 2026 Challenge Leaderboard", css=APP_CSS, js=APP_JS, elem_id="b1k-app") as demo:
        gr.HTML(
            f"""
            <section class="b1k-hero">
              <div class="b1k-hero-header">
                <div>
                  <h1 class="b1k-title">BEHAVIOR-1K 2026 Challenge</h1>
                </div>
                <div class="hero-actions">
                  <a
                    class="registration-link"
                    href="https://forms.gle/Kf4ABLmDKbuK5Yhj6"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Registration Form
                  </a>
                  <a
                    class="portal-link"
                    href="https://behavior-1k-2026-challenge-leaderboard.hf.space/submit?v=validation-fields-3#submission-portal-top"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Submission Portal
                  </a>
                </div>
              </div>
              <div class="leaderboard-deadline-row">
                {countdown_widget}
              </div>
            </section>
            """
        )
        summary = gr.HTML(value=summary_html())

        with gr.Tab("2026 Leaderboard"):
            with gr.Row(elem_classes=["filter-panel"]):
                query = gr.Textbox(label="Search", placeholder="Team, method, or submission ID", scale=4)
                status = gr.Dropdown(label="Status", choices=status_choices(), value="All", scale=1)
                refresh = gr.Button("Refresh", elem_id="leaderboard-refresh", elem_classes=["refresh-button"])
            leaderboard = gr.Dataframe(
                value=display_leaderboard(None, "All"),
                headers=LEADERBOARD_DISPLAY_COLUMNS,
                datatype=LEADERBOARD_DATATYPES,
                column_widths=LEADERBOARD_COLUMN_WIDTHS,
                max_height=RESULT_TABLE_MAX_HEIGHT,
                interactive=False,
                wrap=True,
                elem_classes=["compact-table", "table-frame", "leaderboard-2026"],
            )
            query.change(fn=display_leaderboard, inputs=[query, status], outputs=leaderboard)
            status.change(fn=display_leaderboard, inputs=[query, status], outputs=leaderboard)
            refresh.click(fn=refresh_main, inputs=[query, status], outputs=[summary, status, leaderboard])

        with gr.Tab("2026 Per-Task Results", visible=False):
            with gr.Row(elem_classes=["filter-panel"]):
                selector = gr.Dropdown(label="Submission", choices=submission_choices(), value=None, scale=4)
                refresh_per_task = gr.Button("Refresh", elem_classes=["refresh-button"])
            per_task = gr.Dataframe(
                value=display_per_task(None),
                headers=PER_TASK_DISPLAY_COLUMNS,
                max_height=RESULT_TABLE_MAX_HEIGHT,
                interactive=False,
                wrap=True,
                elem_classes=["compact-table", "table-frame"],
            )
            selector.change(fn=display_per_task, inputs=selector, outputs=per_task)
            refresh_per_task.click(
                fn=lambda: (gr.update(choices=submission_choices(), value=None), display_per_task(None)),
                outputs=[selector, per_task],
            )

        with gr.Tab("2025 Leaderboard"):
            gr.HTML(
                """
                <div class="archive-note">
                  Final results from the 1st BEHAVIOR Challenge at NeurIPS 2025.
                  Public-validation scores are self-reported; held-out test scores
                  are verified by the organizers. SR denotes full-task success rate.
                  Missing verified scores are left blank.
                </div>
                """
            )
            with gr.Row(elem_classes=["filter-panel"]):
                query_2025 = gr.Textbox(label="Search", placeholder="Team or affiliation", scale=4)
                track_2025 = gr.Dropdown(
                    label="Track", choices=["All", "Standard", "Privileged"], value="All", scale=1
                )
            leaderboard_2025 = gr.Dataframe(
                value=display_2025_leaderboard(None, "All"),
                headers=LEADERBOARD_2025_DISPLAY_COLUMNS,
                datatype=LEADERBOARD_2025_DATATYPES,
                column_widths=LEADERBOARD_2025_COLUMN_WIDTHS,
                max_height=RESULT_TABLE_MAX_HEIGHT,
                interactive=False,
                wrap=True,
                elem_classes=["compact-table", "table-frame"],
            )
            query_2025.change(
                fn=display_2025_leaderboard,
                inputs=[query_2025, track_2025],
                outputs=leaderboard_2025,
            )
            track_2025.change(
                fn=display_2025_leaderboard,
                inputs=[query_2025, track_2025],
                outputs=leaderboard_2025,
            )

    with demo.route("Submit Portal", "/submit"):
        render_submission_portal(gr)

    return demo


if __name__ == "__main__":
    build_app().launch(ssr_mode=False)
```
