import base64
import csv
import hashlib
import html
import importlib
import io
import os
import time

import streamlit as st
import streamlit.components.v2 as components_v2
from PIL import Image

# Streamlit keeps imported modules cached between script reruns. Configuration
# must be refreshed before modules that import newly added settings.
import config as _config

_REQUIRED_CONFIG_NAMES = {
    "ALLOW_DEMO_MODE",
    "MAX_UPLOAD_MB",
    "MAX_VIDEO_SECONDS",
    "MAX_VIDEO_SOURCE_DIMENSION",
    "ROBOFLOW_API_URL",
    "ROBOFLOW_CLASSIFICATION_MODEL_ID",
    "ROBOFLOW_DETECTION_MODEL_ID",
}
if not all(hasattr(_config, name) for name in _REQUIRED_CONFIG_NAMES):
    _config = importlib.reload(_config)

import detection as _detection
if not all(
    hasattr(_detection, name)
    for name in ("ALLOW_DEMO_MODE", "ROBOFLOW_DETECTION_MODEL_ID")
):
    _detection = importlib.reload(_detection)

import classification as _classification
if not hasattr(_classification, "ROBOFLOW_CLASSIFICATION_MODEL_ID"):
    _classification = importlib.reload(_classification)
import evaluation as _evaluation
import roboflow_metrics as _roboflow_metrics
if getattr(
    _roboflow_metrics,
    "ROBOFLOW_DETECTION_MODEL_ID",
    None,
) != _config.ROBOFLOW_DETECTION_MODEL_ID:
    _roboflow_metrics = importlib.reload(_roboflow_metrics)
import tracking as _tracking
import video_processing as _video_processing

if os.getenv("MARINE_DEV_RELOAD", "").strip() == "1":
    importlib.reload(_config)
    importlib.reload(_detection)
    importlib.reload(_classification)
    importlib.reload(_evaluation)
    importlib.reload(_roboflow_metrics)
    importlib.reload(_tracking)
    importlib.reload(_video_processing)

from classification import classify_ship, get_classification_status
from config import FINE_CLASSES, MAX_UPLOAD_MB, MAX_VIDEO_SECONDS
from evaluation import (
    calculate_classification_metrics,
    calculate_tracking_metrics,
    read_csv_rows,
)
from roboflow_metrics import get_detection_metrics, get_display_metrics
from video_processing import process_video_bytes


TONNAGE_CLASS_OPTIONS = tuple(
    sorted(
        (label for labels in FINE_CLASSES.values() for label in labels),
        key=lambda label: int(label.removesuffix("톤급").replace(",", "")),
    )
)
CLASSIFICATION_EDITOR_FIELDS = ("sample_id", "true_label", "predicted_label")
TRACKING_EDITOR_FIELDS = ("video_id", "true_count", "predicted_count", "id_switches")


clickable_detection_image = components_v2.component(
    "clickable_detection_image",
    html='<div class="clickable-image-root"></div>',
    css="""
    .clickable-image-root { position: relative; width: 100%; line-height: 0; }
    .clickable-image-root img { width: 100%; height: auto; border-radius: 0; display: block; }
    .clickable-image-root button { position: absolute; z-index: 5; padding: 0; margin: 0;
        border: 0; background: transparent; cursor: pointer; box-sizing: border-box; }
    .clickable-image-root button:hover { background: rgba(126, 169, 205, .12); outline: 2px solid #8fb9dc; }
    """,
    js="""
    export default function(component) {
      const { data, parentElement, setStateValue } = component;
      const root = parentElement.querySelector('.clickable-image-root');
      root.replaceChildren();
      const image = document.createElement('img');
      image.src = data.image;
      root.appendChild(image);
      for (const box of data.boxes) {
        const button = document.createElement('button');
        button.type = 'button';
        button.title = `${box.object_class} 선택`;
        button.style.left = `${box.left}%`;
        button.style.top = `${box.top}%`;
        button.style.width = `${box.width}%`;
        button.style.height = `${box.height}%`;
        button.addEventListener('click', () => setStateValue('selected_box', box.index));
        root.appendChild(button);
      }
    }
    """,
)

from detection import (
    draw_predictions,
    format_json_output,
    get_model_status,
    pipeline_to_json,
    read_image_from_bytes,
    run_pipeline,
    summarize_predictions,
)

# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="해양 침투 객체 감시 시스템",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

:root {
    --ocean-deep: #061525;
    --ocean-mid: #0c2847;
    --ocean-light: #134074;
    --accent-teal: #00c9a7;
    --accent-cyan: #38bdf8;
    --accent-glow: rgba(0, 201, 167, 0.35);
    --card-bg: rgba(12, 40, 71, 0.55);
    --card-border: rgba(56, 189, 248, 0.18);
    --text-primary: #e8f4fc;
    --text-muted: #7da8c4;
    --success: #34d399;
    --warning: #fbbf24;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 201, 167, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(56, 189, 248, 0.08) 0%, transparent 50%),
        linear-gradient(180deg, #061525 0%, #0a1e35 50%, #061525 100%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

[data-testid="column"] {
    gap: 1rem;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(12, 40, 71, 0.9) 0%, rgba(6, 21, 37, 0.95) 100%);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 20px 28px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255,255,255,0.05);
}

.hero-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-cyan), var(--accent-teal));
    background-size: 200% 100%;
    animation: shimmer 4s ease infinite;
}

@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 201, 167, 0.12);
    border: 1px solid rgba(0, 201, 167, 0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 500;
    color: var(--accent-teal);
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

.hero-title-block {
    flex: 1;
    min-width: 240px;
}

.main-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    letter-spacing: -0.4px;
}

.sub-title {
    color: var(--text-muted);
    font-size: 0.88rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.5;
}

.hero-stats {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
    padding-top: 4px;
}

.hero-stat-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-muted);
    font-size: 0.82rem;
}

.hero-stat-item span {
    color: var(--accent-cyan);
    font-weight: 600;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(6, 21, 37, 0.6);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 6px;
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 9px 22px;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border: none;
    transition: all 0.25s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 201, 167, 0.2), rgba(56, 189, 248, 0.15)) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(0, 201, 167, 0.3) !important;
    box-shadow: 0 2px 12px rgba(0, 201, 167, 0.15);
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 8px;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Section Cards ── */
.section-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    margin-bottom: 16px;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 4px 0;
}

.section-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 0 0 14px 0;
}

.section-header-block {
    margin-bottom: 4px;
}

/* ── Video / Upload Area ── */
.video-placeholder {
    height: 380px;
    border: 2px dashed rgba(56, 189, 248, 0.25);
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background:
        radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.06) 0%, transparent 70%),
        rgba(6, 21, 37, 0.8);
    position: relative;
    overflow: hidden;
}

.video-placeholder::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(56, 189, 248, 0.02) 2px,
        rgba(56, 189, 248, 0.02) 4px
    );
    pointer-events: none;
}

.video-placeholder-icon {
    font-size: 2.5rem;
    opacity: 0.6;
}

.video-placeholder-text {
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 400;
}

.video-placeholder-sub {
    font-size: 0.75rem;
    color: rgba(125, 168, 196, 0.6);
}

/* ── Workflow (Roboflow-style) ── */
.workflow-panel {
    background: rgba(6, 21, 37, 0.75);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 14px 12px;
    min-height: 420px;
}

.workflow-panel-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 12px;
}

.workflow-block {
    background: rgba(12, 40, 71, 0.65);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 0.78rem;
    color: var(--text-primary);
    font-weight: 500;
}

.workflow-block.purple { border-left-color: #a855f7; }
.workflow-block.green { border-left-color: #22c55e; }
.workflow-block.orange { border-left-color: #f97316; }
.workflow-block.blue { border-left-color: #3b82f6; }
.workflow-block.cyan { border-left-color: #06b6d4; }

.workflow-connector {
    width: 2px;
    height: 10px;
    background: rgba(56, 189, 248, 0.2);
    margin: 0 auto 8px;
}

.io-panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 16px;
    min-height: 420px;
    display: flex;
    flex-direction: column;
}

.io-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
}

.io-panel-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
}

.io-panel-subtitle {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.media-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(6, 21, 37, 0.65);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 10px;
}

.media-thumb {
    width: 52px;
    height: 52px;
    border-radius: 8px;
    object-fit: cover;
    border: 1px solid rgba(56, 189, 248, 0.15);
    flex-shrink: 0;
}

.media-name {
    font-size: 0.82rem;
    color: var(--text-primary);
    word-break: break-all;
}

.media-meta {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 2px;
}

.output-visual-wrap {
    background: rgba(6, 21, 37, 0.55);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    padding: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    overflow: hidden;
}

.output-visual-wrap.empty {
    height: 220px;
    min-height: 220px;
    flex-direction: column;
    box-sizing: border-box;
}

.output-json-wrap {
    background: rgba(6, 21, 37, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    padding: 0;
    min-height: 0;
    max-height: 70vh;
    overflow: auto;
}

.detection-stats-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 12px;
}

.stat-chip {
    background: rgba(6, 21, 37, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}

.stat-chip-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.stat-chip-value {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2px;
}

.view-toggle label {
    min-width: 72px;
    text-align: center;
}

/* ── Dashboard Panel ── */
.dashboard-panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 18px;
    backdrop-filter: blur(12px);
    min-height: 380px;
    display: flex;
    flex-direction: column;
}

.dashboard-panel-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
    flex-shrink: 0;
}

.metric-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 10px;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(6, 21, 37, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 0;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(0, 201, 167, 0.25);
    box-shadow: 0 0 16px rgba(0, 201, 167, 0.08);
}

.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-value.accent { color: var(--accent-teal); }
.metric-value.cyan { color: var(--accent-cyan); }

.metric-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
}

.metric-card.compact {
    padding: 18px;
    text-align: center;
}

.metric-card.compact .metric-value {
    font-size: 1.6rem;
}

.metric-card.compact .metric-label {
    font-size: 0.78rem;
    text-transform: none;
    letter-spacing: 0;
}

.compact-divider {
    border: none;
    border-top: 1px solid rgba(56, 189, 248, 0.1);
    margin: 12px 0;
}

/* ── Status Badge ── */
.active-status {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.12), rgba(0, 201, 167, 0.08));
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    margin-top: auto;
    font-weight: 600;
    color: var(--success);
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 2s ease infinite;
    box-shadow: 0 0 8px var(--success);
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* ── Environment Badge ── */
.env-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 8px 16px;
    color: var(--accent-cyan);
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 14px;
}

.tab2-controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 6px;
    flex-wrap: wrap;
}

/* ── Object Tags ── */
.object-tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 4px;
}

.object-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(6, 21, 37, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.78rem;
    color: var(--text-muted);
}

.object-label {
    font-size: 0.72rem;
    color: #7da8c4;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}

.object-tag strong {
    color: var(--accent-teal);
    font-weight: 600;
}

/* ── Model Status Grid ── */
.tab3-layout {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 8px 0 16px;
}

.model-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.model-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 201, 167, 0.1);
}

.model-card-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.model-card-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
}

.model-card-value.active {
    color: var(--success);
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 16px 0 8px;
    color: rgba(125, 168, 196, 0.5);
    font-size: 0.75rem;
    border-top: 1px solid rgba(56, 189, 248, 0.08);
    margin-top: 20px;
}

/* ── Streamlit Overrides ── */
[data-testid="stMetric"] {
    background: rgba(6, 21, 37, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 12px;
    padding: 12px 16px;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.4rem !important;
}

[data-testid="stFileUploader"] {
    background: rgba(6, 21, 37, 0.55);
    border: 2px dashed rgba(56, 189, 248, 0.28);
    border-radius: 12px;
    padding: 0;
    margin-top: 48px !important;
    margin-bottom: 10px !important;
    height: 220px;
    min-height: 220px;
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
    position: relative;
}

[data-testid="stFileUploader"] section {
    padding: 0 !important;
    height: 216px !important;
    min-height: 216px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 0 !important;
    border-radius: 10px !important;
    background: transparent !important;
    box-sizing: border-box;
    position: relative;
    flex-direction: column;
    gap: 10px;
}

[data-testid="stFileUploader"] section::before {
    content: '📁';
    font-size: 3.2rem;
    line-height: 1;
    filter: saturate(0.7);
    opacity: 0.55;
    pointer-events: none;
    position: absolute;
    left: 50%;
    top: calc(50% - 14px);
    transform: translate(-50%, -50%);
}

[data-testid="stFileUploader"] section::after {
    content: 'JPG · PNG · MP4 · AVI · MOV';
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 500;
    line-height: 1;
    opacity: 0.65;
    pointer-events: none;
    position: absolute;
    left: 50%;
    top: calc(50% + 30px);
    transform: translate(-50%, -50%);
    white-space: nowrap;
}

[data-testid="stFileUploader"] section > div {
    display: none !important;
}

[data-testid="stFileUploader"] section button {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    border: 0 !important;
    opacity: 0 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    background: transparent !important;
}

[data-testid="stFileUploaderFile"] {
    position: absolute !important;
    top: 12px;
    right: 12px;
    z-index: 5;
    width: auto !important;
    min-width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
}

[data-testid="stFileUploaderFile"] > div:not(:last-child) {
    display: none !important;
}

[data-testid="stFileUploaderDeleteBtn"],
[data-testid="stFileUploaderFile"] button {
    position: static !important;
    width: 30px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 !important;
    opacity: 1 !important;
    color: #e8f4fc !important;
    background: rgba(239, 68, 68, 0.18) !important;
    border: 1px solid rgba(248, 113, 113, 0.45) !important;
    border-radius: 50% !important;
    box-shadow: none !important;
}

[data-testid="stFileUploaderDeleteBtn"] svg,
[data-testid="stFileUploaderFile"] button svg {
    display: none !important;
}

[data-testid="stFileUploaderDeleteBtn"]::before,
[data-testid="stFileUploaderFile"] button::before {
    content: '×';
    font-size: 1.25rem;
    line-height: 1;
}

.st-key-live_upload_card,
.st-key-intrusion_upload_card {
    position: relative;
}

.st-key-live_upload_card .media-card,
.st-key-intrusion_upload_card .media-card {
    padding-right: 48px;
}

.st-key-live_upload_card [data-testid="stButton"],
.st-key-intrusion_upload_card [data-testid="stButton"] {
    position: absolute;
    top: 50%;
    right: 12px;
    transform: translateY(-50%);
    z-index: 10;
}

.st-key-live_upload_card [data-testid="stButton"] button,
.st-key-intrusion_upload_card [data-testid="stButton"] button {
    width: 30px;
    height: 30px;
    min-height: 30px;
    padding: 0;
    color: #fca5a5;
    font-size: 1.2rem;
    line-height: 1;
    background: rgba(239, 68, 68, 0.16);
    border: 1px solid rgba(248, 113, 113, 0.45);
    border-radius: 50%;
}

.st-key-ship_result_box,
.st-key-intrusion_result_box {
    min-height: 420px;
    overflow: auto;
    box-sizing: border-box;
}

.st-key-ship_result_box [data-testid="stVideo"] {
    width: 100%;
    max-height: 420px;
}

.st-key-ship_result_box [data-testid="stVideo"] video {
    width: 100%;
    max-height: 398px;
    object-fit: contain;
}

.st-key-ship_result_box [data-testid="stImage"] img,
.st-key-intrusion_result_box [data-testid="stImage"] img {
    height: 398px !important;
    max-height: 398px !important;
    object-fit: contain !important;
}

.st-key-intrusion_result_box [data-testid="stVideo"],
.st-key-intrusion_result_box [data-testid="stVideo"] video {
    width: 100%;
    max-height: 398px;
    object-fit: contain;
}

.result-file-bar { color: var(--text-muted); font-size: 0.78rem; margin: 2px 0 8px; }
.evaluation-grid { display: flex; flex-direction: column; gap: 12px; margin-top: 14px; }
.evaluation-card { display: grid; grid-template-columns: 170px 150px 1fr; align-items: center; gap: 18px; background: rgba(6, 21, 37, 0.72); border: 1px solid rgba(56, 189, 248, 0.16); border-radius: 12px; padding: 18px 22px; min-height: 92px; }
.evaluation-name { color: var(--accent-cyan); font-size: 0.9rem; font-weight: 700; }
.evaluation-value { color: var(--text-primary); font-size: 1.25rem; font-weight: 700; }
.evaluation-desc { color: var(--text-muted); font-size: 0.78rem; line-height: 1.55; }
@media (max-width: 900px) { .evaluation-card { grid-template-columns: 1fr; gap: 8px; } }

.st-key-replace_live_upload button,
.st-key-replace_intrusion_upload button {
    min-height: 34px !important;
    padding: 5px 14px !important;
    color: var(--accent-cyan) !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.28) !important;
    border-radius: 9px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

.st-key-replace_live_upload button:hover,
.st-key-replace_intrusion_upload button:hover {
    color: var(--text-primary) !important;
    background: rgba(0, 201, 167, 0.12) !important;
    border-color: rgba(0, 201, 167, 0.42) !important;
}

.dropzone-hint {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 6px;
    text-align: center;
}

.mode-badge {
    display: inline-block;
    background: rgba(0, 201, 167, 0.12);
    border: 1px solid rgba(0, 201, 167, 0.25);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.68rem;
    color: var(--accent-teal);
    margin-left: 8px;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 201, 167, 0.4);
}

.stAlert {
    background: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 12px !important;
    color: var(--text-muted) !important;
}

hr {
    border-color: rgba(56, 189, 248, 0.1) !important;
    margin: 12px 0 !important;
}

[data-testid="stRadio"] > div {
    gap: 8px;
}

[data-testid="stRadio"] label {
    background: rgba(6, 21, 37, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 10px !important;
    padding: 7px 14px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stRadio"] label:hover {
    border-color: rgba(0, 201, 167, 0.3) !important;
}

.st-key-intrusion_environment [data-testid="stRadio"] > div {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.st-key-intrusion_environment [data-testid="stRadio"] label {
    color: #e8f4fc !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
}

.st-key-intrusion_environment [data-testid="stRadio"] label p,
.st-key-intrusion_environment [data-testid="stRadio"] label span {
    color: #e8f4fc !important;
    font-weight: 600 !important;
}

.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) {
    color: #67e8f9 !important;
    background: rgba(56, 189, 248, 0.14) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.1);
}

.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) p,
.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) span {
    color: #67e8f9 !important;
}

[data-testid="stImage"] img {
    border-radius: 8px;
    border: none;
    max-height: none !important;
    object-fit: contain;
    width: 100%;
}

[data-testid="stVideo"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--card-border);
    max-height: 380px;
}

[data-testid="stVideo"] video {
    max-height: 380px;
}

h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.stSubheader, [data-testid="stSubheader"] {
    color: var(--text-primary) !important;
}

/* ── Refined maritime console theme ────────────────────────── */
:root {
    --ocean-deep: #07111d;
    --ocean-mid: #0d1826;
    --ocean-light: #142235;
    --accent-teal: #9bbcd6;
    --accent-cyan: #78a6cd;
    --accent-glow: transparent;
    --card-bg: #0d1826;
    --card-border: #233247;
    --text-primary: #f2f5f8;
    --text-muted: #8997a9;
    --success: #7fc6a4;
    --warning: #d9b879;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    letter-spacing: -0.015em;
}

.stApp {
    background: #07111d;
    color: var(--text-primary);
}

.block-container {
    max-width: 1320px;
    padding: 28px 32px 20px;
}

[data-testid="stVerticalBlock"] { gap: .65rem !important; }

.hero-header {
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--card-border);
    border-radius: 0;
    padding: 12px 2px 24px;
    margin-bottom: 22px;
    box-shadow: none;
    overflow: visible;
}

.hero-header::before { display: none; }

.hero-badge {
    gap: 8px;
    margin: 0 0 13px;
    padding: 0;
    background: transparent;
    border: 0;
    border-radius: 0;
    color: #7990a8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .16em;
}

.hero-badge::before {
    content: '';
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--success);
}

.hero-top { align-items: flex-end; gap: 28px; }

.main-title {
    color: #f5f7fa;
    font-size: clamp(1.55rem, 2.3vw, 2.15rem);
    font-weight: 600;
    letter-spacing: -.045em;
    line-height: 1.25;
}

.sub-title {
    margin-top: 8px;
    color: var(--text-muted);
    font-size: .8rem;
}

.hero-stats {
    gap: 0;
    padding: 0 0 3px;
    flex-wrap: nowrap;
}

.hero-stat-item {
    gap: 5px;
    padding: 0 13px;
    border-right: 1px solid var(--card-border);
    color: #708095;
    font-size: .73rem;
    white-space: nowrap;
}

.hero-stat-item:first-child { padding-left: 0; }
.hero-stat-item:last-child { padding-right: 0; border-right: 0; }
.hero-stat-item span { color: #c5d0db; font-weight: 500; }

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    min-height: 42px;
    margin-bottom: 20px;
    padding: 0;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--card-border);
    border-radius: 0;
}

.stTabs [data-baseweb="tab"] {
    min-height: 42px;
    padding: 0 2px 13px;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0;
    color: #718096;
    font-size: .84rem;
    font-weight: 500;
    transition: color .16s ease, border-color .16s ease;
}

.stTabs [data-baseweb="tab"]:hover { color: #c8d1da; }

.stTabs [aria-selected="true"] {
    background: transparent !important;
    border-color: #89abc8 !important;
    box-shadow: none;
    color: #eef3f7 !important;
}

.stTabs [data-baseweb="tab-panel"] { padding-top: 4px; }

[data-testid="stFileUploader"] {
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] section {
    min-height: 230px;
    padding: 52px 28px 38px !important;
    background: #0a1522 !important;
    border: 1px dashed #34465b !important;
    border-radius: 10px !important;
    transition: background .16s ease, border-color .16s ease;
}

[data-testid="stFileUploader"] section:hover {
    background: #0d1927 !important;
    border-color: #6684a0 !important;
}

[data-testid="stFileUploader"] section::before {
    content: 'MEDIA INPUT';
    top: 38px;
    color: #67788c;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .16em;
}

[data-testid="stFileUploader"] section::after {
    content: 'JPG, PNG, MP4, AVI, MOV';
    bottom: 30px;
    color: #536275;
    font-size: 10px;
    letter-spacing: .07em;
}

[data-testid="stFileUploader"] section button,
.stButton button {
    min-height: 36px !important;
    padding: 7px 15px !important;
    background: #152336 !important;
    border: 1px solid #304259 !important;
    border-radius: 7px !important;
    box-shadow: none !important;
    color: #d9e2ea !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] section button:hover,
.stButton button:hover {
    background: #1a2b40 !important;
    border-color: #58738e !important;
    color: #ffffff !important;
}

.output-visual-wrap {
    background: #0a1522;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    box-shadow: none;
}

.output-visual-wrap.empty {
    min-height: 360px;
    background: #0a1522;
    border: 1px solid var(--card-border);
}

.video-placeholder-text {
    color: #b7c3cf;
    font-size: .9rem;
    font-weight: 500;
}

.video-placeholder-sub { color: #637388; font-size: .74rem; }

.dashboard-panel {
    padding: 20px;
    background: #0d1826;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    box-shadow: none;
}

.dashboard-panel-title {
    margin-bottom: 18px;
    padding-bottom: 13px;
    border-bottom: 1px solid var(--card-border);
    color: #c8d2dc;
    font-size: .76rem;
    font-weight: 600;
    letter-spacing: .04em;
}

.metric-grid-2 { gap: 8px; }

.metric-card {
    padding: 13px 14px;
    background: #101d2c;
    border: 1px solid #223249;
    border-radius: 8px;
    box-shadow: none;
}

.metric-card:hover {
    background: #101d2c;
    border-color: #344861;
    transform: none;
    box-shadow: none;
}

.metric-label {
    color: #748399;
    font-size: .69rem;
    font-weight: 500;
    letter-spacing: .01em;
}

.metric-value {
    margin-top: 7px;
    color: #eef2f6;
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: -.03em;
}

.metric-value.accent,
.metric-value.cyan { color: #a8c2d8; }

.compact-divider,
hr { border-color: #223047 !important; margin: 13px 0 !important; }

.object-label {
    margin-bottom: 9px;
    color: #78889d;
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: .04em;
}

.object-tags { gap: 6px; }

.object-tag {
    padding: 6px 9px;
    background: #111f30;
    border: 1px solid #263950;
    border-radius: 6px;
    color: #8e9db0;
    font-size: .7rem;
}

.object-tag strong { color: #dce4eb; font-weight: 600; }

.active-status {
    margin-top: 13px;
    padding: 9px 11px;
    background: rgba(127, 198, 164, .06);
    border: 1px solid rgba(127, 198, 164, .2);
    border-radius: 7px;
    color: #88bca3;
    font-size: .68rem;
    letter-spacing: .06em;
}

.status-dot {
    width: 5px;
    height: 5px;
    background: var(--success);
    box-shadow: none;
    animation: none;
}

[data-testid="stRadio"] label,
.st-key-intrusion_environment [data-testid="stRadio"] label {
    padding: 7px 12px !important;
    background: transparent !important;
    border: 1px solid #2a3a4f !important;
    border-radius: 7px !important;
    color: #9eacbb !important;
    font-size: .76rem !important;
}

[data-testid="stRadio"] label:hover,
.st-key-intrusion_environment [data-testid="stRadio"] label:hover {
    background: #0f1d2d !important;
    border-color: #516a84 !important;
}

[data-testid="stRadio"] label:has(input:checked),
.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) {
    background: #17273a !important;
    border-color: #6f91af !important;
    box-shadow: none !important;
}

[data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stRadio"] label:has(input:checked) span,
.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) p,
.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) span {
    color: #e4ebf1 !important;
    font-weight: 500 !important;
}

.result-file-bar {
    margin: 4px 0 10px;
    color: #8291a4;
    font-size: .75rem;
}

.st-key-ship_result_box,
.st-key-intrusion_result_box,
[data-testid="stVideo"] {
    background: #08121e;
    border-color: var(--card-border) !important;
    border-radius: 9px !important;
    box-shadow: none;
}

[data-testid="stImage"] img { border-radius: 7px; }

[data-testid="stMetric"] {
    padding: 12px 14px;
    background: #0f1c2b;
    border: 1px solid #223249;
    border-radius: 8px;
}

[data-testid="stMetricLabel"] { color: #7d8da1 !important; }
[data-testid="stMetricValue"] { color: #eef2f6 !important; font-weight: 600 !important; }

.stAlert {
    background: #0f1c2b !important;
    border: 1px solid #293b51 !important;
    border-radius: 8px !important;
    color: #aeb9c5 !important;
}

/* ── Processing state ── */
.processing-state {
    position: relative;
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr) auto;
    align-items: center;
    gap: 15px;
    min-height: 86px;
    margin: 2px 0 12px;
    padding: 17px 20px;
    overflow: hidden;
    background: #0b1724;
    border: 1px solid #2a3a4f;
    border-radius: 2px;
}

.processing-state::before {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    width: 42px;
    height: 1px;
    background: #8aaac4;
}

.processing-indicator {
    position: relative;
    width: 28px;
    height: 28px;
    border: 1px solid #30445b;
    border-radius: 50%;
}

.processing-indicator::before {
    content: '';
    position: absolute;
    inset: 4px;
    border: 2px solid transparent;
    border-top-color: #9bbbd3;
    border-right-color: #607f9b;
    border-radius: 50%;
    animation: processing-rotate .9s linear infinite;
}

.processing-copy { min-width: 0; }

.processing-kicker {
    margin-bottom: 4px;
    color: #6f8499;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .16em;
}

.processing-title {
    color: #edf2f6;
    font-size: .86rem;
    font-weight: 600;
    letter-spacing: -.015em;
}

.processing-desc {
    margin-top: 4px;
    color: #7e8da0;
    font-size: .69rem;
}

.processing-meta {
    color: #64778b;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: .11em;
    white-space: nowrap;
}

.processing-track {
    position: absolute;
    right: 0;
    bottom: 0;
    left: 0;
    height: 2px;
    overflow: hidden;
    background: #132236;
}

.processing-track::after {
    content: '';
    position: absolute;
    inset: 0;
    width: 28%;
    background: linear-gradient(90deg, transparent, #7399b8, transparent);
    animation: processing-scan 1.8s ease-in-out infinite;
}

@keyframes processing-rotate { to { transform: rotate(360deg); } }
@keyframes processing-scan {
    from { transform: translateX(-110%); }
    to { transform: translateX(390%); }
}

[data-testid="stSpinner"] {
    padding: 12px 14px !important;
    background: #0b1724 !important;
    border: 1px solid #2a3a4f !important;
    border-radius: 2px !important;
}

[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span { color: #dfe7ee !important; }

.evaluation-grid { gap: 8px; margin-top: 12px; }

.evaluation-card {
    grid-template-columns: minmax(140px, 170px) minmax(120px, 145px) 1fr;
    gap: 20px;
    min-height: 78px;
    padding: 16px 20px;
    background: #0d1826;
    border: 1px solid var(--card-border);
    border-radius: 8px;
}

.evaluation-name {
    color: #91abc2;
    font-size: .78rem;
    font-weight: 600;
}

.evaluation-value {
    color: #f0f4f7;
    font-size: 1.1rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

.evaluation-desc { color: #78879a; font-size: .74rem; }

.dropzone-hint { color: #68788c; font-size: .7rem; }

.app-footer {
    margin-top: 28px;
    padding: 18px 0 6px;
    border-top: 1px solid #1d2b3d;
    color: #526176;
    font-size: .65rem;
    letter-spacing: .08em;
}

@media (max-width: 900px) {
    .block-container { padding: 18px 18px 14px; }
    .hero-top { align-items: flex-start; }
    .hero-stats { width: 100%; overflow-x: auto; }
    .evaluation-card { grid-template-columns: 1fr 1fr; gap: 7px 14px; }
    .evaluation-desc { grid-column: 1 / -1; }
}

@media (max-width: 640px) {
    .hero-header { padding-top: 4px; }
    .hero-stat-item { padding: 0 9px; }
    .stTabs [data-baseweb="tab-list"] { gap: 18px; }
    .output-visual-wrap.empty { min-height: 280px; padding: 24px; }
    .evaluation-card { grid-template-columns: 1fr; }
    .evaluation-desc { grid-column: auto; }
    .processing-state { grid-template-columns: 30px 1fr; padding: 15px 16px; }
    .processing-meta { display: none; }
}

@media (prefers-reduced-motion: reduce) {
    .processing-indicator::before,
    .processing-track::after { animation: none; }
}

/* ── Technical edge & ambient navigation field ────────────── */
html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #07111d !important;
}

[data-testid="stMain"] {
    background-image:
        linear-gradient(rgba(126, 159, 188, .038) 1px, transparent 1px),
        linear-gradient(90deg, rgba(126, 159, 188, .038) 1px, transparent 1px),
        linear-gradient(rgba(126, 159, 188, .016) 1px, transparent 1px),
        linear-gradient(90deg, rgba(126, 159, 188, .016) 1px, transparent 1px),
        radial-gradient(circle at 82% 8%, rgba(71, 112, 148, .10), transparent 34%),
        radial-gradient(circle at 18% 88%, rgba(35, 72, 101, .08), transparent 30%);
    background-size: 48px 48px, 48px 48px, 12px 12px, 12px 12px, auto, auto;
}

.section-card,
.workflow-panel,
.io-panel,
.media-card,
.output-visual-wrap,
.output-visual-wrap.empty,
.dashboard-panel,
.metric-card,
.object-tag,
.active-status,
.evaluation-card,
[data-testid="stMetric"],
.stAlert,
[data-testid="stVideo"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section button,
.stButton button,
.stDownloadButton button,
[data-testid="stRadio"] label,
.st-key-intrusion_environment [data-testid="stRadio"] label,
.st-key-ship_result_box,
.st-key-intrusion_result_box {
    border-radius: 0 !important;
}

[data-testid="stImage"] img,
.clickable-image-root img,
.media-thumb { border-radius: 1px !important; }

.dashboard-panel,
.evaluation-card,
.output-visual-wrap,
.st-key-ship_result_box,
.st-key-intrusion_result_box {
    position: relative;
}

.dashboard-panel::before,
.evaluation-card::before {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    width: 34px;
    height: 1px;
    background: #789ab8;
    pointer-events: none;
}

.hero-header {
    border-bottom-color: #304158;
}

.hero-header::after {
    content: '';
    position: absolute;
    right: 2px;
    bottom: -18px;
    color: #435369;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: .12em;
}

.stDownloadButton button {
    min-height: 38px;
    background: #101f30 !important;
    border: 1px solid #30445b !important;
    border-radius: 0 !important;
    color: #dce7ee !important;
    font-size: .78rem !important;
    box-shadow: none !important;
}

.stDownloadButton button:hover {
    background: #162a3e !important;
    border-color: #6f91af !important;
}

.stTabs [aria-selected="true"] {
    border-bottom-color: #9bb8cf !important;
}

/* ── User-provided quantitative evaluation ──────────────── */
.quant-eval-intro {
    margin: 24px 0 12px;
    padding: 18px 20px;
    background: #0a1623;
    border: 1px solid #2a3d52;
    border-left: 3px solid #789ab8;
}

.quant-eval-kicker {
    margin-bottom: 7px;
    color: #789ab8;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .14em;
}

.quant-eval-title {
    margin-bottom: 7px;
    color: #eef3f7;
    font-size: 1.05rem;
    font-weight: 650;
}

.quant-eval-copy {
    max-width: 920px;
    color: #9aa9b8;
    font-size: .88rem;
    line-height: 1.7;
}

.st-key-offline_evaluation_type [data-testid="stRadio"] {
    margin-bottom: 8px;
}

.st-key-offline_evaluation_type [data-testid="stRadio"] > label {
    color: #aab8c6 !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
}

.st-key-offline_evaluation_type [data-testid="stRadio"] > div {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 220px));
    gap: 8px;
}

.st-key-offline_evaluation_type [data-testid="stRadio"] label[data-baseweb="radio"] {
    min-height: 42px;
    padding: 9px 14px !important;
    background: #0d1927 !important;
    border: 1px solid #30445b !important;
    color: #aebdca !important;
}

.st-key-offline_evaluation_type [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: #142537 !important;
    border-color: #7d9cb8 !important;
    color: #f2f6f9 !important;
    box-shadow: inset 3px 0 0 #7d9cb8;
}

.st-key-offline_evaluation_type [data-testid="stRadio"] label[data-baseweb="radio"] p,
.st-key-offline_evaluation_type [data-testid="stRadio"] label[data-baseweb="radio"] span {
    color: inherit !important;
    font-size: .86rem !important;
    font-weight: 600 !important;
}

.quant-eval-guide {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin: 10px 0 12px;
    border-top: 1px solid #26394d;
    border-bottom: 1px solid #26394d;
    background: #091522;
}

.quant-eval-guide-item {
    min-height: 76px;
    padding: 14px 16px;
}

.quant-eval-guide-item + .quant-eval-guide-item {
    border-left: 1px solid #26394d;
}

.quant-eval-guide-label {
    display: block;
    margin-bottom: 8px;
    color: #708399;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .1em;
}

.quant-eval-guide-value {
    color: #cbd6df;
    font-size: .86rem;
    line-height: 1.6;
}

.quant-eval-guide-value code {
    padding: 2px 5px;
    background: #111f2e;
    border: 1px solid #2a4055;
    border-radius: 0;
    color: #dbe7ef;
    font-size: .78rem;
}

.quant-eval-result-title {
    margin: 18px 0 9px;
    padding-bottom: 8px;
    border-bottom: 1px solid #26394d;
    color: #cfd9e1;
    font-size: .82rem;
    font-weight: 650;
    letter-spacing: .04em;
}

.st-key-classification_eval_editor_import [data-testid="stFileUploader"],
.st-key-tracking_eval_editor_import [data-testid="stFileUploader"] {
    height: auto !important;
    min-height: 0 !important;
    margin-top: 10px !important;
    margin-bottom: 12px !important;
    overflow: visible !important;
}

.st-key-classification_eval_editor_import [data-testid="stFileUploader"] > label,
.st-key-tracking_eval_editor_import [data-testid="stFileUploader"] > label {
    color: #aab8c6 !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
}

.st-key-classification_eval_editor_import [data-testid="stFileUploader"] section,
.st-key-tracking_eval_editor_import [data-testid="stFileUploader"] section {
    height: 126px !important;
    min-height: 126px !important;
    padding: 0 !important;
    background: #091522 !important;
    border-color: #3a526a !important;
}

.st-key-classification_eval_editor_import [data-testid="stFileUploader"] section::before,
.st-key-tracking_eval_editor_import [data-testid="stFileUploader"] section::before {
    content: 'OPTIONAL CSV IMPORT';
    top: 35px;
    bottom: auto;
    color: #8ba4ba;
    font-size: .72rem;
    letter-spacing: .16em;
    transform: translateX(-50%);
}

.quant-editor-heading {
    margin: 18px 0 8px;
    padding-left: 12px;
    border-left: 2px solid #789ab8;
}

.quant-editor-title {
    color: #e5ebf0;
    font-size: .92rem;
    font-weight: 650;
}

.quant-editor-desc {
    margin-top: 4px;
    color: #7f90a2;
    font-size: .78rem;
    line-height: 1.55;
}

[class*="st-key-classification_eval_editor_"] [data-testid="stDataFrame"],
[class*="st-key-tracking_eval_editor_"] [data-testid="stDataFrame"] {
    border: 1px solid #30445b !important;
    border-radius: 0 !important;
}

.quant-editor-status {
    min-height: 38px;
    display: flex;
    align-items: center;
    padding: 8px 12px;
    background: #0a1623;
    border: 1px solid #293d52;
    color: #90a2b4;
    font-size: .78rem;
}

.quant-editor-status strong {
    margin: 0 4px;
    color: #e2eaf0;
    font-size: .9rem;
}

.st-key-classification_eval_editor_import [data-testid="stFileUploader"] section::after,
.st-key-tracking_eval_editor_import [data-testid="stFileUploader"] section::after {
    content: '클릭하여 파일 선택 · CSV / UTF-8';
    top: 76px;
    bottom: auto;
    color: #73869a;
    font-size: .76rem;
    letter-spacing: .02em;
    transform: translateX(-50%);
}

@media (max-width: 640px) {
    .quant-eval-intro { padding: 16px; }
    .st-key-offline_evaluation_type [data-testid="stRadio"] > div,
    .quant-eval-guide { grid-template-columns: 1fr; }
    .quant-eval-guide-item + .quant-eval-guide-item {
        border-top: 1px solid #26394d;
        border-left: 0;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

def render_workflow_sidebar():
    st.markdown("""
    <div class="workflow-panel">
        <div class="workflow-panel-title">Pipeline</div>
        <div class="workflow-block purple">Input · Media Upload</div>
        <div class="workflow-connector"></div>
        <div class="workflow-block purple">YOLOv26 Object Detection</div>
        <div class="workflow-block" style="margin:-4px 0 8px 12px;font-size:0.68rem;color:#7da8c4;border:none;background:transparent;padding:0;">
            함정 · 드론 · 사람
        </div>
        <div class="workflow-connector"></div>
        <div class="workflow-block green">Dynamic Crop</div>
        <div class="workflow-connector"></div>
        <div class="workflow-block orange">ViT Classification</div>
        <div class="workflow-block" style="margin:-4px 0 8px 12px;font-size:0.68rem;color:#7da8c4;border:none;background:transparent;padding:0;">
            함정 톤급 세부 분류
        </div>
        <div class="workflow-connector"></div>
        <div class="workflow-block blue">Bounding Box Visualization</div>
        <div class="workflow-connector"></div>
        <div class="workflow-block">Output · Visual / JSON</div>
    </div>
    """, unsafe_allow_html=True)


def clear_uploaded_file(widget_key: str):
    st.session_state.pop(widget_key, None)
    st.session_state.pop(f"{widget_key}_saved", None)
    st.session_state.pop("processed_upload_key", None)
    st.session_state.pop("processed_upload_result", None)


def get_upload_cache_key(uploaded_file, task: str) -> str:
    content_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    return ":".join(
        (
            "processing-v2",
            task,
            str(uploaded_file.name),
            str(uploaded_file.type or ""),
            content_hash,
        )
    )


def render_clickable_detections(image: Image.Image, predictions: list) -> int | None:
    """이미지 위 bbox를 클릭 가능한 컴포넌트로 렌더링한다."""
    annotated = draw_predictions(image, predictions)
    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    width, height = image.size
    boxes = []
    for index, pred in enumerate(predictions):
        boxes.append({
            "index": index,
            "object_class": pred.object_class,
            "left": pred.x / width * 100,
            "top": pred.y / height * 100,
            "width": pred.width / width * 100,
            "height": pred.height / height * 100,
        })
    selection = clickable_detection_image(
        key="detection_bbox_selector",
        data={"image": f"data:image/jpeg;base64,{encoded}", "boxes": boxes},
        default={"selected_box": None},
        on_selected_box_change=lambda: None,
    )
    return selection.selected_box


@st.cache_data(show_spinner="선택한 함정의 톤급을 분류하고 있습니다...")
def classify_crop_bytes(crop_bytes: bytes) -> tuple[str, float]:
    crop = read_image_from_bytes(crop_bytes)
    return classify_ship(crop)


def render_selected_crop(image: Image.Image, predictions: list, selected_index: int | None) -> None:
    if selected_index is None:
        st.info("이미지의 바운딩 박스를 클릭하면 해당 영역의 crop과 분류 결과가 표시됩니다.")
        return
    try:
        selected_index = int(selected_index)
        prediction = predictions[selected_index]
    except (TypeError, ValueError, IndexError):
        st.warning("선택한 바운딩 박스를 찾을 수 없습니다. 다시 선택해 주세요.")
        return

    crop = image.crop((prediction.x, prediction.y, prediction.x2, prediction.y2))
    st.markdown('<div class="section-title">선택 영역 Classification</div>', unsafe_allow_html=True)
    crop_col, result_col = st.columns([1.2, 2], gap="medium")
    with crop_col:
        st.image(crop, caption=f"Crop · {prediction.object_class}", use_container_width=True)
    with result_col:
        st.metric("탐지 객체", prediction.object_class)
        st.metric("Detection Confidence", f"{prediction.detection_confidence:.2f}")
        if prediction.object_class != "ship":
            st.warning("함정 톤급 Classification 모델은 함정 바운딩 박스에만 적용됩니다.")
            return
        crop_buffer = io.BytesIO()
        crop.save(crop_buffer, format="PNG")
        try:
            class_name, confidence = classify_crop_bytes(crop_buffer.getvalue())
            st.metric("함정 톤급", class_name)
            st.metric("Classification Confidence", f"{confidence:.4f}")
        except Exception as exc:
            st.error(f"Classification 처리 중 오류: {exc}")


def render_media_card(filename: str, meta: str, thumbnail: Image.Image, uploader_key: str):
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    with st.container(key=f"{uploader_key}_card"):
        safe_filename = html.escape(str(filename))
        safe_meta = html.escape(str(meta))
        st.markdown(f"""
        <div class="media-card">
            <img class="media-thumb" src="data:image/jpeg;base64,{encoded}" alt="thumbnail">
            <div>
                <div class="media-name">{safe_filename}</div>
                <div class="media-meta">{safe_meta}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "×",
            key=f"clear_{uploader_key}",
            help="업로드 파일 지우기",
            on_click=clear_uploaded_file,
            args=(uploader_key,),
        )


def render_detection_stats(summary: dict, mode: str):
    counts = summary["size_counts"]
    st.markdown(f"""
    <div class="detection-stats-bar">
        <div class="stat-chip">
            <div class="stat-chip-label">Total</div>
            <div class="stat-chip-value">{summary["total"]}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">대형</div>
            <div class="stat-chip-value">{counts.get("대형", 0)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">중형</div>
            <div class="stat-chip-value">{counts.get("중형", 0)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">소형</div>
            <div class="stat-chip-value">{counts.get("소형", 0)}</div>
        </div>
    </div>
    <div class="dropzone-hint">처리 모드: {mode} · 탐지 {summary["avg_detection_confidence"]} · 분류 {summary["avg_classification_confidence"]}</div>
    """, unsafe_allow_html=True)


def render_object_detection_stats(summary: dict, mode: str):
    counts = summary.get("object_counts", {})
    st.markdown(f"""
    <div class="detection-stats-bar">
        <div class="stat-chip">
            <div class="stat-chip-label">Total</div>
            <div class="stat-chip-value">{summary["total"]}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Ship</div>
            <div class="stat-chip-value">{counts.get("ship", 0)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Drone</div>
            <div class="stat-chip-value">{counts.get("drone", 0)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Person</div>
            <div class="stat-chip-value">{counts.get("person", 0)}</div>
        </div>
    </div>
    <div class="dropzone-hint">처리 모드: {mode} · 평균 탐지 Confidence {summary["avg_detection_confidence"]}</div>
    """, unsafe_allow_html=True)


def process_upload(
    uploaded_file,
    task: str = "detection",
    environment: str | None = None,
    progress_callback=None,
):
    data = uploaded_file.getvalue()
    upload_size_mb = len(data) / (1024 * 1024)
    if upload_size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"파일 크기는 최대 {MAX_UPLOAD_MB}MB까지 지원합니다. "
            f"현재 파일은 {upload_size_mb:.1f}MB입니다."
        )
    name = uploaded_file.name
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

    image_suffixes = {".jpg", ".jpeg", ".png"}
    video_suffixes = {".mp4", ".avi", ".mov"}
    mime_type = uploaded_file.type or ""

    if mime_type.startswith("image") or suffix in image_suffixes:
        image = read_image_from_bytes(data)
        frame_index = None
        media_meta = f"Image · {image.size[0]} x {image.size[1]}"
        pipeline_result = run_pipeline(image, task=task)
        predictions = pipeline_result.predictions
        summary = summarize_predictions(predictions)
        annotated = draw_predictions(image, predictions)
        video_bytes = None
        analyzed_frames = None
    elif mime_type.startswith("video") or suffix in video_suffixes:
        (
            video_bytes,
            image,
            pipeline_result,
            predictions,
            frame_count,
            analyzed_frames,
            summary,
            output_fps,
            preview_frame_index,
        ) = process_video_bytes(
            data,
            suffix or ".mp4",
            task,
            _progress_callback=progress_callback,
        )
        frame_index = preview_frame_index
        media_meta = (
            f"Video · {frame_count} frames · "
            f"{analyzed_frames} analyzed · {output_fps:.1f} fps output"
        )
        annotated = None
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    json_payload = pipeline_to_json(
        pipeline_result,
        image.size,
        name,
        frame_index=frame_index,
    )
    if environment is not None:
        json_payload["environment"] = environment
    if analyzed_frames is not None:
        json_payload["video_analysis"] = {
            "source_frames": frame_count,
            "analyzed_frames": analyzed_frames,
            "output_fps": round(output_fps, 2),
            "preview_frame_index": preview_frame_index,
        }
        json_payload["tracking"] = {
            "diagnostics": summary.get("tracking_diagnostics", {}),
            "tracks": summary.get("tracks", []),
        }

    return {
        "image": image,
        "annotated": annotated,
        "video_bytes": video_bytes,
        "summary": summary,
        "predictions": predictions,
        "pipeline_result": pipeline_result,
        "json_text": format_json_output(json_payload),
        "media_meta": media_meta,
        "is_video": mime_type.startswith("video") or suffix in video_suffixes,
        "mode": pipeline_result.mode,
    }


def build_summary_csv(filename: str, summary: dict, mode: str) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["section", "name", "value"])
    writer.writerow(["media", "filename", filename])
    writer.writerow(["pipeline", "mode", mode])
    writer.writerow(["summary", "unique_objects", summary.get("total", 0)])
    writer.writerow(
        ["summary", "mean_detection_confidence", summary.get("avg_detection_confidence", 0.0)]
    )
    for object_class, count in summary.get("object_counts", {}).items():
        writer.writerow(["object_count", object_class, count])
    for name, value in summary.get("tracking_diagnostics", {}).items():
        writer.writerow(["tracking", name, value])
    for track in summary.get("tracks", []):
        track_id = track.get("track_id", "")
        for name, value in track.items():
            if name != "track_id":
                writer.writerow([f"track_{track_id}", name, value])
    return buffer.getvalue().encode("utf-8-sig")


def build_evaluation_csv(rows: list[dict], fieldnames: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: "" if row.get(field) is None else row.get(field)
                for field in fieldnames
            }
        )
    return buffer.getvalue().encode("utf-8-sig")


def _blank_editor_rows(fieldnames: tuple[str, ...], count: int = 5) -> list[dict]:
    return [{field: None for field in fieldnames} for _ in range(count)]


def get_evaluation_editor_seed(
    prefix: str,
    fieldnames: tuple[str, ...],
) -> tuple[list[dict], int]:
    seed_key = f"{prefix}_editor_seed"
    version_key = f"{prefix}_editor_version"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = _blank_editor_rows(fieldnames)
    if version_key not in st.session_state:
        st.session_state[version_key] = 0
    return st.session_state[seed_key], int(st.session_state[version_key])


def reset_evaluation_editor(prefix: str, fieldnames: tuple[str, ...]) -> None:
    st.session_state[f"{prefix}_editor_seed"] = _blank_editor_rows(fieldnames)
    st.session_state[f"{prefix}_editor_version"] = (
        int(st.session_state.get(f"{prefix}_editor_version", 0)) + 1
    )
    st.session_state.pop(f"{prefix}_editor_import", None)
    st.session_state.pop(f"{prefix}_editor_import_hash", None)


def _parse_optional_nonnegative_integer(value, field_label: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_label}에는 0 이상의 정수를 입력해야 합니다.") from exc
    if not number.is_integer() or number < 0:
        raise ValueError(f"{field_label}에는 0 이상의 정수를 입력해야 합니다.")
    return int(number)


def _normalize_classification_import_row(row: dict, row_number: int) -> dict:
    normalized = {
        "sample_id": str(row.get("sample_id", "") or "").strip(),
        "true_label": str(row.get("true_label", "") or "").strip() or None,
        "predicted_label": str(row.get("predicted_label", "") or "").strip() or None,
    }
    for field in ("true_label", "predicted_label"):
        label = normalized[field]
        if label is not None and label not in TONNAGE_CLASS_OPTIONS:
            raise ValueError(
                f"{row_number}행의 {field} 값 '{label}'은 프로젝트 톤급 클래스가 아닙니다."
            )
    return normalized


def _normalize_tracking_import_row(row: dict, row_number: int) -> dict:
    return {
        "video_id": str(row.get("video_id", "") or "").strip(),
        "true_count": _parse_optional_nonnegative_integer(
            row.get("true_count"), f"{row_number}행 실제 객체 수"
        ),
        "predicted_count": _parse_optional_nonnegative_integer(
            row.get("predicted_count"), f"{row_number}행 추적 결과 수"
        ),
        "id_switches": _parse_optional_nonnegative_integer(
            row.get("id_switches"), f"{row_number}행 ID 변경 횟수"
        ),
    }


def import_evaluation_csv(
    prefix: str,
    uploaded_file,
    fieldnames: tuple[str, ...],
    required_fields: set[str],
    row_normalizer,
) -> int | None:
    if uploaded_file is None:
        return None
    content = uploaded_file.getvalue()
    content_hash = hashlib.sha256(content).hexdigest()
    hash_key = f"{prefix}_editor_import_hash"
    if st.session_state.get(hash_key) == content_hash:
        return None

    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    available_fields = set(reader.fieldnames or [])
    missing_fields = required_fields - available_fields
    if missing_fields:
        missing_text = ", ".join(sorted(missing_fields))
        raise ValueError(f"CSV에 필수 열이 없습니다: {missing_text}")

    imported_rows = [
        row_normalizer(row, row_number)
        for row_number, row in enumerate(reader, start=2)
    ]
    st.session_state[f"{prefix}_editor_seed"] = (
        imported_rows or _blank_editor_rows(fieldnames)
    )
    st.session_state[f"{prefix}_editor_version"] = (
        int(st.session_state.get(f"{prefix}_editor_version", 0)) + 1
    )
    st.session_state[hash_key] = content_hash
    return len(imported_rows)


def collect_classification_editor_rows(rows: list[dict]) -> tuple[list[dict], int]:
    complete_rows = []
    incomplete_rows = 0
    for row in rows:
        sample_id = str(row.get("sample_id", "") or "").strip()
        true_label = str(row.get("true_label", "") or "").strip()
        predicted_label = str(row.get("predicted_label", "") or "").strip()
        if not sample_id and not true_label and not predicted_label:
            continue
        if not true_label or not predicted_label:
            incomplete_rows += 1
            continue
        complete_rows.append(
            {
                "sample_id": sample_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
            }
        )
    return complete_rows, incomplete_rows


def collect_tracking_editor_rows(rows: list[dict]) -> tuple[list[dict], int]:
    complete_rows = []
    incomplete_rows = 0
    for row_number, row in enumerate(rows, start=1):
        video_id = str(row.get("video_id", "") or "").strip()
        true_count = _parse_optional_nonnegative_integer(
            row.get("true_count"), f"{row_number}행 실제 객체 수"
        )
        predicted_count = _parse_optional_nonnegative_integer(
            row.get("predicted_count"), f"{row_number}행 추적 결과 수"
        )
        id_switches = _parse_optional_nonnegative_integer(
            row.get("id_switches"), f"{row_number}행 ID 변경 횟수"
        )
        if (
            not video_id
            and true_count is None
            and predicted_count is None
            and id_switches is None
        ):
            continue
        if true_count is None or predicted_count is None:
            incomplete_rows += 1
            continue
        complete_rows.append(
            {
                "video_id": video_id,
                "true_count": true_count,
                "predicted_count": predicted_count,
                "id_switches": id_switches or 0,
            }
        )
    return complete_rows, incomplete_rows


def render_classification_evaluation(rows: list[dict]) -> None:
    classification_evaluation = calculate_classification_metrics(rows)
    st.markdown(
        '<div class="quant-eval-result-title">분류 평가 결과</div>',
        unsafe_allow_html=True,
    )
    class_metric_columns = st.columns(3)
    class_metric_columns[0].metric(
        "Accuracy",
        f"{classification_evaluation['accuracy']:.4f}",
    )
    class_metric_columns[1].metric(
        "Macro F1",
        f"{classification_evaluation['macro_f1']:.4f}",
    )
    class_metric_columns[2].metric(
        "평가 샘플",
        f"{classification_evaluation['samples']}개",
    )
    label_order = {label: index for index, label in enumerate(TONNAGE_CLASS_OPTIONS)}
    labels = sorted(
        classification_evaluation["labels"],
        key=lambda label: (label_order.get(label, len(label_order)), label),
    )
    confusion_rows = []
    for true_label in labels:
        row = {"실제 클래스": true_label}
        row.update(
            {
                predicted_label: classification_evaluation["confusion_matrix"][
                    true_label
                ][predicted_label]
                for predicted_label in labels
            }
        )
        confusion_rows.append(row)
    st.caption("혼동행렬 · 행은 실제 클래스, 열은 예측 클래스입니다.")
    st.dataframe(
        confusion_rows,
        width="stretch",
        hide_index=True,
        column_order=["실제 클래스", *labels],
    )


def render_tracking_evaluation(rows: list[dict]) -> None:
    tracking_evaluation = calculate_tracking_metrics(rows)
    st.markdown(
        '<div class="quant-eval-result-title">추적 평가 결과</div>',
        unsafe_allow_html=True,
    )
    track_metric_columns = st.columns(3)
    track_metric_columns[0].metric(
        "Count MAE",
        f"{tracking_evaluation['count_mae']:.3f}",
    )
    track_metric_columns[1].metric(
        "정확 집계율",
        f"{tracking_evaluation['exact_count_rate'] * 100:.1f}%",
    )
    track_metric_columns[2].metric(
        "ID 변경 횟수",
        tracking_evaluation["total_id_switches"],
    )
    st.caption(f"평가에 사용된 영상 {tracking_evaluation['videos']}개")


def render_dashboard_tab1(summary: dict | None = None, mode: str = "demo"):
    counts = (summary or {}).get("object_counts", {})
    total = str((summary or {}).get("total", 0))
    avg_det = str((summary or {}).get("avg_detection_confidence", 0.0))
    count_label = (summary or {}).get("count_label", "전체 탐지 수")
    breakdown_label = (summary or {}).get("breakdown_label", "객체별 탐지 수")
    ship = str(counts.get("ship", 0))
    drone = str(counts.get("drone", 0))
    person = str(counts.get("person", 0))

    st.markdown(f"""
    <div class="dashboard-panel">
        <div class="dashboard-panel-title">통합 객체 탐지 현황</div>
        <div class="metric-grid-2">
            <div class="metric-card">
                <div class="metric-label">{count_label}</div>
                <div class="metric-value cyan">{total}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">처리 모드</div>
                <div class="metric-value accent">{mode}</div>
            </div>
        </div>
        <hr class="compact-divider">
        <div class="object-label">{breakdown_label} (Object Detection)</div>
        <div class="object-tags">
            <div class="object-tag">함정 <strong>{ship}</strong></div>
            <div class="object-tag">드론 <strong>{drone}</strong></div>
            <div class="object-tag">사람 <strong>{person}</strong></div>
        </div>
        <hr class="compact-divider">
        <div class="metric-card">
            <div class="metric-label">평균 탐지 Confidence</div>
            <div class="metric-value accent">{avg_det}</div>
        </div>
        <div class="active-status">
            <span class="status-dot"></span>
            Detection Status : ACTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard_tab2(summary: dict | None = None, mode: str = "demo"):
    counts = (summary or {}).get("size_counts", {})
    fine_counts = (summary or {}).get("fine_counts", {})
    total = (summary or {}).get("total", 0)
    avg_confidence = (summary or {}).get("avg_classification_confidence", 0.0)
    tonnage_tags = "".join(
        f'<div class="object-tag">{label} <strong>{count}</strong></div>'
        for label, count in fine_counts.items()
    ) or '<div class="dropzone-hint">분류 결과 없음</div>'
    st.markdown(f"""
    <div class="dashboard-panel">
        <div class="dashboard-panel-title">함정 톤급 분류 현황</div>
        <div class="metric-card">
            <div class="metric-label">분류 함정 수</div>
            <div class="metric-value accent">{total}</div>
        </div>
        <hr class="compact-divider">
        <div class="metric-grid-2">
            <div class="metric-card">
                <div class="metric-label">대형</div>
                <div class="metric-value">{counts.get("대형", 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">중형</div>
                <div class="metric-value">{counts.get("중형", 0)}</div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">소형</div>
            <div class="metric-value">{counts.get("소형", 0)}</div>
        </div>
        <hr class="compact-divider">
        <div class="object-label">톤급별 분류 결과</div>
        <div class="object-tags">{tonnage_tags}</div>
        <hr class="compact-divider">
        <div class="metric-card">
            <div class="metric-label">평균 분류 Confidence</div>
            <div class="metric-value cyan">{avg_confidence:.2f}</div>
        </div>
        <div class="dropzone-hint">처리 모드: {mode}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Hero Header
# ============================================================

model_status = get_model_status()
classification_status = get_classification_status()
detection_ready = bool(
    model_status["roboflow_configured"] or model_status["yolo_loaded"]
)
classification_ready = bool(classification_status["configured"])
if detection_ready and classification_ready:
    system_status = "Configured"
    system_status_color = "#7fc6a4"
elif detection_ready:
    system_status = "Detection only"
    system_status_color = "#d9b879"
elif model_status["demo_mode_allowed"]:
    system_status = "Demo enabled"
    system_status_color = "#d9b879"
else:
    system_status = "Not configured"
    system_status_color = "#e88c8c"

st.markdown(f"""
<div class="hero-header">
    <div class="hero-badge">MARITIME MONITORING</div>
    <div class="hero-top">
        <div class="hero-title-block">
            <div class="main-title">해양 침투 객체 감시 시스템</div>
            <div class="sub-title">사진/영상 기반 객체 탐지 및 함정 분류 통합 관제</div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat-item">Detection <span>YOLOv26</span></div>
            <div class="hero-stat-item">Classification <span>ViT</span></div>
            <div class="hero-stat-item">Status <span style="color:{system_status_color}">{system_status}</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Tabs
# ============================================================

tab_detection, tab_performance, tab_quantitative = st.tabs([
    "객체 탐지",
    "모델 성능평가",
    "정량평가",
])


# ============================================================
# TAB 1 — 실시간 감시
# ============================================================

with tab_detection:
    result = None
    uploaded_file = st.session_state.get("live_upload_saved")
    if uploaded_file is None:
        uploaded_file = st.file_uploader(
            "파일을 드래그 앤 드롭하거나 클릭하여 선택",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
            label_visibility="collapsed",
            key="live_upload",
        )
        if uploaded_file is not None:
            st.session_state["live_upload_saved"] = uploaded_file
            st.rerun()
    if uploaded_file is not None:
        upload_cache_key = get_upload_cache_key(uploaded_file, "detection")
        if st.session_state.get("processed_upload_key") == upload_cache_key:
            result = st.session_state.get("processed_upload_result")
        else:
            upload_name = uploaded_file.name.lower()
            is_processing_video = (
                (uploaded_file.type or "").startswith("video")
                or upload_name.endswith((".mp4", ".avi", ".mov"))
            )
            processing_title = (
                "영상을 분석하고 있습니다"
                if is_processing_video
                else "이미지를 분석하고 있습니다"
            )
            processing_desc = (
                "프레임 선별 · 객체 탐지 · 결과 영상 구성 순서로 처리됩니다"
                if is_processing_video
                else "객체 탐지와 결과 시각화를 준비하고 있습니다"
            )
            processing_meta = "FRAME PIPELINE" if is_processing_video else "IMAGE PIPELINE"
            processing_placeholder = st.empty()
            progress_placeholder = st.empty()
            progress_started_at = time.perf_counter()

            def update_video_progress(progress: float, current_frame: int, total_frames: int):
                elapsed = max(0.0, time.perf_counter() - progress_started_at)
                if progress > 0 and progress < 1:
                    remaining = elapsed * (1 - progress) / progress
                    time_text = f"경과 {elapsed:.0f}초 · 예상 {remaining:.0f}초 남음"
                elif progress >= 1:
                    time_text = f"완료 · {elapsed:.0f}초"
                else:
                    time_text = "영상 정보를 확인하고 있습니다"
                frame_text = (
                    f"{min(current_frame, total_frames):,} / {total_frames:,} 프레임"
                    if total_frames > 0
                    else f"{current_frame:,} 프레임"
                )
                progress_placeholder.progress(
                    progress,
                    text=f"{frame_text} · {time_text}",
                )

            processing_placeholder.markdown(
                f"""
                <div class="processing-state" role="status" aria-live="polite">
                    <div class="processing-indicator" aria-hidden="true"></div>
                    <div class="processing-copy">
                        <div class="processing-kicker">ANALYSIS IN PROGRESS</div>
                        <div class="processing-title">{processing_title}</div>
                        <div class="processing-desc">{processing_desc}</div>
                    </div>
                    <div class="processing-meta">{processing_meta}</div>
                    <div class="processing-track" aria-hidden="true"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                result = process_upload(
                    uploaded_file,
                    task="detection",
                    progress_callback=update_video_progress if is_processing_video else None,
                )
                st.session_state["processed_upload_key"] = upload_cache_key
                st.session_state["processed_upload_result"] = result
            except Exception as exc:
                st.session_state.pop("processed_upload_key", None)
                st.session_state.pop("processed_upload_result", None)
                st.error(f"파일 처리 중 오류: {exc}")
            finally:
                processing_placeholder.empty()
                progress_placeholder.empty()

    if result is not None:
        info_col, replace_col = st.columns([7, 1.4])
        with info_col:
            safe_filename = html.escape(str(uploaded_file.name))
            safe_media_meta = html.escape(str(result["media_meta"]))
            st.markdown(
                f'<div class="result-file-bar">{safe_filename} · {safe_media_meta}</div>',
                unsafe_allow_html=True,
            )
        with replace_col:
            st.button("↻ 파일 변경", key="replace_live_upload", on_click=clear_uploaded_file, args=("live_upload",), use_container_width=True)

        output_col, status_col = st.columns([3.2, 1.25], gap="medium")
        with output_col:
            view_mode = st.radio("출력 형식", ["Visual", "JSON"], horizontal=True, label_visibility="collapsed", key="output_view")
            selected_box = None
            if view_mode == "Visual":
                with st.container(border=True, key="ship_result_box"):
                    if result["is_video"]:
                        st.video(
                            result["video_bytes"],
                            format="video/mp4",
                            autoplay=True,
                            muted=True,
                        )
                        st.caption("아래 대표 탐지 프레임의 바운딩 박스를 클릭해 분류할 영역을 선택하세요.")
                        selected_box = render_clickable_detections(result["image"], result["predictions"])
                    else:
                        selected_box = render_clickable_detections(result["image"], result["predictions"])
            else:
                with st.container(border=True, key="ship_result_box"):
                    st.code(result["json_text"], language="json")
        with status_col:
            render_dashboard_tab1(result["summary"], result["mode"])

        safe_base_name = "".join(
            character
            for character in os.path.splitext(uploaded_file.name)[0]
            if character.isalnum() or character in {"-", "_"}
        ) or "marine_result"
        download_columns = st.columns(3, gap="small")
        with download_columns[0]:
            if result["is_video"]:
                st.download_button(
                    "결과 영상 다운로드",
                    data=result["video_bytes"],
                    file_name=f"{safe_base_name}_detected.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            else:
                image_buffer = io.BytesIO()
                result["annotated"].save(image_buffer, format="PNG")
                st.download_button(
                    "결과 이미지 다운로드",
                    data=image_buffer.getvalue(),
                    file_name=f"{safe_base_name}_detected.png",
                    mime="image/png",
                    use_container_width=True,
                )
        with download_columns[1]:
            st.download_button(
                "JSON 다운로드",
                data=result["json_text"].encode("utf-8"),
                file_name=f"{safe_base_name}_result.json",
                mime="application/json",
                use_container_width=True,
            )
        with download_columns[2]:
            st.download_button(
                "CSV 요약 다운로드",
                data=build_summary_csv(
                    uploaded_file.name,
                    result["summary"],
                    result["mode"],
                ),
                file_name=f"{safe_base_name}_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if view_mode == "Visual":
            render_selected_crop(result["image"], result["predictions"], selected_box)
    elif uploaded_file is not None:
        st.button("파일 다시 선택", key="retry_live_upload", on_click=clear_uploaded_file, args=("live_upload",))
    else:
        st.markdown(f"""
        <div class="output-visual-wrap empty">
            <div class="video-placeholder-text">파일을 업로드하면 함정 · 드론 · 사람 통합 탐지 결과를 표시합니다</div>
            <div class="video-placeholder-sub">최대 {MAX_UPLOAD_MB}MB · 영상 {MAX_VIDEO_SECONDS}초 · JPG, PNG, MP4, AVI, MOV</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 2 — 모델 성능평가
# ============================================================

with tab_performance:
    st.markdown('<div class="object-label">평가 대상</div>', unsafe_allow_html=True)

    evaluation_options = ["전체", "대형 함정", "중형 함정", "소형 함정", "드론", "사람"]
    if st.session_state.get("evaluation_target") not in evaluation_options:
        st.session_state["evaluation_target"] = "전체"

    evaluation_target = st.radio(
        "평가 대상",
        evaluation_options,
        horizontal=True,
        label_visibility="collapsed",
        key="evaluation_target",
    )

    try:
        detection_metrics = get_detection_metrics()
        display_metrics = get_display_metrics(evaluation_target)
        target_metrics = display_metrics["metrics"]

        precision = target_metrics.get("precision", 0.0)
        recall = target_metrics.get("recall", 0.0)
        ap50 = target_metrics.get("map50", 0.0)
        ap50_95 = target_metrics.get("map50_95", 0.0)
        images = target_metrics.get("images", 0)
        targets = target_metrics.get("targets", 0)
        is_overall = display_metrics["type"] == "overall"
        metric_prefix = "mAP" if is_overall else "AP"
        source = detection_metrics.get("overall_source", "-") if is_overall else display_metrics["class_name"]
        scope_text = "전체 모델" if is_overall else "선택 클래스"

        st.markdown(f"""
        <div class="tab3-layout">
            <div class="object-label">선택된 평가 대상: {evaluation_target}</div>
            <div class="evaluation-grid">
                <div class="evaluation-card">
                    <div class="evaluation-name">Precision</div>
                    <div class="evaluation-value">{precision:.4f}</div>
                    <div class="evaluation-desc">{scope_text}의 정밀도입니다.</div>
                </div>
                <div class="evaluation-card">
                    <div class="evaluation-name">Recall</div>
                    <div class="evaluation-value">{recall:.4f}</div>
                    <div class="evaluation-desc">{scope_text}의 재현율입니다.</div>
                </div>
                <div class="evaluation-card">
                    <div class="evaluation-name">{metric_prefix}@0.5</div>
                    <div class="evaluation-value">{ap50:.4f}</div>
                    <div class="evaluation-desc">{scope_text}의 IoU 0.5 기준 Average Precision입니다.</div>
                </div>
                <div class="evaluation-card">
                    <div class="evaluation-name">{metric_prefix}@0.5:0.95</div>
                    <div class="evaluation-value">{ap50_95:.4f}</div>
                    <div class="evaluation-desc">{scope_text}에서 IoU 0.5~0.95를 반영한 Average Precision입니다.</div>
                </div>
                <div class="evaluation-card">
                    <div class="evaluation-name">평가 데이터</div>
                    <div class="evaluation-value">{images if not is_overall else detection_metrics["split"]}</div>
                    <div class="evaluation-desc">{f'Images {images} · Targets {targets}' if not is_overall else f'전체 모델 성능 · Source {source}'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as exc:
        if not model_status["roboflow_configured"]:
            st.info(
                "Roboflow API가 설정되면 실제 모델 성능평가 결과가 표시됩니다."
            )
        else:
            st.error(f"Roboflow 성능평가 연동 오류: {exc}")

    st.markdown('<div class="section-title">영상 추적 진단</div>', unsafe_allow_html=True)
    if result is not None and result["is_video"]:
        diagnostics = result["summary"].get("tracking_diagnostics", {})
        tracking_columns = st.columns(4, gap="small")
        tracking_columns[0].metric(
            "확정 트랙",
            diagnostics.get("confirmed_tracks", 0),
        )
        tracking_columns[1].metric(
            "잠정 트랙",
            diagnostics.get("tentative_tracks", 0),
        )
        tracking_columns[2].metric(
            "트랙당 평균 관측",
            diagnostics.get("mean_observations_per_track", 0.0),
        )
        tracking_columns[3].metric(
            "분석 프레임",
            diagnostics.get("analyzed_steps", 0),
        )
        st.caption(
            "현재 업로드 영상에서 계산한 운영 진단입니다. "
            "정확도 수치가 아니라 추적 지속성과 잠정 탐지 규모를 보여줍니다."
        )
    else:
        st.info("객체 탐지 탭에서 영상을 분석하면 실제 추적 진단이 표시됩니다.")

# ============================================================
# TAB 3 — 사용자 검증 데이터 정량평가
# ============================================================

with tab_quantitative:
    st.markdown("""
    <div class="quant-eval-intro">
        <div class="quant-eval-kicker">USER VALIDATION</div>
        <div class="quant-eval-title">사용자 검증 데이터 정량평가</div>
        <div class="quant-eval-copy">
            위의 Detection 성능 지표와 별도로, 직접 준비한 정답과 모델 결과를 비교합니다.
            CSV는 평가 계산에만 사용되며 객체 탐지와 드론 추적 결과에는 영향을 주지 않습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    evaluation_type = st.radio(
        "평가 방식",
        ["톤급 분류", "영상 추적"],
        horizontal=True,
        key="offline_evaluation_type",
    )

    if evaluation_type == "톤급 분류":
        st.markdown("""
        <div class="quant-eval-guide">
            <div class="quant-eval-guide-item">
                <span class="quant-eval-guide-label">표 입력 항목</span>
                <div class="quant-eval-guide-value">
                    샘플 ID(선택) · <code>true_label</code> 실제 톤급 ·
                    <code>predicted_label</code> 예측 톤급
                </div>
            </div>
            <div class="quant-eval-guide-item">
                <span class="quant-eval-guide-label">계산 결과</span>
                <div class="quant-eval-guide-value">Accuracy · Macro F1 · 클래스별 혼동행렬</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        classification_eval_file = st.file_uploader(
            "기존 분류 CSV 불러오기 (선택)",
            type=["csv"],
            key="classification_eval_editor_import",
            help="기존 CSV를 표로 불러와 수정할 수 있습니다. 새로 작성하려면 건너뛰세요.",
        )
        try:
            imported_count = import_evaluation_csv(
                "classification_eval",
                classification_eval_file,
                CLASSIFICATION_EDITOR_FIELDS,
                {"true_label", "predicted_label"},
                _normalize_classification_import_row,
            )
            if imported_count is not None:
                if imported_count:
                    st.success(f"CSV {imported_count}개 행을 편집표로 불러왔습니다.")
                else:
                    st.success("CSV 열을 확인했습니다. 빈 편집표를 준비했습니다.")
        except Exception as exc:
            st.error(f"분류 CSV 불러오기 오류: {exc}")

        st.markdown("""
        <div class="quant-editor-heading">
            <div class="quant-editor-title">분류 평가 데이터 편집</div>
            <div class="quant-editor-desc">
                셀을 선택해 직접 입력하세요. 표 아래의 + 버튼으로 행을 추가하고,
                행 왼쪽 메뉴에서 삭제할 수 있습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
        classification_seed, classification_editor_version = get_evaluation_editor_seed(
            "classification_eval",
            CLASSIFICATION_EDITOR_FIELDS,
        )
        classification_editor_rows = st.data_editor(
            classification_seed,
            width="stretch",
            height=280,
            hide_index=True,
            num_rows="dynamic",
            key=f"classification_eval_editor_{classification_editor_version}",
            column_order=CLASSIFICATION_EDITOR_FIELDS,
            column_config={
                "sample_id": st.column_config.TextColumn(
                    "샘플 ID",
                    width="medium",
                    help="이미지 파일명이나 관리 번호를 입력합니다. 평가 계산에는 사용하지 않습니다.",
                ),
                "true_label": st.column_config.SelectboxColumn(
                    "실제 톤급",
                    options=TONNAGE_CLASS_OPTIONS,
                    required=True,
                    width="medium",
                    help="정답으로 확인한 함정 톤급입니다.",
                ),
                "predicted_label": st.column_config.SelectboxColumn(
                    "예측 톤급",
                    options=TONNAGE_CLASS_OPTIONS,
                    required=True,
                    width="medium",
                    help="ViT 모델이 예측한 함정 톤급입니다.",
                ),
            },
        )
        classification_rows, incomplete_classification_rows = (
            collect_classification_editor_rows(classification_editor_rows)
        )
        class_status_col, class_download_col, class_reset_col = st.columns(
            [2.2, 1, 0.8], gap="small"
        )
        with class_status_col:
            st.markdown(
                f'<div class="quant-editor-status">평가 가능 <strong>{len(classification_rows)}</strong>건'
                f' · 미완성 <strong>{incomplete_classification_rows}</strong>건</div>',
                unsafe_allow_html=True,
            )
        with class_download_col:
            st.download_button(
                "작성 CSV 다운로드",
                data=build_evaluation_csv(
                    classification_rows,
                    CLASSIFICATION_EDITOR_FIELDS,
                ),
                file_name="classification_evaluation.csv",
                mime="text/csv",
                disabled=not classification_rows,
                use_container_width=True,
                key="classification_eval_download",
            )
        with class_reset_col:
            st.button(
                "표 비우기",
                use_container_width=True,
                key="classification_eval_reset",
                on_click=reset_evaluation_editor,
                args=("classification_eval", CLASSIFICATION_EDITOR_FIELDS),
            )
        if incomplete_classification_rows:
            st.warning("실제 톤급과 예측 톤급이 모두 입력되지 않은 행은 평가에서 제외됩니다.")
        if classification_rows:
            render_classification_evaluation(classification_rows)
        else:
            st.info("실제 톤급과 예측 톤급을 한 행 이상 입력하면 평가 결과가 자동 계산됩니다.")
    else:
        st.markdown("""
        <div class="quant-eval-guide">
            <div class="quant-eval-guide-item">
                <span class="quant-eval-guide-label">표 입력 항목</span>
                <div class="quant-eval-guide-value">
                    영상 ID(선택) · <code>true_count</code> 실제 객체 수 ·
                    <code>predicted_count</code> 추적 결과 수 ·
                    <code>id_switches</code> ID 변경 횟수(선택)
                </div>
            </div>
            <div class="quant-eval-guide-item">
                <span class="quant-eval-guide-label">계산 결과</span>
                <div class="quant-eval-guide-value">Count MAE · 정확 집계율 · 전체 ID 변경 횟수</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        tracking_eval_file = st.file_uploader(
            "기존 추적 CSV 불러오기 (선택)",
            type=["csv"],
            key="tracking_eval_editor_import",
            help="기존 CSV를 표로 불러와 수정할 수 있습니다. 새로 작성하려면 건너뛰세요.",
        )
        try:
            imported_count = import_evaluation_csv(
                "tracking_eval",
                tracking_eval_file,
                TRACKING_EDITOR_FIELDS,
                {"true_count", "predicted_count"},
                _normalize_tracking_import_row,
            )
            if imported_count is not None:
                if imported_count:
                    st.success(f"CSV {imported_count}개 행을 편집표로 불러왔습니다.")
                else:
                    st.success("CSV 열을 확인했습니다. 빈 편집표를 준비했습니다.")
        except Exception as exc:
            st.error(f"추적 CSV 불러오기 오류: {exc}")

        st.markdown("""
        <div class="quant-editor-heading">
            <div class="quant-editor-title">영상 추적 평가 데이터 편집</div>
            <div class="quant-editor-desc">
                영상별 정답 객체 수와 추적 결과 수를 입력하세요.
                ID가 바뀐 횟수를 모르면 비워 두어도 됩니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
        tracking_seed, tracking_editor_version = get_evaluation_editor_seed(
            "tracking_eval",
            TRACKING_EDITOR_FIELDS,
        )
        tracking_editor_rows = st.data_editor(
            tracking_seed,
            width="stretch",
            height=280,
            hide_index=True,
            num_rows="dynamic",
            key=f"tracking_eval_editor_{tracking_editor_version}",
            column_order=TRACKING_EDITOR_FIELDS,
            column_config={
                "video_id": st.column_config.TextColumn(
                    "영상 ID",
                    width="medium",
                    help="영상 파일명이나 관리 번호입니다. 평가 계산에는 사용하지 않습니다.",
                ),
                "true_count": st.column_config.NumberColumn(
                    "실제 객체 수",
                    min_value=0,
                    step=1,
                    required=True,
                    format="%d",
                    help="영상에 실제로 등장한 고유 객체 수입니다.",
                ),
                "predicted_count": st.column_config.NumberColumn(
                    "추적 결과 수",
                    min_value=0,
                    step=1,
                    required=True,
                    format="%d",
                    help="시스템이 고유 ID로 집계한 객체 수입니다.",
                ),
                "id_switches": st.column_config.NumberColumn(
                    "ID 변경 횟수",
                    min_value=0,
                    step=1,
                    format="%d",
                    help="같은 객체의 추적 ID가 바뀐 횟수입니다. 미입력 시 0으로 계산합니다.",
                ),
            },
        )
        try:
            tracking_rows, incomplete_tracking_rows = collect_tracking_editor_rows(
                tracking_editor_rows
            )
            tracking_input_error = None
        except ValueError as exc:
            tracking_rows = []
            incomplete_tracking_rows = 0
            tracking_input_error = str(exc)

        track_status_col, track_download_col, track_reset_col = st.columns(
            [2.2, 1, 0.8], gap="small"
        )
        with track_status_col:
            st.markdown(
                f'<div class="quant-editor-status">평가 가능 <strong>{len(tracking_rows)}</strong>건'
                f' · 미완성 <strong>{incomplete_tracking_rows}</strong>건</div>',
                unsafe_allow_html=True,
            )
        with track_download_col:
            st.download_button(
                "작성 CSV 다운로드",
                data=build_evaluation_csv(tracking_rows, TRACKING_EDITOR_FIELDS),
                file_name="tracking_evaluation.csv",
                mime="text/csv",
                disabled=not tracking_rows,
                use_container_width=True,
                key="tracking_eval_download",
            )
        with track_reset_col:
            st.button(
                "표 비우기",
                use_container_width=True,
                key="tracking_eval_reset",
                on_click=reset_evaluation_editor,
                args=("tracking_eval", TRACKING_EDITOR_FIELDS),
            )
        if tracking_input_error:
            st.error(tracking_input_error)
        elif incomplete_tracking_rows:
            st.warning("실제 객체 수와 추적 결과 수가 모두 입력되지 않은 행은 평가에서 제외됩니다.")
        if tracking_rows:
            render_tracking_evaluation(tracking_rows)
        elif not tracking_input_error:
            st.info("실제 객체 수와 추적 결과 수를 한 행 이상 입력하면 평가 결과가 자동 계산됩니다.")


# ============================================================
# Footer
# ============================================================

st.markdown("""
<div class="app-footer">
    Marine Ship Detection &amp; Classification System &nbsp;·&nbsp; YOLOv26 + ViT
</div>
""", unsafe_allow_html=True)
