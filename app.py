from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
CHARTS_DIR = ROOT / "charts"
ANALYSIS_SCRIPT = ROOT / "analysis.py"

SENTIMENT_ORDER = [
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed",
]

SENTIMENT_COLORS = {
    "Extreme Fear": "#C95A4D",
    "Fear": "#C67C3D",
    "Neutral": "#8A8E91",
    "Greed": "#1D6B67",
    "Extreme Greed": "#0E5D55",
}

ARCHETYPE_COLORS = {
    "Patient Position Builders": "#1D6B67",
    "Aggressive Takers": "#C95A4D",
    "Selective Rotators": "#506D9A",
    "Impulse Scalpers": "#C67C3D",
}

CHART_LABELS = {
    "01_performance_by_sentiment.png": "Performance by Sentiment",
    "02_timeline_sentiment_pnl.png": "Matched Event Timeline",
    "03_behavioral_fingerprint.png": "Behavioral Fingerprint",
    "04_event_coverage.png": "Event Coverage Heatmap",
    "05_segmentation_deep_dive.png": "Segmentation Deep Dive",
    "06_directional_bias.png": "Directional Bias",
    "07_archetypes.png": "Trader Archetypes",
    "08_cluster_profiles.png": "Cluster Profiles",
    "09_robustness_checks.png": "Robustness Checks",
    "10_strategy_playbook.png": "Strategy Playbook",
    "11_drawdown_analysis.png": "Drawdown Analysis",
}

REQUIRED_OUTPUTS = [
    "performance_by_sentiment.csv",
    "behavior_by_sentiment.csv",
    "account_summary.csv",
    "cluster_profiles.csv",
    "leverage_segmentation.csv",
    "frequency_segmentation.csv",
    "consistency_segmentation.csv",
    "execution_segmentation.csv",
    "event_summary.csv",
    "robustness_checks.csv",
    "strategy_playbook.csv",
    "ui_metrics.json",
]

PLOTLY_AXIS = "rgba(24, 36, 45, 0.12)"
PLOTLY_GRID = "rgba(24, 36, 45, 0.08)"
PLOTLY_PAPER = "rgba(0, 0, 0, 0)"
PLOTLY_PLOT = "rgba(255, 255, 255, 0.60)"


st.set_page_config(
    page_title="Hyperliquid Signal Atlas",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #f7f1e8;
            --bg-soft: #efe3d4;
            --panel: rgba(255, 251, 246, 0.86);
            --panel-strong: rgba(255, 255, 255, 0.92);
            --line: rgba(43, 34, 25, 0.10);
            --line-strong: rgba(43, 34, 25, 0.18);
            --ink: #1b1713;
            --muted: #655d56;
            --accent: #9a7141;
            --accent-deep: #6a4826;
            --teal: #365d58;
            --rose: #8b574f;
            --blue: #4e627d;
            --gold: #b4935d;
            --shadow: 0 24px 80px rgba(89, 69, 43, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(198, 124, 61, 0.14), transparent 26%),
                radial-gradient(circle at 88% 12%, rgba(29, 107, 103, 0.12), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(80, 109, 154, 0.08), transparent 22%),
                linear-gradient(180deg, #faf6f0 0%, #f6f0e8 45%, #efe6da 100%);
            color: var(--ink);
        }

        html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
            font-family: "Manrope", sans-serif;
        }

        h1, h2, h3, .hero-title, .section-title, .metric-value, .panel-title, .note-value {
            font-family: "Cormorant Garamond", serif !important;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(253, 248, 242, 0.98) 0%, rgba(244, 234, 221, 0.98) 100%);
            border-right: 1px solid var(--line);
        }

        .sidebar-brand {
            padding: 1.1rem 1rem 1.05rem 1rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(248, 239, 229, 0.88) 100%);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .sidebar-kicker,
        .section-kicker,
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.70rem;
            font-weight: 800;
            color: var(--accent-deep);
        }

        .sidebar-title {
            font-size: 1.6rem;
            line-height: 1.02;
            letter-spacing: -0.03em;
            margin: 0.45rem 0 0.35rem 0;
            color: var(--ink);
        }

        .sidebar-copy,
        .section-copy {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.72;
            margin: 0;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            border-radius: 34px;
            padding: 2rem 2.15rem 2.1rem 2.15rem;
            background:
                linear-gradient(180deg, rgba(255, 252, 248, 0.92) 0%, rgba(248, 241, 232, 0.84) 100%);
            border: 1px solid var(--line);
            box-shadow: 0 34px 110px rgba(82, 62, 37, 0.11);
            margin-bottom: 1.1rem;
        }

        .hero-card::before {
            content: "";
            position: absolute;
            inset: 16px;
            border-radius: 24px;
            border: 1px solid rgba(154, 113, 65, 0.10);
            pointer-events: none;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 420px;
            height: 420px;
            right: -140px;
            top: -160px;
            background: radial-gradient(circle, rgba(180, 147, 93, 0.17), transparent 62%);
        }

        .masthead-strip {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            padding-bottom: 1rem;
            margin-bottom: 1.4rem;
            border-bottom: 1px solid rgba(43, 34, 25, 0.10);
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.70rem;
            font-weight: 800;
            color: var(--muted);
        }

        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1.6fr) minmax(260px, 0.84fr);
            gap: 1.2rem;
            align-items: end;
        }

        .hero-title {
            font-size: clamp(3rem, 5vw, 5.2rem);
            line-height: 0.90;
            letter-spacing: -0.045em;
            margin: 0.6rem 0 0.9rem 0;
            color: var(--ink);
            max-width: 840px;
        }

        .hero-copy {
            max-width: 760px;
            font-size: 1.02rem;
            color: var(--muted);
            line-height: 1.82;
            margin: 0 0 1.3rem 0;
        }

        .hero-note {
            padding: 1.2rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(24, 36, 45, 0.08);
            backdrop-filter: blur(8px);
        }

        .note-label {
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 800;
            color: var(--teal);
        }

        .note-value {
            font-size: 1.6rem;
            line-height: 1.08;
            color: var(--ink);
            margin: 0.35rem 0 0.45rem 0;
        }

        .note-copy {
            font-size: 0.93rem;
            color: var(--muted);
            line-height: 1.7;
            margin: 0;
        }

        .chip-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
        }

        .insight-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.62rem 0.82rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(24, 36, 45, 0.08);
            color: var(--ink);
            font-size: 0.86rem;
            font-weight: 700;
            box-shadow: 0 10px 25px rgba(89, 68, 42, 0.06);
        }

        .stat-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.6rem;
            margin: 1.1rem 0 1.1rem 0;
        }

        .stat-item,
        .metric-card,
        .panel-card,
        .quote-block {
            border-radius: 24px;
            border: 1px solid rgba(24, 36, 45, 0.08);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.90), rgba(250, 243, 233, 0.82));
            box-shadow: var(--shadow);
        }

        .stat-item {
            padding: 1rem 1rem 0.95rem 1rem;
        }

        .stat-label,
        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 800;
            color: var(--muted);
        }

        .stat-number,
        .metric-value {
            font-size: 2rem;
            line-height: 1;
            margin: 0.55rem 0 0.42rem 0;
            color: var(--ink);
            letter-spacing: -0.04em;
        }

        .stat-copy,
        .metric-detail,
        .panel-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.58;
            margin: 0;
        }

        .metric-card,
        .panel-card {
            padding: 1.2rem 1.2rem 1.08rem 1.2rem;
        }

        .panel-title {
            font-size: 1.15rem;
            line-height: 1.08;
            margin: 0.3rem 0 0.45rem 0;
            color: var(--ink);
        }

        .metric-accent,
        .panel-accent {
            width: 54px;
            height: 4px;
            border-radius: 999px;
            margin-bottom: 0.9rem;
        }

        .accent-copper {
            background: linear-gradient(90deg, #c67c3d, #e8b07b);
        }

        .accent-teal {
            background: linear-gradient(90deg, #1d6b67, #63a7a3);
        }

        .accent-rose {
            background: linear-gradient(90deg, #c95a4d, #ec9a90);
        }

        .quote-block {
            position: relative;
            padding: 1.4rem 1.4rem 1.2rem 1.45rem;
        }

        .quote-mark {
            position: absolute;
            top: 0.8rem;
            left: 1rem;
            font-size: 3rem;
            line-height: 1;
            color: rgba(29, 107, 103, 0.24);
        }

        .quote-text {
            position: relative;
            padding-left: 0.6rem;
            color: var(--ink);
            font-size: 1.02rem;
            line-height: 1.74;
            margin: 0 0 0.9rem 0;
        }

        .quote-source {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 800;
            padding-left: 0.6rem;
        }

        .section-shell {
            margin: 1rem 0 1rem 0;
        }

        .section-rule {
            width: 78px;
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(29, 107, 103, 0.85), rgba(198, 124, 61, 0.85));
            margin-bottom: 0.9rem;
        }

        .section-title {
            font-size: 1.95rem;
            line-height: 1.04;
            letter-spacing: -0.03em;
            margin: 0.3rem 0 0.5rem 0;
            color: var(--ink);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.55);
            padding: 0.35rem;
            border-radius: 999px;
            border: 1px solid rgba(24, 36, 45, 0.08);
            width: fit-content;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.55rem 1rem;
            font-weight: 700;
            color: var(--muted);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(29, 107, 103, 0.16), rgba(198, 124, 61, 0.14));
            color: var(--ink);
        }

        [data-testid="stButton"] button,
        [data-testid="baseButton-secondary"] {
            border-radius: 999px;
            border: 1px solid rgba(24, 36, 45, 0.08);
            font-weight: 800;
            min-height: 2.9rem;
            box-shadow: 0 10px 28px rgba(91, 70, 44, 0.08);
        }

        [data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #18242d 0%, #2c3e4d 100%);
            color: #fffdf8;
        }

        [data-testid="stDataFrame"] {
            border-radius: 24px;
            overflow: hidden;
            border: 1px solid rgba(43, 34, 25, 0.10);
            box-shadow: var(--shadow);
        }

        @media (max-width: 980px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .stat-strip {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 680px) {
            .stat-strip {
                grid-template-columns: 1fr;
            }
        }

        :root {
            --bg: #f6f1e8;
            --bg-soft: #eadfd1;
            --panel: rgba(255, 255, 255, 0.68);
            --panel-strong: rgba(255, 255, 255, 0.82);
            --line: rgba(12, 26, 38, 0.10);
            --line-strong: rgba(12, 26, 38, 0.18);
            --ink: #0f2232;
            --muted: #5d6772;
            --accent: #b47b38;
            --accent-deep: #88551b;
            --teal: #1f6e68;
            --rose: #c56b55;
            --blue: #3f5f86;
            --gold: #cfac6b;
            --shadow: 0 28px 90px rgba(15, 34, 50, 0.12);
            --shadow-soft: 0 18px 45px rgba(15, 34, 50, 0.09);
        }

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(197, 107, 85, 0.16), transparent 28%),
                radial-gradient(circle at 100% 12%, rgba(31, 110, 104, 0.17), transparent 25%),
                radial-gradient(circle at 40% 100%, rgba(63, 95, 134, 0.14), transparent 25%),
                linear-gradient(135deg, #fbf8f2 0%, #f6efe5 48%, #ece1d2 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1rem;
            padding-bottom: 5rem;
        }

        [data-testid="stSidebar"] {
            backdrop-filter: blur(18px);
            background:
                linear-gradient(180deg, rgba(255, 251, 247, 0.92) 0%, rgba(242, 232, 219, 0.95) 100%);
        }

        .sidebar-brand {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            border: 1px solid rgba(255, 255, 255, 0.55);
            box-shadow: 0 24px 70px rgba(15, 34, 50, 0.12);
        }

        .sidebar-brand::after {
            content: "";
            position: absolute;
            inset: auto -28px -60px auto;
            width: 150px;
            height: 150px;
            background: radial-gradient(circle, rgba(207, 172, 107, 0.30), transparent 68%);
            pointer-events: none;
        }

        .hero-card {
            overflow: visible;
            padding: 2.2rem 2.2rem 2rem 2.2rem;
            background:
                linear-gradient(160deg, rgba(255, 255, 255, 0.88) 0%, rgba(247, 238, 226, 0.68) 58%, rgba(236, 225, 210, 0.82) 100%);
            border: 1px solid rgba(255, 255, 255, 0.58);
            box-shadow: 0 40px 120px rgba(15, 34, 50, 0.14);
        }

        .hero-card::before {
            inset: 14px;
            border: 1px solid rgba(255, 255, 255, 0.45);
        }

        .hero-card::after {
            width: 520px;
            height: 520px;
            right: -160px;
            top: -210px;
            background: radial-gradient(circle, rgba(207, 172, 107, 0.28), transparent 64%);
        }

        .masthead-strip {
            padding-bottom: 1.05rem;
            margin-bottom: 1.55rem;
            color: rgba(15, 34, 50, 0.68);
            border-bottom: 1px solid rgba(15, 34, 50, 0.10);
        }

        .hero-grid {
            grid-template-columns: minmax(0, 1.22fr) minmax(330px, 0.96fr);
            gap: 1.4rem;
            align-items: center;
        }

        .hero-title {
            max-width: 900px;
            text-wrap: balance;
            font-size: clamp(3.2rem, 5vw, 5.6rem);
        }

        .hero-copy {
            max-width: 720px;
            font-size: 1.01rem;
            line-height: 1.88;
        }

        .chip-row {
            gap: 0.7rem;
        }

        .insight-pill {
            padding: 0.66rem 0.9rem;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(12px);
        }

        .stat-strip {
            gap: 0.8rem;
            margin: 1.15rem 0 1.35rem 0;
        }

        .stat-item,
        .metric-card,
        .panel-card,
        .quote-block {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.58);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(246, 237, 225, 0.72));
            box-shadow: var(--shadow-soft);
        }

        .stat-item::after,
        .metric-card::after,
        .panel-card::after,
        .quote-block::after {
            content: "";
            position: absolute;
            inset: auto -30px -58px auto;
            width: 120px;
            height: 120px;
            background: radial-gradient(circle, rgba(207, 172, 107, 0.20), transparent 68%);
            pointer-events: none;
        }

        .stTabs [data-baseweb="tab-list"] {
            padding: 0.42rem;
            background: rgba(255, 255, 255, 0.52);
            border: 1px solid rgba(255, 255, 255, 0.60);
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(12px);
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.65rem 1.1rem;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(140deg, rgba(31, 110, 104, 0.20), rgba(197, 107, 85, 0.18), rgba(207, 172, 107, 0.22));
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.38);
        }

        [data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #0f2232 0%, #244059 100%);
        }

        [data-testid="stPlotlyChart"],
        [data-testid="stImage"] {
            border-radius: 30px;
            padding: 0.7rem;
            border: 1px solid rgba(255, 255, 255, 0.58);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(243, 234, 222, 0.72));
            box-shadow: var(--shadow-soft);
        }

        [data-testid="stPlotlyChart"] > div,
        [data-testid="stImage"] img {
            border-radius: 24px;
            overflow: hidden;
        }

        [data-testid="stExpander"] {
            border-radius: 28px;
            border: 1px solid rgba(255, 255, 255, 0.58);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(243, 234, 222, 0.65));
            box-shadow: var(--shadow-soft);
            overflow: hidden;
        }

        [data-testid="stDataFrame"] {
            border-radius: 28px;
            border: 1px solid rgba(255, 255, 255, 0.58);
            box-shadow: var(--shadow-soft);
            overflow: hidden;
        }

        .accent-blue {
            background: linear-gradient(90deg, #3f5f86, #85a6d0);
        }

        .signal-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1.15rem 0 1.45rem 0;
        }

        .signal-card {
            position: relative;
            overflow: hidden;
            padding: 1.12rem 1.1rem 1.05rem 1.1rem;
            border-radius: 26px;
            border: 1px solid rgba(255, 255, 255, 0.58);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(244, 235, 224, 0.72));
            box-shadow: var(--shadow-soft);
        }

        .signal-card::before {
            content: "";
            position: absolute;
            inset: -1px auto auto -1px;
            width: 44%;
            height: 4px;
            border-radius: 999px;
            background: var(--signal-accent, linear-gradient(90deg, #1f6e68, #cfac6b));
        }

        .signal-card::after {
            content: "";
            position: absolute;
            right: -28px;
            bottom: -40px;
            width: 120px;
            height: 120px;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.72), transparent 72%);
            pointer-events: none;
        }

        .signal-label {
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-size: 0.7rem;
            font-weight: 800;
            color: rgba(15, 34, 50, 0.56);
        }

        .signal-value {
            margin: 0.55rem 0 0.45rem 0;
            font-family: "Cormorant Garamond", serif;
            font-size: 2.2rem;
            line-height: 0.96;
            letter-spacing: -0.045em;
            color: var(--ink);
        }

        .signal-copy {
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.62;
        }

        .tone-teal {
            --signal-accent: linear-gradient(90deg, #1f6e68, #6bc6b5);
        }

        .tone-copper {
            --signal-accent: linear-gradient(90deg, #b47b38, #dfb271);
        }

        .tone-rose {
            --signal-accent: linear-gradient(90deg, #c56b55, #f0a08b);
        }

        .tone-blue {
            --signal-accent: linear-gradient(90deg, #3f5f86, #89a8cf);
        }

        .orbital-shell {
            position: relative;
            min-height: 360px;
        }

        .orbital-caption {
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.68rem;
            font-weight: 800;
            color: rgba(15, 34, 50, 0.58);
        }

        .orbital-stage {
            position: relative;
            min-height: 360px;
            border-radius: 34px;
            overflow: hidden;
            perspective: 1800px;
            transform-style: preserve-3d;
            border: 1px solid rgba(255, 255, 255, 0.58);
            background:
                radial-gradient(circle at 30% 28%, rgba(197, 107, 85, 0.16), transparent 26%),
                radial-gradient(circle at 76% 35%, rgba(31, 110, 104, 0.16), transparent 24%),
                linear-gradient(180deg, rgba(12, 26, 38, 0.96) 0%, rgba(20, 38, 54, 0.94) 100%);
            box-shadow: 0 40px 120px rgba(15, 34, 50, 0.24);
        }

        .orbital-stage::before {
            content: "";
            position: absolute;
            inset: 16px;
            border-radius: 26px;
            border: 1px solid rgba(255, 255, 255, 0.10);
            pointer-events: none;
        }

        .orbital-glow {
            position: absolute;
            inset: auto auto -120px -60px;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(207, 172, 107, 0.34), transparent 68%);
            filter: blur(18px);
            animation: pulse-glow 9s ease-in-out infinite;
        }

        .orbital-ring {
            position: absolute;
            top: 50%;
            left: 50%;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            transform-style: preserve-3d;
        }

        .ring-a {
            width: 250px;
            height: 250px;
            margin-left: -125px;
            margin-top: -125px;
            transform: rotateX(74deg) rotateY(16deg) translateZ(28px);
            animation: orbital-spin-a 18s linear infinite;
        }

        .ring-b {
            width: 328px;
            height: 328px;
            margin-left: -164px;
            margin-top: -164px;
            border-color: rgba(107, 198, 181, 0.22);
            transform: rotateX(78deg) rotateZ(26deg);
            animation: orbital-spin-b 24s linear infinite reverse;
        }

        .ring-c {
            width: 395px;
            height: 395px;
            margin-left: -197px;
            margin-top: -197px;
            border-color: rgba(197, 107, 85, 0.18);
            transform: rotateY(74deg) rotateX(84deg);
            animation: orbital-spin-c 28s linear infinite;
        }

        .orbital-core {
            position: absolute;
            top: 50%;
            left: 50%;
            width: min(72%, 285px);
            padding: 1.25rem 1.2rem 1.1rem 1.2rem;
            border-radius: 28px;
            transform: translate(-50%, -50%) translateZ(80px);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(235, 242, 246, 0.74));
            border: 1px solid rgba(255, 255, 255, 0.62);
            box-shadow: 0 28px 80px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(20px);
        }

        .orbital-core-kicker {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.7rem;
            font-weight: 800;
            color: rgba(15, 34, 50, 0.58);
        }

        .orbital-core-value {
            margin: 0.55rem 0 0.38rem 0;
            font-family: "Cormorant Garamond", serif;
            font-size: 2.7rem;
            line-height: 0.95;
            letter-spacing: -0.05em;
            color: #0f2232;
        }

        .orbital-core-copy {
            margin: 0;
            color: #5b6772;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .scene-tile {
            position: absolute;
            width: 170px;
            padding: 0.9rem 0.95rem 0.82rem 0.95rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.08));
            color: rgba(248, 252, 255, 0.90);
            box-shadow: 0 24px 48px rgba(0, 0, 0, 0.20);
            backdrop-filter: blur(16px);
            transform-style: preserve-3d;
            animation: float-card 7s ease-in-out infinite;
        }

        .scene-tile span {
            display: block;
            font-size: 0.68rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 800;
            color: rgba(238, 244, 247, 0.70);
        }

        .scene-tile strong {
            display: block;
            margin: 0.45rem 0 0.28rem 0;
            font-family: "Cormorant Garamond", serif;
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1;
            letter-spacing: -0.045em;
        }

        .scene-tile p {
            margin: 0;
            font-size: 0.82rem;
            line-height: 1.45;
            color: rgba(238, 244, 247, 0.72);
        }

        .tile-a {
            top: 26px;
            right: 18px;
            transform: translateZ(75px) rotateX(9deg) rotateY(-12deg);
        }

        .tile-b {
            bottom: 34px;
            right: 34px;
            transform: translateZ(95px) rotateX(7deg) rotateY(-16deg);
            animation-delay: -1.8s;
        }

        .tile-c {
            bottom: 30px;
            left: 18px;
            transform: translateZ(58px) rotateX(5deg) rotateY(12deg);
            animation-delay: -3.4s;
        }

        .tile-d {
            top: 78px;
            left: 10px;
            transform: translateZ(62px) rotateX(9deg) rotateY(14deg);
            animation-delay: -5.2s;
        }

        @keyframes orbital-spin-a {
            from {
                transform: rotateX(74deg) rotateY(16deg) rotateZ(0deg) translateZ(28px);
            }
            to {
                transform: rotateX(74deg) rotateY(16deg) rotateZ(360deg) translateZ(28px);
            }
        }

        @keyframes orbital-spin-b {
            from {
                transform: rotateX(78deg) rotateZ(26deg);
            }
            to {
                transform: rotateX(78deg) rotateZ(386deg);
            }
        }

        @keyframes orbital-spin-c {
            from {
                transform: rotateY(74deg) rotateX(84deg) rotateZ(0deg);
            }
            to {
                transform: rotateY(74deg) rotateX(84deg) rotateZ(360deg);
            }
        }

        @keyframes float-card {
            0%, 100% {
                translate: 0 0;
            }
            50% {
                translate: 0 -10px;
            }
        }

        @keyframes pulse-glow {
            0%, 100% {
                opacity: 0.75;
                transform: scale(1);
            }
            50% {
                opacity: 1;
                transform: scale(1.08);
            }
        }

        @media (max-width: 1180px) {
            .signal-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 980px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .orbital-shell {
                min-height: 330px;
            }

            .tile-a,
            .tile-b,
            .tile-c,
            .tile-d {
                width: 150px;
            }
        }

        @media (max-width: 720px) {
            .signal-grid {
                grid-template-columns: 1fr;
            }

            .orbital-stage {
                min-height: 330px;
            }

            .scene-tile {
                width: 140px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def outputs_exist() -> bool:
    return all((OUTPUTS_DIR / name).exists() for name in REQUIRED_OUTPUTS)


@st.cache_data(show_spinner=False)
def load_output_csv(filename: str, **kwargs: Any) -> pd.DataFrame:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def load_summary() -> dict[str, Any]:
    summary_path = OUTPUTS_DIR / "ui_metrics.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


@st.cache_data(show_spinner=False)
def list_chart_paths() -> list[Path]:
    return sorted(CHARTS_DIR.glob("*.png"))


def format_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{int(value):,}"


def format_compact_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def format_currency(value: Any, *, compact: bool = False) -> str:
    if value is None or pd.isna(value):
        return "--"
    value = float(value)
    if compact:
        return "$" + format_compact_number(value)
    return f"${value:,.0f}" if abs(value) >= 100 else f"${value:,.2f}"


def format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) * 100:.1f}%"


def format_timestamp(value: Any) -> str:
    if not value:
        return "--"
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "--"
    return timestamp.strftime("%b %d, %Y | %H:%M UTC")


def metric_card(label: str, value: str, detail: str, accent: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-accent accent-{accent}"></div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <p class="metric-detail">{detail}</p>
    </div>
    """


def panel_card(title: str, copy: str, accent: str) -> str:
    return f"""
    <div class="panel-card">
        <div class="panel-accent accent-{accent}"></div>
        <div class="panel-title">{title}</div>
        <p class="panel-copy">{copy}</p>
    </div>
    """


def stat_strip(items: list[tuple[str, str, str]]) -> str:
    blocks = "".join(
        f"""
        <div class="stat-item">
            <div class="stat-label">{label}</div>
            <div class="stat-number">{value}</div>
            <div class="stat-copy">{copy}</div>
        </div>
        """
        for label, value, copy in items
    )
    return f'<div class="stat-strip">{blocks}</div>'


def signal_grid(items: list[tuple[str, str, str, str]]) -> str:
    cards = "".join(
        f"""
        <div class="signal-card tone-{tone}">
            <div class="signal-label">{label}</div>
            <div class="signal-value">{value}</div>
            <p class="signal-copy">{copy}</p>
        </div>
        """
        for label, value, copy, tone in items
    )
    return f'<div class="signal-grid">{cards}</div>'


def orbital_scene(
    primary_label: str,
    primary_value: str,
    primary_copy: str,
    tiles: list[tuple[str, str, str]],
) -> str:
    tile_classes = ["tile-a", "tile-b", "tile-c", "tile-d"]
    tile_markup = "".join(
        f"""
        <div class="scene-tile {tile_class}">
            <span>{label}</span>
            <strong>{value}</strong>
            <p>{copy}</p>
        </div>
        """
        for tile_class, (label, value, copy) in zip(tile_classes, tiles)
    )
    return f"""
    <div class="orbital-shell">
        <div class="orbital-caption">3D Signal Vault</div>
        <div class="orbital-stage">
            <div class="orbital-glow"></div>
            <div class="orbital-ring ring-a"></div>
            <div class="orbital-ring ring-b"></div>
            <div class="orbital-ring ring-c"></div>
            <div class="orbital-core">
                <div class="orbital-core-kicker">{primary_label}</div>
                <div class="orbital-core-value">{primary_value}</div>
                <p class="orbital-core-copy">{primary_copy}</p>
            </div>
            {tile_markup}
        </div>
    </div>
    """


def quote_block(quote: str, source: str) -> str:
    return f"""
    <div class="quote-block">
        <div class="quote-mark">"</div>
        <div class="quote-text">{quote}</div>
        <div class="quote-source">{source}</div>
    </div>
    """


def section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-rule"></div>
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_archetype_colors(values: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for value in values:
        base = value.split(" (")[0]
        resolved[value] = ARCHETYPE_COLORS.get(base, "#7D878E")
    return resolved


def base_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_PLOT,
        font={"family": "Manrope, sans-serif", "color": "#0F2232", "size": 13},
        title={"x": 0.02, "font": {"family": "Cormorant Garamond, serif", "size": 25, "color": "#0F2232"}},
        margin={"l": 18, "r": 18, "t": 74, "b": 18},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255, 255, 255, 0.56)",
            "bordercolor": "rgba(15, 34, 50, 0.08)",
            "borderwidth": 1,
        },
        hoverlabel={"bgcolor": "#FFF7EF", "font_size": 13, "font_family": "Manrope"},
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor=PLOTLY_AXIS,
        gridcolor=PLOTLY_GRID,
        zeroline=False,
        title_standoff=14,
    )
    fig.update_yaxes(
        showline=False,
        gridcolor=PLOTLY_GRID,
        zeroline=False,
        title_standoff=14,
    )
    return fig


def build_sentiment_overview(perf: pd.DataFrame) -> go.Figure:
    ordered = perf.copy()
    ordered["classification"] = pd.Categorical(
        ordered["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    ordered = ordered.sort_values("classification")

    colors = [SENTIMENT_COLORS.get(name, "#7D878E") for name in ordered["classification"]]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_bar(
        x=ordered["classification"],
        y=ordered["median_pnl"],
        name="Median PnL",
        marker={"color": colors, "line": {"color": "#FFFFFF", "width": 0.5}},
        hovertemplate="%{x}<br>Median PnL: $%{y:,.0f}<extra></extra>",
    )
    figure.add_scatter(
        x=ordered["classification"],
        y=ordered["win_rate"] * 100,
        name="Realized Win Rate",
        mode="lines+markers",
        line={"color": "#18242D", "width": 3},
        marker={"size": 10, "color": "#F6F0E8", "line": {"color": "#18242D", "width": 2}},
        hovertemplate="%{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
        secondary_y=True,
    )
    figure.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(24, 36, 45, 0.32)",
        secondary_y=False,
    )
    figure.update_yaxes(title_text="Median Daily PnL (USD)", tickprefix="$", secondary_y=False)
    win_values = (ordered["win_rate"] * 100).dropna()
    if not win_values.empty:
        low = max(0, float(win_values.min()) - 8)
        high = min(100, float(win_values.max()) + 6)
    else:
        low, high = 0, 100
    figure.update_yaxes(title_text="Realized Win Rate", ticksuffix="%", secondary_y=True, range=[low, high])
    figure.update_layout(title="Performance differs across the matched sentiment events")
    return base_layout(figure)


def build_behavior_scatter(behavior: pd.DataFrame) -> go.Figure:
    frame = behavior.copy()
    frame["classification"] = pd.Categorical(
        frame["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    frame = frame.sort_values("classification")
    frame["crossed_share_pct"] = frame["crossed_share"] * 100

    figure = px.scatter(
        frame,
        x="avg_risk_size",
        y="crossed_share_pct",
        size="avg_notional",
        color="classification",
        size_max=42,
        color_discrete_map=SENTIMENT_COLORS,
        hover_data={
            "avg_trades": ":.2f",
            "avg_risk_size": ":,.0f",
            "crossed_share_pct": ":.1f",
            "avg_notional": ":,.0f",
            "close_share": ":.1%",
        },
    )
    figure.add_hline(
        y=50,
        line_dash="dot",
        line_color="rgba(24, 36, 45, 0.28)",
    )
    figure.update_layout(title="Sentiment shifts ticket size and execution style")
    figure.update_xaxes(title="Median Ticket Size (USD)", tickprefix="$")
    figure.update_yaxes(title="Crossed Share", ticksuffix="%")
    return base_layout(figure)


def build_event_timeline(event_summary: pd.DataFrame) -> go.Figure:
    frame = event_summary.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_bar(
        x=frame["date"],
        y=frame["total_pnl"],
        name="Total PnL",
        marker={"color": [SENTIMENT_COLORS.get(label, "#7D878E") for label in frame["classification"]]},
        hovertemplate="%{x|%Y-%m-%d}<br>Total PnL: $%{y:,.0f}<extra></extra>",
    )
    figure.add_scatter(
        x=frame["date"],
        y=frame["value"],
        name="Fear/Greed value",
        mode="lines+markers+text",
        text=frame["classification"],
        textposition="top center",
        line={"color": "#18242D", "width": 3},
        marker={"size": 10, "color": "#F6F0E8", "line": {"color": "#18242D", "width": 2}},
        hovertemplate="%{x|%Y-%m-%d}<br>Fear/Greed: %{y:.0f}<extra></extra>",
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Total PnL (USD)", tickprefix="$", secondary_y=False)
    figure.update_yaxes(title_text="Fear/Greed Value", secondary_y=True, range=[0, 100])
    figure.update_layout(title="Matched event-day timeline")
    return base_layout(figure)


def build_archetype_scatter(accounts: pd.DataFrame) -> go.Figure:
    figure = px.scatter(
        accounts,
        x="avg_risk_size",
        y="mean_pnl",
        color="archetype",
        size="avg_trades_day",
        size_max=32,
        color_discrete_map=resolve_archetype_colors(sorted(accounts["archetype"].dropna().unique())),
        hover_name="account",
        hover_data={
            "win_rate": ":.1%",
            "crossed_share": ":.1%",
            "max_drawdown": ":,.0f",
            "avg_trades_day": ":.2f",
        },
    )
    figure.add_hline(y=0, line_dash="dot", line_color="rgba(24, 36, 45, 0.28)")
    figure.update_layout(title="Trader DNA: ticket size, alpha, and operating tempo")
    figure.update_xaxes(title="Average Ticket Size (USD)", tickprefix="$")
    figure.update_yaxes(title="Mean Daily PnL", tickprefix="$")
    return base_layout(figure)


def build_trader_constellation(accounts: pd.DataFrame) -> go.Figure:
    frame = accounts.copy()
    max_trades = float(frame["avg_trades_day"].max()) if not frame.empty else 1.0
    frame["tempo_size"] = 14 + 26 * (frame["avg_trades_day"] / max_trades).pow(0.42)
    frame["win_rate_pct"] = frame["win_rate"] * 100
    frame["crossed_share_pct"] = frame["crossed_share"] * 100

    figure = px.scatter_3d(
        frame,
        x="avg_risk_size",
        y="avg_trades_day",
        z="mean_pnl",
        color="archetype",
        size="tempo_size",
        size_max=28,
        color_discrete_map=resolve_archetype_colors(sorted(frame["archetype"].dropna().unique())),
        hover_name="account",
        hover_data={
            "avg_risk_size": ":,.0f",
            "avg_trades_day": ":.2f",
            "mean_pnl": ":,.0f",
            "win_rate_pct": ":.1f",
            "crossed_share_pct": ":.1f",
            "sharpe_proxy": ":.2f",
            "max_drawdown": ":,.0f",
            "tempo_size": False,
        },
    )
    figure.update_traces(
        marker={
            "opacity": 0.95,
            "line": {"color": "rgba(255, 255, 255, 0.68)", "width": 1.5},
            "symbol": "circle",
        }
    )
    figure.update_layout(
        title="3D trader constellation: size, pace, and alpha in one room",
        paper_bgcolor=PLOTLY_PAPER,
        font={"family": "Manrope, sans-serif", "color": "#0F2232", "size": 13},
        hoverlabel={"bgcolor": "#FFF7EF", "font_size": 13, "font_family": "Manrope"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255, 255, 255, 0.56)",
            "bordercolor": "rgba(15, 34, 50, 0.08)",
            "borderwidth": 1,
        },
        scene={
            "aspectmode": "manual",
            "aspectratio": {"x": 1.05, "y": 0.95, "z": 0.92},
            "bgcolor": "rgba(0, 0, 0, 0)",
            "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.86}},
            "xaxis": {
                "title": "Average Ticket Size (USD)",
                "tickprefix": "$",
                "gridcolor": "rgba(15, 34, 50, 0.12)",
                "zerolinecolor": "rgba(15, 34, 50, 0.10)",
                "showbackground": True,
                "backgroundcolor": "rgba(15, 34, 50, 0.05)",
            },
            "yaxis": {
                "title": "Trades / Day",
                "gridcolor": "rgba(15, 34, 50, 0.12)",
                "zerolinecolor": "rgba(15, 34, 50, 0.10)",
                "showbackground": True,
                "backgroundcolor": "rgba(15, 34, 50, 0.04)",
            },
            "zaxis": {
                "title": "Mean Daily PnL",
                "tickprefix": "$",
                "gridcolor": "rgba(15, 34, 50, 0.12)",
                "zerolinecolor": "rgba(15, 34, 50, 0.10)",
                "showbackground": True,
                "backgroundcolor": "rgba(15, 34, 50, 0.06)",
            },
        },
        margin={"l": 0, "r": 0, "t": 74, "b": 0},
    )
    return figure


def build_archetype_donut(accounts: pd.DataFrame) -> go.Figure:
    counts = (
        accounts["archetype"]
        .value_counts()
        .rename_axis("archetype")
        .reset_index(name="count")
    )
    figure = go.Figure(
        data=[
            go.Pie(
                labels=counts["archetype"],
                values=counts["count"],
                hole=0.62,
                sort=False,
                marker={
                    "colors": [
                        resolve_archetype_colors(counts["archetype"].tolist())[name]
                        for name in counts["archetype"]
                    ]
                },
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value} traders<extra></extra>",
            )
        ]
    )
    figure.update_layout(title="Archetype mix")
    return base_layout(figure)


def build_segment_bar(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_sequence: list[str],
    tickprefix: str = "",
    ticksuffix: str = "",
) -> go.Figure:
    figure = px.bar(
        frame,
        x=x_col,
        y=y_col,
        color=x_col,
        color_discrete_sequence=color_sequence,
        text_auto=".2s",
    )
    figure.update_layout(showlegend=False, title=title)
    figure.update_xaxes(title="")
    figure.update_yaxes(title="", tickprefix=tickprefix, ticksuffix=ticksuffix)
    return base_layout(figure)


def build_gallery_columns(chart_paths: list[Path]) -> None:
    columns = st.columns(2, gap="large")
    for index, chart_path in enumerate(chart_paths):
        with columns[index % 2]:
            st.image(
                str(chart_path),
                caption=CHART_LABELS.get(chart_path.name, chart_path.name),
                width="stretch",
            )


def run_analysis_pipeline() -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, ANALYSIS_SCRIPT.name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    output = completed.stdout.strip()
    errors = completed.stderr.strip()
    log = "\n\n".join(part for part in [output, errors] if part)
    return completed.returncode, log or "No output was captured."


def show_empty_state() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">UI Ready</div>
            <div class="hero-grid">
                <div>
                    <h1 class="hero-title">Generate the research package first</h1>
                    <p class="hero-copy">
                        This dashboard sits on top of the analysis pipeline. Run the project once and it will
                        populate the event-study UI with the matched sentiment tables, trader archetypes,
                        strategy playbook, and chart archive.
                    </p>
                    <div class="chip-row">
                        <span class="insight-pill">Reads outputs/ and charts/</span>
                        <span class="insight-pill">Run with python analysis.py</span>
                        <span class="insight-pill">Or use the sidebar action</span>
                    </div>
                </div>
                <div class="hero-note">
                    <div class="note-label">Launch path</div>
                    <div class="note-value">streamlit run app.py</div>
                    <p class="note-copy">
                        Use the sidebar action to regenerate every chart and CSV from the raw files.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

if "analysis_log" not in st.session_state:
    st.session_state["analysis_log"] = ""

if "run_message" in st.session_state:
    st.success(st.session_state.pop("run_message"))

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-kicker">Signal Atlas</div>
            <div class="sidebar-title">Hyperliquid Command Deck</div>
            <p class="sidebar-copy">
                A premium control room for the real assignment data. Refresh the pipeline,
                inspect the matched event days, and move between 3D atlas views, narrative, strategy, and evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rerun = st.button("Run Full Analysis", type="primary", width="stretch")
    st.caption("This regenerates every chart, CSV, and summary metric from the raw data files.")

    if rerun:
        with st.spinner("Running analysis.py and refreshing the dashboard outputs..."):
            exit_code, run_log = run_analysis_pipeline()
        st.session_state["analysis_log"] = run_log
        if exit_code == 0:
            st.cache_data.clear()
            st.session_state["run_message"] = "Analysis refreshed successfully."
            st.rerun()
        st.error("The analysis run failed. Open the log below for details.")

    if st.session_state["analysis_log"]:
        with st.expander("Latest run log", expanded=False):
            st.code(st.session_state["analysis_log"], language="text")


if not outputs_exist():
    show_empty_state()
    st.stop()


summary = load_summary()
performance = load_output_csv("performance_by_sentiment.csv")
behavior = load_output_csv("behavior_by_sentiment.csv")
accounts = load_output_csv("account_summary.csv")
cluster_profiles = load_output_csv("cluster_profiles.csv")
risk_seg = load_output_csv("leverage_segmentation.csv")
freq_seg = load_output_csv("frequency_segmentation.csv")
consistency_seg = load_output_csv("consistency_segmentation.csv")
execution_seg = load_output_csv("execution_segmentation.csv")
event_summary = load_output_csv("event_summary.csv")
robustness = load_output_csv("robustness_checks.csv")
strategy_playbook = load_output_csv("strategy_playbook.csv")
chart_paths = list_chart_paths()

archetype_options = sorted(accounts["archetype"].dropna().unique().tolist())
max_risk = float(accounts["avg_risk_size"].max())

with st.sidebar:
    st.markdown("---")
    st.markdown("#### Controls")
    focus_metric = st.selectbox(
        "Spotlight metric",
        options=["mean_pnl", "win_rate", "avg_risk_size", "crossed_share"],
        format_func=lambda value: {
            "mean_pnl": "Mean Daily PnL",
            "win_rate": "Win Rate",
            "avg_risk_size": "Ticket Size",
            "crossed_share": "Crossed Share",
        }[value],
    )
    selected_archetypes = st.multiselect(
        "Archetypes",
        options=archetype_options,
        default=archetype_options,
    )
    risk_window = st.slider(
        "Ticket size window",
        min_value=0.0,
        max_value=round(max_risk + 500, 0),
        value=(0.0, round(max_risk + 500, 0)),
        step=100.0,
    )
    sort_metric = st.selectbox(
        "Trader table sort",
        options=["mean_pnl", "win_rate", "avg_risk_size", "avg_trades_day", "max_drawdown", "crossed_share"],
        format_func=lambda value: {
            "mean_pnl": "Mean Daily PnL",
            "win_rate": "Win Rate",
            "avg_risk_size": "Ticket Size",
            "avg_trades_day": "Trades per Day",
            "max_drawdown": "Max Drawdown",
            "crossed_share": "Crossed Share",
        }[value],
    )

filtered_accounts = accounts[
    accounts["archetype"].isin(selected_archetypes)
    & accounts["avg_risk_size"].between(risk_window[0], risk_window[1])
].copy()

if filtered_accounts.empty:
    st.warning("The current filters exclude every trader. Adjust the archetype or ticket-size filters.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.download_button(
        "Download Filtered Trader Table",
        data=filtered_accounts.to_csv(index=False).encode("utf-8"),
        file_name="hyperliquid_filtered_accounts.csv",
        mime="text/csv",
        width="stretch",
    )


generated_at = format_timestamp(summary.get("generated_at_utc"))
top_archetype = summary.get("top_archetype") or "Unavailable"
top_archetype_count = summary.get("top_archetype_count")
matched_days_text = format_int(summary.get("matched_days"))
paired_count_text = format_int(summary.get("paired_account_count"))
trade_count_text = format_compact_number(summary.get("matched_trades") or summary.get("total_trades"))
trader_count_text = format_int(summary.get("unique_traders"))
chart_count_text = format_int(summary.get("chart_count"))
table_count_text = format_int(summary.get("table_count"))

execution_edge = None
if summary.get("patient_executor_mean_pnl") is not None and summary.get("aggressive_executor_mean_pnl") is not None:
    execution_edge = summary["patient_executor_mean_pnl"] - summary["aggressive_executor_mean_pnl"]

greed_ticket_edge = None
if summary.get("greed_small_ticket_median_pnl") is not None and summary.get("greed_large_ticket_median_pnl") is not None:
    greed_ticket_edge = summary["greed_small_ticket_median_pnl"] - summary["greed_large_ticket_median_pnl"]

hero_tiles = [
    ("Matched Days", matched_days_text, "The joined dataset behaves like a focused event study."),
    ("Paired Accounts", paired_count_text, "Accounts active in both fear and greed windows."),
    ("Greed Ticket Cap", format_currency(summary.get("greed_ticket_threshold"), compact=True), "Above this level, greed-day results soften materially."),
    ("Execution Threshold", format_pct(summary.get("execution_threshold")), "Median crossed-share split between patient and aggressive routing."),
]

command_cards = [
    (
        "Execution Premium",
        format_currency(execution_edge, compact=True),
        "Patient execution materially out-earns aggressive crossing in the matched sample.",
        "teal",
    ),
    (
        "Greed Ticket Delta",
        format_currency(greed_ticket_edge, compact=True),
        "Smaller greed-day tickets hold up better than oversized aggression.",
        "copper",
    ),
    (
        "Research Stance",
        f"{matched_days_text} dates",
        "The interface is built like a signal room because the data honestly supports an event study, not a daily backtest.",
        "rose",
    ),
    (
        "Dominant Archetype",
        top_archetype,
        f"{format_int(top_archetype_count)} traders sit inside the largest operating cluster.",
        "blue",
    ),
]

story_stats = [
    ("Matched Trades", trade_count_text, "Exact date overlaps feeding the signal atlas."),
    ("Paired Accounts", paired_count_text, "Accounts seen in both fear and greed windows."),
    ("Archive Plates", chart_count_text, "Generated visuals preserved in the evidence vault."),
    ("Data Ledgers", table_count_text, "Structured tables ready for download and review."),
]

st.markdown(
    f"""
    <div class="hero-card">
        <div class="masthead-strip">
            <span>Issue 03 | Signal Atlas</span>
            <span>Hyperliquid x Fear-Greed</span>
            <span>{generated_at}</span>
        </div>
        <div class="hero-grid">
            <div>
                <div class="eyebrow">Premium Research Interface</div>
                <h1 class="hero-title">A cinematic command deck for Hyperliquid's real event days.</h1>
                <p class="hero-copy">
                    Built on the actual assignment files, this experience turns a thin matched panel into a premium signal atlas.
                    Only six calendar dates overlap between the trade export and the sentiment feed, so the product leans into
                    what the data truly supports: execution quality, ticket discipline, trader archetypes, and the shape of edge
                    when the market regime is emotionally charged.
                </p>
                <div class="chip-row">
                    <span class="insight-pill">Coverage {summary.get("date_start", "--")} to {summary.get("date_end", "--")}</span>
                    <span class="insight-pill">{matched_days_text} matched days</span>
                    <span class="insight-pill">{trade_count_text} matched trades</span>
                    <span class="insight-pill">{trader_count_text} traders</span>
                    <span class="insight-pill">{chart_count_text} visual plates</span>
                </div>
            </div>
            {orbital_scene(
                "Execution Spread",
                format_currency(execution_edge, compact=True),
                "The cleanest repeatable edge in the real export is patient routing over aggressive crossing.",
                hero_tiles,
            )}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(signal_grid(command_cards), unsafe_allow_html=True)
st.markdown(stat_strip(story_stats), unsafe_allow_html=True)

lead_left, lead_right = st.columns([1.2, 0.8], gap="large")
with lead_left:
    st.markdown(
        quote_block(
            "The cleanest advantage in the real file is not 'greed good, fear bad'. "
            "It is that patient execution and smaller greed-day tickets preserve edge better than aggression.",
            "Editorial Summary",
        ),
        unsafe_allow_html=True,
    )
with lead_right:
    st.markdown(
        panel_card(
            "Front-of-book framing",
            f"The report covers {trade_count_text} matched trades across {trader_count_text} traders. "
            "Instead of pretending this is a full daily backtest, the dashboard leans into what the sample actually is: "
            "a compact event study on how execution style and ticket sizing interact with sentiment snapshots.",
            accent="copper",
        ),
        unsafe_allow_html=True,
    )


tabs = st.tabs(["Command Deck", "3D Atlas", "Cuts", "Playbook", "Archive"])

with tabs[0]:
    section_header(
        "Command Deck",
        "The market regime story, framed with restraint and product-level polish",
        "This opening deck keeps the honest read front and center. The matched panel is thin in time, so the strongest story is not broad sentiment prophecy. "
        "It is the interaction between execution style, ticket sizing, and what traders do when the tape gets emotional.",
    )

    chart_left, chart_right = st.columns([1.1, 0.9], gap="large")
    with chart_left:
        st.plotly_chart(build_sentiment_overview(performance), width="stretch")
    with chart_right:
        st.plotly_chart(build_behavior_scatter(behavior), width="stretch")

    timeline_left, timeline_right = st.columns([1.1, 0.9], gap="large")
    with timeline_left:
        st.plotly_chart(build_event_timeline(event_summary), width="stretch")
    with timeline_right:
        if focus_metric == "mean_pnl":
            spotlight = performance.sort_values("mean_pnl", ascending=False).iloc[0]
            spotlight_copy = (
                f"{spotlight['classification']} posts the highest mean account-day PnL "
                f"at {format_currency(spotlight['mean_pnl'])} in the matched sample."
            )
        elif focus_metric == "win_rate":
            spotlight = performance.sort_values("win_rate", ascending=False).iloc[0]
            spotlight_copy = (
                f"{spotlight['classification']} leads on realized win rate at {format_pct(spotlight['win_rate'])}."
            )
        elif focus_metric == "avg_risk_size":
            spotlight = behavior.sort_values("avg_risk_size", ascending=False).iloc[0]
            spotlight_copy = (
                f"{spotlight['classification']} carries the largest median ticket size "
                f"at {format_currency(spotlight['avg_risk_size'])}."
            )
        else:
            spotlight = behavior.sort_values("crossed_share", ascending=False).iloc[0]
            spotlight_copy = (
                f"{spotlight['classification']} shows the highest crossed-order share "
                f"at {format_pct(spotlight['crossed_share'])}."
            )

        st.markdown(
            panel_card(
                "Spotlight line",
                spotlight_copy,
                accent="rose",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            panel_card(
                "Data reality check",
                f"{matched_days_text} matched dates means the assignment behaves like an event study, not a full daily panel. "
                "That constraint is explicit throughout the charts and write-up.",
                accent="teal",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            panel_card(
                "Execution quality matters",
                f"Patient executors average {format_currency(summary.get('patient_executor_mean_pnl'))}, while "
                f"aggressive takers average {format_currency(summary.get('aggressive_executor_mean_pnl'))}.",
                accent="copper",
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Source tables"):
        table_left, table_right = st.columns(2, gap="large")
        with table_left:
            st.dataframe(
                performance,
                width="stretch",
                hide_index=True,
                column_config={
                    "median_pnl": st.column_config.NumberColumn("Median PnL", format="$%.2f"),
                    "mean_pnl": st.column_config.NumberColumn("Mean PnL", format="$%.2f"),
                    "profit_day_rate": st.column_config.NumberColumn("Profit-day Rate", format="%.2f"),
                    "win_rate": st.column_config.NumberColumn("Realized Win Rate", format="%.2f"),
                    "crossed_share": st.column_config.NumberColumn("Crossed Share", format="%.2f"),
                },
            )
        with table_right:
            st.dataframe(
                behavior,
                width="stretch",
                hide_index=True,
                column_config={
                    "avg_trades": st.column_config.NumberColumn("Median Trades", format="%.2f"),
                    "avg_risk_size": st.column_config.NumberColumn("Median Ticket Size", format="$%.0f"),
                    "long_ratio": st.column_config.NumberColumn("Buy Share", format="%.2f"),
                    "avg_notional": st.column_config.NumberColumn("Median Total Volume", format="$%.0f"),
                    "crossed_share": st.column_config.NumberColumn("Crossed Share", format="%.2f"),
                    "close_share": st.column_config.NumberColumn("Close-share", format="%.2f"),
                    "fee_bps": st.column_config.NumberColumn("Fee bps", format="%.2f"),
                },
            )

with tabs[1]:
    section_header(
        "3D Atlas",
        "Trader temperament mapped as a premium spatial constellation",
        "This atlas turns the account universe into a depth scene. The 3D view makes ticket size, operating tempo, and alpha legible in one move, "
        "while the flat projection and leaderboard keep the picture practical for review.",
    )

    st.plotly_chart(build_trader_constellation(filtered_accounts), width="stretch")

    dna_left, dna_right = st.columns([1.2, 0.8], gap="large")
    with dna_left:
        st.plotly_chart(build_archetype_scatter(filtered_accounts), width="stretch")
    with dna_right:
        st.plotly_chart(build_archetype_donut(filtered_accounts), width="stretch")

        sort_labels = {
            "mean_pnl": "Mean Daily PnL",
            "win_rate": "Win Rate",
            "avg_risk_size": "Ticket Size",
            "avg_trades_day": "Trades per Day",
            "max_drawdown": "Max Drawdown",
            "crossed_share": "Crossed Share",
        }
        atlas_leader = filtered_accounts.sort_values(sort_metric, ascending=False).iloc[0]
        if sort_metric == "mean_pnl":
            atlas_value = format_currency(atlas_leader[sort_metric])
        elif sort_metric == "avg_risk_size":
            atlas_value = format_currency(atlas_leader[sort_metric])
        elif sort_metric == "avg_trades_day":
            atlas_value = f"{atlas_leader[sort_metric]:.1f}"
        elif sort_metric == "max_drawdown":
            atlas_value = format_currency(atlas_leader[sort_metric])
        else:
            atlas_value = format_pct(atlas_leader[sort_metric])

        atlas_account = f"{atlas_leader['account'][:10]}...{atlas_leader['account'][-6:]}"
        st.markdown(
            panel_card(
                "Atlas reading",
                f"{atlas_account} leads the filtered view on {sort_labels[sort_metric]} at {atlas_value}. "
                f"The 3D scene makes it easy to spot how {atlas_leader['archetype']} separates on tempo, ticket size, and outcome.",
                accent="blue",
            ),
            unsafe_allow_html=True,
        )

    leaderboard = filtered_accounts.sort_values(sort_metric, ascending=False).head(15).copy()
    leaderboard["win_rate"] = leaderboard["win_rate"].round(4)
    leaderboard["crossed_share"] = leaderboard["crossed_share"].round(4)

    st.dataframe(
        leaderboard[
            [
                "account",
                "archetype",
                "mean_pnl",
                "win_rate",
                "avg_risk_size",
                "avg_trades_day",
                "crossed_share",
                "max_drawdown",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "mean_pnl": st.column_config.NumberColumn("Mean Daily PnL", format="$%.2f"),
            "win_rate": st.column_config.NumberColumn("Win Rate", format="%.2f"),
            "avg_risk_size": st.column_config.NumberColumn("Avg Ticket Size", format="$%.0f"),
            "avg_trades_day": st.column_config.NumberColumn("Trades / Day", format="%.2f"),
            "crossed_share": st.column_config.NumberColumn("Crossed Share", format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("Max Drawdown", format="$%.0f"),
        },
    )

with tabs[2]:
    section_header(
        "Cuts",
        "The cleanest cross-sections once the trader universe is sliced apart",
        "These comparative cuts focus on the three strongest lenses in the real export: ticket size, activity, and consistency. "
        "Execution style sits beside them as the practical bridge from observation to action.",
    )

    def ensure_segment(frame: pd.DataFrame) -> pd.DataFrame:
        updated = frame.rename(columns={"Unnamed: 0": "segment"}).copy()
        if "segment" not in updated.columns:
            updated = updated.rename(columns={updated.columns[0]: "segment"})
        return updated

    risk_frame = ensure_segment(risk_seg)
    freq_frame = ensure_segment(freq_seg).sort_values("win_rate", ascending=False)
    consistency_frame = ensure_segment(consistency_seg)

    seg_col_1, seg_col_2, seg_col_3 = st.columns(3, gap="large")
    with seg_col_1:
        st.plotly_chart(
            build_segment_bar(
                risk_frame,
                "segment",
                "mean_pnl",
                "Ticket-size cohorts by mean daily PnL",
                ["#506D9A", "#C6A15E", "#1D6B67"],
                tickprefix="$",
            ),
            width="stretch",
        )
    with seg_col_2:
        st.plotly_chart(
            build_segment_bar(
                freq_frame,
                "segment",
                "win_rate",
                "Activity split by win rate",
                ["#C67C3D", "#506D9A"],
            ),
            width="stretch",
        )
    with seg_col_3:
        st.plotly_chart(
            build_segment_bar(
                consistency_frame,
                "segment",
                "mean_pnl",
                "Consistency cohorts by mean daily PnL",
                ["#C95A4D", "#A6A39F", "#1D6B67"],
                tickprefix="$",
            ),
            width="stretch",
        )

    cut_left, cut_right = st.columns([0.95, 1.05], gap="large")
    with cut_left:
        st.markdown(
            panel_card(
                "Editors' reading of the cuts",
                "Frequent traders are bigger winners in this event-driven export, but the cleaner repeatable edge comes from execution discipline. "
                "Patient executors keep fee drag lower, and greed-day performance is materially stronger when traders scale with smaller tickets.",
                accent="teal",
            ),
            unsafe_allow_html=True,
        )
        if not execution_seg.empty:
            st.dataframe(
                execution_seg,
                width="stretch",
                hide_index=True,
                column_config={
                    "mean_pnl": st.column_config.NumberColumn("Mean PnL", format="$%.2f"),
                    "win_rate": st.column_config.NumberColumn("Win Rate", format="%.2f"),
                    "avg_risk_size": st.column_config.NumberColumn("Avg Ticket Size", format="$%.0f"),
                    "crossed_share": st.column_config.NumberColumn("Crossed Share", format="%.2f"),
                    "fee_bps": st.column_config.NumberColumn("Fee bps", format="%.2f"),
                },
            )
    with cut_right:
        st.dataframe(
            cluster_profiles,
            width="stretch",
            hide_index=True,
            column_config={
                "avg_risk_size": st.column_config.NumberColumn("Avg Ticket Size", format="$%.0f"),
                "avg_trades_day": st.column_config.NumberColumn("Trades / Day", format="%.2f"),
                "win_rate": st.column_config.NumberColumn("Win Rate", format="%.2f"),
                "crossed_share": st.column_config.NumberColumn("Crossed Share", format="%.2f"),
                "mean_pnl": st.column_config.NumberColumn("Mean PnL", format="$%.2f"),
                "max_drawdown": st.column_config.NumberColumn("Max Drawdown", format="$%.0f"),
            },
        )

with tabs[3]:
    section_header(
        "Playbook",
        "Robustness first, then action",
        "This final spread turns the research into practical rules. The focus is on which comparisons survive the thin matched panel and which thresholds are concrete enough to use as rules of thumb.",
    )

    play_1, play_2, play_3, play_4 = st.columns(4, gap="large")
    with play_1:
        st.markdown(
            metric_card(
                "Matched Days",
                matched_days_text,
                "The true time coverage of the joined dataset.",
                accent="teal",
            ),
            unsafe_allow_html=True,
        )
    with play_2:
        st.markdown(
            metric_card(
                "Paired Accounts",
                paired_count_text,
                "Accounts active in both fear and greed windows.",
                accent="copper",
            ),
            unsafe_allow_html=True,
        )
    with play_3:
        st.markdown(
            metric_card(
                "Crossed Threshold",
                f"{summary.get('execution_threshold', 0):.2f}" if summary.get("execution_threshold") is not None else "--",
                "Median crossed-share split used to separate patient from aggressive execution.",
                accent="rose",
            ),
            unsafe_allow_html=True,
        )
    with play_4:
        st.markdown(
            metric_card(
                "Greed Ticket Cap",
                format_currency(summary.get("greed_ticket_threshold"), compact=True),
                "Median greed-day ticket size used in the playbook split.",
                accent="teal",
            ),
            unsafe_allow_html=True,
        )

    play_left, play_right = st.columns(2, gap="large")
    with play_left:
        robustness_chart = CHARTS_DIR / "09_robustness_checks.png"
        if robustness_chart.exists():
            st.image(str(robustness_chart), caption=CHART_LABELS[robustness_chart.name], width="stretch")
    with play_right:
        strategy_chart = CHARTS_DIR / "10_strategy_playbook.png"
        if strategy_chart.exists():
            st.image(str(strategy_chart), caption=CHART_LABELS[strategy_chart.name], width="stretch")

    table_left, table_right = st.columns(2, gap="large")
    with table_left:
        st.dataframe(
            robustness,
            width="stretch",
            hide_index=True,
            column_config={
                "effect_value": st.column_config.NumberColumn("Effect", format="$%.2f"),
                "pvalue": st.column_config.NumberColumn("p-value", format="%.4f"),
            },
        )
    with table_right:
        st.dataframe(
            strategy_playbook,
            width="stretch",
            hide_index=True,
            column_config={
                "sample_size": st.column_config.NumberColumn("Sample", format="%d"),
                "median_pnl": st.column_config.NumberColumn("Median PnL", format="$%.2f"),
                "mean_pnl": st.column_config.NumberColumn("Mean PnL", format="$%.2f"),
                "profit_day_rate": st.column_config.NumberColumn("Profit-day Rate", format="%.2f"),
                "win_rate": st.column_config.NumberColumn("Win Rate", format="%.2f"),
                "avg_trade_usd": st.column_config.NumberColumn("Avg Ticket Size", format="$%.0f"),
                "crossed_share": st.column_config.NumberColumn("Crossed Share", format="%.2f"),
            },
        )

with tabs[4]:
    section_header(
        "Archive",
        "Every original research plate, collected like an appendix",
        "The dashboard adds interaction, but the static figures are still the core evidence package. "
        "Use the archive when you want the original visuals exactly as the analysis generated them.",
    )
    build_gallery_columns(chart_paths)
