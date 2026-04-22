from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
CHARTS_DIR = ROOT / "charts"

OUTPUTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

GENERATED_CHARTS = [
    "01_performance_by_sentiment.png",
    "02_timeline_sentiment_pnl.png",
    "03_behavioral_fingerprint.png",
    "04_event_coverage.png",
    "05_segmentation_deep_dive.png",
    "06_directional_bias.png",
    "07_archetypes.png",
    "08_cluster_profiles.png",
    "09_robustness_checks.png",
    "10_strategy_playbook.png",
    "11_drawdown_analysis.png",
]

GENERATED_TABLES = [
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
]

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
    "Aggressive Takers": "#C95A4D",
    "Patient Position Builders": "#1D6B67",
    "Selective Rotators": "#506D9A",
    "Impulse Scalpers": "#B4935D",
}

BACKGROUND = "#F8F4EE"
PANEL = "#FFFDFC"
INK = "#231B17"
MUTED = "#6A6158"
GRID = "#D9CFC3"

plt.rcParams.update(
    {
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BACKGROUND,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.35,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.frameon": False,
        "font.size": 10,
        "figure.dpi": 120,
    }
)

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


@dataclass
class LoadedDataset:
    sentiment: pd.DataFrame
    trades: pd.DataFrame
    sentiment_path: Path
    trades_path: Path
    schema: str


def print_rule(title: str) -> None:
    line = "=" * 84
    print(f"\n{line}\n{title}\n{line}")


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: charts/{name}.png")


def clean_previous_outputs() -> None:
    for path in CHARTS_DIR.glob("*.png"):
        path.unlink()

    for path in OUTPUTS_DIR.glob("*.csv"):
        path.unlink()

    metrics_path = OUTPUTS_DIR / "ui_metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()


def pick_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find any of the expected files: {joined}")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: (
            column.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def infer_epoch_unit(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "ms"
    max_value = float(numeric.abs().max())
    if max_value > 1e14:
        return "ns"
    if max_value > 1e11:
        return "ms"
    return "s"


def load_sentiment_dataset() -> tuple[pd.DataFrame, Path]:
    candidates = [
        DATA_DIR / "fear_greed_index.csv",
        ROOT / "real_data" / "fear_greed_download",
        DATA_DIR / "fear_greed.csv",
    ]
    path = pick_existing(candidates)
    frame = normalize_columns(pd.read_csv(path))

    if "date" not in frame.columns and "timestamp" in frame.columns:
        unit = infer_epoch_unit(frame["timestamp"])
        frame["date"] = pd.to_datetime(frame["timestamp"], unit=unit, errors="coerce")
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    frame["value"] = pd.to_numeric(frame.get("value"), errors="coerce")
    frame["classification"] = frame["classification"].astype(str).str.strip()
    frame = frame[["date", "value", "classification"]].dropna(subset=["date", "classification"])
    frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()
    frame = frame.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame, path


def load_trade_dataset() -> tuple[pd.DataFrame, Path, str]:
    candidates = [
        DATA_DIR / "historical_data.csv",
        ROOT / "real_data" / "historical_download",
        DATA_DIR / "trader_data.csv",
    ]
    path = pick_existing(candidates)
    frame = normalize_columns(pd.read_csv(path, low_memory=False))

    alias_map = {
        "coin": "symbol",
        "closedpnl": "closed_pnl",
        "size": "size_tokens",
        "time": "time",
        "execution_price": "execution_price",
        "size_tokens": "size_tokens",
        "size_usd": "size_usd",
        "timestamp_ist": "timestamp_ist",
        "start_position": "start_position",
        "direction": "direction",
        "crossed": "crossed",
        "fee": "fee",
        "timestamp": "timestamp",
        "account": "account",
        "symbol": "symbol",
        "side": "side",
        "event": "event",
        "leverage": "leverage",
    }
    frame = frame.rename(columns={column: alias_map[column] for column in frame.columns if column in alias_map})

    schema = "hyperliquid_export" if "size_usd" in frame.columns and "timestamp" in frame.columns else "synthetic_placeholder"

    if "timestamp" in frame.columns:
        unit = infer_epoch_unit(frame["timestamp"])
        frame["time"] = pd.to_datetime(frame["timestamp"], unit=unit, errors="coerce")
    elif "timestamp_ist" in frame.columns:
        frame["time"] = pd.to_datetime(frame["timestamp_ist"], errors="coerce", dayfirst=True)
    else:
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")

    if "direction" not in frame.columns:
        if "event" in frame.columns:
            frame["direction"] = np.where(
                frame["event"].astype(str).str.contains("LIQ", case=False, na=False),
                "Liquidation",
                np.where(frame["side"].astype(str).str.upper() == "BUY", "Open Long", "Open Short"),
            )
        else:
            frame["direction"] = np.nan

    numeric_columns = [
        "execution_price",
        "size_tokens",
        "size_usd",
        "start_position",
        "closed_pnl",
        "fee",
        "leverage",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "size_usd" not in frame.columns:
        frame["size_usd"] = frame["execution_price"].fillna(0) * frame["size_tokens"].fillna(0)
    if "start_position" not in frame.columns:
        frame["start_position"] = 0.0
    if "fee" not in frame.columns:
        frame["fee"] = 0.0
    if "crossed" not in frame.columns:
        frame["crossed"] = False
    if "symbol" not in frame.columns:
        frame["symbol"] = "UNKNOWN"
    if "side" not in frame.columns:
        frame["side"] = "UNKNOWN"

    frame["crossed"] = frame["crossed"].astype(str).str.lower().map({"true": True, "false": False}).fillna(frame["crossed"])
    frame["side"] = frame["side"].astype(str).str.upper()
    frame["direction"] = frame["direction"].astype(str)
    frame["account"] = frame["account"].astype(str)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce").dt.tz_localize(None)
    frame["date"] = frame["time"].dt.normalize()

    columns = [
        "account",
        "symbol",
        "execution_price",
        "size_tokens",
        "size_usd",
        "side",
        "time",
        "date",
        "start_position",
        "direction",
        "closed_pnl",
        "crossed",
        "fee",
        "leverage",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = frame[columns].dropna(subset=["account", "time", "date", "closed_pnl"]).reset_index(drop=True)
    return frame, path, schema


def load_datasets() -> LoadedDataset:
    sentiment, sentiment_path = load_sentiment_dataset()
    trades, trades_path, schema = load_trade_dataset()
    return LoadedDataset(
        sentiment=sentiment,
        trades=trades,
        sentiment_path=sentiment_path,
        trades_path=trades_path,
        schema=schema,
    )


def realized_win_rate(series: pd.Series) -> float:
    realized = pd.to_numeric(series, errors="coerce").dropna()
    realized = realized[realized != 0]
    if realized.empty:
        return np.nan
    return float((realized > 0).mean())


def positive_sum(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float(numeric[numeric > 0].sum())


def negative_abs_sum(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float(-numeric[numeric < 0].sum())


def safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.median())


def drawdown_from_curve(curve: pd.Series) -> float:
    if curve.empty:
        return np.nan
    running_peak = curve.cummax()
    return float((curve - running_peak).min())


def classify_binary(label: str) -> str:
    if "Fear" in label:
        return "Fear"
    if "Greed" in label:
        return "Greed"
    return "Neutral"


def safe_mannwhitney(left: pd.Series, right: pd.Series) -> float:
    left_clean = pd.to_numeric(left, errors="coerce").dropna()
    right_clean = pd.to_numeric(right, errors="coerce").dropna()
    if len(left_clean) < 2 or len(right_clean) < 2:
        return np.nan
    return float(stats.mannwhitneyu(left_clean, right_clean, alternative="two-sided").pvalue)


def safe_wilcoxon(diff: pd.Series) -> float:
    clean = pd.to_numeric(diff, errors="coerce").dropna()
    clean = clean[clean != 0]
    if len(clean) < 2:
        return np.nan
    try:
        return float(stats.wilcoxon(clean).pvalue)
    except ValueError:
        return np.nan


def quantile_segment(series: pd.Series, labels: list[str]) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels)


def nice_round(value: float, base: int) -> float:
    if pd.isna(value):
        return np.nan
    return float(base * round(float(value) / base))


def format_big_currency(value: float) -> str:
    if pd.isna(value):
        return "--"
    absolute = abs(float(value))
    if absolute >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def currency_formatter(_: float, __: int) -> str:
    return format_big_currency(_)


def label_order(values: list[str]) -> list[str]:
    available = [label for label in SENTIMENT_ORDER if label in values]
    return available


def build_frames(dataset: LoadedDataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = dataset.trades.merge(dataset.sentiment[["date", "value", "classification"]], on="date", how="inner")
    merged = merged.sort_values(["account", "time"]).reset_index(drop=True)

    merged["is_realized"] = merged["closed_pnl"].ne(0)
    merged["is_buy"] = merged["side"].eq("BUY")
    merged["crossed_flag"] = merged["crossed"].fillna(False).astype(bool)
    merged["direction_text"] = merged["direction"].fillna("").astype(str)
    merged["open_flag"] = merged["direction_text"].str.startswith("Open")
    merged["close_flag"] = merged["direction_text"].str.startswith("Close")
    merged["flip_flag"] = merged["direction_text"].isin(["Long > Short", "Short > Long"])
    merged["abs_start_exposure"] = (merged["start_position"].abs() * merged["execution_price"]).clip(lower=0)
    merged["turnover_ratio"] = merged["size_usd"] / (merged["abs_start_exposure"] + 1.0)
    merged.loc[~np.isfinite(merged["turnover_ratio"]), "turnover_ratio"] = np.nan

    account_day = (
        merged.groupby(["account", "date", "classification", "value"])
        .agg(
            n_trades=("closed_pnl", "count"),
            realized_trades=("is_realized", "sum"),
            daily_pnl=("closed_pnl", "sum"),
            gross_profit=("closed_pnl", positive_sum),
            gross_loss=("closed_pnl", negative_abs_sum),
            win_rate_realized=("closed_pnl", realized_win_rate),
            avg_trade_usd=("size_usd", "mean"),
            median_trade_usd=("size_usd", "median"),
            total_volume=("size_usd", "sum"),
            long_ratio=("is_buy", "mean"),
            crossed_share=("crossed_flag", "mean"),
            open_share=("open_flag", "mean"),
            close_share=("close_flag", "mean"),
            flip_share=("flip_flag", "mean"),
            fee_total=("fee", "sum"),
            turnover_ratio=("turnover_ratio", safe_median),
        )
        .reset_index()
        .sort_values(["account", "date"])
        .reset_index(drop=True)
    )
    account_day["profit_day"] = account_day["daily_pnl"] > 0
    account_day["sentiment_binary"] = account_day["classification"].map(classify_binary)
    account_day["fee_bps"] = 10000 * account_day["fee_total"] / account_day["total_volume"].replace(0, np.nan)
    account_day["cum_pnl"] = account_day.groupby("account")["daily_pnl"].cumsum()

    drawdown = (
        account_day.groupby("account")["cum_pnl"]
        .apply(drawdown_from_curve)
        .rename("max_drawdown")
        .reset_index()
    )

    account = (
        account_day.groupby("account")
        .agg(
            total_pnl=("daily_pnl", "sum"),
            mean_pnl=("daily_pnl", "mean"),
            pnl_std=("daily_pnl", "std"),
            win_rate=("win_rate_realized", "mean"),
            profit_day_rate=("profit_day", "mean"),
            avg_risk_size=("avg_trade_usd", "mean"),
            avg_trades_day=("n_trades", "mean"),
            n_trading_days=("date", "nunique"),
            total_volume=("total_volume", "sum"),
            crossed_share=("crossed_share", "mean"),
            long_ratio=("long_ratio", "mean"),
            open_share=("open_share", "mean"),
            close_share=("close_share", "mean"),
            flip_share=("flip_share", "mean"),
            fee_bps=("fee_bps", "mean"),
            turnover_ratio=("turnover_ratio", "mean"),
            gross_profit=("gross_profit", "sum"),
            gross_loss=("gross_loss", "sum"),
        )
        .reset_index()
        .merge(drawdown, on="account", how="left")
    )
    account["sharpe_proxy"] = account["mean_pnl"] / account["pnl_std"].replace(0, np.nan)
    account["calmar_proxy"] = account["mean_pnl"] / account["max_drawdown"].abs().replace(0, np.nan)
    account["profit_factor"] = account["gross_profit"] / account["gross_loss"].replace(0, np.nan)
    account["loss_day_rate"] = 1.0 - account["profit_day_rate"]

    account["risk_segment"] = quantile_segment(
        account["avg_risk_size"], ["Low Ticket", "Mid Ticket", "High Ticket"]
    )
    account["activity_segment"] = np.where(
        account["avg_trades_day"] > account["avg_trades_day"].median(),
        "Frequent",
        "Selective",
    )
    account["consistency_segment"] = quantile_segment(
        account["win_rate"].fillna(account["win_rate"].median()),
        ["Inconsistent", "Mixed", "Consistent"],
    )
    account["execution_segment"] = np.where(
        account["crossed_share"] > account["crossed_share"].median(),
        "Aggressive Taker",
        "Patient Executor",
    )

    return merged, account_day, account


def cluster_accounts(account: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, float]:
    feature_columns = [
        "avg_risk_size",
        "avg_trades_day",
        "win_rate",
        "crossed_share",
        "fee_bps",
        "turnover_ratio",
        "close_share",
        "mean_pnl",
    ]
    features = account[feature_columns].copy()
    features = features.fillna(features.median(numeric_only=True))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    candidate_scores: list[tuple[int, float]] = []
    candidate_models: dict[int, KMeans] = {}
    for k in range(2, min(5, len(account))):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        cluster_sizes = pd.Series(labels).value_counts()
        if cluster_sizes.min() < 2:
            continue
        score = silhouette_score(scaled, labels)
        candidate_scores.append((k, float(score)))
        candidate_models[k] = model

    if candidate_scores:
        best_k, best_score = max(candidate_scores, key=lambda item: item[1])
        model = candidate_models[best_k]
    else:
        best_k = 2
        best_score = np.nan
        model = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(scaled)

    account = account.copy()
    account["cluster"] = model.predict(scaled)

    centroids = (
        account.groupby("cluster")[feature_columns]
        .mean()
        .reset_index()
        .sort_values("mean_pnl", ascending=False)
        .reset_index(drop=True)
    )
    crossed_median = centroids["crossed_share"].median()
    close_median = centroids["close_share"].median()
    risk_median = centroids["avg_risk_size"].median()
    trade_median = centroids["avg_trades_day"].median()

    labels_by_cluster: dict[int, str] = {}
    for _, row in centroids.iterrows():
        cluster_id = int(row["cluster"])
        if row["crossed_share"] >= crossed_median:
            label = "Aggressive Takers"
        elif row["close_share"] >= close_median and row["avg_risk_size"] >= risk_median:
            label = "Patient Position Builders"
        elif row["avg_trades_day"] < trade_median:
            label = "Selective Rotators"
        else:
            label = "Impulse Scalpers"
        suffix = 2
        base_label = label
        while label in labels_by_cluster.values():
            label = f"{base_label} {suffix}"
            suffix += 1
        labels_by_cluster[cluster_id] = label

    account["archetype"] = account["cluster"].map(labels_by_cluster)

    cluster_profiles = (
        account.groupby("archetype")[feature_columns + ["total_pnl", "max_drawdown"]]
        .mean()
        .reset_index()
        .sort_values("mean_pnl", ascending=False)
    )

    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(scaled)
    plot_frame = account[["account", "archetype", "mean_pnl", "avg_risk_size", "avg_trades_day", "crossed_share", "fee_bps"]].copy()
    plot_frame["pc1"] = embedding[:, 0]
    plot_frame["pc2"] = embedding[:, 1]
    return account, cluster_profiles, plot_frame, best_k, best_score


def build_event_summary(merged: pd.DataFrame) -> pd.DataFrame:
    return (
        merged.groupby(["date", "classification", "value"])
        .agg(
            trade_count=("account", "count"),
            trader_count=("account", "nunique"),
            total_pnl=("closed_pnl", "sum"),
            total_volume=("size_usd", "sum"),
            avg_trade_usd=("size_usd", "mean"),
            crossed_share=("crossed_flag", "mean"),
            long_ratio=("is_buy", "mean"),
            close_share=("close_flag", "mean"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_output_tables(account_day: pd.DataFrame, account: pd.DataFrame, cluster_profiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    performance = (
        account_day.groupby("classification")
        .agg(
            n_obs=("account", "count"),
            median_pnl=("daily_pnl", "median"),
            mean_pnl=("daily_pnl", "mean"),
            pnl_std=("daily_pnl", "std"),
            profit_day_rate=("profit_day", "mean"),
            win_rate=("win_rate_realized", "mean"),
            avg_trade_usd=("avg_trade_usd", "median"),
            crossed_share=("crossed_share", "mean"),
        )
        .reset_index()
    )
    performance["classification"] = pd.Categorical(
        performance["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    performance = performance.sort_values("classification").reset_index(drop=True)

    behavior = (
        account_day.groupby("classification")
        .agg(
            avg_trades=("n_trades", "median"),
            avg_risk_size=("avg_trade_usd", "median"),
            avg_notional=("total_volume", "median"),
            long_ratio=("long_ratio", "mean"),
            crossed_share=("crossed_share", "mean"),
            close_share=("close_share", "mean"),
            fee_bps=("fee_bps", "mean"),
            turnover_ratio=("turnover_ratio", "mean"),
        )
        .reset_index()
    )
    behavior["classification"] = pd.Categorical(
        behavior["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    behavior = behavior.sort_values("classification").reset_index(drop=True)

    risk_segmentation = (
        account.groupby("risk_segment")
        .agg(
            n=("account", "count"),
            mean_pnl=("mean_pnl", "mean"),
            win_rate=("win_rate", "mean"),
            crossed_share=("crossed_share", "mean"),
            fee_bps=("fee_bps", "mean"),
            avg_risk_size=("avg_risk_size", "mean"),
            max_drawdown=("max_drawdown", "mean"),
        )
        .reset_index()
        .rename(columns={"risk_segment": "segment"})
    )

    frequency_segmentation = (
        account.groupby("activity_segment")
        .agg(
            n=("account", "count"),
            mean_pnl=("mean_pnl", "mean"),
            win_rate=("win_rate", "mean"),
            avg_risk_size=("avg_risk_size", "mean"),
            crossed_share=("crossed_share", "mean"),
            fee_bps=("fee_bps", "mean"),
        )
        .reset_index()
        .rename(columns={"activity_segment": "segment"})
    )

    consistency_segmentation = (
        account.groupby("consistency_segment")
        .agg(
            n=("account", "count"),
            mean_pnl=("mean_pnl", "mean"),
            win_rate=("win_rate", "mean"),
            avg_risk_size=("avg_risk_size", "mean"),
            profit_day_rate=("profit_day_rate", "mean"),
            max_drawdown=("max_drawdown", "mean"),
        )
        .reset_index()
        .rename(columns={"consistency_segment": "segment"})
    )

    execution_segmentation = (
        account.groupby("execution_segment")
        .agg(
            n=("account", "count"),
            mean_pnl=("mean_pnl", "mean"),
            win_rate=("win_rate", "mean"),
            avg_risk_size=("avg_risk_size", "mean"),
            avg_trades_day=("avg_trades_day", "mean"),
            crossed_share=("crossed_share", "mean"),
            fee_bps=("fee_bps", "mean"),
        )
        .reset_index()
        .rename(columns={"execution_segment": "segment"})
    )

    return {
        "performance_by_sentiment": performance,
        "behavior_by_sentiment": behavior,
        "account_summary": account.sort_values("total_pnl", ascending=False).reset_index(drop=True),
        "cluster_profiles": cluster_profiles,
        "leverage_segmentation": risk_segmentation,
        "frequency_segmentation": frequency_segmentation,
        "consistency_segmentation": consistency_segmentation,
        "execution_segmentation": execution_segmentation,
    }


def build_robustness_table(account_day: pd.DataFrame, account: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    fear = account_day.loc[account_day["sentiment_binary"] == "Fear"]
    greed = account_day.loc[account_day["sentiment_binary"] == "Greed"]

    paired = (
        account_day.loc[account_day["sentiment_binary"].isin(["Fear", "Greed"])]
        .groupby(["account", "sentiment_binary"])
        .agg(
            pnl=("daily_pnl", "median"),
            crossed_share=("crossed_share", "mean"),
            avg_risk_size=("avg_trade_usd", "mean"),
        )
        .reset_index()
        .pivot(index="account", columns="sentiment_binary")
    )
    paired.columns = [f"{metric}_{sentiment}" for metric, sentiment in paired.columns]
    paired = paired.dropna().reset_index()

    execution_threshold = account["crossed_share"].median()
    patient = account.loc[account["crossed_share"] <= execution_threshold]
    aggressive = account.loc[account["crossed_share"] > execution_threshold]

    greed_threshold = nice_round(greed["avg_trade_usd"].median(), 100) if not greed.empty else np.nan
    greed_small = greed.loc[greed["avg_trade_usd"] <= greed_threshold] if not pd.isna(greed_threshold) else greed.iloc[0:0]
    greed_large = greed.loc[greed["avg_trade_usd"] > greed_threshold] if not pd.isna(greed_threshold) else greed.iloc[0:0]

    rows = [
        {
            "finding": "Greed vs Fear account-day PnL",
            "group_a": "Greed account-days",
            "group_b": "Fear account-days",
            "n_a": len(greed),
            "n_b": len(fear),
            "effect_value": float(greed["daily_pnl"].median() - fear["daily_pnl"].median()) if not greed.empty and not fear.empty else np.nan,
            "pvalue": safe_mannwhitney(greed["daily_pnl"], fear["daily_pnl"]),
            "test": "Mann-Whitney U",
            "note": "Interpret carefully: the matched set includes 4 greed dates but only 1 fear date.",
        },
        {
            "finding": "Paired greed-fear PnL by account",
            "group_a": "Greed median PnL",
            "group_b": "Fear median PnL",
            "n_a": len(paired),
            "n_b": len(paired),
            "effect_value": float((paired["pnl_Greed"] - paired["pnl_Fear"]).median()) if not paired.empty else np.nan,
            "pvalue": safe_wilcoxon(paired["pnl_Greed"] - paired["pnl_Fear"]) if not paired.empty else np.nan,
            "test": "Wilcoxon signed-rank",
            "note": "Uses only accounts active in both fear and greed windows.",
        },
        {
            "finding": "Paired greed-fear crossed share by account",
            "group_a": "Greed crossed share",
            "group_b": "Fear crossed share",
            "n_a": len(paired),
            "n_b": len(paired),
            "effect_value": float((paired["crossed_share_Greed"] - paired["crossed_share_Fear"]).median()) if not paired.empty else np.nan,
            "pvalue": safe_wilcoxon(paired["crossed_share_Greed"] - paired["crossed_share_Fear"]) if not paired.empty else np.nan,
            "test": "Wilcoxon signed-rank",
            "note": "Positive values mean more aggressive crossing on greed days.",
        },
        {
            "finding": "Patient vs aggressive executors",
            "group_a": "Patient executors",
            "group_b": "Aggressive takers",
            "n_a": len(patient),
            "n_b": len(aggressive),
            "effect_value": float(patient["mean_pnl"].mean() - aggressive["mean_pnl"].mean()) if not patient.empty and not aggressive.empty else np.nan,
            "pvalue": safe_mannwhitney(patient["mean_pnl"], aggressive["mean_pnl"]),
            "test": "Mann-Whitney U",
            "note": "Split on median account-level crossed share.",
        },
        {
            "finding": "Greed small-ticket vs large-ticket days",
            "group_a": "Small-ticket greed days",
            "group_b": "Large-ticket greed days",
            "n_a": len(greed_small),
            "n_b": len(greed_large),
            "effect_value": float(greed_small["daily_pnl"].median() - greed_large["daily_pnl"].median()) if not greed_small.empty and not greed_large.empty else np.nan,
            "pvalue": safe_mannwhitney(greed_small["daily_pnl"], greed_large["daily_pnl"]),
            "test": "Mann-Whitney U",
            "note": "Threshold is the median greed-day ticket size.",
        },
    ]
    return pd.DataFrame(rows), paired, execution_threshold, greed_threshold


def build_strategy_table(account_day: pd.DataFrame, account: pd.DataFrame, execution_threshold: float, greed_threshold: float) -> pd.DataFrame:
    all_days = account_day.copy()
    all_days["rule"] = np.where(
        all_days["crossed_share"] <= execution_threshold,
        f"Crossed share <= {execution_threshold:.2f}",
        f"Crossed share > {execution_threshold:.2f}",
    )

    greed_days = account_day.loc[account_day["sentiment_binary"] == "Greed"].copy()
    if not greed_days.empty and not pd.isna(greed_threshold):
        greed_days["rule"] = np.where(
            greed_days["avg_trade_usd"] <= greed_threshold,
            f"Greed ticket size <= {format_big_currency(greed_threshold)}",
            f"Greed ticket size > {format_big_currency(greed_threshold)}",
        )
    else:
        greed_days["rule"] = "No greed sample"

    global_rules = (
        all_days.groupby("rule")
        .agg(
            scope=("rule", lambda _: "All matched days"),
            sample_size=("account", "count"),
            median_pnl=("daily_pnl", "median"),
            mean_pnl=("daily_pnl", "mean"),
            profit_day_rate=("profit_day", "mean"),
            win_rate=("win_rate_realized", "mean"),
            avg_trade_usd=("avg_trade_usd", "mean"),
            crossed_share=("crossed_share", "mean"),
            fee_bps=("fee_bps", "mean"),
        )
        .reset_index()
    )
    greed_rules = (
        greed_days.groupby("rule")
        .agg(
            scope=("rule", lambda _: "Greed and Extreme Greed"),
            sample_size=("account", "count"),
            median_pnl=("daily_pnl", "median"),
            mean_pnl=("daily_pnl", "mean"),
            profit_day_rate=("profit_day", "mean"),
            win_rate=("win_rate_realized", "mean"),
            avg_trade_usd=("avg_trade_usd", "mean"),
            crossed_share=("crossed_share", "mean"),
            fee_bps=("fee_bps", "mean"),
        )
        .reset_index()
    )
    strategy = pd.concat([global_rules, greed_rules], ignore_index=True)
    return strategy.sort_values(["scope", "median_pnl"], ascending=[True, False]).reset_index(drop=True)


def plot_performance(performance: pd.DataFrame, account_day: pd.DataFrame) -> None:
    order = label_order(performance["classification"].astype(str).tolist())
    data = account_day.loc[account_day["classification"].isin(order)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    fig.suptitle("Chart 1 - Performance Across Matched Sentiment Events", fontsize=15, y=1.02)

    sns.boxplot(
        data=data,
        x="classification",
        y="daily_pnl",
        order=order,
        palette=[SENTIMENT_COLORS[label] for label in order],
        ax=axes[0],
        showfliers=False,
    )
    sns.stripplot(
        data=data,
        x="classification",
        y="daily_pnl",
        order=order,
        color="#2F2B28",
        size=4.2,
        alpha=0.55,
        ax=axes[0],
    )
    axes[0].axhline(0, color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_title("Account-day PnL")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Daily PnL")
    axes[0].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[0].tick_params(axis="x", rotation=10)

    rate_frame = performance.set_index("classification").reindex(order)
    axes[1].bar(
        rate_frame.index,
        rate_frame["profit_day_rate"] * 100,
        color=[SENTIMENT_COLORS[label] for label in order],
        edgecolor="#D6C9BB",
    )
    axes[1].axhline(50, color=MUTED, linestyle="--", linewidth=1)
    axes[1].set_title("Profit-day Rate")
    axes[1].set_ylabel("Share of profitable account-days (%)")
    axes[1].tick_params(axis="x", rotation=10)

    axes[2].bar(
        rate_frame.index,
        rate_frame["win_rate"] * 100,
        color=[SENTIMENT_COLORS[label] for label in order],
        edgecolor="#D6C9BB",
    )
    axes[2].axhline(50, color=MUTED, linestyle="--", linewidth=1)
    axes[2].set_title("Realized Win Rate")
    axes[2].set_ylabel("Winning realized trades (%)")
    axes[2].tick_params(axis="x", rotation=10)

    for axis in axes:
        axis.set_facecolor(PANEL)

    savefig("01_performance_by_sentiment")


def plot_event_timeline(event_summary: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 5.8))
    ax2 = ax1.twinx()

    ax1.plot(
        event_summary["date"],
        event_summary["value"],
        color="#1D6B67",
        linewidth=2.5,
        marker="o",
        label="Fear/Greed value",
    )
    bars = ax2.bar(
        event_summary["date"],
        event_summary["total_pnl"],
        width=14,
        color=[SENTIMENT_COLORS[label] for label in event_summary["classification"]],
        alpha=0.45,
        label="Total PnL",
    )

    for date_value, score, label in zip(
        event_summary["date"], event_summary["value"], event_summary["classification"]
    ):
        ax1.text(date_value, score + 1.2, label, fontsize=8.5, ha="center", color=INK)

    ax1.set_title("Chart 2 - Timeline of Matched Hyperliquid Event Days")
    ax1.set_ylabel("Fear/Greed Index")
    ax2.set_ylabel("Total PnL")
    ax2.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax1.set_xlabel("")
    ax1.grid(False)
    ax2.grid(False)
    ax1.set_ylim(0, max(100, event_summary["value"].max() + 10))

    handles = [
        Line2D([0], [0], color="#1D6B67", linewidth=2.5, marker="o", label="Fear/Greed value"),
        bars,
    ]
    labels = ["Fear/Greed value", "Total PnL"]
    ax1.legend(handles, labels, loc="upper left")
    savefig("02_timeline_sentiment_pnl")


def plot_behavior_fingerprint(behavior: pd.DataFrame) -> None:
    order = label_order(behavior["classification"].astype(str).tolist())
    behavior = behavior.set_index("classification").reindex(order).reset_index()
    metrics = [
        ("avg_trades", "Median Trades / Account-day", ""),
        ("avg_risk_size", "Median Ticket Size", "$"),
        ("avg_notional", "Median Total Volume", "$"),
        ("crossed_share", "Crossed Share", "%"),
        ("close_share", "Close-share", "%"),
        ("fee_bps", "Fees per $10k Traded", "bps"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Chart 3 - Behavioral Fingerprint by Sentiment", fontsize=15, y=1.01)

    for axis, (column, title, unit) in zip(axes.flat, metrics):
        values = behavior[column].copy()
        if unit == "%":
            values = values * 100
        axis.bar(
            behavior["classification"],
            values,
            color=[SENTIMENT_COLORS[label] for label in behavior["classification"]],
            edgecolor="#D6C9BB",
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=12)
        if unit == "$":
            axis.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        elif unit == "%":
            axis.set_ylabel("%")
        elif unit == "bps":
            axis.set_ylabel("basis points")

    savefig("03_behavioral_fingerprint")


def plot_event_coverage(merged: pd.DataFrame, event_summary: pd.DataFrame) -> None:
    trade_counts = (
        merged.groupby(["account", "date"])["closed_pnl"]
        .count()
        .unstack(fill_value=0)
        .sort_values(by=list(event_summary["date"]), ascending=False)
    )
    short_index = [account[-6:] for account in trade_counts.index]

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        trade_counts,
        cmap="YlGnBu",
        linewidths=0.35,
        linecolor="#F3EADF",
        cbar_kws={"label": "Trades"},
        yticklabels=short_index,
    )
    plt.title("Chart 4 - Event Coverage Heatmap (Accounts x Matched Dates)")
    plt.xlabel("Matched event day")
    plt.ylabel("Account suffix")
    savefig("04_event_coverage")


def plot_segmentation(account: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> None:
    risk = outputs["leverage_segmentation"]
    activity = outputs["frequency_segmentation"]
    consistency = outputs["consistency_segmentation"]
    execution = outputs["execution_segmentation"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Chart 5 - Segmentation Deep Dive", fontsize=15, y=1.01)

    axes[0, 0].bar(
        risk["segment"],
        risk["mean_pnl"],
        color=["#506D9A", "#B4935D", "#1D6B67"],
        edgecolor="#D6C9BB",
    )
    axes[0, 0].set_title("Ticket Size Segments by Mean PnL")
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[0, 0].tick_params(axis="x", rotation=10)

    axes[0, 1].bar(
        activity["segment"],
        activity["mean_pnl"],
        color=["#C67C3D", "#1D6B67"],
        edgecolor="#D6C9BB",
    )
    axes[0, 1].set_title("Activity Segments by Mean PnL")
    axes[0, 1].yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    axes[1, 0].bar(
        execution["segment"],
        execution["mean_pnl"],
        color=["#C95A4D", "#1D6B67"],
        edgecolor="#D6C9BB",
    )
    axes[1, 0].set_title("Execution Style by Mean PnL")
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[1, 0].tick_params(axis="x", rotation=10)

    axes[1, 1].bar(
        consistency["segment"],
        consistency["mean_pnl"],
        color=["#C95A4D", "#A6A39F", "#1D6B67"],
        edgecolor="#D6C9BB",
    )
    axes[1, 1].set_title("Consistency Segments by Mean PnL")
    axes[1, 1].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[1, 1].tick_params(axis="x", rotation=10)

    savefig("05_segmentation_deep_dive")


def plot_directional_bias(behavior: pd.DataFrame) -> None:
    order = label_order(behavior["classification"].astype(str).tolist())
    behavior = behavior.set_index("classification").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Chart 6 - Directional and Position Management Mix", fontsize=14, y=1.02)

    axes[0].barh(
        behavior["classification"],
        behavior["long_ratio"] * 100,
        color=[SENTIMENT_COLORS[label] for label in behavior["classification"]],
        edgecolor="#D6C9BB",
    )
    axes[0].axvline(50, color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_title("Buy-side Share")
    axes[0].set_xlabel("% BUY trades")

    axes[1].barh(
        behavior["classification"],
        behavior["close_share"] * 100,
        color=[SENTIMENT_COLORS[label] for label in behavior["classification"]],
        edgecolor="#D6C9BB",
    )
    axes[1].set_title("Close-share")
    axes[1].set_xlabel("% closing trades")

    savefig("06_directional_bias")


def plot_archetypes(plot_frame: pd.DataFrame, account: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    fig.suptitle("Chart 7 - Trader Archetypes", fontsize=15, y=1.02)

    for archetype, frame in plot_frame.groupby("archetype"):
        color = ARCHETYPE_COLORS.get(archetype, "#506D9A")
        axes[0].scatter(
            frame["pc1"],
            frame["pc2"],
            s=85,
            alpha=0.85,
            color=color,
            edgecolor="#F5F0E9",
            linewidth=0.8,
            label=archetype,
        )
        axes[1].scatter(
            frame["avg_risk_size"],
            frame["mean_pnl"],
            s=np.clip(frame["avg_trades_day"] / 20, 35, 220),
            alpha=0.82,
            color=color,
            edgecolor="#F5F0E9",
            linewidth=0.8,
            label=archetype,
        )

    axes[0].set_title("PCA projection of account behavior")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].set_title("Ticket size vs mean PnL")
    axes[1].set_xlabel("Average ticket size")
    axes[1].set_ylabel("Mean daily PnL")
    axes[1].xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[1].yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    axes[0].legend(loc="best", fontsize=9)
    savefig("07_archetypes")


def plot_cluster_profiles(cluster_profiles: pd.DataFrame) -> None:
    frame = cluster_profiles.copy()
    metrics = ["avg_risk_size", "avg_trades_day", "win_rate", "crossed_share", "fee_bps", "close_share", "mean_pnl"]
    scaled = frame[metrics].copy()
    for column in scaled.columns:
        column_values = scaled[column]
        spread = column_values.max() - column_values.min()
        scaled[column] = 0 if spread == 0 else (column_values - column_values.min()) / spread
    scaled.index = frame["archetype"]

    plt.figure(figsize=(11, 5))
    sns.heatmap(
        scaled,
        cmap="YlGnBu",
        annot=frame[metrics].round(2),
        fmt="",
        linewidths=0.35,
        linecolor="#F3EADF",
        cbar_kws={"label": "Min-max normalized"},
    )
    plt.title("Chart 8 - Cluster Profiles")
    plt.xlabel("Feature")
    plt.ylabel("Archetype")
    savefig("08_cluster_profiles")


def plot_robustness(paired: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle("Chart 9 - Robustness Checks on Matched Fear vs Greed Accounts", fontsize=14.5, y=1.02)

    if paired.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No matched fear/greed pairs available", ha="center", va="center")
            axis.axis("off")
        savefig("09_robustness_checks")
        return

    for _, row in paired.iterrows():
        axes[0].plot([0, 1], [row["pnl_Fear"], row["pnl_Greed"]], color="#C7BBB0", linewidth=1.1, alpha=0.8)
    axes[0].scatter(np.zeros(len(paired)), paired["pnl_Fear"], color="#C67C3D", s=40, label="Fear")
    axes[0].scatter(np.ones(len(paired)), paired["pnl_Greed"], color="#1D6B67", s=40, label="Greed")
    axes[0].set_xticks([0, 1], ["Fear", "Greed"])
    axes[0].set_title("Paired median PnL per account")
    axes[0].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[0].legend()

    for _, row in paired.iterrows():
        axes[1].plot(
            [0, 1],
            [row["crossed_share_Fear"] * 100, row["crossed_share_Greed"] * 100],
            color="#C7BBB0",
            linewidth=1.1,
            alpha=0.8,
        )
    axes[1].scatter(np.zeros(len(paired)), paired["crossed_share_Fear"] * 100, color="#C67C3D", s=40)
    axes[1].scatter(np.ones(len(paired)), paired["crossed_share_Greed"] * 100, color="#1D6B67", s=40)
    axes[1].set_xticks([0, 1], ["Fear", "Greed"])
    axes[1].set_title("Paired crossed-share per account")
    axes[1].set_ylabel("Crossed share (%)")

    savefig("09_robustness_checks")


def plot_strategy_playbook(strategy: pd.DataFrame, execution_threshold: float, greed_threshold: float) -> None:
    all_days = strategy.loc[strategy["scope"] == "All matched days"].copy()
    greed_days = strategy.loc[strategy["scope"] == "Greed and Extreme Greed"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    fig.suptitle("Chart 10 - Strategy Playbook", fontsize=15, y=1.02)

    if not all_days.empty:
        axes[0].bar(
            all_days["rule"],
            all_days["mean_pnl"],
            color=["#1D6B67" if "<=" in rule else "#C95A4D" for rule in all_days["rule"]],
            edgecolor="#D6C9BB",
        )
        axes[0].set_title(f"Execution discipline split at crossed share {execution_threshold:.2f}")
        axes[0].set_ylabel("Mean daily PnL")
        axes[0].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        axes[0].tick_params(axis="x", rotation=10)

    if not greed_days.empty:
        axes[1].bar(
            greed_days["rule"],
            greed_days["median_pnl"],
            color=["#1D6B67" if "<=" in rule else "#C95A4D" for rule in greed_days["rule"]],
            edgecolor="#D6C9BB",
        )
        axes[1].set_title(f"Greed-day ticket sizing split at {format_big_currency(greed_threshold)}")
        axes[1].set_ylabel("Median daily PnL")
        axes[1].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        axes[1].tick_params(axis="x", rotation=10)

    savefig("10_strategy_playbook")


def plot_drawdowns(account: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle("Chart 11 - Drawdowns and Risk Control", fontsize=14.8, y=1.02)

    sns.boxplot(
        data=account,
        x="risk_segment",
        y="max_drawdown",
        palette=["#506D9A", "#B4935D", "#1D6B67"],
        ax=axes[0],
        showfliers=False,
    )
    axes[0].set_title("Max drawdown by ticket-size segment")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Max drawdown")
    axes[0].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[0].tick_params(axis="x", rotation=10)

    palette = {
        "Patient Executor": "#1D6B67",
        "Aggressive Taker": "#C95A4D",
    }
    for segment, frame in account.groupby("execution_segment"):
        axes[1].scatter(
            frame["max_drawdown"],
            frame["mean_pnl"],
            s=np.clip(frame["avg_trades_day"] / 20, 40, 220),
            color=palette.get(segment, "#506D9A"),
            alpha=0.8,
            edgecolor="#F5F0E9",
            linewidth=0.8,
            label=segment,
        )
    axes[1].axhline(0, color=MUTED, linestyle="--", linewidth=1)
    axes[1].set_title("Mean PnL vs max drawdown")
    axes[1].set_xlabel("Max drawdown")
    axes[1].set_ylabel("Mean daily PnL")
    axes[1].xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[1].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[1].legend()

    savefig("11_drawdown_analysis")


def write_outputs(
    outputs: dict[str, pd.DataFrame],
    event_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    strategy: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    outputs["performance_by_sentiment"].to_csv(OUTPUTS_DIR / "performance_by_sentiment.csv", index=False)
    outputs["behavior_by_sentiment"].to_csv(OUTPUTS_DIR / "behavior_by_sentiment.csv", index=False)
    outputs["account_summary"].to_csv(OUTPUTS_DIR / "account_summary.csv", index=False)
    outputs["cluster_profiles"].to_csv(OUTPUTS_DIR / "cluster_profiles.csv", index=False)
    outputs["leverage_segmentation"].to_csv(OUTPUTS_DIR / "leverage_segmentation.csv", index=False)
    outputs["frequency_segmentation"].to_csv(OUTPUTS_DIR / "frequency_segmentation.csv", index=False)
    outputs["consistency_segmentation"].to_csv(OUTPUTS_DIR / "consistency_segmentation.csv", index=False)
    outputs["execution_segmentation"].to_csv(OUTPUTS_DIR / "execution_segmentation.csv", index=False)
    event_summary.to_csv(OUTPUTS_DIR / "event_summary.csv", index=False)
    robustness.to_csv(OUTPUTS_DIR / "robustness_checks.csv", index=False)
    strategy.to_csv(OUTPUTS_DIR / "strategy_playbook.csv", index=False)

    with (OUTPUTS_DIR / "ui_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def build_summary(
    dataset: LoadedDataset,
    merged: pd.DataFrame,
    account_day: pd.DataFrame,
    account: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    robustness: pd.DataFrame,
    strategy: pd.DataFrame,
    best_k: int,
    silhouette: float,
    execution_threshold: float,
    greed_threshold: float,
) -> dict[str, Any]:
    performance = outputs["performance_by_sentiment"]
    fear_row = performance.loc[performance["classification"] == "Fear"]
    greed_row = performance.loc[performance["classification"] == "Greed"]
    archetype_counts = account["archetype"].value_counts()

    patient = account.loc[account["execution_segment"] == "Patient Executor", "mean_pnl"]
    aggressive = account.loc[account["execution_segment"] == "Aggressive Taker", "mean_pnl"]

    greed_rules = strategy.loc[strategy["scope"] == "Greed and Extreme Greed"].copy()
    small_ticket = greed_rules.loc[greed_rules["rule"].str.contains("<=", regex=False)]
    large_ticket = greed_rules.loc[greed_rules["rule"].str.contains(">", regex=False)]

    return {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "date_start": str(merged["date"].min().date()),
        "date_end": str(merged["date"].max().date()),
        "sentiment_source": dataset.sentiment_path.name,
        "trader_source": dataset.trades_path.name,
        "schema": dataset.schema,
        "total_trades": int(len(dataset.trades)),
        "matched_trades": int(len(merged)),
        "unique_traders": int(merged["account"].nunique()),
        "matched_days": int(account_day["date"].nunique()),
        "matched_account_rows": int(len(account_day)),
        "fear_median_pnl": float(fear_row["median_pnl"].iloc[0]) if not fear_row.empty else None,
        "greed_median_pnl": float(greed_row["median_pnl"].iloc[0]) if not greed_row.empty else None,
        "median_pnl_lift": float(greed_row["median_pnl"].iloc[0] - fear_row["median_pnl"].iloc[0])
        if not fear_row.empty and not greed_row.empty
        else None,
        "paired_account_count": int(robustness.loc[robustness["finding"] == "Paired greed-fear PnL by account", "n_a"].iloc[0]),
        "execution_threshold": float(execution_threshold),
        "greed_ticket_threshold": float(greed_threshold) if not pd.isna(greed_threshold) else None,
        "patient_executor_mean_pnl": float(patient.mean()) if not patient.empty else None,
        "aggressive_executor_mean_pnl": float(aggressive.mean()) if not aggressive.empty else None,
        "greed_small_ticket_median_pnl": float(small_ticket["median_pnl"].iloc[0]) if not small_ticket.empty else None,
        "greed_large_ticket_median_pnl": float(large_ticket["median_pnl"].iloc[0]) if not large_ticket.empty else None,
        "low_risk_mean_pnl": float(outputs["leverage_segmentation"].iloc[0]["mean_pnl"]) if not outputs["leverage_segmentation"].empty else None,
        "high_risk_mean_pnl": float(outputs["leverage_segmentation"].iloc[-1]["mean_pnl"]) if not outputs["leverage_segmentation"].empty else None,
        "cluster_k": int(best_k),
        "silhouette_score": float(silhouette) if not pd.isna(silhouette) else None,
        "top_archetype": archetype_counts.index[0] if not archetype_counts.empty else None,
        "top_archetype_count": int(archetype_counts.iloc[0]) if not archetype_counts.empty else None,
        "analysis_note": "Real assignment export overlaps sentiment on 6 matched event days, so results are event-driven rather than a continuous daily backtest.",
        "cv_auc_mean": None,
        "cv_auc_std": None,
        "test_auc": None,
        "test_accuracy": None,
        "chart_count": len(GENERATED_CHARTS),
        "table_count": len(GENERATED_TABLES),
    }


def main() -> None:
    clean_previous_outputs()
    print_rule("LOAD DATA")
    dataset = load_datasets()
    print(f"Sentiment source : {dataset.sentiment_path}")
    print(f"Trader source    : {dataset.trades_path}")
    print(f"Detected schema  : {dataset.schema}")
    print(f"Sentiment rows   : {len(dataset.sentiment):,}")
    print(f"Trader rows      : {len(dataset.trades):,}")

    merged, account_day, account = build_frames(dataset)

    print_rule("MATCHED PANEL")
    print(f"Matched trades        : {len(merged):,}")
    print(f"Matched dates         : {account_day['date'].nunique()}")
    print(f"Matched account-days  : {len(account_day):,}")
    print(f"Unique traders        : {merged['account'].nunique()}")
    print("Matched dates summary:")
    matched_dates = (
        merged[["date", "classification", "value"]]
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(matched_dates.to_string(index=False))

    account, cluster_profiles, plot_frame, best_k, silhouette = cluster_accounts(account)
    outputs = build_output_tables(account_day, account, cluster_profiles)
    event_summary = build_event_summary(merged)
    robustness, paired, execution_threshold, greed_threshold = build_robustness_table(account_day, account)
    strategy = build_strategy_table(account_day, account, execution_threshold, greed_threshold)

    print_rule("KEY TABLES")
    print("Performance by sentiment:")
    print(outputs["performance_by_sentiment"].round(4).to_string(index=False))
    print("\nExecution segmentation:")
    print(outputs["execution_segmentation"].round(4).to_string(index=False))
    print("\nRobustness checks:")
    print(robustness.round(4).to_string(index=False))

    print_rule("PLOT CHARTS")
    plot_performance(outputs["performance_by_sentiment"], account_day)
    plot_event_timeline(event_summary)
    plot_behavior_fingerprint(outputs["behavior_by_sentiment"])
    plot_event_coverage(merged, event_summary)
    plot_segmentation(account, outputs)
    plot_directional_bias(outputs["behavior_by_sentiment"])
    plot_archetypes(plot_frame, account)
    plot_cluster_profiles(cluster_profiles)
    plot_robustness(paired)
    plot_strategy_playbook(strategy, execution_threshold, greed_threshold)
    plot_drawdowns(account)

    summary = build_summary(
        dataset=dataset,
        merged=merged,
        account_day=account_day,
        account=account,
        outputs=outputs,
        robustness=robustness,
        strategy=strategy,
        best_k=best_k,
        silhouette=silhouette,
        execution_threshold=execution_threshold,
        greed_threshold=greed_threshold,
    )
    write_outputs(outputs, event_summary, robustness, strategy, summary)

    print_rule("DONE")
    print("Charts generated : 11")
    print("Tables generated : 10 CSV files + 1 JSON summary")
    print(summary["analysis_note"])


if __name__ == "__main__":
    main()
