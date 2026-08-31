"""Check the generated Hyperliquid evidence snapshot without third-party dependencies."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CHARTS = ROOT / "charts"

EXPECTED_SUMMARY = {
    "total_trades": 136255,
    "matched_trades": 116751,
    "unique_traders": 21,
    "matched_days": 6,
    "matched_account_rows": 55,
    "paired_account_count": 20,
    "chart_count": 11,
    "table_count": 11,
}

REQUIRED_TABLES = {
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
}

REQUIRED_CHARTS = {
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
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"OUTPUT CHECK FAILED: {message}")


def main() -> None:
    summary_path = OUTPUTS / "ui_metrics.json"
    require(summary_path.exists(), "outputs/ui_metrics.json is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for key, expected in EXPECTED_SUMMARY.items():
        require(summary.get(key) == expected, f"{key}={summary.get(key)!r}; expected {expected!r}")

    require(summary.get("cv_auc_mean") is None, "cv_auc_mean must remain null without a validated model run")
    require(summary.get("test_auc") is None, "test_auc must remain null without a validated model run")
    require(summary.get("analysis_note", "").startswith("Real assignment export overlaps sentiment on 6 matched event days"), "event-study limitation is missing")

    actual_tables = {path.name for path in OUTPUTS.glob("*.csv")}
    actual_charts = {path.name for path in CHARTS.glob("*.png")}
    require(REQUIRED_TABLES <= actual_tables, f"missing tables: {sorted(REQUIRED_TABLES - actual_tables)}")
    require(REQUIRED_CHARTS <= actual_charts, f"missing charts: {sorted(REQUIRED_CHARTS - actual_charts)}")

    with (OUTPUTS / "robustness_checks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 5, f"robustness row count is {len(rows)}; expected 5")
    paired = next((row for row in rows if row.get("finding") == "Paired greed-fear PnL by account"), None)
    require(paired is not None and paired.get("n_a") == "20" and paired.get("n_b") == "20", "paired robustness sample is not 20 x 20")

    print(
        "OUTPUT CHECK PASSED: "
        f"{summary['matched_trades']:,} matched trades / "
        f"{summary['matched_days']} days / "
        f"{summary['chart_count']} charts / {summary['table_count']} tables"
    )


if __name__ == "__main__":
    main()
