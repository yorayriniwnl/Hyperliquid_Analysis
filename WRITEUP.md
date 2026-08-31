# Hyperliquid Signal Atlas — Current Evidence Dossier

`VERIFIED DATA PIPELINE` · `EVENT STUDY` · `NO LIVE TRADING`

This dossier describes the outputs generated from the files currently checked into the repository. It replaces an earlier narrative that cited a larger trade universe and an ML score not present in the current pipeline.

## Scope and method

The pipeline joins `data/fear_greed_index.csv` with `data/historical_data.csv` after column normalization, timestamp/date coercion, and account-day aggregation. The trader export contains **136,255 trades**. The date join produces **116,751 matched trades**, **21 unique traders**, **55 account-day rows**, and **six matched calendar days** between 2023-03-28 and 2025-02-19.

Each account-day record carries activity, ticket-size, notional, execution, directional, fee, PnL, and drawdown-proxy features. The analysis then produces percentile-based segments, a standardized K-Means account clustering view, non-parametric comparisons, and static chart plates for the local app.

The current generated snapshot is [`outputs/ui_metrics.json`](outputs/ui_metrics.json). The dependency-free integrity check is [`scripts/verify_outputs.py`](scripts/verify_outputs.py).

## Findings from the current snapshot

### 1. The observed sentiment split is not a broad market forecast

The Fear account-day median PnL is **$64,510.17** and the Greed account-day median is **$20,925.51**, producing a Greed-minus-Fear median lift of **-$43,584.66**. The account-day Mann–Whitney comparison has 28 Greed observations and 21 Fear observations (`p = 0.0079`), but the matched set includes only four Greed dates and one Fear date. Calendar coverage is therefore the primary interpretive constraint.

A paired comparison on the 20 accounts active in both windows estimates a Greed-minus-Fear median PnL effect of **-$26,913.24** (`p = 0.0441`). The paired result is useful as a robustness view, not as proof that sentiment causes the difference.

### 2. Execution discipline is the cleanest operational contrast

The account-level split uses median crossed share as its boundary. Patient executors average **$218,845.98** mean daily PnL and aggressive takers average **$94,629.37**. The comparison includes 11 patient and 10 aggressive accounts, with `p = 0.4181`; the difference is descriptive and not statistically significant in this sample.

The all-window execution cut shows a median daily PnL of **$43,099.22** for account-days at or below the `0.57` crossed-share threshold versus **$20,607.45** above it. The dashboard presents this as an observed split, not a guaranteed edge.

### 3. Smaller Greed-day tickets are a follow-up hypothesis

The Greed/Extreme Greed ticket-size threshold is the median **$2,500** average ticket. The smaller-ticket cohort has a **$38,752.35** median daily PnL; the larger-ticket cohort has **$5,109.76**. The comparison is not statistically significant (`p = 0.2590`) and should not be converted into a sizing rule without a longer, independent sample.

### 4. Clustering is descriptive structure, not trader identity

The current standardized K-Means run selects `k = 2`, with a silhouette score of **0.3294**. The largest archetype is **Aggressive Takers** with 18 of the 21 matched traders. The labels are generated from behavioral features and are not stable personal identities, recommendations, or proof of a persistent strategy class.

## What is intentionally not claimed

- The repository does **not** contain a continuous daily backtest for the joined panel.
- The current generated summary has `cv_auc_mean`, `cv_auc_std`, `test_auc`, and `test_accuracy` set to `null`. There is no validated Gradient Boosting ROC-AUC result in this snapshot, so no ML performance claim is made.
- No liquidation-rate, Sharpe, Calmar, causal, out-of-sample, profitability, or future-return claim is promoted unless it is present in the current checked-in outputs.
- The dashboard does not place orders or connect to a live exchange. It is a local research interface over static exports.

## Reproducibility

```powershell
python -m pip install -r requirements.txt
python analysis.py
python scripts/verify_outputs.py
streamlit run app.py
```

The analysis writes 11 CSV tables, 11 PNG chart plates, and the `outputs/ui_metrics.json` snapshot. The app consumes those generated artifacts and exposes the same limitations in its hero framing and method notes.

## Interpretation and safety

These results are descriptive observations from a thin, event-driven overlap. A low p-value does not repair limited coverage, establish causality, or guarantee future performance. Any rule suggested by the cuts would require a longer sample, explicit out-of-sample validation, realistic costs, and risk controls.

This project is for educational research and interface exploration. It is **not investment, trading, risk, or financial advice**.
