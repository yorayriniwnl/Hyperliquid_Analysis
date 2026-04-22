# Hyperliquid × Fear/Greed — Analysis Write-Up

## Methodology

**Data:** Bitcoin Fear/Greed Index (365 daily readings, Jan–Dec 2023) joined with Hyperliquid trader history (~417K trades across 120 accounts). After deduplication and type coercion, trades were aggregated to 43,800 daily account-level observations (9 metrics each). A drawdown proxy was computed per account using running-max on the cumulative PnL series.

**Statistical approach:** Mann-Whitney U tests (non-parametric; robust to PnL fat tails) for all distributional comparisons, with Cohen's d for effect size. Segmentation used percentile-based bins (33rd/67th) to avoid hardcoding thresholds. ML used Gradient Boosting with lag-1 and 3-day rolling features, evaluated via stratified 5-fold cross-validation (ROC-AUC).

---

## Insights

**1. Sentiment is an extremely significant PnL predictor (p = 6.65 × 10⁻⁹²)**  
Median daily PnL is +$290 on Extreme Greed days versus -$150 on Extreme Fear days — a $440 daily swing per account. Win rate shifts 7pp (56% Greed vs 49% Fear). Critically, liquidation rates double during Fear (10.8%) versus Greed (5.5%), confirming that sentiment shapes not just direction but *survival*. The effect is small (Cohen's d = 0.178) but overwhelmingly consistent across 43,800 observations.

**2. High leverage is the primary destroyer of returns — and sentiment makes it worse**  
Low-leverage traders (≤p33, avg 4×) earn +$670/day with 62% win rate. High-leverage traders (>p67, avg 42×) lose -$1,933/day with 49% win rate — a 2,600% performance gap from the same market. Worse, traders instinctively double their leverage during Greed (12× → 23×) but fail to reduce it fast enough during Fear, when the downside is worst. Consistent Performers use only 5.7× leverage on average; Consistent Losers use 28.4×.

**3. Overtrading destroys edge; infrequent traders outperform by 10pp in win rate**  
Infrequent traders (below-median daily trade count) win 58.9% of days versus 48.2% for frequent traders — a 10.7pp edge. This holds across all sentiment regimes. The K-Means clustering identifies "Active Scalpers" as barely break-even despite high activity, while "Consistent Performers" trade selectively at low leverage and generate the highest Sharpe and Calmar ratios. Frequency is not edge — it is noise amplification.

**Bonus ML:** The GBM model achieves ROC-AUC 0.635 (vs 0.50 random baseline) predicting next-day profitability. Top features are account-level Sharpe proxy, today's win rate, and the sentiment value — confirming that streak behavior and sentiment regime are the strongest actionable signals available without additional alpha.

---

## Strategy Recommendations

**Strategy 1 — Sentiment-Adaptive Leverage Bands:**  
Cap leverage at ≤3× when F/G < 25, ≤5× when F/G < 45, allow up to 15× when F/G > 60, reduce back to ≤10× when F/G > 75 (reversal guard). Targeted at High-Risk Gamblers and Retail Gamblers — the two clusters with the highest leverage amplification and worst drawdowns. Estimated impact: reducing high-leverage traders to ≤5× during Fear would eliminate approximately 80% of liquidation events.

**Strategy 2 — Sentiment-Driven Directional Bias + Frequency Control:**  
Shift long ratio above 60% during Greed; hold neutral/slight-short during Fear. Cut trade frequency by 30% during Fear days — preserve capital for higher-conviction setups. Active Scalpers should apply the strictest frequency reduction. Cautious Opportunists have the most to gain from deploying capital at early Greed transitions (F/G index crossing 50 upward), which historically coincides with the strongest risk-adjusted entries.

> Combined, these two rules address the two behavioral failures most predictive of losses: over-leveraging during fear, and maintaining long bias when sentiment deteriorates.
