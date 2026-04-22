"""
Realistic synthetic data generator for Hyperliquid × Fear/Greed analysis.
Creates genuine trader heterogeneity so segmentation & clustering are meaningful.

Trader archetypes baked in:
  - Whale Scalpers    (20 accounts): very high leverage 30-50×, many trades, mixed PnL
  - Smart Money       (25 accounts): low leverage 2-6×, selective, consistent winners
  - Retail Gamblers   (40 accounts): mid-high leverage 10-25×, frequent, mostly losers
  - Swing Traders     (25 accounts): low leverage 3-8×, infrequent, moderate PnL
  - Degen Losers      (10 accounts): extreme leverage 40-100×, frequent liquidations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ─── Fear/Greed Index (Jan 2023 – Dec 2023) ──────────────────────────────
dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
# Realistic BTC cycle: Jan–Feb fear, Mar–Jun recovery, Jul–Oct greed, Nov–Dec mixed
raw = (
    np.concatenate([
        np.linspace(20, 35, 60),   # Jan-Feb: Fear
        np.linspace(35, 65, 60),   # Mar-Apr: Recovery
        np.linspace(65, 80, 61),   # May-Jul: Greed
        np.linspace(80, 55, 61),   # Aug-Sep: Pullback
        np.linspace(55, 75, 62),   # Oct-Nov: Greed
        np.linspace(75, 40, 61),   # Dec: Fear
    ])[:365]
    + np.random.normal(0, 7, 365)
)
raw = np.clip(raw, 5, 97).astype(int)

def classify(v):
    if v <= 24:   return "Extreme Fear"
    elif v <= 44: return "Fear"
    elif v <= 54: return "Neutral"
    elif v <= 74: return "Greed"
    else:          return "Extreme Greed"

fg_df = pd.DataFrame({
    "date":           dates,
    "value":          raw,
    "classification": [classify(v) for v in raw],
})
fg_df.to_csv("data/fear_greed.csv", index=False)
print("Fear/Greed saved:", fg_df.shape)
print(fg_df["classification"].value_counts().to_string())

# ─── Trader Archetypes ────────────────────────────────────────────────────
def hex_addr(i): return f"0x{i:040x}"

archetypes = [
    # (name, n, lev_mu, lev_sig, trade_mu, edge, pnl_vol, liq_p, long_bias_greed)
    ("Whale Scalpers",   20, 35,  8,  15, +0.001, 0.05, 0.08, 0.58),
    ("Smart Money",      25,  4,  1,   3, +0.008, 0.02, 0.01, 0.60),
    ("Retail Gamblers",  40, 16,  5,   8, -0.004, 0.06, 0.12, 0.55),
    ("Swing Traders",    25,  5,  2,   2, +0.003, 0.025, 0.02, 0.58),
    ("Degen Losers",     10, 60, 15,  20, -0.012, 0.09, 0.30, 0.52),
]

symbols = ["BTC-USD","ETH-USD","SOL-USD","ARB-USD","DOGE-USD"]
sym_prices = {"BTC-USD":28000,"ETH-USD":1800,"SOL-USD":22,"ARB-USD":1.2,"DOGE-USD":0.08}

accounts_meta = {}  # addr → archetype info
all_rows = []
acct_id = 1

for (arch_name, n_accts, lev_mu, lev_sig, trade_mu,
     edge, pnl_vol, liq_p_base, long_bias_greed) in archetypes:

    for _ in range(n_accts):
        addr = hex_addr(acct_id); acct_id += 1
        # Each account has its own fixed leverage tendency
        acct_lev = max(1, np.random.lognormal(np.log(lev_mu), lev_sig/lev_mu))
        acct_lev = min(100, acct_lev)
        accounts_meta[addr] = dict(
            archetype=arch_name, base_lev=acct_lev,
            edge=edge + np.random.normal(0, 0.002),   # slight per-account variation
            pnl_vol=pnl_vol, liq_p=liq_p_base,
            trade_mu=trade_mu, long_bias_greed=long_bias_greed
        )

for day in dates:
    row = fg_df[fg_df["date"]==day].iloc[0]
    fg_val = row["value"]
    cls    = row["classification"]
    is_greed = cls in ("Greed","Extreme Greed")
    is_fear  = cls in ("Fear","Extreme Fear")

    for addr, meta in accounts_meta.items():
        # Adjust trade count by sentiment volatility
        vol_mult = 1 + abs(fg_val - 50) / 80
        n_trades = max(1, int(np.random.poisson(meta["trade_mu"] * vol_mult)))

        for _ in range(n_trades):
            sym  = np.random.choice(symbols, p=[0.55,0.25,0.10,0.05,0.05])
            base = sym_prices[sym] * (1 + np.random.normal(0, 0.015))

            # Leverage: accounts boost during greed, cut during fear
            lev_mult = 1.3 if is_greed else (0.7 if is_fear else 1.0)
            lev = int(np.clip(meta["base_lev"] * lev_mult * np.random.lognormal(0, 0.2), 1, 100))

            # Long/short bias
            long_p = meta["long_bias_greed"] if is_greed else (1 - meta["long_bias_greed"] + 0.05)
            side = "BUY" if np.random.random() < long_p else "SELL"

            size = abs(np.random.lognormal(0.5, 0.8))
            notional = base * size

            # PnL: sentiment modulates edge
            sentiment_edge = 0.003 if is_greed else (-0.003 if is_fear else 0)
            pnl_pct = np.random.normal(meta["edge"] + sentiment_edge, meta["pnl_vol"])
            closed_pnl = round(notional * pnl_pct, 4)

            # Liquidation events
            liq_p = meta["liq_p"] * (1.5 if is_fear else 0.7) / n_trades
            event = "LIQUIDATION" if np.random.random() < liq_p else "TRADE"
            if event == "LIQUIDATION":
                closed_pnl = -abs(notional / lev) * 0.9  # wipe margin

            ts = day + timedelta(hours=int(np.random.uniform(0,24)),
                                 minutes=int(np.random.uniform(0,60)))

            all_rows.append({
                "account":         addr,
                "symbol":          sym,
                "execution_price": round(base, 4),
                "size":            round(size, 4),
                "side":            side,
                "time":            ts,
                "start_position":  round(np.random.uniform(-10,10), 4),
                "event":           event,
                "closedPnL":       closed_pnl,
                "leverage":        lev,
                "fee":             round(abs(notional)*0.0004, 6),
                "crossed":         np.random.choice([True,False], p=[0.3,0.7]),
            })

tr = pd.DataFrame(all_rows)
tr.to_csv("data/trader_data.csv", index=False)

# Also save ground-truth archetype for validation
arch_df = pd.DataFrame([
    {"account": addr, "true_archetype": meta["archetype"],
     "base_leverage": round(meta["base_lev"],2)}
    for addr, meta in accounts_meta.items()
])
arch_df.to_csv("data/account_archetypes.csv", index=False)

print(f"\nTrader data saved: {tr.shape}")
print("Event distribution:", tr["event"].value_counts().to_dict())
print("Leverage range:", tr["leverage"].min(), "–", tr["leverage"].max())
print("ClosedPnL range:", tr["closedPnL"].min().round(2), "–", tr["closedPnL"].max().round(2))
