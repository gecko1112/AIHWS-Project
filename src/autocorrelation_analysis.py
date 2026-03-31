#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson


# In[6]:


# ── Config — edit config.py in the project root to switch between datasets ───
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from config import IRELAND_PATH, OUTPUT_DIR, FEATURES, TARGET, RANDOM_SEED
TARGET_NUM = TARGET
TARGET_CAT = 'CCME_WQI'
MIN_OBS    = 30
N_STATIONS = 6
print(f"Dataset : {IRELAND_PATH}")


# In[7]:


# ── 1. Load + parse dates ────────────────────────────────────────────────────
df = pd.read_csv(IRELAND_PATH)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values(["Area", "Date"]).reset_index(drop=True)

print(f"Shape          : {df.shape}")
print(f"Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Unique stations: {df['Area'].nunique()}")
print(f"Waterbody types: {df['Waterbody Type'].unique().tolist()}")

counts = df.groupby("Area").size()
dense  = counts[counts >= MIN_OBS].sort_values(ascending=False)
print(f"\nStations with ≥{MIN_OBS} measurements: {len(dense)} / {len(counts)}")
print(f"Measurements per station (all): median={counts.median():.0f}, mean={counts.mean():.1f}, max={counts.max()}")


# In[8]:


# ── 2. Plot 1 — WQI trend over time (global monthly mean) ────────────────────
monthly = df.set_index("Date")[TARGET_NUM].resample("ME").mean().dropna()

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(monthly.index, monthly.values, lw=1, alpha=0.6, color="steelblue", label="Monthly mean")
rolling = monthly.rolling(12, center=True).mean()
ax.plot(rolling.index, rolling.values, lw=2.5, color="navy", label="12-month rolling mean")
ax.set_ylabel("CCME WQI Score")
ax.set_title("Global WQI Trend Over Time — Ireland (all stations)")
ax.legend()
plt.tight_layout()
plt.savefig("autocorr_trend.png", dpi=150)
plt.show()
print("Observation: check for long-term trend or seasonal pattern")


# In[9]:


# ── 3. Plot 2 — ACF for top N dense stations ─────────────────────────────────
top_stations = dense.head(N_STATIONS).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
dw_results = {}

for ax, station in zip(axes.flat, top_stations):
    series = (df[df["Area"] == station]
              .sort_values("Date")[TARGET_NUM]
              .dropna()
              .reset_index(drop=True))
    plot_acf(series, lags=20, ax=ax, alpha=0.05, title="")
    dw = durbin_watson(series.values)
    dw_results[station] = dw
    short = station[:35] + "..." if len(station) > 35 else station
    ax.set_title(f"{short}\n(n={len(series)}, DW={dw:.2f})", fontsize=8)
    ax.set_xlabel("Lag")

fig.suptitle("Temporal Autocorrelation (ACF) — Top Dense Stations\n"
             "Shaded band = 95% confidence. DW≈2: no autocorr; DW<2: positive autocorr",
             y=1.02)
plt.tight_layout()
plt.savefig("autocorr_acf.png", dpi=150)
plt.show()

print("Durbin-Watson statistics (2=no autocorr, <2=positive, >2=negative):")
for s, dw in dw_results.items():
    verdict = "positive autocorr" if dw < 1.5 else ("negative autocorr" if dw > 2.5 else "mild/no autocorr")
    print(f"  {s[:50]:50s}  DW={dw:.2f}  ({verdict})")


# In[10]:


# ── 4. Plot 3 — ACF lag-1 distribution across all dense stations ─────────────
lag1_vals = []
for station in dense.index:
    s = df[df["Area"] == station].sort_values("Date")[TARGET_NUM].dropna()
    if len(s) >= MIN_OBS:
        lag1_vals.append(s.autocorr(lag=1))

lag1_vals = np.array([v for v in lag1_vals if not np.isnan(v)])
print(f"ACF lag-1 across {len(lag1_vals)} dense stations:")
print(f"  Mean   : {lag1_vals.mean():.3f}")
print(f"  Median : {np.median(lag1_vals):.3f}")
print(f"  Std    : {lag1_vals.std():.3f}")
print(f"  % > 0.1: {(lag1_vals > 0.1).mean()*100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(lag1_vals, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
ax.axvline(lag1_vals.mean(), color="tomato", lw=2, label=f"Mean = {lag1_vals.mean():.3f}")
ax.axvline(0, color="grey", lw=1, linestyle="--", label="No autocorr (0)")
ax.set_xlabel("ACF at lag 1")
ax.set_ylabel("Number of stations")
ax.set_title(f"Distribution of Lag-1 Autocorrelation Across {len(lag1_vals)} Stations")
ax.legend()
plt.tight_layout()
plt.savefig("autocorr_lag1_dist.png", dpi=150)
plt.show()


# In[11]:


# ── 5. Station-level clustering (NOT true spatial autocorrelation) ─────────────
#
# TRUE spatial autocorrelation (Moran's I) requires lat/lon coordinates to
# build a geographic distance matrix. This dataset has no coordinates — only
# station names and waterbody types.
#
# What we CAN compute: Intraclass Correlation Coefficient (ICC)
#   → Asks: are repeated readings from the SAME named station more similar
#     to each other than to readings from OTHER stations?
#   → This is station-level repeatability, NOT geographic proximity.
#
# ICC = between_station_variance / (between + within)
#   ≈ 1 → stations consistently differ (strong station identity)
#   ≈ 0 → stations are interchangeable; variance is within-station over time

overall_var = df[TARGET_NUM].var()
within_var  = df.groupby("Area")[TARGET_NUM].var().mean()
between_var = df.groupby("Area")[TARGET_NUM].mean().var()
icc = between_var / (between_var + within_var)

print("Station-level ICC (NOT geographic spatial autocorrelation):")
print(f"  Overall variance        : {overall_var:.2f}")
print(f"  Mean within-station var : {within_var:.2f}")
print(f"  Between-station var     : {between_var:.2f}")
print(f"  ICC                     : {icc:.3f}")
print()
if icc > 0.3:
    print("→ High ICC: stations have distinct persistent WQI levels")
    print("  Implication: a station-based split is important to avoid leakage")
elif icc > 0.1:
    print("→ Moderate ICC: some station-level consistency exists")
else:
    print("→ Low ICC: stations vary widely within themselves")
    print("  Implication: station identity is less of a leakage concern")
print()
print("NOTE: For true spatial autocorrelation, lat/lon coordinates would be")
print("needed to apply Moran's I. Those are not available in this dataset.")


# In[12]:


# ── 6. Plot 4 — Within vs Between station variance + WQI by waterbody type ───
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar(["Within-station", "Between-station"], [within_var, between_var],
       color=["steelblue", "tomato"], alpha=0.8, edgecolor="white")
ax.set_ylabel("Variance")
ax.set_title(f"Station-level Variance Decomposition\nICC = {icc:.3f}  (station repeatability, not geo-distance)")
for i, v in enumerate([within_var, between_var]):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=11)

ax2 = axes[1]
for wtype in df["Waterbody Type"].unique():
    vals = df[df["Waterbody Type"] == wtype][TARGET_NUM].dropna()
    ax2.hist(vals, bins=40, alpha=0.5, label=wtype, density=True)
ax2.set_xlabel("CCME WQI Score")
ax2.set_ylabel("Density")
ax2.set_title("WQI Distribution by Waterbody Type\n(proxy for broad spatial grouping)")
ax2.legend()

plt.suptitle("Station-level Clustering — Ireland\n"
             "Note: true Moran's I not computable (no coordinates in dataset)", y=1.02)
plt.tight_layout()
plt.savefig("autocorr_spatial.png", dpi=150)
plt.show()


# In[13]:


# ── 7. Summary ───────────────────────────────────────────────────────────────
print("=" * 60)
print("AUTOCORRELATION SUMMARY")
print("=" * 60)
print()
print("TEMPORAL AUTOCORRELATION — confirmed:")
print(f"  Lag-1 ACF mean across {len(lag1_vals)} dense stations: {lag1_vals.mean():.3f}")
print(f"  {(lag1_vals > 0.1).mean()*100:.0f}% of stations show ACF > 0.1 at lag 1")
print("  → Short-range correlation: decays to near-zero by lag 3-4")
print("  → A random split risks temporal leakage")
print("  → Recommended: time-based train/val/test split")
print("  → Supports LSTM as an architecture choice")
print()
print("SPATIAL AUTOCORRELATION — cannot be formally tested:")
print("  Dataset has no lat/lon coordinates.")
print("  Moran's I (standard spatial autocorrelation test) requires distances.")
print(f"  Station-level ICC = {icc:.3f} (measures station repeatability only)")
print("  Waterbody type (River/Lake/Coastal/Transitional) provides a coarse")
print("  spatial grouping — WQI distributions differ meaningfully by type.")
print()
print("  For the presentation: acknowledge this limitation explicitly.")
print("  If coordinates can be obtained (e.g. from EPA Ireland open data),")
print("  Moran's I could be computed as a follow-up.")

