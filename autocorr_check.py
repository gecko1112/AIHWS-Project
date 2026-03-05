import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/werwo/Desktop/Master/2nd Semester/AIHWS/data/Dataset/Country-Wise Data/Ireland_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values(["Area", "Date"])

print("Date range:", df["Date"].min(), "->", df["Date"].max())
print("Unique areas:", df["Area"].nunique())
print("Waterbody types:", df["Waterbody Type"].unique())

counts = df.groupby("Area").size()
print("\nMeasurements per area:")
print(counts.describe().round(1))

top_area = counts.idxmax()
sub = df[df["Area"] == top_area].sort_values("Date")
print("\nTop area:", top_area, "(", len(sub), "rows)")
gaps = sub["Date"].diff().dt.days.dropna()
print("Avg gap:", round(gaps.mean(), 1), "days | Median:", round(gaps.median(), 1), "days")

acf_vals = [sub["CCME_Values"].autocorr(lag=i) for i in range(1, 7)]
print("Temporal ACF lags 1-6:", [round(v, 3) for v in acf_vals])

# Spatial: ICC proxy
overall_var = df["CCME_Values"].var()
within_var  = df.groupby("Area")["CCME_Values"].var().mean()
print("\nOverall variance:", round(overall_var, 2))
print("Mean within-area variance:", round(within_var, 2))
print("ICC proxy (between/total):", round(1 - within_var / overall_var, 3))

# Also check a few more areas' ACF
print("\nACF lag-1 across sample of areas:")
sample_areas = counts[counts >= 20].sample(10, random_state=42).index
for area in sample_areas:
    s = df[df["Area"] == area]["CCME_Values"]
    print(f"  {area[:40]:40s} acf(1)={s.autocorr(lag=1):.3f}")
