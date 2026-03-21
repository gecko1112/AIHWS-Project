import os
import pandas as pd

DATA_DIR = "data/Dataset/Country-Wise Data"
OUT_DIR  = "output"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "Ireland_dataset.csv"),
                 usecols=["Area", "Waterbody Type"])

counts = df.groupby(["Area", "Waterbody Type"]).size().reset_index(name="n_measurements")
top150 = (counts[counts["n_measurements"] >= 30]
          .sort_values("n_measurements", ascending=False)
          .head(150)
          .reset_index(drop=True))

top150["latitude"]  = ""
top150["longitude"] = ""

out = os.path.join(OUT_DIR, "ireland_stations_geocode.csv")
top150.to_csv(out, index=False)
print(f"Saved {len(top150)} stations -> {out}")
print(top150.head(10).to_string(index=False))
