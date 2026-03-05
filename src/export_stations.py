import os
import pandas as pd

DATA_DIR = "data/Dataset/Country-Wise Data"
OUT_DIR  = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Ireland — real place names, geocodable on Geonames
ireland = pd.read_csv(
    os.path.join(DATA_DIR, "Ireland_dataset.csv"),
    usecols=["Country", "Area", "Waterbody Type"]
).drop_duplicates().sort_values(["Waterbody Type", "Area"]).reset_index(drop=True)
ireland.to_csv(os.path.join(OUT_DIR, "ireland_stations.csv"), index=False)
print(f"Ireland: {len(ireland)} unique stations -> output/ireland_stations.csv")

# China — single station, included for completeness
china = pd.read_csv(
    os.path.join(DATA_DIR, "China_dataset.csv"),
    usecols=["Country", "Area", "Waterbody Type"]
).drop_duplicates().reset_index(drop=True)
china.to_csv(os.path.join(OUT_DIR, "china_stations.csv"), index=False)
print(f"China  : {len(china)} unique station  -> output/china_stations.csv")

print()
print("England/Canada/USA skipped — their 'Area' columns are internal codes,")
print("not place names, so they cannot be resolved on Geonames.")
