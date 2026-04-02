"""
Bulk geocode all Ireland stations with >=MIN_OBS measurements via Geonames API.
Resumes from existing output — safe to re-run if interrupted.

Usage:
    uv run python src/geocode_bulk.py

Rate limit: Geonames free tier = 1000 req/hour. Script sleeps 3.7s between
requests to stay safe (~970/hour). For 2188 stations expect ~2.5 hours.
Already-geocoded stations are skipped.
"""

import os, sys, time, re
import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
MIN_OBS      = 20
SLEEP_SEC    = 3.7        # 3.7s → ~970 req/hour (under 1000 limit)
IRELAND_CSV  = os.path.join(os.path.dirname(__file__), '..', 'data', 'Dataset',
                             'Country-Wise Data', 'Ireland_dataset.csv')
OUT_CSV      = os.path.join(os.path.dirname(__file__), '..', 'output',
                             'ireland_stations_geocode_bulk.csv')
GEONAMES_USER = os.getenv("GEONAMES_USER", "gecko1112")   # your Geonames username

# ── Query cleaning ────────────────────────────────────────────────────────────
def clean_query(area: str) -> str:
    """Strip station codes, sub-catchment prefixes, and noise words."""
    # Remove trailing codes like _060, _010
    q = re.sub(r'_\d{3,}', '', area)
    # Remove sub-catchment prefix (everything before last comma if > 1 comma)
    parts = [p.strip() for p in q.split(',')]
    q = parts[-1] if len(parts) > 1 else parts[0]
    # Remove parenthetical qualifiers
    q = re.sub(r'\(.*?\)', '', q).strip()
    # Collapse whitespace
    q = ' '.join(q.split())
    return q

# ── Geonames lookup ───────────────────────────────────────────────────────────
def geonames_search(query: str) -> tuple:
    url = "http://api.geonames.org/searchJSON"
    base = {"q": query + " Ireland", "maxRows": 1, "country": "IE",
            "username": GEONAMES_USER}
    for params in [
        {**base, "featureClass": "H"},   # hydrological first
        {**base},                          # fallback: any feature
    ]:
        try:
            r = requests.get(url, params=params, timeout=10)
            hits = r.json().get("geonames", [])
            if hits:
                return float(hits[0]["lat"]), float(hits[0]["lng"])
        except Exception as e:
            print(f"    API error for '{query}': {e}")
            time.sleep(5)
    return None, None

# ── Build station list ────────────────────────────────────────────────────────
df_raw = pd.read_csv(IRELAND_CSV)
counts = df_raw.groupby('Area').agg(
    n_measurements=('Area', 'count'),
    waterbody_type=('Waterbody Type', 'first')
).reset_index()
counts = counts[counts['n_measurements'] >= MIN_OBS].sort_values(
    'n_measurements', ascending=False).reset_index(drop=True)
print(f"Stations with >= {MIN_OBS} measurements: {len(counts)}")

# Add query column
counts['query'] = counts['Area'].apply(clean_query)

# ── Resume from existing output ───────────────────────────────────────────────
if os.path.exists(OUT_CSV):
    existing = pd.read_csv(OUT_CSV)
    done = set(existing[existing['latitude'].notna()]['Area'].tolist())
    print(f"Resuming — {len(done)} already geocoded, {len(counts)-len(done)} remaining")
    # Merge existing coords back
    counts = counts.merge(
        existing[['Area', 'latitude', 'longitude']],
        on='Area', how='left'
    )
else:
    done = set()
    counts['latitude']  = None
    counts['longitude'] = None

# ── Also seed from the 150-station file ──────────────────────────────────────
old_geo = os.path.join(os.path.dirname(__file__), '..', 'output',
                        'ireland_stations_geocode_coded.csv')
if os.path.exists(old_geo):
    old = pd.read_csv(old_geo)[['Area', 'latitude', 'longitude']]
    old = old[old['latitude'].notna()]
    for _, row in old.iterrows():
        mask = counts['Area'] == row['Area']
        if mask.any() and pd.isna(counts.loc[mask, 'latitude'].values[0]):
            counts.loc[mask, 'latitude']  = row['latitude']
            counts.loc[mask, 'longitude'] = row['longitude']
            done.add(row['Area'])
    print(f"Seeded from existing 150-station file — {len(done)} total known")

# ── Geocode remaining ─────────────────────────────────────────────────────────
todo = counts[counts['latitude'].isna()].index.tolist()
print(f"To geocode: {len(todo)} stations\n")

for i, idx in enumerate(todo):
    row   = counts.loc[idx]
    query = row['query']
    lat, lng = geonames_search(query)
    counts.at[idx, 'latitude']  = lat
    counts.at[idx, 'longitude'] = lng
    status = f"OK ({lat:.3f}, {lng:.3f})" if lat else "NOT FOUND"
    pct = (i + 1) / len(todo) * 100
    print(f"  [{i+1:4d}/{len(todo)}] {query[:45]:45s} -> {status}  ({pct:.1f}%)")

    # Save every 50 stations so progress is preserved
    if (i + 1) % 50 == 0:
        counts.to_csv(OUT_CSV, index=False)
        found = counts['latitude'].notna().sum()
        print(f"  [checkpoint] Saved — {found}/{len(counts)} geocoded so far")

    time.sleep(SLEEP_SEC)

# ── Final save ────────────────────────────────────────────────────────────────
counts.to_csv(OUT_CSV, index=False)
found = counts['latitude'].notna().sum()
print(f"\nDone. Geocoded {found} / {len(counts)} stations ({found/len(counts)*100:.1f}%)")
print(f"Saved to {OUT_CSV}")