#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import time
import requests
import pandas as pd
from ipyleaflet import Map, Marker, MarkerCluster, Popup
from ipywidgets import HTML

import os
import dotenv
dotenv.load_dotenv()


# In[3]:


# ── Config ───────────────────────────────────────────────────────────────────
# To run this section you have to register at:https://www.geonames.org/login
# Enable free web services at: https://www.geonames.org/manageaccount
GEONAMES_USER = os.getenv("USERNAME")  # from .env file

STATIONS_CSV = "../output/ireland_stations_geocode_prep.csv"
OUT_CSV      = "../output/ireland_stations_geocode_coded.csv"  # overwrite with coords filled


# In[ ]:


# ── 1. Clean station name -> search query ────────────────────────────────────
def make_query(area: str) -> str:
    """Extract a geocodable place name from the Ireland station Area string.

    Pattern: 'CatchmentName, SpecificStation_CODE'
    Strategy: prefer the specific station part (after comma), strip codes.
    """
    # Split on comma - take specific station part if available
    parts = area.split(",", 1)
    name = parts[1].strip() if len(parts) > 1 else parts[0].strip()

    # Remove trailing _NNN measurement codes  e.g. _010, _060
    name = re.sub(r'_\d+$', '', name).strip()
    # Remove leading codes like 'FLESK (KERRY)' -> keep 'Flesk Kerry'
    name = re.sub(r'[()_]', ' ', name)
    # Collapse whitespace
    name = ' '.join(name.split())
    return name

# Preview
df = pd.read_csv(STATIONS_CSV)
df["query"] = df["Area"].apply(make_query)
print(df[["Area", "query"]].head(10).to_string(index=False))


# In[5]:


# ── 2. Geonames API lookup ───────────────────────────────────────────────────
def geonames_search(query: str, username: str) -> tuple:
    """Return (lat, lng) for the top Geonames result in Ireland, or (None, None)."""
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q"          : query + " Ireland",
        "maxRows"    : 1,
        "country"    : "IE",
        "username"   : username,
        "featureClass": "H",   # H = stream/lake/river features
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        hits = data.get("geonames", [])
        if hits:
            return float(hits[0]["lat"]), float(hits[0]["lng"])
        # Retry without feature class filter if no hydrological result
        params.pop("featureClass")
        r = requests.get(url, params=params, timeout=10)
        hits = r.json().get("geonames", [])
        if hits:
            return float(hits[0]["lat"]), float(hits[0]["lng"])
    except Exception as e:
        print(f"  Error for '{query}': {e}")
    return None, None


# Only geocode rows that don't have coords yet
todo = df[df["latitude"].isna() | (df["latitude"] == "")].index
print(f"Geocoding {len(todo)} stations...")

for i, idx in enumerate(todo):
    q = df.at[idx, "query"]
    lat, lng = geonames_search(q, GEONAMES_USER)
    df.at[idx, "latitude"]  = lat
    df.at[idx, "longitude"] = lng
    status = f"OK ({lat:.3f}, {lng:.3f})" if lat else "NOT FOUND"
    print(f"  [{i+1}/{len(todo)}] {q[:50]:50s} -> {status}")
    time.sleep(0.5)   # stay well under 1000 req/hour free limit

df.to_csv(OUT_CSV, index=False)
found = df["latitude"].notna() & (df["latitude"] != "")
print(f"\nGeocoded {found.sum()} / {len(df)} stations. Saved to {OUT_CSV}")


# In[6]:


# ── 3. ipyleaflet map ────────────────────────────────────────────────────────
df_map = df.dropna(subset=["latitude", "longitude"]).copy()
df_map = df_map[df_map["latitude"] != ""]
df_map["latitude"]  = pd.to_numeric(df_map["latitude"],  errors="coerce")
df_map["longitude"] = pd.to_numeric(df_map["longitude"], errors="coerce")
df_map = df_map.dropna(subset=["latitude", "longitude"])

print(f"Plotting {len(df_map)} stations on map")

m = Map(center=[53.3, -8.0], zoom=7)

markers = []
for _, row in df_map.iterrows():
    popup_html = HTML(value=(
        f"<b>{row['Area']}</b><br>"
        f"Type: {row['Waterbody Type']}<br>"
        f"Measurements: {row['n_measurements']}"
    ))
    marker = Marker(
        location=(row["latitude"], row["longitude"]),
        title=row["Area"],
        draggable=False,
    )
    marker.popup = popup_html
    markers.append(marker)

cluster = MarkerCluster(markers=markers)
m.add(cluster)
m

