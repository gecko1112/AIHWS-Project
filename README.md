# Water Quality Index — AI Classification & Spatial Analysis

Multi-country water quality classification using a deep neural network on CCME WQI data, with temporal/spatial autocorrelation analysis and interactive geospatial visualisation.

---

## Dataset

| Country | Stations | Rows |
|---------|----------|------|
| Ireland | ~9,700 | 235k |
| England | ~48,600 | 2.1M |
| USA | ~955 | 414k |
| Canada | ~2,350 | 4k |
| China | 1 (Hou Bay) | 46k |

**Features:** Ammonia, BOD, Dissolved Oxygen, Orthophosphate, pH, Temperature, Nitrogen, Nitrate
**Target:** CCME WQI class — `Excellent / Good / Fair / Marginal / Poor`

---

## Project Structure

```
AIHWS/
├── main.ipynb                        # Main model: training, evaluation, saving
├── geocode_map.ipynb                 # Geonames geocoding + ipyleaflet map
├── src/
│   ├── autocorrelation_analysis.ipynb  # Temporal ACF + station-level ICC
│   ├── minority_analysis.ipynb         # Minority class deep-dive
│   └── china_eval.ipynb                # China single-station evaluation
├── data/
│   └── Dataset/Country-Wise Data/    # Per-country CSVs
├── output/                           # Generated plots and CSVs
├── export_stations.py                # Exports top-150 Ireland stations for geocoding
├── wqi_model.keras                   # Saved model
├── scaler.joblib                     # Fitted StandardScaler
└── label_encoder.joblib              # Fitted LabelEncoder
```

---

## Model

Dense neural network (TensorFlow/Keras):
- `Dense(128) → BatchNorm → Dropout(0.3)`
- `Dense(64)  → BatchNorm → Dropout(0.3)`
- `Dense(5, softmax)` — 5 WQI classes

Class imbalance handled with `compute_class_weight`. Early stopping + ReduceLROnPlateau on val_loss.

---

## Analysis Notebooks

### `autocorrelation_analysis.ipynb`
- **Temporal:** ACF plots per station, Durbin-Watson test — confirms short-range temporal correlation (lag 1–3). Motivates a **time-based train/val/test split** to avoid leakage.
- **Spatial:** No lat/lon in the dataset, so Moran's I cannot be computed directly. Station-level ICC reported instead (are readings from the same station more similar than across stations?).

### `geocode_map.ipynb`
- Exports top 150 Ireland stations by measurement count
- Queries the **Geonames API** to resolve station names to coordinates
- 108/150 stations successfully geocoded
- Renders an interactive **ipyleaflet** map with clustered markers

---

## Ideas / Planned

- [ ] **Moran's I spatial autocorrelation** — now that 108 Ireland stations are geocoded, build a spatial weights matrix and run a proper spatial autocorrelation test on mean WQI per station
- [ ] **Time-based split** — replace random stratified split with a chronological split to respect temporal structure found in ACF analysis
- [ ] **LSTM** — reframe as a sequence model using each station's time series; temporal autocorrelation justifies sequence modelling
- [ ] **Cross-country generalisation** — train on Ireland, evaluate zero-shot on England/USA to test transferability

---

## Setup

```bash
uv sync
uv run jupyter notebook
```
