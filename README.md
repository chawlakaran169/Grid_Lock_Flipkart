# Traffic Demand Prediction

Predict traffic demand at a given location and timestamp using gradient boosted tree ensembles.

**Metric:** `max(0, 100 * r2_score(actual, predicted))`

---

## Dataset

| File | Shape |
|------|-------|
| train.csv | 77,299 × 11 |
| test.csv | 41,778 × 10 |
| sample_submission.csv | 5 × 2 |

**Target column:** `demand` — traffic demand at a location and timestamp

---

## How to Run

```bash
pip install lightgbm xgboost catboost pygeohash scikit-learn pandas numpy matplotlib seaborn
python traffic_demand_prediction.py
```

Or open `traffic_demand_prediction.ipynb` in Jupyter and run all cells top to bottom.

---

## Approach

### Feature Engineering

**Geohash (location)**
- Decoded geohash strings to latitude and longitude
- Created 4 spatial prefix levels: geo3, geo4, geo5, geo6 (coarse to fine)

**Timestamp (time)**
- Extracted hour, minute, minute-of-day
- Peak hour flags: morning rush (7–9 AM), evening rush (5–7 PM), night, midday
- Cyclical encoding using sin/cos so hour 23 and hour 0 are treated as adjacent

**Day of Week**
- Mapped day names to integers (Monday=0 to Sunday=6)
- Weekend flag, cyclical sin/cos encoding

**Interaction Features**
- geo4 × hour, geo4 × day, day × hour
- geo4 × weekend, geo4 × peak_am, geo4 × peak_pm

**Road Features**
- NumberofLanes as numeric, high capacity flag (≥4 lanes)
- LargeVehicles and Landmarks binarised (Yes/No → 1/0)

**Weather and Temperature**
- Temperature coerced to numeric, missing values imputed with median
- High/low temperature flags (above 75th / below 25th percentile)
- Weather label encoded

**Target Aggregation Features**
- Mean, median, std of demand grouped by: geo3, geo4, geo5, geo6, hour, day, day×hour, geo4×hour, geo4×day, RoadType, Weather
- Computed on training data only and merged onto both train and test

### Models

Three models trained with 5-fold cross validation, then blended by OOF score weights:

- **LightGBM** — num_leaves=255, lr=0.02, n_estimators=3000, early stopping=150
- **XGBoost** — max_depth=8, lr=0.02, n_estimators=3000, early stopping=150
- **CatBoost** — depth=8, lr=0.02, iterations=3000, early stopping=150

Final predictions are clipped to 0 (demand cannot be negative).

---

## Output Files

| File | Description |
|------|-------------|
| submission.csv | 41,778 × 2 (Index, demand) |
| feature_importance.png | Top 30 features by importance |
| oof_diagnostics.png | Actual vs predicted plot |
