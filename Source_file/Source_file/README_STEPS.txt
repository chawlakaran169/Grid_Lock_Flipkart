======================================================================
  STEP-BY-STEP GUIDE — HOW TO RUN & SUBMIT
======================================================================

PREREQUISITES
─────────────
  Python 3.8+  (Anaconda / standard install both fine)

STEP 1 — INSTALL DEPENDENCIES
──────────────────────────────
  Open your terminal / Anaconda Prompt and run:

    pip install lightgbm xgboost catboost pygeohash scikit-learn pandas numpy matplotlib seaborn

  This takes ~2–3 minutes. Do this once.

STEP 2 — ORGANISE YOUR FOLDER
──────────────────────────────
  Create a folder (e.g., "traffic_project") and put these files in it:

    traffic_project/
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    └── traffic_demand_prediction.py   ← from this zip

STEP 3 — RUN THE SCRIPT
─────────────────────────
  In your terminal, navigate to the folder and run:

    cd traffic_project
    python traffic_demand_prediction.py

  What happens:
    ✓ Loads and explores the data
    ✓ Engineers all features (geohash, time, interactions, aggs)
    ✓ Trains LightGBM (5-fold)
    ✓ Trains XGBoost (5-fold)
    ✓ Trains CatBoost (5-fold)
    ✓ Blends predictions optimally
    ✓ Saves  submission.csv

  Expected runtime: 20–40 minutes (CPU, depending on your machine)

  Output files created in the same folder:
    submission.csv          ← UPLOAD THIS to the competition
    feature_importance.png  ← top features chart
    eda_demand_dist.png     ← demand distribution chart
    oof_diagnostics.png     ← model quality chart

STEP 4 — VERIFY SUBMISSION FILE
─────────────────────────────────
  Open submission.csv and confirm:
    • Exactly 41,778 rows (+ 1 header row)
    • Columns: Index, demand
    • No missing values

STEP 5 — SUBMIT
─────────────────
  1. Go to the competition portal
  2. Upload submission.csv under "Upload File" (prediction)
  3. Upload traffic_demand_prediction.py under "Upload Source Code"
  4. Write any comments in the "Your Answer" section
  5. Click Submit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIPS TO IMPROVE SCORE FURTHER (if you have time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Log-transform target: add np.log1p(y) before training, then
     np.expm1() on predictions. Helps if demand is very skewed.

  2. Hyperparameter tuning with Optuna:
       pip install optuna
     Then run a quick 50-trial search on LightGBM num_leaves,
     learning_rate, and min_child_samples.

  3. Add more interaction features:
       geo5_hour = geo5 + '_h' + hour
       geo4_weather = geo4 + '_w' + Weather

  4. SHAP analysis: identifies which features actually hurt —
     remove those to reduce noise.
       pip install shap

  5. Increase N_FOLDS to 10 for more stable CV estimate.

======================================================================
