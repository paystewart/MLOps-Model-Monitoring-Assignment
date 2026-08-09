# Assignment #4: Model Monitoring

**Payton Stewart**

Regression model (`RandomForestRegressor`) predicting county-level cancer mortality
(`TARGET_deathRate`) from `cancer_reg.csv`, with **Evidently AI** monitoring for input data
drift and prediction/performance drift across three controlled input-shift scenarios.

## Repository layout

```
.
├── ADSP31021_Assignment4_ModelMonitoring.ipynb   # complete, already-executed workflow
├── data/
│   └── cancer_reg.csv                            # provided dataset
├── outputs/
│   ├── plots/
│   │   ├── baseline_eval_plots.png               # predicted-vs-actual + residuals
│   │   └── scenario_metrics_comparison.png        # RMSE/MAE/R2 across scenarios
│   └── evidently_reports/
│       ├── 00_baseline_original_test_vs_reference.html
│       ├── scenario_a_vs_reference.html
│       ├── scenario_aplusb_vs_reference.html
│       └── scenario_aplusbplusc_vs_reference.html
├── evidently_workspace/                          # local Evidently project (all 4 runs logged)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebook

```bash
jupyter notebook ADSP31021_Assignment4_ModelMonitoring.ipynb
```

or, to re-run it end-to-end from the command line with no manual steps:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    ADSP31021_Assignment4_ModelMonitoring.ipynb --ExecutePreprocessor.timeout=500
```

The notebook reads `data/cancer_reg.csv`, trains the model, evaluates all four datasets
(original test set + Scenarios A, A+B, A+B+C), and regenerates every file under `outputs/`
and `evidently_workspace/`.

## Browsing the Evidently monitoring UI (optional)

The individual HTML reports in `outputs/evidently_reports/` can be opened directly in a
browser — no server needed. To browse all logged runs in the interactive self-hosted
Evidently UI instead, run from the project root:

```bash
evidently ui --workspace ./evidently_workspace --port 8000
```

then open `http://localhost:8000`.

## Key decisions (see the notebook for full detail)

- **Target:** `TARGET_deathRate`. **Model:** `RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)`.
- **Split:** 80/20 train/test, `random_state=42`.
- **Excluded columns:** `Geography` (identifier), `binnedInc` (redundant with `medIncome`),
  `PctSomeCol18_24` (~75% missing), `avgAnnCount` and `avgDeathsPerYear` (near-deterministic
  drivers of the target — dropped to avoid leakage).
- **Missing values:** `MedianAge` values above 100 are recoded as data-entry errors; all
  remaining missing values are imputed with the **training-set** median.
- **Monitoring scenarios** (each applied cumulatively to a fresh copy of the untouched test set):
  - Scenario A: `medIncome` − 40,000
  - Scenario A+B: + `povertyPercent` + 20
  - Scenario A+B+C: + `AvgHouseholdSize` + 2
- **Evidently checks:** `DataSummaryPreset` + `DataDriftPreset` (all input features, plus the
  model's `prediction` column so prediction drift is measured directly) and `RegressionPreset`
  (RMSE/MAE/R²/mean error), reference = training data + training predictions.
