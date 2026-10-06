# Crop Yield Prediction using Random Forest, XGBoost and SVR

This implementation follows the model-comparison approach in the supplied abstract and presentation. It predicts a **continuous crop-yield value** and compares Random Forest, XGBoost and Support Vector Regression (SVR) using R², RMSE, MAE and MAPE.

## Important distinction in the supplied documents

The project report describes a separate LightGBM classifier whose target is `risk_category` (Low/Moderate/High/Critical). That is an agriculture-risk classification task, not continuous yield regression. This code implements the yield-regression project described in the title and PPT. It does not claim to reproduce the LightGBM risk results.

## 1. Requirements

- Python 3.9+
- A CSV dataset containing a continuous yield target, or both production and area columns
- Suggested environment: VS Code, Jupyter, or Google Colab

## 2. Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Prepare your dataset

Place your CSV in this folder or note its path. The script attempts to find one of these target columns automatically:

- `yield_ton_per_ha`
- `yield_tons_per_hectare`
- `yield_per_hectare`
- `crop_yield`
- `yield`
- `Yield`

If your target column has a different name, pass it explicitly with `--target`. If no yield target exists but production and area columns are present, the script can derive yield as `production / area` (with area zero treated as missing).

**Avoid target leakage:** do not include columns that would only be known after harvest or that directly encode the target as input features. Inspect the dataset columns and choose the appropriate target before reporting results.

## 4. Train and compare the models

```bash
python train_models.py --data "your_dataset.csv"
```

For a small grid search:

```bash
python train_models.py --data "your_dataset.csv" --tune
```

If the target column is named differently:

```bash
python train_models.py --data "your_dataset.csv" --target "your_yield_column"
```

The script uses an 80:20 train/test split and saves these files to `outputs/`:

- `model_comparison.csv` — R², RMSE, MAE and MAPE for each trained model
- `model_comparison_rmse.png` — RMSE comparison chart
- `best_yield_model.joblib` — best model by test RMSE
- `all_yield_models.joblib` — fitted models
- `model_metadata.json` — target, feature names and split sizes
- `best_model_test_predictions.csv` — held-out actual vs predicted values

The metrics are not known until the code is run on your actual dataset. Do not copy example or expected results into the report.

## 5. Launch the prediction interface

After training:

```bash
streamlit run app.py
```

The app supports a one-row CSV upload with the same input columns used during training. For numeric fields in the manual form, the app uses generic detection; CSV upload is more reliable for datasets with custom feature names.

## 6. Methodology

1. Load the CSV and identify the continuous yield target.
2. Remove rows without a usable target.
3. Split into training and test sets (80:20).
4. Impute numeric missing values using the median.
5. Impute categorical missing values and one-hot encode categorical variables.
6. Scale numeric features for SVR; tree-based models use unscaled numeric values.
7. Train Random Forest, XGBoost and SVR.
8. Compare R², RMSE, MAE and MAPE on the same held-out test split.

## 7. Limitations

- The supplied report names `modern_agriculture_ai_crop_production_2020_2050.csv`, but the CSV itself was not included with the documents used to prepare this code. The exact column names and data quality must be verified when you run it.
- A random split may overestimate generalization when records from the same farm, region or year occur in both sets. Consider region-aware or time-aware validation for a final evaluation.
- MAPE can be misleading when actual yields are zero or close to zero; the implementation excludes near-zero actual values from the MAPE calculation.
- Feature importance is not generated for SVR. Add permutation importance or SHAP if interpretability is required.

## 8. Project title

**Crop Yield Prediction Using Random Forest, XGBoost, and Support Vector Regression Techniques**
