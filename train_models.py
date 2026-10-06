"""
Crop Yield Prediction using Random Forest, XGBoost and Support Vector Regression.
Usage:
    python train_models.py --data path/to/dataset.csv
    python train_models.py --data dataset.csv --target yield_ton_per_ha
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def choose_target(df: pd.DataFrame, requested: str | None) -> tuple[pd.DataFrame, str]:
    """Use an explicitly requested target, otherwise common yield columns or production/area."""
    if requested:
        if requested in df.columns:
            return df, requested
        raise ValueError(f"Target column '{requested}' was not found. Available columns: {list(df.columns)}")

    candidates = [
        "yield_ton_per_ha", "yield_tons_per_hectare", "yield_per_hectare",
        "crop_yield", "yield", "Yield", "Yield_ton_per_ha"
    ]
    for col in candidates:
        if col in df.columns:
            return df, col

    production_col = next((c for c in ["production", "Production", "production_tons"] if c in df.columns), None)
    area_col = next((c for c in ["area", "Area", "farm_size_hectares", "area_hectares"] if c in df.columns), None)
    if production_col and area_col:
        area = pd.to_numeric(df[area_col], errors="coerce").replace(0, np.nan)
        production = pd.to_numeric(df[production_col], errors="coerce")
        df = df.copy()
        df["__derived_yield_ton_per_ha__"] = production / area
        return df, "__derived_yield_ton_per_ha__"

    raise ValueError(
        "Could not identify the yield target. Specify --target COLUMN. "
        "If the data has production and area columns, yield can be derived as production / area."
    )

def mape_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def main():
    parser = argparse.ArgumentParser(description="Train and compare crop yield regression models.")
    parser.add_argument("--data", required=True, help="Path to the agricultural CSV dataset")
    parser.add_argument("--target", default=None, help="Continuous yield target column")
    parser.add_argument("--tune", action="store_true", help="Run a small GridSearchCV for each model (slower)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    df, target = choose_target(df, args.target)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()

    # Drop identifiers and columns that directly encode the target or can cause target leakage.
    leakage_or_id = {
        "record_id", "id", "recommended_action", "risk_category",
        "yield", "Yield", "crop_yield", "yield_ton_per_ha",
        "yield_tons_per_hectare", "yield_per_hectare", "__derived_yield_ton_per_ha__"
    }
    # Keep the chosen target out of X, but retain other legitimate input features.
    drop_cols = [c for c in leakage_or_id if c in df.columns and c != target]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")
    y = df[target].astype(float)

    # Remove entirely empty columns and columns with a single distinct value.
    X = X.loc[:, X.notna().any(axis=0)]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] == 0:
        raise ValueError("No usable input features remain after cleaning.")

    # Numeric/categorical preprocessing is fitted only on training data via Pipeline.
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    numeric_unscaled = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    def preprocessor(scale_numeric=True):
        num_pipe = numeric_scaled if scale_numeric else numeric_unscaled
        return ColumnTransformer([
            ("numeric", num_pipe, numeric_features),
            ("categorical", categorical, categorical_features)
        ], remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    models = {
        "Random Forest": Pipeline([
            ("preprocess", preprocessor(False)),
            ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                            n_jobs=-1, min_samples_leaf=1))
        ]),
        "XGBoost": None,
        "SVR": Pipeline([
            ("preprocess", preprocessor(True)),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1))
        ])
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = Pipeline([
            ("preprocess", preprocessor(False)),
            ("model", XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                   subsample=0.9, colsample_bytree=0.9,
                                   objective="reg:squarederror", random_state=RANDOM_STATE,
                                   n_jobs=-1))
        ])
    except ImportError:
        print("xgboost is not installed. Run: pip install xgboost")
        print("Continuing with Random Forest and SVR.")

    results = []
    fitted = {}
    for name, pipe in models.items():
        if pipe is None:
            continue
        print(f"\nTraining {name}...")
        if args.tune:
            grid = {
                "Random Forest": {"model__n_estimators": [200, 300], "model__max_depth": [None, 20]},
                "XGBoost": {"model__n_estimators": [200, 400], "model__max_depth": [4, 6]},
                "SVR": {"model__C": [1, 10, 50], "model__epsilon": [0.1, 0.2]}
            }[name]
            pipe = GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        row = {
            "Model": name,
            "R2": float(r2_score(y_test, pred)),
            "RMSE": rmse,
            "MAE": float(mean_absolute_error(y_test, pred)),
            "MAPE_percent": mape_percent(y_test, pred)
        }
        results.append(row)
        fitted[name] = pipe
        print(f"  R²={row['R2']:.4f} | RMSE={rmse:.4f} | MAE={row['MAE']:.4f} | MAPE={row['MAPE_percent']:.2f}%")
        if args.tune and hasattr(pipe, "best_params_"):
            print("  Best parameters:", pipe.best_params_)

    if not results:
        raise RuntimeError("No models were trained. Check your dependencies.")

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(OUT / "model_comparison.csv", index=False)
    best_name = str(results_df.iloc[0]["Model"])
    joblib.dump(fitted[best_name], OUT / "best_yield_model.joblib")
    joblib.dump(fitted, OUT / "all_yield_models.joblib")
    with open(OUT / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"target_column": target, "feature_columns": list(X.columns),
                   "best_model_by_rmse": best_name, "training_rows": len(X_train),
                   "test_rows": len(X_test)}, f, indent=2)

    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["RMSE"])
    plt.ylabel("RMSE (lower is better)")
    plt.title("Crop Yield Model Comparison")
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison_rmse.png", dpi=160)
    plt.close()

    # Save actual vs predicted values for reproducible result reporting.
    best_pipe = fitted[best_name]
    best_pred = best_pipe.predict(X_test)
    pd.DataFrame({"actual_yield": y_test.to_numpy(),
                  "predicted_yield": best_pred}).to_csv(OUT / "best_model_test_predictions.csv", index=False)

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model by test RMSE: {best_name}")
    print(f"Outputs saved in: {OUT.resolve()}")
    print("Note: metrics are calculated from your dataset; do not report them before running this script.")

if __name__ == "__main__":
    main()
