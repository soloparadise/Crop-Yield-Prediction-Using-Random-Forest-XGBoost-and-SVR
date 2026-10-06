"""Simple Streamlit interface for the trained crop-yield model."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

OUT = Path("outputs")
MODEL_PATH = OUT / "best_yield_model.joblib"
META_PATH = OUT / "model_metadata.json"

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")
st.title("🌾 Crop Yield Prediction")
st.caption("Random Forest · XGBoost · Support Vector Regression")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.warning("No trained model found. First run the training command in README.md.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
features = metadata["feature_columns"]

st.write("Enter values for the features used during training. Leave unknown fields blank.")
uploaded = st.file_uploader("Optional: upload one-row CSV with the same feature columns", type=["csv"])
if uploaded:
    sample = pd.read_csv(uploaded)
    missing = [c for c in features if c not in sample.columns]
    if missing:
        st.error("Missing required feature columns: " + ", ".join(missing))
        st.stop()
    row = sample[features].iloc[[0]].copy()
else:
    # A safe generic form based on the fitted training schema. Categorical fields accept text;
    # numeric fields accept numbers. For best results, use the one-row CSV upload option.
    values = {}
    for feature in features:
        is_numeric = feature.lower() in {
            "rainfall", "temperature", "avg_temperature", "area", "area_hectares",
            "farm_size_hectares", "fertilizer", "fertilizer_usage", "pesticide",
            "pesticide_usage", "soil_moisture", "nitrogen", "phosphorus", "potassium",
            "n", "p", "k", "year"
        } or any(token in feature.lower() for token in ["rain", "temp", "area", "fertiliz", "pestic", "moisture", "nitrogen", "phosphorus", "potassium"])
        if is_numeric:
            values[feature] = st.number_input(feature.replace("_", " ").title(), value=0.0, format="%.4f")
        else:
            values[feature] = st.text_input(feature.replace("_", " ").title(), value="")
    row = pd.DataFrame([values], columns=features)

if st.button("Predict crop yield", type="primary"):
    try:
        prediction = float(model.predict(row)[0])
        st.success(f"Predicted yield: {prediction:,.4f}")
        st.caption("The output is in the same unit as the target column used to train the model.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Upload a one-row CSV with the exact feature columns used during training.")
