# app.py
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Boston Housing Prediction", page_icon="🏠", layout="wide")
st.title("🏠 Boston Housing Price Prediction")

MODEL_PATH = Path("boston_best_model.pkl")
META_PATH = Path("feature_columns.json")
CARD_PATH = Path("model_card.md")
RESID_PLOT = Path("residual_plot.png")
PVA_PLOT = Path("pred_vs_actual.png")

# ---------- Load artifacts ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("boston_best_model.pkl not found. Please train and save the model first.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    if not META_PATH.exists():
        st.error("feature_columns.json not found. Please train and save the model first.")
        st.stop()
    with open(META_PATH, "r") as f:
        return json.load(f)

model = load_model()
meta = load_meta()

num_feats = meta.get("numeric_features", [])
cat_feats = meta.get("categorical_features", [])
all_feats = meta.get("all_features", [])
target = meta.get("target", "MEDV")

# ---------- Sidebar: single prediction ----------
st.sidebar.header("🔢 Enter Features")
single_inputs = {}

# sensible defaults if you know typical Boston ranges; here we default to 0.0 or empty string
for col in all_feats:
    if col in num_feats:
        single_inputs[col] = st.sidebar.number_input(col, value=0.0, step=0.1, format="%.3f")
    else:
        single_inputs[col] = st.sidebar.text_input(col, value="")

single_df = pd.DataFrame([single_inputs], columns=all_feats)

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("🔮 Single Prediction")
    if st.button("Predict"):
        try:
            pred = model.predict(single_df)[0]
            st.success(f"Estimated {target}: **{pred:.2f}** (MEDV is $1000s)")
            st.caption("Note: The model pipeline includes imputation & scaling, so missing or unseen categories are handled.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- Batch predictions ----------
with col_right:
    st.subheader("📦 Batch Predictions (CSV)")
    st.caption("Upload a CSV with *exactly* these columns (order doesn’t matter):")
    st.code(", ".join(all_feats), language="text")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            missing_cols = [c for c in all_feats if c not in df_in.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                df_in = df_in[all_feats]  # ensure column order
                preds = model.predict(df_in)
                out = df_in.copy()
                out[target + "_PRED"] = preds
                st.dataframe(out.head(20))
                st.download_button(
                    "Download Predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="boston_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# ---------- Diagnostics ----------
st.subheader("🧪 Diagnostics")
diag_cols = st.columns(2)
if RESID_PLOT.exists():
    with diag_cols[0]:
        st.image(str(RESID_PLOT), caption="Residual Plot", use_container_width=True)
if PVA_PLOT.exists():
    with diag_cols[1]:
        st.image(str(PVA_PLOT), caption="Predicted vs Actual", use_container_width=True)
if not RESID_PLOT.exists() and not PVA_PLOT.exists():
    st.caption("Train with the provided script to generate diagnostic plots automatically.")

# ---------- Model Card ----------
st.subheader("📄 Model Card")
if CARD_PATH.exists():
    with open(CARD_PATH, "r") as f:
        st.markdown(f.read())
else:
    st.caption("model_card.md not found. Train with the script to generate a quick summary.")
