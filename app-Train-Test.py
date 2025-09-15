import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Boston Housing — Linear Regression", page_icon="🏠", layout="wide")
st.title("🏠 Boston Housing — Linear Regression")

MODEL_PATH = Path("boston_linear_regression.pkl")
META_PATH = Path("linear_feature_meta.json")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("boston_linear_regression.pkl not found. Train and save the model first.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    if not META_PATH.exists():
        st.error("linear_feature_meta.json not found.")
        st.stop()
    with open(META_PATH, "r") as f:
        return json.load(f)

pipe = load_model()
meta = load_meta()

all_feats = meta["all_features"]
num_feats = meta["numeric_features"]
cat_feats = meta["categorical_features"]
target = meta["target"]

# Sidebar inputs
st.sidebar.header("🔢 Enter Features")
inputs = {}
for col in all_feats:
    if col in num_feats:
        inputs[col] = st.sidebar.number_input(col, value=0.0, step=0.1, format="%.3f")
    else:
        inputs[col] = st.sidebar.text_input(col, value="")

X_user = pd.DataFrame([inputs], columns=all_feats)

left, right = st.columns([1,1])
with left:
    st.subheader("🔮 Single Prediction")
    if st.button("Predict"):
        try:
            pred = pipe.predict(X_user)[0]
            st.success(f"Estimated {target}: **{pred:.2f}**  (MEDV is $1000s)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with right:
    st.subheader("📦 Batch Predictions (CSV)")
    st.caption("CSV must contain exactly these columns (any order):")
    st.code(", ".join(all_feats))
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            missing = [c for c in all_feats if c not in df_in.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df_in = df_in[all_feats]
                preds = pipe.predict(df_in)
                out = df_in.copy()
                out[target + "_PRED"] = preds
                st.dataframe(out.head(20))
                st.download_button(
                    "Download Predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="boston_linear_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
