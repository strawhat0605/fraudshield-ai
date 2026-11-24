"""Streamlit demo for the trained FraudShield model."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "fraudshield_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudShield AI")
st.caption("Educational machine-learning demo • Synthetic data is not real banking data")

st.warning(
    "This is a learning prototype, not a production fraud detector. "
    "Do not enter real customer or payment information."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.info(
        "No trained model was found. From the project root, run:\n\n"
        "1. `python scripts/generate_demo_data.py`\n"
        "2. `python -m fraudshield.train --data data/processed/demo_transactions.csv`"
    )
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
features = metadata["feature_columns"]

st.subheader("Transaction risk scoring")
st.write(f"Loaded model: **{metadata['model_name'].replace('_', ' ').title()}**")

with st.form("transaction_form"):
    values = {}
    cols = st.columns(3)
    for index, feature in enumerate(features):
        with cols[index % 3]:
            default = 100.0 if feature == "Amount" else 0.0
            values[feature] = st.number_input(feature, value=float(default), format="%.5f")
    submitted = st.form_submit_button("Assess transaction")

if submitted:
    row = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(row)[0, 1])
    threshold = float(metadata.get("threshold", 0.5))
    flagged = probability >= threshold
    left, right = st.columns(2)
    left.metric("Estimated fraud probability", f"{probability:.1%}")
    right.metric("Decision at configured threshold", "Flag for review" if flagged else "Not flagged")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        f"Threshold: {threshold:.2f}. A score is a model estimate, not proof of fraud. "
        "Human review and domain-specific validation are required."
    )

with st.expander("Model evaluation (held-out test set)"):
    metrics = metadata.get("test_metrics", {})
    display = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    st.json(display)
    cm = metrics.get("confusion_matrix")
    if cm:
        st.write("Confusion matrix (rows = actual, columns = predicted)")
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual legitimate", "Actual fraud"],
                columns=["Predicted legitimate", "Predicted fraud"],
            )
        )
