# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# added RF_THRESHOLD, LR_THRESHOLD import (safe, required to show thresholds)
from inference import predict_both, align_and_prepare, RF_MODEL, LR_MODEL, RF_THRESHOLD, LR_THRESHOLD
from explain import lr_contributions, rf_shap_explanation, rf_importance_explanation
from utils import save_uploaded_file

ROOT = Path(__file__).resolve().parents[1]
CONCLUSION_PATH = ROOT / "Flight_Delay_Prediction_Final_Conclusion.md"

st.set_page_config(page_title="Flight Delay Predictor — Demo", layout="wide")
st.title("✈️ Flight Delay Predictor — Demo")
st.markdown("Upload a CSV/JSON with model-ready features (features aligned to the saved model). The app will predict with both Logistic Regression and Random Forest and provide explanations.")

# Sidebar
st.sidebar.header("Inputs & Options")

uploaded = st.sidebar.file_uploader("Upload CSV/JSON (one or many rows)", type=["csv", "json"])
if uploaded is not None:
    data_path, df_uploaded = save_uploaded_file(uploaded)
    st.sidebar.success(f"Saved uploaded file to `{data_path}`")
    st.sidebar.write(f"Rows: {df_uploaded.shape[0]} columns: {df_uploaded.shape[1]}")
else:
    st.sidebar.info("Or generate a sample row below to test the app.")
    df_uploaded = None

st.sidebar.markdown("---")
show_report = st.sidebar.checkbox("Show Full Project Report (conclusion)", value=False)

# Sample quick input generator (user-friendly) only when no upload present
if df_uploaded is None:
    st.subheader("Quick single-row input (for demo)")
    c1, c2, c3 = st.columns(3)
    dep_hour = c1.slider("dep_hour", 0, 23, 10)
    precip_in = c1.number_input("precip_in", 0.0, 10.0, 0.0, step=0.01)
    avg_wind_speed_kts = c2.number_input("avg_wind_speed_kts", 0.0, 60.0, 5.0, step=0.1)
    DISTANCE = c2.number_input("DISTANCE", 50, 5000, 800)
    has_turnaround = c3.selectbox("has_turnaround", ["Yes", "No"])
    # Build a minimal DataFrame with common columns - rest will be filled with zeros by align_and_prepare
    df_uploaded = pd.DataFrame([{
        "dep_hour": dep_hour,
        "precip_in": precip_in,
        "avg_wind_speed_kts": avg_wind_speed_kts,
        "DISTANCE": DISTANCE,
        "has_turnaround": 1 if has_turnaround == "Yes" else 0
    }])
    st.write("Preview of the input row:")
    st.dataframe(df_uploaded.T)

# Predict
if st.button("Run Prediction (both models)"):
    with st.spinner("Preparing input and running models..."):
        results_df, X_prepared_rf, X_prepared_lr = predict_both(df_uploaded)
    st.success("Done — results below")

    # Show aggregated results
    st.subheader("Predictions (per-row)")
    display_df = results_df.copy()
    # Nicely format probabilities
    display_df["rf_proba"] = display_df["rf_proba"].round(3)
    display_df["lr_proba"] = display_df["lr_proba"].round(3)
    st.dataframe(display_df)

    # Save predictions to disk
    out_path = ROOT / "data" / "input" / "predictions.csv"
    display_df.to_csv(out_path, index=False)
    st.info(f"Predictions saved to `{out_path}`")

    # Per-row detailed explanation UI
    st.subheader("Detailed Explanation for a selected row")
    idx = st.number_input("Row index (0-based)", min_value=0, max_value=max(0, len(display_df)-1), value=0)
    # LR explanation
    st.markdown("#### Logistic Regression — Contributions (top positive & negative)")
    top_pos, top_neg, full_contrib = lr_contributions(LR_MODEL, X_prepared_lr.iloc[idx], feature_names=X_prepared_lr.columns.tolist(), top_n=8)
    st.write("Top positive contributions (increase delay probability):")
    st.table(top_pos[["feature", "value", "coef", "contribution"]].reset_index(drop=True))
    st.write("Top negative contributions (decrease delay probability):")
    st.table(top_neg[["feature", "value", "coef", "contribution"]].reset_index(drop=True))

    # RF explanation (SHAP preferred)
    st.markdown("#### Random Forest — Explanation (SHAP if available, else feature importances)")
    shap_df = rf_shap_explanation(RF_MODEL, X_prepared_rf, row_idx=idx, top_n=12)
    if shap_df is not None:
        st.write("Top SHAP contributions (magnitude):")
        st.table(shap_df.reset_index(drop=True))
    else:
        st.write("SHAP not available or failed — falling back to global feature importances.")
        fi_df = rf_importance_explanation(RF_MODEL, X_prepared_rf, top_n=12)
        st.table(fi_df.reset_index(drop=True))

    # Combined textual report (auto-generated)
    st.subheader("Auto-generated Decision Report (per-row)")
    prob_rf = float(display_df.loc[idx, "rf_proba"])
    prob_lr = float(display_df.loc[idx, "lr_proba"])
    pred_rf = int(display_df.loc[idx, "rf_pred"])
    pred_lr = int(display_df.loc[idx, "lr_pred"])

    report_lines = []
    report_lines.append(f"**Row index:** {idx}")
    # show thresholds correctly
    try:
        rf_thresh_disp = float(RF_THRESHOLD)
    except Exception:
        rf_thresh_disp = None
    try:
        lr_thresh_disp = float(LR_THRESHOLD)
    except Exception:
        lr_thresh_disp = None

    if rf_thresh_disp is not None:
        report_lines.append(f"**Random Forest**: probability={prob_rf:.3f}, prediction={pred_rf} (threshold={rf_thresh_disp})")
    else:
        report_lines.append(f"**Random Forest**: probability={prob_rf:.3f}, prediction={pred_rf}")

    if lr_thresh_disp is not None:
        report_lines.append(f"**Logistic Regression**: probability={prob_lr:.3f}, prediction={pred_lr} (threshold={lr_thresh_disp})")
    else:
        report_lines.append(f"**Logistic Regression**: probability={prob_lr:.3f}, prediction={pred_lr}")

    report_lines.append("\n**Top LR positive contributors**:")
    for _, r in top_pos.head(6).iterrows():
        report_lines.append(f"- {r['feature']}: value={r['value']}, coef={r['coef']:.3f}, contribution={r['contribution']:.3f}")
    report_lines.append("\n**Top LR negative contributors**:")
    for _, r in top_neg.head(6).iterrows():
        report_lines.append(f"- {r['feature']}: value={r['value']}, coef={r['coef']:.3f}, contribution={r['contribution']:.3f}")
    if shap_df is not None:
        report_lines.append("\n**Top RF SHAP features:**")
        for _, r in shap_df.head(6).iterrows():
            report_lines.append(f"- {r['feature']}: shap_value={r['shap_value']:.3f}, feature_value={r['feature_value']}")
    else:
        report_lines.append("\n**Top RF global importances:**")
        for _, r in fi_df.head(6).iterrows():
            report_lines.append(f"- {r['feature']}: importance={r['importance']:.4f}")

    st.markdown("\n".join(report_lines))

    # -------------------------
    # Final short verdicts (inserted here; uses existing computed variables)
    # -------------------------
    st.markdown("---")
    st.subheader("Final Verdicts (concise)")

    # logistic regression short verdict
    lr_label = "Delay" if pred_lr == 1 else "On time"
    lr_top_feats = top_pos.head(3)["feature"].tolist() if not top_pos.empty else []
    lr_neg_feats = top_neg.head(3)["feature"].tolist() if not top_neg.empty else []
    lr_support = []
    if lr_top_feats:
        lr_support.append("↑ " + ", ".join(lr_top_feats))
    if lr_neg_feats:
        lr_support.append("↓ " + ", ".join(lr_neg_feats))
    lr_support_text = "; ".join(lr_support) if lr_support else "No strong contributors identified."

    st.markdown(
        f"**Logistic Regression — {lr_label}**  \n"
        f"Probability: **{prob_lr:.3f}**  \n"
        f"Support: {lr_support_text}"
    )

    # random forest short verdict
    rf_label = "Delay" if pred_rf == 1 else "On time"
    if shap_df is not None:
        rf_top_feats = shap_df.head(3)["feature"].tolist()
    elif 'fi_df' in locals():
        rf_top_feats = fi_df.head(3)["feature"].tolist()
    else:
        rf_top_feats = []
    rf_support_text = ", ".join(rf_top_feats) if rf_top_feats else "No top features available."

    st.markdown(
        f"**Random Forest — {rf_label}**  \n"
        f"Probability: **{prob_rf:.3f}**  \n"
        f"Support: top features: {rf_support_text}"
    )

    # --- Overall recommendation (simple, explainable) ---
    # Use "normalized confidence" = proba / threshold to compare how strongly each model crosses its own decision boundary
    try:
        lr_norm = prob_lr / (LR_THRESHOLD if LR_THRESHOLD is not None else 0.6)
    except Exception:
        lr_norm = prob_lr / 0.6
    try:
        rf_norm = prob_rf / (RF_THRESHOLD if RF_THRESHOLD is not None else 0.30)
    except Exception:
        rf_norm = prob_rf / 0.30

    if (pred_lr == pred_rf):
        overall_text = f"Both models agree on **{('Delay' if pred_rf==1 else 'On time')}**. Recommend following this outcome."
    else:
        # prefer the model with higher normalized confidence, but if close, prefer LR for interpretability
        diff = rf_norm - lr_norm
        if abs(diff) < 0.15:
            overall_text = (
                "Models disagree, but confidence levels are similar — prefer **Logistic Regression** for a "
                "more interpretable explanation to operations staff."
            )
        elif diff > 0:
            overall_text = (
                "Models disagree. Random Forest has higher relative confidence — recommend trusting **Random Forest** for this case."
            )
        else:
            overall_text = (
                "Models disagree. Logistic Regression has higher relative confidence — recommend trusting **Logistic Regression** for this case."
            )

    st.markdown("**Overall recommendation:** " + overall_text)

# Show full project conclusion if asked
if show_report:
    st.markdown("---")
    st.header("Project Conclusion")
    if CONCLUSION_PATH.exists():
        text = CONCLUSION_PATH.read_text(encoding="utf-8")
        st.markdown(text)
    else:
        st.info("Final conclusion file not found.")
