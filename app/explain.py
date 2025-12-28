# app/explain.py
import numpy as np
import pandas as pd

def lr_contributions(lr_model, X_row: pd.Series, feature_names=None, top_n=8):
    """
    Returns top positive and negative contributions for a single row.
    lr_model: trained sklearn logistic regression
    X_row: Series (feature-aligned, scaled if model expects)
    """
    coefs = lr_model.coef_[0]  # shape (n_features,)
    if feature_names is None:
        feature_names = X_row.index.tolist()
    vals = X_row.values.astype(float)
    contrib = coefs * vals
    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "value": vals,
        "coef": coefs,
        "contribution": contrib
    })
    contrib_df = contrib_df.sort_values("contribution", ascending=False)
    top_pos = contrib_df.head(top_n)
    top_neg = contrib_df.tail(top_n).sort_values("contribution")
    return top_pos, top_neg, contrib_df

def rf_shap_explanation(rf_model, X_prepared, row_idx=0, top_n=10):
    """
    Try to produce SHAP explanations for a specific row.
    If shap is not available or errors, return None.
    """
    try:
        import shap
        # TreeExplainer for tree models
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_prepared)
        # shap_values[1] is for class 1
        shap_row = shap_values[1][row_idx]
        features = X_prepared.columns.tolist()
        shap_df = pd.DataFrame({
            "feature": features,
            "shap_value": shap_row,
            "feature_value": X_prepared.iloc[row_idx].values
        })
        shap_df = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index)
        return shap_df.head(top_n)
    except Exception as e:
        # fallback is None, caller should handle
        return None

def rf_importance_explanation(rf_model, X_prepared, top_n=10):
    """
    Return top feature importances (global) as a DataFrame.
    """
    importances = rf_model.feature_importances_
    feat = X_prepared.columns.tolist()
    df = pd.DataFrame({"feature": feat, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    return df
