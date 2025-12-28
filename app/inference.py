# app/inference.py
import joblib
import pandas as pd
import numpy as np
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load {path}: {e}")

# Load artifacts once
# Tries to load RF feature list first, falls back to LR feature list
RF_MODEL = safe_joblib_load(os.path.join(MODELS_DIR, "random_forest_final.joblib"))
RF_THRESHOLD = safe_joblib_load(os.path.join(MODELS_DIR, "random_forest_final_threshold.joblib"))
try:
    RF_FEATURES = safe_joblib_load(os.path.join(MODELS_DIR, "random_forest_feature_list.joblib"))
except Exception:
    RF_FEATURES = None

LR_MODEL = safe_joblib_load(os.path.join(MODELS_DIR, "log_reg_final_class_weighted.joblib"))
try:
    LR_THRESHOLD = safe_joblib_load(os.path.join(MODELS_DIR, "log_reg_final_threshold.joblib"))
except Exception:
    LR_THRESHOLD = 0.6  # fallback if not saved
try:
    LR_FEATURES = safe_joblib_load(os.path.join(MODELS_DIR, "log_reg_feature_list.joblib"))
except Exception:
    LR_FEATURES = None

# If both exist, make sure features match or prefer RF features
if RF_FEATURES is None and LR_FEATURES is not None:
    FEATURES = LR_FEATURES
elif RF_FEATURES is not None:
    FEATURES = RF_FEATURES
else:
    FEATURES = None

# Optionally load a saved scaler if you saved one as models/scaler.joblib
SCALER = None
scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
if os.path.exists(scaler_path):
    try:
        SCALER = joblib.load(scaler_path)
    except Exception:
        SCALER = None

def align_and_prepare(df: pd.DataFrame, features: list = None):
    """
    Ensure df has all model features in correct order.
    If feature list is missing, try using df columns as-is.
    This function will:
     - reorder columns to features
     - add missing columns filled with 0
     - cast numeric columns and fillna
    """
    X = df.copy()
    if features is None:
        # no known feature list, just try best-effort numeric conversion
        X = X.select_dtypes(include=[np.number]).fillna(0)
        return X
    # Ensure we have strings in features
    features = list(features)
    # Add missing columns
    for f in features:
        if f not in X.columns:
            X[f] = 0
    # Reorder
    X = X[features].copy()
    # Convert to numeric - coerce errors to NaN then fill
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    # If a scaler exists, apply it (scaler expects same column order)
    if SCALER is not None:
        try:
            X_scaled = SCALER.transform(X)
            X = pd.DataFrame(X_scaled, columns=features, index=X.index)
        except Exception:
            # If scaler fails, proceed with raw numeric features
            pass
    return X

def predict_both(X: pd.DataFrame):
    """
    Returns a DataFrame with predictions & probabilities for both models.
    """
    X_prepared = align_and_prepare(X, FEATURES)
    # RF predictions
    rf_proba = RF_MODEL.predict_proba(X_prepared)[:, 1]
    rf_pred = (rf_proba >= RF_THRESHOLD).astype(int)
    # LR predictions
    # Align LR features too; if LR uses different features, align separately
    if LR_FEATURES is not None and LR_FEATURES != FEATURES:
        X_prepared_lr = align_and_prepare(X, LR_FEATURES)
    else:
        X_prepared_lr = X_prepared
    lr_proba = LR_MODEL.predict_proba(X_prepared_lr)[:, 1]
    lr_pred = (lr_proba >= LR_THRESHOLD).astype(int)

    out = X.copy()
    out["rf_proba"] = rf_proba
    out["rf_pred"] = rf_pred
    out["lr_proba"] = lr_proba
    out["lr_pred"] = lr_pred
    return out, X_prepared, X_prepared_lr
