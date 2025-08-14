#!/usr/bin/env python3
"""
scripts/calibrate_model.py

Calibrate probabilities for the latest trained model.

Usage:
  python scripts/calibrate_model.py --meta artifacts/model_meta_20250813_133426.json --test_days 365

Notes:
- This script expects your project to expose the same helper functions used during training:
  fetch_matches(...) and construct_feature_matrix(...).
  If those live inside a module (train.py), we import them. Adjust import paths if necessary.
- It attempts sklearn's CalibratedClassifierCV first; if that fails due to xgboost build issues,
  it falls back to per-class isotonic regression calibration using Booster probabilities.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
import joblib

# Ensure project root is importable (adjust if your repo layout is different)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Try to import your training helpers. Adjust path if your train script is elsewhere.
try:
    # Example path based on earlier conversation: scripts/model_train/train.py
    from scripts.model_train.train import fetch_matches
except Exception:
    # fallback: try top-level import (if your functions are at scripts/train.py)
    try:
        from model_train.train import fetch_matches
    except Exception as e:
        raise ImportError(
            "Could not import fetch_matches / construct_feature_matrix from your training scripts. "
            "Adjust imports in this script to point to where those functions live."
        ) from e

# xgboost imports
import xgboost as xgb
from xgboost import XGBClassifier

def load_meta(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)

def prepare_data_for_calibration(meta_json, test_days):
    # fetch all matches (same as training)
    print("Fetching matches from DB (this may take a moment)...")
    df = fetch_matches(min_date=None)  # reuse training fetch, same table
    print(f"Rows fetched: {len(df):,}")

    # build features using the same function used for training
    X_all, y_all, meta_df = construct_feature_matrix(df, window=5)  # match your training window
    meta_df['date'] = pd.to_datetime(meta_df['date'])

    # train/test split identical to training: last `test_days` used as holdout
    cutoff = meta_df['date'].max() - pd.Timedelta(days=test_days)
    train_mask = meta_df['date'] < cutoff
    test_mask = meta_df['date'] >= cutoff

    X_train_full = X_all.loc[train_mask].copy()
    y_train_full = y_all.loc[train_mask].copy()
    X_test = X_all.loc[test_mask].copy()
    y_test = y_all.loc[test_mask].copy()

    # Now create a time-aware validation split *within training* for calibration:
    # take the last 10% of the training period as calibration validation (time-based).
    train_meta = meta_df.loc[train_mask].copy()
    train_meta_sorted = train_meta.sort_values("date")
    n_val = max(1, int(len(train_meta_sorted) * 0.10))
    val_ids = train_meta_sorted.tail(n_val)['match_id'].values
    val_mask = X_train_full.index.isin(val_ids)

    X_train = X_train_full.loc[~val_mask].copy()
    y_train = y_train_full.loc[~val_mask].copy()
    X_val = X_train_full.loc[val_mask].copy()
    y_val = y_train_full.loc[val_mask].copy()

    print(f"Calibration split: train={len(X_train):,}, val={len(X_val):,}, test(heldout)={len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def calibrate_with_sklearn(base_params, X_train_prep, y_train, X_val_prep, y_val, save_path):
    """
    Train XGBClassifier (no early stopping) then fit CalibratedClassifierCV with cv='prefit'.
    Saves calibrated classifier with joblib.
    """
    # sanitize params: remove keys not accepted by sklearn's fit (if any)
    params = base_params.copy()
    for k in ['use_label_encoder', 'n_jobs', 'verbosity']:
        params.pop(k, None)

    clf = XGBClassifier(**params)
    print("Training base XGBClassifier (no early stopping) for calibration...")
    clf.fit(X_train_prep, y_train)  # should work on builds that fail only on early_stopping keyword

    print("Fitting CalibratedClassifierCV (isotonic)...")
    calib = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    calib.fit(X_val_prep, y_val)

    joblib.dump(calib, save_path)
    print(f"Saved calibrated sklearn classifier to: {save_path}")
    return calib

def calibrate_booster_with_isotonic(booster, X_val_df, y_val, save_prefix):
    """
    booster: xgboost Booster object
    X_val_df: DataFrame with feature names matching metadata['features']
    y_val: Series (0/1/2)
    Returns dict of per-class IsotonicRegression models and saves them with joblib.
    """
    probs_val = booster.predict(xgb.DMatrix(X_val_df, feature_names=X_val_df.columns.tolist()))
    # probs_val shape = (n_samples, n_classes)
    iso_models = {}
    for cls in range(probs_val.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(probs_val[:, cls], (y_val == cls).astype(int))
        iso_models[f'class_{cls}'] = ir

    # Save all isotonic regressors and meta
    out_path = Path(save_prefix).with_suffix('.isotonic.joblib')
    joblib.dump(iso_models, out_path)
    print(f"Saved per-class isotonic regressors to: {out_path}")
    return iso_models, out_path

def main(args):
    meta = load_meta(args.meta)
    print("Loaded meta:", args.meta)

    # Step: recreate X/y and splits
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_calibration(meta, args.test_days)

    # Make sure we select the same raw columns order expected by preprocessor.
    # meta['features'] in your artifact are the numeric feature column names used during training.
    feature_cols = meta['features']
    # Ensure they exist in X_train/X_val/X_test
    for df_name, df in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test)]:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing features in {df_name}: {missing}")

    # Load preprocessor (fitted during training) and transform
    preprocessor = joblib.load(meta['preprocessor_path'])
    print("Loaded preprocessor:", meta['preprocessor_path'])

    # Transform - pass DataFrame with the original columns; ColumnTransformer expects those names
    X_train_df = X_train[feature_cols]
    X_val_df = X_val[feature_cols]
    X_test_df = X_test[feature_cols]

    X_train_prep = preprocessor.transform(X_train_df)
    X_val_prep = preprocessor.transform(X_val_df)
    X_test_prep = preprocessor.transform(X_test_df)

    # Try sklearn calibration route
    calib_save_path = Path("models") / f"xgb_calibrated_{meta.get('created_at','calib')}.joblib"
    try:
        calib = calibrate_with_sklearn(meta['params'], X_train_prep, y_train, X_val_prep, y_val, calib_save_path)
        # Evaluate calibrated model on held-out test
        probs_test = calib.predict_proba(X_test_prep)
        ll = log_loss(y_test, probs_test)
        brier_vals = [brier_score_loss((y_test==cls).astype(int), probs_test[:,cls]) for cls in range(probs_test.shape[1])]
        print(f"Calibrated model test log-loss: {ll:.4f}, brier_mean: {np.mean(brier_vals):.4f}")
        return

    except Exception as e:
        print("sklearn calibration path failed:", e)
        print("Falling back to booster-based calibration (isotonic per class).")

    # Fallback path: use existing booster to get probs on validation and fit isotonic per class
    # Load booster model
    model_path = meta['model_path']
    booster = xgb.Booster()
    print("Loading Booster from:", model_path)
    booster.load_model(model_path)

    # For booster calibration we need DataFrames with named columns (preprocessor produced numeric features)
    # The preprocessor.transform returned numpy arrays; we converted to DataFrame earlier as X_*_df
    # But isotonic calibration needs the predicted probs (before) and the true labels
    # Create DMatrix for val and test
    dval = xgb.DMatrix(X_val_df, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test_df, feature_names=feature_cols)

    iso_models, iso_path = calibrate_booster_with_isotonic(booster, X_val_df, y_val, f"artifacts/iso_{meta.get('created_at','calib')}")
    # Evaluate calibrated probabilities on test
    probs_test_raw = booster.predict(dtest)  # (n_samples, n_classes)
    # apply isotonic mapping per class
    probs_test_cal = np.vstack([iso_models[f'class_{c}'].transform(probs_test_raw[:, c]) for c in range(probs_test_raw.shape[1])]).T
    # re-normalize to sum=1 (clip & renorm)
    probs_test_cal = np.clip(probs_test_cal, 1e-9, 1-1e-9)
    probs_test_cal = probs_test_cal / probs_test_cal.sum(axis=1, keepdims=True)

    ll = log_loss(y_test, probs_test_cal)
    brier_vals = [brier_score_loss((y_test==cls).astype(int), probs_test_cal[:,cls]) for cls in range(probs_test_cal.shape[1])]
    print(f"Booster + isotonic calibration test log-loss: {ll:.4f}, brier_mean: {np.mean(brier_vals):.4f}")

    # Save calibration metadata
    calib_meta = {
        "method": "booster_per_class_isotonic",
        "iso_path": str(iso_path),
        "model_path": model_path,
        "created_at": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "test_log_loss": float(ll),
        "test_brier_mean": float(np.mean(brier_vals))
    }
    meta_out = Path("artifacts") / f"calibration_meta_{meta.get('created_at','calib')}.json"
    with open(meta_out, "w") as f:
        json.dump(calib_meta, f, indent=2)
    print("Saved calibration metadata:", meta_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to model_meta JSON")
    parser.add_argument("--test_days", type=int, default=365, help="Holdout test days used in training (default=365)")
    args = parser.parse_args()
    main(args)
