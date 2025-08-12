# scripts/calibrate.py
from joblib import load, dump
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from fetch_data import fetch_matches
from features import build_feature_matrix

def main(model_path):
    df = fetch_matches()
    X, y, meta = build_feature_matrix(df)
    # split a small holdout for calibration (time-aware ideally)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, shuffle=False)
    base = load(model_path)
    calib = CalibratedClassifierCV(base, method='isotonic', cv='prefit')
    calib.fit(X_cal, y_cal)
    dump(calib, model_path.replace('.joblib','_calib.joblib'))
    print("Saved calibrated model")
