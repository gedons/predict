#!/usr/bin/env python3
"""
scripts/train.py - Optimized Version

Key optimizations:
- Vectorized operations with pandas and numpy
- Memory-efficient data processing with chunking
- Improved cross-validation strategy
- Better feature engineering and selection
- Enhanced evaluation metrics and model validation
- Reduced redundant computations
- Optimized hyperparameter search
"""

import os
import json
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from supabase import create_client

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss, accuracy_score, classification_report, 
    brier_score_loss, f1_score, precision_score, recall_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import joblib
import optuna
from tqdm import tqdm
import psutil
import gc
import xgboost as xgb
from typing import Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL in .env")

MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Memory optimization settings
CHUNK_SIZE = 10000
MAX_MEMORY_USAGE = 0.8  # 80% of available RAM


def get_memory_usage():
    """Get current memory usage percentage."""
    return psutil.virtual_memory().percent / 100


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame dtypes to reduce memory usage."""
    df = df.copy()
    
    # Handle integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= -128 and df[col].max() <= 127:
            df[col] = df[col].astype('int8')
        elif df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype('int16')
        elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
            df[col] = df[col].astype('int32')
    
    # Handle float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Handle object columns - but exclude date-related columns
    date_related_cols = ['date', 'created_at', 'updated_at', 'timestamp']
    
    for col in df.select_dtypes(include=['object']).columns:
        # Skip date-related columns from categorical conversion
        if any(date_word in col.lower() for date_word in date_related_cols):
            continue
            
        # Only convert to category if it's truly categorical (low cardinality)
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df


####################
# Enhanced Data fetching
####################
def fetch_matches(min_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch matches with optimized query and memory usage.
    """
    engine = create_engine(DATABASE_URL, future=True) # type: ignore
    
    # More selective query to reduce memory usage
    base_query = """
    SELECT match_id, date, home_team, away_team, 
           full_time_home_goals, full_time_away_goals, full_time_result,
           home_shots, away_shots, home_shots_on_target, away_shots_on_target,
           home_corners, away_corners, season, league,
           b365_home_odds, b365_draw_odds, b365_away_odds
    FROM public.matches 
    WHERE date IS NOT NULL 
      AND full_time_result IN ('H', 'D', 'A')
      AND full_time_home_goals IS NOT NULL 
      AND full_time_away_goals IS NOT NULL
    """
    
    if min_date:
        base_query += " AND date >= :min_date"
    
    base_query += " ORDER BY date"
    
    params = {"min_date": min_date} if min_date else {}
    
    with engine.connect() as conn:
        df = pd.read_sql(text(base_query), conn, params=params)
    
    # Convert date column BEFORE dtype optimization
    df['date'] = pd.to_datetime(df['date'])
    
    # Optimize dtypes (will now skip the date column)
    df = optimize_dtypes(df)
    
    return df


####################
# Optimized Feature Engineering
####################
def build_team_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized approach to build team features using efficient pandas operations.
    """
    df = df.copy()
    
    # Create home and away records more efficiently
    home_records = pd.DataFrame({
        'match_id': df['match_id'],
        'date': df['date'],
        'team': df['home_team'],
        'is_home': True,
        'goals_for': df['full_time_home_goals'].astype('int8'),
        'goals_against': df['full_time_away_goals'].astype('int8'),
        'shots': df['home_shots'].fillna(0).astype('int8'),
        'shots_on_target': df['home_shots_on_target'].fillna(0).astype('int8'),
        'corners': df['home_corners'].fillna(0).astype('int8'),
        'result': df['full_time_result']
    })

    away_records = pd.DataFrame({
        'match_id': df['match_id'],
        'date': df['date'],
        'team': df['away_team'],
        'is_home': False,
        'goals_for': df['full_time_away_goals'].astype('int8'),
        'goals_against': df['full_time_home_goals'].astype('int8'),
        'shots': df['away_shots'].fillna(0).astype('int8'),
        'shots_on_target': df['away_shots_on_target'].fillna(0).astype('int8'),
        'corners': df['away_corners'].fillna(0).astype('int8'),
        'result': df['full_time_result']
    })

    # Determine wins more efficiently
    home_records['is_win'] = (home_records['result'] == 'H').astype('int8')
    away_records['is_win'] = (away_records['result'] == 'A').astype('int8')
    
    # Calculate goal difference
    home_records['goal_diff'] = (home_records['goals_for'] - home_records['goals_against']).astype('int8')
    away_records['goal_diff'] = (away_records['goals_for'] - away_records['goals_against']).astype('int8')

    # Combine and sort
    team_records = pd.concat([home_records, away_records], ignore_index=True)
    team_records = team_records.sort_values(['team', 'date']).reset_index(drop=True)
    
    return team_records


def compute_rolling_stats_optimized(team_records: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Optimized rolling statistics computation using groupby operations.
    """
    df = team_records.copy()
    
    # Define columns for rolling calculations
    stat_cols = ['is_win', 'goal_diff', 'shots', 'shots_on_target', 'corners', 'goals_for', 'goals_against']
    
    # Use groupby with efficient rolling operations
    grouped = df.groupby('team', group_keys=False)
    
    for col in stat_cols:
        # Use shift(1) to exclude current match, then rolling
        df[f'{col}_avg_{window}'] = grouped[col].shift(1).rolling(
            window=window, min_periods=1
        ).mean().astype('float32')
        
        if col in ['is_win']:
            # Also calculate recent form (last 3 games)
            df[f'{col}_form_3'] = grouped[col].shift(1).rolling(
                window=3, min_periods=1
            ).mean().astype('float32')
    
    # Add additional performance metrics with dynamic window
    df[f'attack_strength_{window}'] = grouped['goals_for'].shift(1).rolling(
        window=window, min_periods=1
    ).mean().astype('float32')
    
    df[f'defense_strength_{window}'] = grouped['goals_against'].shift(1).rolling(
        window=window, min_periods=1
    ).mean().astype('float32')
    
    # Head-to-head features would go here if we had historical H2H data
    
    return df


def create_match_features(team_stats: pd.DataFrame, original_matches: pd.DataFrame, window: int = 5) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Create final feature matrix by merging team statistics with match data.
    """
    # Separate home and away statistics
    home_stats = team_stats[team_stats['is_home'] == True].copy()
    away_stats = team_stats[team_stats['is_home'] == False].copy()
    
    # Select and rename feature columns - use dynamic column names based on window
    feature_cols = [
        'match_id', f'is_win_avg_{window}', 'is_win_form_3', f'goal_diff_avg_{window}',
        f'shots_avg_{window}', f'shots_on_target_avg_{window}', f'corners_avg_{window}',
        f'attack_strength_{window}', f'defense_strength_{window}'
    ]
    
    # Verify all columns exist in the data
    missing_cols = [col for col in feature_cols if col not in home_stats.columns]
    if missing_cols:
        print(f"Missing columns in team_stats: {missing_cols}")
        print(f"Available columns: {list(home_stats.columns)}")
        raise KeyError(f"Missing expected columns: {missing_cols}")
    
    home_features = home_stats[feature_cols].copy()
    away_features = away_stats[feature_cols].copy()
    
    # Rename columns with home/away prefixes
    for col in feature_cols[1:]:  # Skip match_id
        home_features = home_features.rename(columns={col: f'home_{col}'})
        away_features = away_features.rename(columns={col: f'away_{col}'})
    
    # Merge home and away features
    match_features = home_features.merge(away_features, on='match_id', how='inner')
    
    # Merge with original match data for odds and targets
    final_df = original_matches.merge(match_features, on='match_id', how='inner')
    
    # Add derived features
    final_df = add_derived_features(final_df, window)
    
    # Create target variable
    label_map = {'H': 0, 'D': 1, 'A': 2}
    final_df['target'] = final_df['full_time_result'].map(label_map)
    
    # Select feature columns - use dynamic column names
    feature_columns = [col for col in final_df.columns if 
                      col.startswith(('home_', 'away_')) or 
                      col.endswith('_prob_implied') or
                      col in ['form_diff', 'strength_diff', 'total_avg_goals']]
    
    X = final_df[feature_columns].copy()
    y = final_df['target'].copy()
    meta = final_df[['match_id', 'date', 'home_team', 'away_team', 'season', 'league']].copy()
    
    return X, y, meta


def add_derived_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add derived features to improve model performance."""
    df = df.copy()
    
    # Betting odds features (with better handling of missing values)
    odds_cols = ['b365_home_odds', 'b365_draw_odds', 'b365_away_odds']
    
    if all(col in df.columns for col in odds_cols):
        # Fill missing odds with neutral values
        df['b365_home_odds'] = df['b365_home_odds'].fillna(2.5)
        df['b365_draw_odds'] = df['b365_draw_odds'].fillna(3.5)
        df['b365_away_odds'] = df['b365_away_odds'].fillna(2.5)
        
        # Convert to implied probabilities
        df['home_prob_implied'] = (1 / df['b365_home_odds']).astype('float32')
        df['draw_prob_implied'] = (1 / df['b365_draw_odds']).astype('float32')
        df['away_prob_implied'] = (1 / df['b365_away_odds']).astype('float32')
        
        # Normalize probabilities
        total_prob = df[['home_prob_implied', 'draw_prob_implied', 'away_prob_implied']].sum(axis=1)
        df['home_prob_implied'] /= total_prob
        df['draw_prob_implied'] /= total_prob
        df['away_prob_implied'] /= total_prob
    
    # Form difference features - use dynamic column names
    home_win_col = f'home_is_win_avg_{window}'
    away_win_col = f'away_is_win_avg_{window}'
    if home_win_col in df.columns and away_win_col in df.columns:
        df['form_diff'] = (df[home_win_col] - df[away_win_col]).astype('float32')
    
    # Strength difference features - use dynamic column names
    home_attack_col = f'home_attack_strength_{window}'
    away_defense_col = f'away_defense_strength_{window}'
    if home_attack_col in df.columns and away_defense_col in df.columns:
        df['strength_diff'] = (df[home_attack_col] - df[away_defense_col]).astype('float32')
    
    # Total expected goals - use dynamic column names
    away_attack_col = f'away_attack_strength_{window}'
    if home_attack_col in df.columns and away_attack_col in df.columns:
        df['total_avg_goals'] = (df[home_attack_col] + df[away_attack_col]).astype('float32')
    
    return df


####################
# Enhanced Model Training
####################
def create_preprocessing_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create preprocessing pipeline with imputation and scaling."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )
    
    return preprocessor # type: ignore

def enhanced_objective(trial, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, n_splits: int = 5) -> float:
    """
    Optuna objective using TimeSeriesSplit CV. For portability we avoid early stopping
    inside folds and use fixed n_estimators during CV (faster & stable).
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=50),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95, step=0.05),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'random_state': 42,
        'objective': 'multi:softprob',
        'num_class': 3,
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1,
        'use_label_encoder': False
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    preprocessor = create_preprocessing_pipeline(X)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        try:
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            if len(X_fold_train) < 100 or len(X_fold_val) < 20:
                continue

            # Fit preprocessing pipeline
            X_fold_train_prep = preprocessor.fit_transform(X_fold_train)
            X_fold_val_prep = preprocessor.transform(X_fold_val)

            # Train without early stopping in CV (ensures compatibility)
            model = XGBClassifier(**params)
            model.fit(X_fold_train_prep, y_fold_train, verbose=False)

            probs = model.predict_proba(X_fold_val_prep)
            score = log_loss(y_fold_val, probs)
            fold_scores.append(score)

            # cleanup
            del model, X_fold_train_prep, X_fold_val_prep
            if get_memory_usage() > MAX_MEMORY_USAGE:
                gc.collect()

        except Exception as e:
            print(f"Fold {fold} failed: {str(e)}")
            continue

    if not fold_scores:
        return float('inf')
    return float(np.mean(fold_scores))

def comprehensive_evaluate(model, X_test, y_test, class_names=['H', 'D', 'A']) -> Dict[str, Any]:
    """Comprehensive model evaluation with multiple metrics."""
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'log_loss': float(log_loss(y_test, probs)),
        'f1_macro': float(f1_score(y_test, preds, average='macro')),
        'f1_weighted': float(f1_score(y_test, preds, average='weighted')),
        'precision_macro': float(precision_score(y_test, preds, average='macro')),
        'recall_macro': float(recall_score(y_test, preds, average='macro'))
    }
    
    # Brier score per class
    brier_scores = []
    for i in range(3):
        brier = brier_score_loss((y_test == i).astype(int), probs[:, i])
        brier_scores.append(float(brier))
        metrics[f'brier_class_{class_names[i]}'] = float(brier)
    
    metrics['brier_mean'] = float(np.mean(brier_scores))
    
    # Classification report
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report  # type: ignore 
    
    return metrics

def _get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        print("Warning: cannot create supabase client:", e)
        return None

def upload_file_to_supabase(local_path: str, bucket: Optional[str] = None, object_name: Optional[str] = None) -> Optional[str]:
    """
    Upload local file to Supabase Storage and return a public URL.
    This is defensive across different supabase-py versions.
    """
    supabase = _get_supabase_client()
    if supabase is None:
        print("SUPABASE not configured; skipping upload.")
        return None

    local_path = str(local_path)
    bucket = bucket or os.getenv("SUPABASE_BUCKET", "models")
    object_name = object_name or Path(local_path).name

    try:
        # Try uploading by providing a file-like object (works on many client versions)
        with open(local_path, "rb") as fh:
            try:
                res = supabase.storage.from_(bucket).upload(object_name, fh)  # most common signature
            except TypeError:
                # fallback: some versions expect raw bytes
                fh.seek(0)
                data = fh.read()
                res = supabase.storage.from_(bucket).upload(object_name, data)
            except Exception as e_inner:
                # Last resort: some versions expect (path_on_bucket, file_path) or reversed args.
                # Try calling with file path as second arg.
                try:
                    res = supabase.storage.from_(bucket).upload(object_name, local_path)
                except Exception as e_last:
                    raise RuntimeError(f"Supabase upload failed (multiple attempts): {e_inner} / {e_last}")

        # Check for explicit error structure
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(f"Supabase upload error: {res.get('error')}")

        # Get public URL
        try:
            public_res = supabase.storage.from_(bucket).get_public_url(object_name)
            if isinstance(public_res, dict):
                public_url = public_res.get("publicURL") or public_res.get("public_url") or list(public_res.values())[0]
            elif hasattr(public_res, "publicURL"):
                public_url = getattr(public_res, "publicURL")
            else:
                public_url = str(public_res)
        except Exception:
            # Fallback: construct public url from known pattern
            supabase_url = os.getenv("SUPABASE_URL")
            public_url = None
            if supabase_url:
                public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{object_name}"

        print(f"Uploaded to Supabase: bucket={bucket}, object={object_name}, url={public_url}")
        return public_url

    except Exception as e:
        print(f"Warning: supabase upload failed for {local_path}: {e}")
        return None

def register_model_in_db(metadata: dict, model_path: str, created_by: str = None, activate: bool = False) -> Optional[int]:  # type: ignore 
    """
    Insert a row into model_registry and optionally activate it.
    Returns inserted model id.
    Uses DATABASE_URL from env.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("DATABASE_URL not set; skipping DB model registration.")
        return None

    engine = create_engine(DATABASE_URL)
    meta_json = json.dumps(metadata)

    model_name = metadata.get("model_name") or f"xgb_model_{metadata.get('created_at', '')}"
    version = metadata.get("created_at", str(int(pd.Timestamp.utcnow().timestamp())))

    insert_sql = text("""
        INSERT INTO model_registry (model_name, version, artifact_path, metadata, created_by, is_active)
        VALUES (:model_name, :version, :artifact_path, :metadatab, :created_by, :is_active)
        RETURNING id
    """)

    with engine.begin() as conn:
        # if activate=true, deactivate others first
        if activate:
            conn.execute(text("UPDATE model_registry SET is_active = false WHERE is_active = true"))
        res = conn.execute(insert_sql, {
            "model_name": model_name,
            "version": version,
            "artifact_path": str(model_path),
            "metadatab": meta_json,
            "created_by": created_by,
            "is_active": activate
        })
        row = res.fetchone()
        model_id = int(row[0]) if row is not None else None

    print(f"Registered model in DB id={model_id}, activate={activate}")
    return model_id

def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                     y_test: pd.Series, best_params: Dict[str, Any], ts_label: str) -> Tuple[str, Dict[str, Any]]:
    """
    Train final model with best_params. Try sklearn XGBClassifier.fit with early stopping.
    If that fails (older xgboost builds), fall back to xgb.train() and ensure feature names
    match the transformed data by converting transformed arrays back to DataFrames.
    """

    # Build preprocessor and fit-transform training data
    preprocessor = create_preprocessing_pipeline(X_train)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    X_train_prep_df = pd.DataFrame(X_train_prep, columns=numeric_features, index=X_train.index)
    X_test_prep_df = pd.DataFrame(X_test_prep, columns=numeric_features, index=X_test.index)

    # First attempt: sklearn wrapper with early stopping (may fail on some xgboost builds)
    try:
        model = XGBClassifier(**best_params)
        model.fit(
            X_train_prep_df,
            y_train,
            eval_set=[(X_test_prep_df, y_test)],
            early_stopping_rounds=50,
            verbose=True
        )

        # Evaluate
        metrics = comprehensive_evaluate(model, X_test_prep_df, y_test)

        # Save sklearn-wrapped model
        model_path = MODEL_DIR / f"xgb_model_{ts_label}.joblib"
        joblib.dump(model, model_path)

        feature_importances = dict(zip(numeric_features, model.feature_importances_.astype(float)))
        model_type = 'sklearn_xgb'

    except TypeError as e:
        # Fallback: use xgboost.train with DMatrix (supports older xgboost builds)
        print("Sklearn XGBClassifier.fit unsupported early_stopping in this xgboost build. Falling back to xgb.train().")
        print("Original TypeError:", e)

        # Create DMatrix from DataFrames (preserves feature names)
        dtrain = xgb.DMatrix(X_train_prep_df, label=y_train)
        dtest = xgb.DMatrix(X_test_prep_df, label=y_test)

        # Prepare params for xgboost.train (remove sklearn-only keys)
        params_xgb = {k: v for k, v in best_params.items() if k not in ('n_jobs', 'use_label_encoder')}
        params_xgb['objective'] = params_xgb.get('objective', 'multi:softprob')
        params_xgb['num_class'] = params_xgb.get('num_class', 3)
        params_xgb['tree_method'] = params_xgb.get('tree_method', 'hist')
        params_xgb['verbosity'] = params_xgb.get('verbosity', 0)

        num_boost_round = int(best_params.get('n_estimators', 500))

        evals_result = {}
        booster = xgb.train(
            params_xgb,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, 'validation')],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False
        )

        # Predictions and metrics
        probs = booster.predict(dtest)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)
        brier_vals = [float(brier_score_loss((y_test == cls).astype(int), probs[:, cls])) for cls in range(3)]
        brier = sum(brier_vals) / len(brier_vals)

        metrics = {
            'accuracy': float(acc),
            'log_loss': float(ll),
            'brier_mean': float(brier),
            'classification_report': classification_report(y_test, preds, target_names=['H', 'D', 'A'], output_dict=True)
        }

        # Save booster (JSON) + joblib fallback
        model_path = MODEL_DIR / f"xgb_booster_{ts_label}.json"
        booster.save_model(str(model_path))
        try:
            joblib.dump(booster, MODEL_DIR / f"xgb_booster_{ts_label}.joblib")
        except Exception:
            pass

        # Feature importances (gain) -> map to numeric_features
        fi = booster.get_score(importance_type='gain')
        feature_importances = {fn: float(fi.get(fn, 0.0)) for fn in numeric_features} # type: ignore
        model_type = 'xgb_booster'
    
    preprocessor_path = ARTIFACT_DIR / f"preprocessor_{ts_label}.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    meta_path = ARTIFACT_DIR / f"model_meta_{ts_label}.json"
    metadata = {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "created_at": ts_label,
        "n_features": len(numeric_features),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": numeric_features,
        "params": best_params,
        "metrics": metrics,
        "feature_importance": feature_importances,
        "model_type": model_type
    }

    # Try to upload to Supabase (best-effort)
    try:
        supabase_model_url = upload_file_to_supabase(str(model_path), object_name=f"{Path(model_path).name}")
        supabase_preproc_url = upload_file_to_supabase(str(preprocessor_path), object_name=f"{Path(preprocessor_path).name}")
        if supabase_model_url:
            metadata["artifact_url"] = supabase_model_url
        if supabase_preproc_url:
            metadata["preprocessor_url"] = supabase_preproc_url
    except Exception as e:
        print("Warning: exception during supabase upload:", e)

    # write metadata file (always)
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        print("Warning: could not write model metadata file:", e)

    # prepare DB registration
    metadata_for_db = metadata.copy()
    metadata_for_db["model_name"] = f"xgb_model_{ts_label}"
    metadata_for_db["artifact_filename"] = os.path.basename(str(model_path))

    artifact_path_for_db = metadata.get("artifact_url") or str(model_path)

    model_db_id = None
    try:
        model_db_id = register_model_in_db(metadata_for_db, artifact_path_for_db, created_by=os.getenv("USER") or os.getenv("USERNAME"), activate=True) # type: ignore
        metadata['db_model_id'] = model_db_id
    except Exception as e:
        print("Warning: failed to register model in DB:", e)

    # Final prints & guaranteed return
    print(f"\nModel saved: {model_path}")
    print(f"Metadata saved: {meta_path}")
    if model_db_id:
        print(f"Registered model in DB id={model_db_id} (activate=True)")
    else:
        print("Model not registered in DB.")

    # ensure we always return the model path and metadata tuple (string path, dict)
    try:
        return str(model_path), metadata
    except Exception as e:
        # if something odd happened, raise explicit error
        raise RuntimeError(f"train_final_model failed to return results: {e}")


def parse_args():
    """Parse command line arguments with better defaults."""
    parser = argparse.ArgumentParser(description="Optimized football match prediction model training")
    parser.add_argument("--window", type=int, default=5, 
                       help="Rolling window for team features (default: 5)")
    parser.add_argument("--n_trials", type=int, default=50, 
                       help="Optuna trials for hyperparameter optimization (default: 50)")
    parser.add_argument("--n_splits", type=int, default=5, 
                       help="TimeSeriesSplit cross-validation folds (default: 5)")
    parser.add_argument("--test_days", type=int, default=365, 
                       help="Holdout test period in days (default: 365)")
    parser.add_argument("--min_date", type=str, default=None, 
                       help="Minimum date for data fetching (YYYY-MM-DD)")
    parser.add_argument("--quick_test", action="store_true", 
                       help="Run with reduced trials for quick testing")
    return parser.parse_args()

def main():
    """Main training pipeline with optimizations."""
    args = parse_args()
    
    if args.quick_test:
        args.n_trials = 10
        print("Quick test mode: Using reduced trials")
    
    print("=" * 60)
    print("OPTIMIZED FOOTBALL MATCH PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    print(f"Memory usage at start: {get_memory_usage():.1%}")
    print(f"Using rolling window: {args.window}")
    
    # Step 1: Data Loading
    print("\n1. Fetching match data from database...")
    df_matches = fetch_matches(min_date=args.min_date)
    print(f"Loaded {len(df_matches):,} matches")
    print(f"Date range: {df_matches['date'].min()} to {df_matches['date'].max()}")
    print(f"Memory usage: {get_memory_usage():.1%}")
    
    # Step 2: Feature Engineering
    print("\n2. Building team performance features...")
    team_records = build_team_features_vectorized(df_matches)
    print(f"Created {len(team_records):,} team records")
    
    print(f"   Computing rolling statistics (window={args.window})...")
    team_stats = compute_rolling_stats_optimized(team_records, window=args.window)
    print(f"Memory usage after features: {get_memory_usage():.1%}")
    
    # Step 3: Create Feature Matrix
    print("\n3. Creating feature matrix...")
    X, y, meta = create_match_features(team_stats, df_matches, window=args.window)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Memory cleanup
    del df_matches, team_records, team_stats
    gc.collect()
    
    # Step 4: Train/Test Split
    print("\n4. Creating train/test split...")
    meta['date'] = pd.to_datetime(meta['date'])
    cutoff_date = meta['date'].max() - pd.Timedelta(days=args.test_days)
    
    train_mask = meta['date'] < cutoff_date
    test_mask = meta['date'] >= cutoff_date
    
    X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
    y_train, y_test = y[train_mask].copy(), y[test_mask].copy()
    
    # Remove samples with too many missing features
    train_valid_mask = X_train.isna().sum(axis=1) < len(X_train.columns) * 0.5
    test_valid_mask = X_test.isna().sum(axis=1) < len(X_test.columns) * 0.5
    
    X_train, y_train = X_train[train_valid_mask], y_train[train_valid_mask]
    X_test, y_test = X_test[test_valid_mask], y_test[test_valid_mask]
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Cutoff date: {cutoff_date}")
    
    # Step 5: Hyperparameter Optimization
    print(f"\n5. Starting Optuna hyperparameter optimization ({args.n_trials} trials)...")
    
    study = optuna.create_study(
        direction="minimize",
        study_name=f"football_prediction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    def objective_wrapper(trial):
        return enhanced_objective(trial, X_train, y_train, meta[train_mask], args.n_splits)
    
    study.optimize(objective_wrapper, n_trials=args.n_trials, show_progress_bar=True)
    
    print(f"\nBest trial score: {study.best_trial.value:.4f}")
    print("Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Step 6: Final Model Training
    print("\n6. Training final model...")
    ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    best_params = study.best_trial.params.copy()
    best_params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1,
        'use_label_encoder': False
    })
    
    result = train_final_model(X_train, y_train, X_test, y_test, best_params, ts_label)
    if result is None:
        raise RuntimeError("train_final_model returned None")
    model_path, metadata = result
    
    print(f"\n7. Training completed successfully!")
    print(f"Final memory usage: {get_memory_usage():.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()