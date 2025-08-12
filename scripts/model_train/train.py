#!/usr/bin/env python3
"""
scripts/train_optuna.py

- Fetches match rows from Postgres (Supabase) using DATABASE_URL from .env
- Builds vectorized rolling features per team (window configurable)
- Runs Optuna tuning using TimeSeriesSplit CV (time-aware)
- Trains final XGBoost model with best params and saves model + metadata
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, classification_report, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import optuna
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL in .env")

MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Constants for better maintainability
RESULT_MAPPING = {'H': 0, 'D': 1, 'A': 2}
RESULT_LABELS = ['H', 'D', 'A']
RANDOM_STATE = 42


####################
# Data fetching
####################
def fetch_matches(min_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch all rows from public.matches with non-null date, ordered by date.
    
    Args:
        min_date: Optional minimum date filter (YYYY-MM-DD format)
        
    Returns:
        DataFrame with match data
    """
    engine = create_engine(DATABASE_URL, future=True)
    
    if min_date:
        sql = """
        SELECT * FROM public.matches 
        WHERE date IS NOT NULL AND date >= :min_date 
        ORDER BY date
        """
        params = {"min_date": min_date}
    else:
        sql = "SELECT * FROM public.matches WHERE date IS NOT NULL ORDER BY date"
        params = {}
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        print(f"Fetched {len(df)} matches from database")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data from database: {e}")


####################
# Feature engineering (vectorized and optimized)
####################
def build_team_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long-form table with one row per team per match (home/away).
    Optimized with vectorized operations.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Create both tables simultaneously with vectorized operations
    base_cols = ['match_id', 'date', 'full_time_result']
    optional_cols = {
        'home_shots': 'shots',
        'away_shots': 'shots', 
        'home_shots_on_target': 'shots_on_target',
        'away_shots_on_target': 'shots_on_target',
        'home_corners': 'corners',
        'away_corners': 'corners'
    }
    
    # Home team data
    home_data = {
        'match_id': df['match_id'],
        'date': df['date'],
        'team': df['home_team'],
        'is_home': True,
        'goals_for': df['full_time_home_goals'],
        'goals_against': df['full_time_away_goals'],
        'result': df['full_time_result']
    }
    
    # Away team data  
    away_data = {
        'match_id': df['match_id'],
        'date': df['date'],
        'team': df['away_team'],
        'is_home': False,
        'goals_for': df['full_time_away_goals'],
        'goals_against': df['full_time_home_goals'],
        'result': df['full_time_result']
    }
    
    # Add optional columns if they exist
    for home_col, target_col in optional_cols.items():
        if home_col in df.columns:
            home_data[target_col] = df[home_col]
        if home_col.replace('home_', 'away_') in df.columns:
            away_data[target_col] = df[home_col.replace('home_', 'away_')]
    
    home_df = pd.DataFrame(home_data)
    away_df = pd.DataFrame(away_data)

    # Vectorized win calculation
    home_df['is_win'] = (home_df['result'] == 'H').astype(int)
    away_df['is_win'] = (away_df['result'] == 'A').astype(int)

    # Vectorized goal difference calculation
    home_df['goal_diff'] = home_df['goals_for'] - home_df['goals_against']
    away_df['goal_diff'] = away_df['goals_for'] - away_df['goals_against']

    # Combine and sort efficiently
    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df.sort_values(['team', 'date'], inplace=True)
    
    return long_df


def compute_rolling_features(long_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Vectorized rolling features per team using groupby operations.
    More efficient than the original apply-based approach.
    """
    df = long_df.copy()
    
    # Columns to compute rolling statistics for
    rolling_cols = ['is_win', 'goal_diff', 'shots', 'shots_on_target', 'corners']
    existing_cols = [col for col in rolling_cols if col in df.columns]
    
    # Vectorized rolling computation
    for col in existing_cols:
        # Shift to exclude current match, then compute rolling mean
        df[f'{col}_roll_mean_{window}'] = (
            df.groupby('team')[col]
            .shift()  # Exclude current match
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)  # Remove team level from index
        )
    
    return df


def pivot_features_to_matches(rolled_long: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Pivot home and away features to per-match rows (match_id).
    Optimized with better column detection and handling.
    """
    # Dynamic column detection based on window parameter
    rolling_suffix = f'_roll_mean_{window}'
    
    # Find all rolling columns
    rolling_cols = {
        'form': f'is_win{rolling_suffix}',
        'goal_diff': f'goal_diff{rolling_suffix}',
        'shots': f'shots{rolling_suffix}',
        'shots_on_target': f'shots_on_target{rolling_suffix}',
        'corners': f'corners{rolling_suffix}'
    }
    
    # Filter to only existing columns
    existing_rolling_cols = {k: v for k, v in rolling_cols.items() if v in rolled_long.columns}
    
    if not existing_rolling_cols:
        raise ValueError(f"No rolling columns found with suffix {rolling_suffix}")
    
    # Separate home and away data more efficiently
    home_mask = rolled_long['is_home'] == True
    away_mask = rolled_long['is_home'] == False
    
    # Base columns for both
    base_cols = ['match_id', 'team'] + list(existing_rolling_cols.values())
    
    # Home features
    home_rename = {'team': 'home_team'}
    for feature, col in existing_rolling_cols.items():
        home_rename[col] = f'home_{feature}'
    
    home_feats = (
        rolled_long.loc[home_mask, base_cols]
        .rename(columns=home_rename)
        .set_index('match_id')
    )
    
    # Away features  
    away_rename = {'team': 'away_team'}
    for feature, col in existing_rolling_cols.items():
        away_rename[col] = f'away_{feature}'
        
    away_feats = (
        rolled_long.loc[away_mask, base_cols]
        .rename(columns=away_rename)
        .set_index('match_id')
    )
    
    # Join efficiently
    merged = home_feats.join(away_feats, how='outer')
    merged.reset_index(inplace=True)
    
    return merged


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add implied probability features from betting odds.
    Handles missing odds gracefully.
    """
    odds_cols = ['b365_home_odds', 'b365_draw_odds', 'b365_away_odds']
    
    # Check if all odds columns exist
    if all(col in df.columns for col in odds_cols):
        # Vectorized implied probability calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            home_implied = 1 / df['b365_home_odds']
            draw_implied = 1 / df['b365_draw_odds'] 
            away_implied = 1 / df['b365_away_odds']
            
            # Normalize probabilities (handle division by zero)
            total_implied = home_implied + draw_implied + away_implied
            valid_mask = (total_implied > 0) & np.isfinite(total_implied)
            
            df['home_prob_implied'] = np.where(valid_mask, home_implied / total_implied, np.nan)
            df['draw_prob_implied'] = np.where(valid_mask, draw_implied / total_implied, np.nan)
            df['away_prob_implied'] = np.where(valid_mask, away_implied / total_implied, np.nan)
    else:
        # Set to NaN if odds not available
        df['home_prob_implied'] = np.nan
        df['draw_prob_implied'] = np.nan  
        df['away_prob_implied'] = np.nan
        
    return df


def construct_feature_matrix(matches_df: pd.DataFrame, window: int = 5) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Full feature engineering pipeline with optimizations.
    
    Returns:
        X: Feature matrix
        y: Target labels  
        meta: Metadata (match_id, date, teams, etc.)
    """
    print(f"Building features with window={window}...")
    
    matches = matches_df.copy()
    matches['date'] = pd.to_datetime(matches['date'])
    
    # Filter valid matches early
    valid_mask = (
        matches['match_id'].notna() & 
        matches['date'].notna() &
        matches['full_time_result'].isin(RESULT_LABELS)
    )
    matches = matches.loc[valid_mask].sort_values('date').reset_index(drop=True)
    
    print(f"Processing {len(matches)} valid matches...")
    
    # Feature engineering pipeline
    long_df = build_team_long_table(matches)
    rolled_df = compute_rolling_features(long_df, window=window)
    features_df = pivot_features_to_matches(rolled_df, window=window)
    
    # Merge with original matches
    merged = matches.merge(features_df, on='match_id', how='left')
    merged = add_odds_features(merged)
    
    # Create target variable
    merged['y'] = merged['full_time_result'].map(RESULT_MAPPING)
    
    # Define feature columns dynamically
    feature_cols = []
    
    # Team form features
    form_features = [f'{side}_{feat}' for side in ['home', 'away'] 
                    for feat in ['form', 'goal_diff', 'shots', 'shots_on_target', 'corners']
                    if f'{side}_{feat}' in merged.columns]
    feature_cols.extend(form_features)
    
    # Odds features
    odds_features = ['home_prob_implied', 'draw_prob_implied', 'away_prob_implied']
    odds_features = [col for col in odds_features if col in merged.columns]
    feature_cols.extend(odds_features)
    
    print(f"Selected {len(feature_cols)} features: {feature_cols}")
    
    # Create final datasets
    X = merged[feature_cols].copy()
    y = merged['y'].copy()
    
    # Metadata for tracking
    meta_cols = ['match_id', 'date', 'home_team', 'away_team']
    if 'season' in merged.columns:
        meta_cols.append('season')
    if 'league' in merged.columns:
        meta_cols.append('league')
        
    meta = merged[meta_cols].copy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    return X, y, meta


####################
# Model training and evaluation
####################
def create_preprocessing_pipeline() -> Pipeline:
    """Create preprocessing pipeline with imputation and scaling."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """
    Optuna objective with improved parameter space and early stopping.
    """
    # Improved parameter suggestions
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        
        # Fixed parameters
        'random_state': RANDOM_STATE,
        'objective': 'multi:softprob',
        'num_class': 3,
        'tree_method': 'hist',
        'verbosity': 0,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        try:
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Preprocess data
            X_train_processed = preprocessor.fit_transform(X_train_fold)
            X_val_processed = preprocessor.transform(X_val_fold)
            
            # Train model with early stopping
            model = XGBClassifier(**params)
            
            # Use a portion of training data for early stopping validation
            if len(X_train_fold) > 100:
                split_idx = int(len(X_train_fold) * 0.9)
                X_train_es = X_train_processed[:split_idx]
                X_val_es = X_train_processed[split_idx:]
                y_train_es = y_train_fold.iloc[:split_idx]
                y_val_es = y_train_fold.iloc[split_idx:]
                
                model.fit(
                    X_train_es, y_train_es,
                    eval_set=[(X_val_es, y_val_es)],
                    early_stopping_rounds=20,
                    verbose=False
                )
            else:
                model.fit(X_train_processed, y_train_fold)
            
            # Predict and evaluate
            y_pred_proba = model.predict_proba(X_val_processed)
            fold_score = log_loss(y_val_fold, y_pred_proba)
            fold_scores.append(fold_score)
            
        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}")
            # Return a penalty score for failed folds
            return 10.0
    
    if not fold_scores:
        return 10.0
        
    mean_score = np.mean(fold_scores)
    
    # Report intermediate results for pruning
    trial.report(mean_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
        
    return mean_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, preprocessor) -> Dict[str, float]:
    """Comprehensive model evaluation."""
    X_test_processed = preprocessor.transform(X_test)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test_processed)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    
    # Brier score for each class
    brier_scores = []
    for class_idx in range(3):
        y_binary = (y_test == class_idx).astype(int)
        brier = brier_score_loss(y_binary, y_pred_proba[:, class_idx])
        brier_scores.append(brier)
    
    brier_mean = np.mean(brier_scores)
    
    metrics = {
        'accuracy': float(accuracy),
        'log_loss': float(logloss), 
        'brier_mean': float(brier_mean),
        'brier_home': float(brier_scores[0]),
        'brier_draw': float(brier_scores[1]),
        'brier_away': float(brier_scores[2])
    }
    
    # Print detailed results
    print(f"\nFinal Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Brier Score (mean): {brier_mean:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=RESULT_LABELS))
    
    return metrics


def train_and_save(X_train: pd.DataFrame, y_train: pd.Series, 
                   X_test: pd.DataFrame, y_test: pd.Series,
                   best_params: Dict[str, Any], ts_label: str) -> Tuple[str, Dict[str, Any]]:
    """
    Train final model and save artifacts.
    """
    print("Training final model...")
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train final model
    model = XGBClassifier(**best_params)
    model.fit(
        X_train_processed, y_train,
        eval_set=[(X_test_processed, y_test)],
        early_stopping_rounds=50,
        verbose=True
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, preprocessor)
    
    # Save artifacts
    model_path = MODEL_DIR / f"xgb_model_{ts_label}.joblib"
    preprocessor_path = ARTIFACT_DIR / f"preprocessor_{ts_label}.joblib"
    meta_path = ARTIFACT_DIR / f"model_meta_{ts_label}.json"
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save metadata
    metadata = {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "created_at": ts_label,
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X_train.columns),
        "params": best_params,
        **metrics  # Include all evaluation metrics
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved: {model_path}")
    print(f"Preprocessor saved: {preprocessor_path}")
    print(f"Metadata saved: {meta_path}")
    
    return str(model_path), metadata


####################
# Main execution
####################
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train football match prediction model with Optuna")
    parser.add_argument("--window", type=int, default=5, 
                       help="Rolling window size for team features")
    parser.add_argument("--n_trials", type=int, default=50, 
                       help="Number of Optuna optimization trials")  
    parser.add_argument("--n_splits", type=int, default=5,
                       help="Number of TimeSeriesSplit CV folds")
    parser.add_argument("--test_days", type=int, default=365,
                       help="Number of days for holdout test set")
    parser.add_argument("--min_date", type=str, default=None,
                       help="Minimum date for data filtering (YYYY-MM-DD)")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout for Optuna study in seconds")
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("="*60)
    print("Football Match Prediction Model Training")
    print("="*60)
    print(f"Configuration:")
    print(f"  Window size: {args.window}")
    print(f"  Optuna trials: {args.n_trials}")
    print(f"  CV folds: {args.n_splits}")
    print(f"  Test period: {args.test_days} days")
    print(f"  Min date: {args.min_date or 'None'}")
    print("="*60)

    # Fetch and prepare data
    df_all = fetch_matches(min_date=args.min_date)
    
    if len(df_all) == 0:
        raise ValueError("No data fetched from database")
    
    # Build features
    X_all, y_all, meta = construct_feature_matrix(df_all, window=args.window)
    
    if len(X_all) == 0:
        raise ValueError("No features could be constructed")
    
    # Split data temporally
    meta['date'] = pd.to_datetime(meta['date'])
    last_date = meta['date'].max()
    cutoff = last_date - pd.Timedelta(days=args.test_days)
    
    train_mask = meta['date'] < cutoff
    test_mask = meta['date'] >= cutoff
    
    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} matches (up to {cutoff.date()})")
    print(f"  Testing: {len(X_test)} matches (from {cutoff.date()})")
    
    if len(X_train) < 100:
        raise ValueError(f"Insufficient training data: {len(X_train)} matches")
    
    # Clean data - remove rows with all missing features
    train_valid = ~X_train.isna().all(axis=1)
    test_valid = ~X_test.isna().all(axis=1)
    
    X_train, y_train = X_train.loc[train_valid], y_train.loc[train_valid]
    X_test, y_test = X_test.loc[test_valid], y_test.loc[test_valid]
    
    print(f"  After cleaning: {len(X_train)} train, {len(X_test)} test")
    
    # Optuna hyperparameter optimization
    print(f"\nStarting Optuna optimization ({args.n_trials} trials)...")
    
    study = optuna.create_study(
        direction="minimize",
        study_name=f"xgb_football_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    def optuna_objective(trial):
        return objective(trial, X_train, y_train, n_splits=args.n_splits)
    
    study.optimize(
        optuna_objective, 
        n_trials=args.n_trials, 
        timeout=args.timeout,
        show_progress_bar=True
    )
    
    print(f"\nOptimization completed!")
    print(f"Best score: {study.best_trial.value:.4f}")
    print(f"Best parameters: {study.best_trial.params}")
    
    # Prepare final model parameters
    best_params = study.best_trial.params.copy()
    best_params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'use_label_encoder': False,
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'verbosity': 1,
        'eval_metric': 'mlogloss'
    })
    
    # Train and save final model
    ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path, metadata = train_and_save(X_train, y_train, X_test, y_test, best_params, ts_label)
    
    print("="*60)
    print("Training completed successfully!")
    print(f"Model accuracy: {metadata['accuracy']:.4f}")
    print(f"Model log loss: {metadata['log_loss']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()