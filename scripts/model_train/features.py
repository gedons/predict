# scripts/features.py
"""
Optimized feature engineering module for football match prediction.

Key optimizations:
- Vectorized operations using pandas groupby and rolling functions
- Efficient memory usage with optimized data types
- Improved rolling window calculations
- Enhanced feature engineering with additional metrics
- Better handling of missing data and edge cases
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types to reduce memory usage."""
    df = df.copy()
    
    # Optimize integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype('int32')
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert string columns to categorical if they have low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['date'] and df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    return df


def prepare_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced basic data preparation with validation and optimization.
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Clean team names more thoroughly
    team_cols = ['home_team', 'away_team']
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    
    # Sort by date for proper time series handling
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    # Fill missing numerical stats with 0 (makes sense for shots, corners, etc.)
    stat_cols = [
        'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
        'home_corners', 'away_corners', 'full_time_home_goals', 'full_time_away_goals'
    ]
    
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Optimize data types
    df = optimize_dtypes(df)
    
    return df


def add_result_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add result-based columns with enhanced metrics.
    """
    df = df.copy()
    
    # Basic result indicators
    if 'full_time_result' in df.columns:
        df['home_win'] = (df['full_time_result'] == 'H').astype('int8')
        df['draw'] = (df['full_time_result'] == 'D').astype('int8')
        df['away_win'] = (df['full_time_result'] == 'A').astype('int8')
    
    # Goal-based metrics
    if 'full_time_home_goals' in df.columns and 'full_time_away_goals' in df.columns:
        df['goal_diff_home'] = (df['full_time_home_goals'] - df['full_time_away_goals']).astype('int8')
        df['total_goals'] = (df['full_time_home_goals'] + df['full_time_away_goals']).astype('int8')
        
        # Goal-based performance indicators
        df['home_clean_sheet'] = (df['full_time_away_goals'] == 0).astype('int8')
        df['away_clean_sheet'] = (df['full_time_home_goals'] == 0).astype('int8')
        df['both_teams_scored'] = ((df['full_time_home_goals'] > 0) & 
                                  (df['full_time_away_goals'] > 0)).astype('int8')
    
    # Shot efficiency (if shot data available)
    for side in ['home', 'away']:
        shots_col = f'{side}_shots'
        sot_col = f'{side}_shots_on_target'
        goals_col = f'full_time_{side}_goals'
        
        if all(col in df.columns for col in [shots_col, sot_col, goals_col]):
            # Shot accuracy
            df[f'{side}_shot_accuracy'] = np.where(
                df[shots_col] > 0,
                df[sot_col] / df[shots_col],
                0
            ).astype('float32')
            
            # Conversion rate (goals per shot on target)
            df[f'{side}_conversion_rate'] = np.where(
                df[sot_col] > 0,
                df[goals_col] / df[sot_col],
                0
            ).astype('float32')
    
    return df


def create_team_records_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create team performance records using vectorized operations.
    Much faster than the original iterative approach.
    """
    # Create home team records
    home_records = pd.DataFrame({
        'match_id': df['match_id'],
        'date': df['date'],
        'team': df['home_team'],
        'is_home': True,
        'opponent': df['away_team'],
        'goals_for': df['full_time_home_goals'].astype('int8'),
        'goals_against': df['full_time_away_goals'].astype('int8'),
        'shots': df.get('home_shots', 0).astype('int8'),
        'shots_on_target': df.get('home_shots_on_target', 0).astype('int8'),
        'corners': df.get('home_corners', 0).astype('int8'),
        'result': df['full_time_result']
    })
    
    # Create away team records
    away_records = pd.DataFrame({
        'match_id': df['match_id'],
        'date': df['date'],
        'team': df['away_team'],
        'is_home': False,
        'opponent': df['home_team'],
        'goals_for': df['full_time_away_goals'].astype('int8'),
        'goals_against': df['full_time_home_goals'].astype('int8'),
        'shots': df.get('away_shots', 0).astype('int8'),
        'shots_on_target': df.get('away_shots_on_target', 0).astype('int8'),
        'corners': df.get('away_corners', 0).astype('int8'),
        'result': df['full_time_result']
    })
    
    # Determine match outcomes from team perspective
    home_records['is_win'] = (home_records['result'] == 'H').astype('int8')
    home_records['is_draw'] = (home_records['result'] == 'D').astype('int8')
    home_records['is_loss'] = (home_records['result'] == 'A').astype('int8')
    
    away_records['is_win'] = (away_records['result'] == 'A').astype('int8')
    away_records['is_draw'] = (away_records['result'] == 'D').astype('int8')
    away_records['is_loss'] = (away_records['result'] == 'H').astype('int8')
    
    # Calculate goal difference from team perspective
    home_records['goal_diff'] = (home_records['goals_for'] - home_records['goals_against']).astype('int8')
    away_records['goal_diff'] = (away_records['goals_for'] - away_records['goals_against']).astype('int8')
    
    # Add performance metrics
    for records in [home_records, away_records]:
        records['clean_sheet'] = (records['goals_against'] == 0).astype('int8')
        records['scored'] = (records['goals_for'] > 0).astype('int8')
        records['shot_accuracy'] = np.where(
            records['shots'] > 0,
            records['shots_on_target'] / records['shots'],
            0
        ).astype('float32')
    
    # Combine records
    team_records = pd.concat([home_records, away_records], ignore_index=True)
    team_records = team_records.sort_values(['team', 'date']).reset_index(drop=True)
    
    return team_records


def compute_rolling_features_optimized(team_records: pd.DataFrame, 
                                     windows: list = [3, 5, 10]) -> pd.DataFrame:
    """
    Compute rolling features using efficient pandas operations.
    Supports multiple window sizes for different time horizons.
    """
    df = team_records.copy()
    
    # Define columns for rolling calculations
    rolling_cols = {
        'is_win': 'mean',
        'is_draw': 'mean',
        'goal_diff': 'mean',
        'goals_for': 'mean',
        'goals_against': 'mean',
        'shots': 'mean',
        'shots_on_target': 'mean',
        'corners': 'mean',
        'clean_sheet': 'mean',
        'scored': 'mean',
        'shot_accuracy': 'mean'
    }
    
    # Group by team for efficient rolling calculations
    grouped = df.groupby('team', group_keys=False)
    
    for window in windows:
        window_suffix = f'_{window}'
        
        for col, agg_func in rolling_cols.items():
            if col in df.columns:
                # Use shift(1) to exclude current match, then rolling
                if agg_func == 'mean':
                    df[f'{col}_avg{window_suffix}'] = grouped[col].shift(1).rolling(
                        window=window, min_periods=1
                    ).mean().astype('float32')
                elif agg_func == 'sum':
                    df[f'{col}_sum{window_suffix}'] = grouped[col].shift(1).rolling(
                        window=window, min_periods=1
                    ).sum().astype('float32')
    
    # Add recent form (weighted toward recent matches)
    def weighted_form(series, window=5):
        """Calculate weighted form giving more weight to recent matches."""
        weights = np.exp(np.linspace(-1, 0, window))
        weights = weights / weights.sum()
        
        def apply_weights(x):
            if len(x) == 0:
                return 0
            actual_weights = weights[-len(x):]
            actual_weights = actual_weights / actual_weights.sum()
            return np.average(x, weights=actual_weights)
        
        return series.shift(1).rolling(window=window, min_periods=1).apply(apply_weights)
    
    # Add weighted form for win rate
    df['form_weighted_5'] = grouped['is_win'].transform(
        lambda x: weighted_form(x, 5)
    ).astype('float32')
    
    # Add home/away specific performance
    home_mask = df['is_home'] == True
    away_mask = df['is_home'] == False
    
    for mask, venue in [(home_mask, 'home'), (away_mask, 'away')]:
        if mask.sum() > 0:
            venue_grouped = df[mask].groupby('team', group_keys=False)
            df.loc[mask, f'form_{venue}_5'] = venue_grouped['is_win'].shift(1).rolling(
                window=5, min_periods=1
            ).mean().astype('float32')
    
    # Forward fill venue-specific stats for opposite venue
    df['form_home_5'] = df.groupby('team')['form_home_5'].fillna(method='ffill')
    df['form_away_5'] = df.groupby('team')['form_away_5'].fillna(method='ffill')
    
    return df


def create_head_to_head_features(df: pd.DataFrame, team_records: pd.DataFrame) -> pd.DataFrame:
    """
    Create head-to-head historical features between teams.
    """
    h2h_features = []
    
    for _, match in df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        match_date = match['date']
        
        # Get historical matches between these teams
        h2h_matches = df[
            (((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
             ((df['home_team'] == away_team) & (df['away_team'] == home_team))) &
            (df['date'] < match_date)
        ].tail(10)  # Last 10 H2H matches
        
        if len(h2h_matches) > 0:
            # Calculate H2H stats from home team perspective
            home_wins = len(h2h_matches[
                ((h2h_matches['home_team'] == home_team) & (h2h_matches['full_time_result'] == 'H')) |
                ((h2h_matches['away_team'] == home_team) & (h2h_matches['full_time_result'] == 'A'))
            ])
            
            draws = len(h2h_matches[h2h_matches['full_time_result'] == 'D'])
            away_wins = len(h2h_matches) - home_wins - draws
            
            h2h_home_win_rate = home_wins / len(h2h_matches)
            h2h_draw_rate = draws / len(h2h_matches)
            
            # Average goals in H2H
            avg_total_goals = h2h_matches[['full_time_home_goals', 'full_time_away_goals']].sum(axis=1).mean()
        else:
            h2h_home_win_rate = 0.33  # Neutral prior
            h2h_draw_rate = 0.33
            avg_total_goals = 2.5  # League average
        
        h2h_features.append({
            'match_id': match['match_id'],
            'h2h_home_win_rate': h2h_home_win_rate,
            'h2h_draw_rate': h2h_draw_rate,
            'h2h_avg_goals': avg_total_goals,
            'h2h_matches_count': len(h2h_matches) if len(h2h_matches) > 0 else 0
        })
    
    return pd.DataFrame(h2h_features)


def rolling_team_features_optimized(df: pd.DataFrame, 
                                  windows: list = [3, 5, 10],
                                  include_h2h: bool = True) -> pd.DataFrame:
    """
    Optimized rolling team features with multiple window sizes and enhanced metrics.
    """
    print("Creating team performance records...")
    team_records = create_team_records_vectorized(df)
    
    print("Computing rolling statistics...")
    team_records = compute_rolling_features_optimized(team_records, windows)
    
    print("Creating match-level features...")
    
    # Separate home and away records
    home_records = team_records[team_records['is_home'] == True].copy()
    away_records = team_records[team_records['is_home'] == False].copy()
    
    # Create feature columns mapping
    feature_mappings = {}
    for window in windows:
        base_features = [
            f'is_win_avg_{window}', f'is_draw_avg_{window}', f'goal_diff_avg_{window}',
            f'goals_for_avg_{window}', f'goals_against_avg_{window}',
            f'shots_avg_{window}', f'shots_on_target_avg_{window}',
            f'corners_avg_{window}', f'clean_sheet_avg_{window}',
            f'scored_avg_{window}', f'shot_accuracy_avg_{window}'
        ]
        
        for feature in base_features:
            if feature in team_records.columns:
                feature_mappings[f'home_{feature}'] = feature
                feature_mappings[f'away_{feature}'] = feature
    
    # Add special features
    special_features = ['form_weighted_5', 'form_home_5', 'form_away_5']
    for feature in special_features:
        if feature in team_records.columns:
            feature_mappings[f'home_{feature}'] = feature
            feature_mappings[f'away_{feature}'] = feature
    
    # Create home features DataFrame
    home_features = home_records[['match_id'] + list(feature_mappings.values())].copy()
    home_features.columns = ['match_id'] + [f'home_{col}' for col in feature_mappings.values()]
    
    # Create away features DataFrame  
    away_features = away_records[['match_id'] + list(feature_mappings.values())].copy()
    away_features.columns = ['match_id'] + [f'away_{col}' for col in feature_mappings.values()]
    
    # Merge home and away features
    match_features = home_features.merge(away_features, on='match_id', how='inner')
    
    # Add current match statistics
    current_stats = df[[
        'match_id', 'home_shots', 'away_shots',
        'home_shots_on_target', 'away_shots_on_target',
        'home_corners', 'away_corners',
        'b365_home_odds', 'b365_draw_odds', 'b365_away_odds'
    ]].copy()
    
    match_features = match_features.merge(current_stats, on='match_id', how='left')
    
    # Add head-to-head features if requested
    if include_h2h:
        print("Computing head-to-head features...")
        h2h_features = create_head_to_head_features(df, team_records)
        match_features = match_features.merge(h2h_features, on='match_id', how='left')
    
    # Process betting odds
    odds_cols = ['b365_home_odds', 'b365_draw_odds', 'b365_away_odds']
    prob_cols = ['home_prob_implied', 'draw_prob_implied', 'away_prob_implied']
    
    if all(col in match_features.columns for col in odds_cols):
        # Handle missing odds with league averages
        match_features['b365_home_odds'] = match_features['b365_home_odds'].fillna(2.5)
        match_features['b365_draw_odds'] = match_features['b365_draw_odds'].fillna(3.2)
        match_features['b365_away_odds'] = match_features['b365_away_odds'].fillna(2.8)
        
        # Convert to implied probabilities
        for odds_col, prob_col in zip(odds_cols, prob_cols):
            match_features[prob_col] = (1 / match_features[odds_col]).astype('float32')
        
        # Normalize probabilities (remove bookmaker margin)
        total_prob = match_features[prob_cols].sum(axis=1)
        for prob_col in prob_cols:
            match_features[prob_col] = (match_features[prob_col] / total_prob).astype('float32')
        
        # Add betting value features
        match_features['home_odds_value'] = match_features['home_prob_implied'] - (1 / match_features['b365_home_odds'])
        match_features['draw_odds_value'] = match_features['draw_prob_implied'] - (1 / match_features['b365_draw_odds'])
        match_features['away_odds_value'] = match_features['away_prob_implied'] - (1 / match_features['b365_away_odds'])
    
    return match_features


def build_feature_matrix(df_matches: pd.DataFrame, 
                        windows: list = [3, 5, 10],
                        include_h2h: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Enhanced feature matrix building with comprehensive feature engineering.
    
    Returns:
        X: Feature matrix
        y: Target vector  
        meta: Metadata DataFrame
    """
    print("Preparing basic data...")
    df = prepare_basic(df_matches)
    df = add_result_cols(df)
    
    print("Building rolling features...")
    feat = rolling_team_features_optimized(df, windows=windows, include_h2h=include_h2h)
    
    print("Finalizing feature matrix...")
    
    # Add metadata
    meta_cols = ['match_id', 'date', 'home_team', 'away_team']
    if 'season' in df.columns:
        meta_cols.append('season')
    if 'league' in df.columns:
        meta_cols.append('league')
    
    meta_df = df[meta_cols].copy()
    feat = feat.merge(meta_df, on='match_id', how='left')
    
    # Create target variable
    label_map = df.set_index('match_id')['full_time_result']
    feat['target'] = feat['match_id'].map(label_map)
    feat = feat.dropna(subset=['target'])
    
    # Encode target: H=0, D=1, A=2
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    feat['y'] = feat['target'].map(target_mapping)
    
    # Select feature columns (exclude metadata and targets)
    exclude_cols = ['match_id', 'target', 'y', 'date', 'home_team', 'away_team', 'season', 'league']
    feature_cols = [col for col in feat.columns if col not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} features")
    
    X = feat[feature_cols].copy()
    y = feat['y'].copy()
    meta = feat[meta_cols + ['target', 'y']].copy()
    
    # Enhanced imputation strategy
    X = enhanced_imputation(X)
    
    return X, y, meta


def enhanced_imputation(X: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced imputation strategy that's more sophisticated than simple mean filling.
    """
    X = X.copy()
    
    # Group columns by type for different imputation strategies
    form_cols = [col for col in X.columns if 'form' in col or 'win' in col or 'draw' in col]
    stats_cols = [col for col in X.columns if any(stat in col for stat in ['shots', 'corners', 'goals'])]
    odds_cols = [col for col in X.columns if 'odds' in col or 'prob' in col]
    other_cols = [col for col in X.columns if col not in form_cols + stats_cols + odds_cols]
    
    # Impute form columns with 0.33 (neutral form)
    for col in form_cols:
        X[col] = X[col].fillna(0.33)
    
    # Impute stats columns with 0 (no shots/corners if missing)
    for col in stats_cols:
        X[col] = X[col].fillna(0)
    
    # Impute odds columns with market averages
    odds_defaults = {
        'home_prob_implied': 0.4, 'draw_prob_implied': 0.27, 'away_prob_implied': 0.33,
        'home_odds_value': 0, 'draw_odds_value': 0, 'away_odds_value': 0
    }
    
    for col in odds_cols:
        default_val = odds_defaults.get(col, X[col].median())
        X[col] = X[col].fillna(default_val)
    
    # Impute remaining columns with median
    for col in other_cols:
        X[col] = X[col].fillna(X[col].median())
    
    # Final check: any remaining NaNs get filled with 0
    X = X.fillna(0)
    
    return X


def get_feature_importance_groups() -> Dict[str, list]:
    """
    Return feature groups for analysis and interpretation.
    """
    return {
        'recent_form': [col for col in ['home_form_weighted_5', 'away_form_weighted_5'] if col],
        'historical_form': [col for col in ['home_is_win_avg_5', 'away_is_win_avg_5'] if col],
        'attacking_stats': [col for col in ['home_goals_for_avg_5', 'away_goals_for_avg_5', 
                                          'home_shots_avg_5', 'away_shots_avg_5'] if col],
        'defensive_stats': [col for col in ['home_goals_against_avg_5', 'away_goals_against_avg_5',
                                          'home_clean_sheet_avg_5', 'away_clean_sheet_avg_5'] if col],
        'betting_odds': [col for col in ['home_prob_implied', 'draw_prob_implied', 'away_prob_implied'] if col],
        'head_to_head': [col for col in ['h2h_home_win_rate', 'h2h_draw_rate', 'h2h_avg_goals'] if col],
        'venue_specific': [col for col in ['home_form_home_5', 'away_form_away_5'] if col]
    }