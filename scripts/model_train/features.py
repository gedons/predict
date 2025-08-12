# scripts/features.py
import pandas as pd
import numpy as np

def prepare_basic(df):
    # Ensure types
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    # canonicalize team names whitespace
    df['home_team'] = df['home_team'].str.strip()
    df['away_team'] = df['away_team'].str.strip()
    return df

def add_result_cols(df):
    # result: H/D/A (already present as full_time_result) but create numeric for modelling
    df['home_win'] = (df['full_time_result'] == 'H').astype(int)
    df['draw'] = (df['full_time_result'] == 'D').astype(int)
    df['away_win'] = (df['full_time_result'] == 'A').astype(int)
    df['goal_diff_home'] = df['full_time_home_goals'] - df['full_time_away_goals']
    return df

def rolling_team_features(df, window=5, decay=None):
    """
    Compute rolling features per team (form over last N matches).
    returns DataFrame with features for each match (home_..., away_...)
    """
    df = df.copy()
    teams = pd.concat([df[['date','home_team','home_win','goal_diff_home']].rename(columns={'home_team':'team','home_win':'is_win','goal_diff_home':'goal_diff'}),
                       df[['date','away_team','away_win']].rename(columns={'away_team':'team','away_win':'is_win'})],
                      ignore_index=True, sort=False)

    teams.sort_values(['team','date'], inplace=True)

    # compute rolling stats per team: last N matches
    teams['is_win'] = teams['is_win'].fillna(0)
    # rolling count & mean per team
    rolled = teams.groupby('team').rolling(window, on='date', closed='left').agg({
        'is_win': 'mean',
        'goal_diff': 'mean'
    }).reset_index().rename(columns={'is_win':'form_last_%d'%(window),'goal_diff':'avg_goal_diff_last_%d'%(window)})
    # merge rolled back to matches for home and away
    # For performance, create helper dict of last stats by (team,date)
    rolled['date_key'] = rolled['date'].dt.strftime('%Y-%m-%d')
    # create a lookup by team+date to get stats BEFORE that match
    rolled_lookup = rolled.set_index(['team','date_key'])

    def get_team_stats(team, date):
        key = (team, date.strftime('%Y-%m-%d'))
        # find last available rolled stat for this team before date
        # simple approach: take max date < date for team
        # here we will query rolled where team matches and rolled.date < date
        try:
            # find all entries for team with date < date
            s = rolled.loc[(rolled['team'] == team) & (rolled['date'] < date)]
            if s.empty:
                return {'form': np.nan, 'goal_diff': np.nan}
            last = s.iloc[-1]
            return {'form': last[f'form_last_{window}'], 'goal_diff': last[f'avg_goal_diff_last_{window}']}
        except Exception:
            return {'form': np.nan, 'goal_diff': np.nan}

    # Build feature rows
    rows = []
    for _, r in df.iterrows():
        hstats = get_team_stats(r['home_team'], r['date'])
        astats = get_team_stats(r['away_team'], r['date'])
        rows.append({
            'match_id': r['match_id'],
            'date': r['date'],
            'home_form': hstats['form'],
            'away_form': astats['form'],
            'home_avg_goal_diff': hstats['goal_diff'],
            'away_avg_goal_diff': astats['goal_diff'],
            'home_shots': r.get('home_shots'),
            'away_shots': r.get('away_shots'),
            'home_shots_on_target': r.get('home_shots_on_target'),
            'away_shots_on_target': r.get('away_shots_on_target'),
            'home_corners': r.get('home_corners'),
            'away_corners': r.get('away_corners'),
            'b365_home_odds': r.get('b365_home_odds'),
            'b365_draw_odds': r.get('b365_draw_odds'),
            'b365_away_odds': r.get('b365_away_odds'),
        })
    feat = pd.DataFrame(rows)
    # compute implied probabilities from odds (simple)
    for col,o in [('b365_home_odds','home_prob_implied'),('b365_draw_odds','draw_prob_implied'),('b365_away_odds','away_prob_implied')]:
        feat[o] = 1 / feat[col]
    # normalize implied to sum=1 (remove vig approx)
    s = feat[['home_prob_implied','draw_prob_implied','away_prob_implied']].sum(axis=1)
    feat['home_prob_implied'] /= s
    feat['draw_prob_implied'] /= s
    feat['away_prob_implied'] /= s

    return feat

def build_feature_matrix(df_matches):
    df = prepare_basic(df_matches)
    df = add_result_cols(df)
    feat = rolling_team_features(df, window=5)
    # merge label (target) from df
    label_map = df.set_index('match_id')['full_time_result']
    feat['target'] = feat['match_id'].map(label_map)
    # drop rows without target (if predicting future matches we'd avoid)
    feat = feat.dropna(subset=['target'])
    # encode target to ints: H=0, D=1, A=2 (consistent)
    mapping = {'H':0, 'D':1, 'A':2}
    feat['y'] = feat['target'].map(mapping)
    # choose feature columns
    feature_cols = [
        'home_form','away_form',
        'home_avg_goal_diff','away_avg_goal_diff',
        'home_shots','away_shots',
        'home_shots_on_target','away_shots_on_target',
        'home_corners','away_corners',
        'home_prob_implied','draw_prob_implied','away_prob_implied'
    ]
    X = feat[feature_cols]
    y = feat['y']
    # simple imputation: fillna with column mean
    X = X.fillna(X.mean())
    return X, y, feat
