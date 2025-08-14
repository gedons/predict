# app/api/predict.py
import json
import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "artifacts/model_meta_latest.json")  

# module-level globals populated at startup
MODEL = None
PREPROCESSOR = None
MODEL_META = None
BOOSTER = None
SKLEARN_MODEL = None

router = APIRouter()

class MatchRequest(BaseModel):
    home_team: str = Field(..., example="Man United")
    away_team: str = Field(..., example="Fulham")
    match_date: str = Field(..., example="2025-08-16")  # YYYY-MM-DD
    mode: str = Field("auto", description="auto | server | features", example="auto")
    # if mode == "features", supply feature dict below
    features: Dict[str, float] = None

class PredictResponse(BaseModel):
    match_id: str
    features: Dict[str, float]
    probabilities: Dict[str, float]
    implied_odds: Dict[str, float]
    edge: Dict[str, float]
    model_meta: Dict[str, Any]

def load_model_at_startup():
    """
    Robust loader for model_meta + assets (preprocessor & model).
    Tries multiple likely candidate paths (meta dir, meta_dir parent, cwd, cwd/app, etc.)
    and prints helpful diagnostics if nothing is found.
    """
    global MODEL_META, PREPROCESSOR, BOOSTER, SKLEARN_MODEL, MODEL

    meta_path_env = os.getenv("MODEL_META_PATH")
    if meta_path_env and os.path.exists(meta_path_env):
        meta_path = os.path.normpath(meta_path_env)
    else:
        from glob import glob
        candidates_meta = sorted(glob("app/artifacts/model_meta_*.json")) + sorted(glob("artifacts/model_meta_*.json"))
        if not candidates_meta:
            raise RuntimeError(
                "No model_meta JSON found. Set MODEL_META_PATH in .env to the correct path "
                "(e.g. app/artifacts/model_meta_<ts>.json) or place the JSON in app/artifacts/ or artifacts/."
            )
        meta_path = os.path.normpath(candidates_meta[-1])

    meta_dir = os.path.dirname(meta_path)
    meta_parent = os.path.normpath(os.path.join(meta_dir, ".."))
    cwd = os.getcwd()

    with open(meta_path, "r") as f:
        MODEL_META = json.load(f)

    def resolve_asset_path(raw_path: str, meta_dir: str) -> str:
        raw_norm = os.path.normpath(raw_path)
        basename = os.path.basename(raw_norm)

        candidates = []

        # 1) Absolute raw path
        if os.path.isabs(raw_norm):
            candidates.append(raw_norm)

        # 2) meta_dir / raw_path (common)
        candidates.append(os.path.normpath(os.path.join(meta_dir, raw_norm)))
        # 3) meta_dir / basename (if raw_path included "artifacts/" or "models/")
        candidates.append(os.path.normpath(os.path.join(meta_dir, basename)))

        # 4) meta_dir parent (e.g., app/) + raw_path (covers app/models when meta in app/artifacts)
        candidates.append(os.path.normpath(os.path.join(meta_parent, raw_norm)))
        candidates.append(os.path.normpath(os.path.join(meta_parent, basename)))

        # 5) cwd + raw_path (project-root relative)
        candidates.append(os.path.normpath(os.path.join(cwd, raw_norm)))
        candidates.append(os.path.normpath(os.path.join(cwd, basename)))

        # 6) cwd/app + raw_path and cwd/app + basename (cover cases where assets are under app/)
        candidates.append(os.path.normpath(os.path.join(cwd, "app", raw_norm)))
        candidates.append(os.path.normpath(os.path.join(cwd, "app", basename)))

        # 7) finally raw_norm as-is (relative)
        candidates.append(raw_norm)

        # dedupe preserving order
        seen = set(); uniq = []
        for p in candidates:
            if p not in seen:
                uniq.append(p); seen.add(p)

        for p in uniq:
            if os.path.exists(p):
                return p

        # Nothing found — raise diagnostic error
        raise RuntimeError(
            f"Could not resolve asset path for '{raw_path}'.\nTried the following locations:\n" +
            "\n".join(f"  - {p}" for p in uniq) +
            f"\n\nMeta file used: {meta_path}\nCurrent working dir: {cwd}"
        )

    # Resolve preprocessor
    preproc_raw = MODEL_META.get("preprocessor_path")
    if not preproc_raw:
        raise RuntimeError("preprocessor_path not found in model_meta JSON")
    resolved_preproc = resolve_asset_path(preproc_raw, meta_dir)
    PREPROCESSOR = joblib.load(resolved_preproc)

    # Resolve model
    model_raw = MODEL_META.get("model_path")
    if not model_raw:
        raise RuntimeError("model_path not found in model_meta JSON")
    resolved_model = resolve_asset_path(model_raw, meta_dir)

    model_type = MODEL_META.get("model_type", "xgb_booster")
    if model_type == "sklearn_xgb":
        SKLEARN_MODEL = joblib.load(resolved_model)
        MODEL = SKLEARN_MODEL
        BOOSTER = None
    else:
        BOOSTER = xgb.Booster()
        BOOSTER.load_model(resolved_model)
        MODEL = BOOSTER
        SKLEARN_MODEL = None

    print(f"Loaded model: {resolved_model} (type={model_type})")
    return



# ---- helpers to build features from DB or accept client-provided ----

def fetch_recent_team_matches(team: str, cutoff_date: str, limit: int = 10):
    """Fetch last N matches for team before cutoff_date from database (both home and away)."""
    engine = create_engine(DATABASE_URL)
    # we need matches where either home_team or away_team equals team and date < cutoff_date
    sql = text("""
        SELECT date, home_team, away_team,
               full_time_home_goals, full_time_away_goals,
               home_shots, away_shots, home_shots_on_target, away_shots_on_target,
               home_corners, away_corners, full_time_result
        FROM public.matches
        WHERE date < :cutoff
          AND (home_team = :team OR away_team = :team)
        ORDER BY date DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"cutoff": cutoff_date, "team": team, "limit": limit})
    return df

def compute_rolling_features_for_team(team_df: pd.DataFrame, team_name: str, window: int = 5, is_home_perspective=True):
    """Compute aggregates used in training for a single team from its past matches (team_df expected sorted desc)."""
    if team_df is None or team_df.empty:
        # produce NaNs for all features
        return {
            "is_win_avg": np.nan,
            "is_win_form_3": np.nan,
            "goal_diff_avg": np.nan,
            "shots_avg": np.nan,
            "shots_on_target_avg": np.nan,
            "corners_avg": np.nan,
            "attack_strength": np.nan,
            "defense_strength": np.nan
        }
    # convert dates & compute perspective-specific columns
    df = team_df.copy()
    # We need to interpret goals/shots/corners from the perspective of `team_name`
    def get_for(row):
        if row["home_team"] == team_name:
            gf = row["full_time_home_goals"]
            ga = row["full_time_away_goals"]
            shots = row.get("home_shots")
            sots = row.get("home_shots_on_target")
            corners = row.get("home_corners")
            result = row["full_time_result"]
            # result codes: H home win, A away win, D draw
            is_win = 1 if result == "H" else 0
        else:
            gf = row["full_time_away_goals"]
            ga = row["full_time_home_goals"]
            shots = row.get("away_shots")
            sots = row.get("away_shots_on_target")
            corners = row.get("away_corners")
            result = row["full_time_result"]
            is_win = 1 if result == "A" else 0
        return pd.Series({
            "gf": gf, "ga": ga, "goal_diff": gf - ga,
            "shots": shots if pd.notnull(shots) else np.nan,
            "sots": sots if pd.notnull(sots) else np.nan,
            "corners": corners if pd.notnull(corners) else np.nan,
            "is_win": is_win
        })

    stats = df.apply(get_for, axis=1)
    # use last `window` matches (already ordered desc)
    lastk = stats.head(window)
    res = {}
    res["is_win_avg"] = float(lastk["is_win"].mean()) if not lastk.empty else np.nan
    # form over last 3
    res["is_win_form_3"] = float(lastk["is_win"].head(3).mean()) if not lastk.empty else np.nan
    res["goal_diff_avg"] = float(lastk["goal_diff"].mean()) if not lastk.empty else np.nan
    res["shots_avg"] = float(lastk["shots"].mean()) if not lastk.empty else np.nan
    res["shots_on_target_avg"] = float(lastk["sots"].mean()) if not lastk.empty else np.nan
    res["corners_avg"] = float(lastk["corners"].mean()) if not lastk.empty else np.nan
    # attack/defense strength approximated by gf avg and ga avg
    res["attack_strength"] = float(lastk["gf"].mean()) if "gf" in lastk.columns and not lastk.empty else np.nan
    res["defense_strength"] = float(lastk["ga"].mean()) if "ga" in lastk.columns and not lastk.empty else np.nan
    return res

def build_feature_vector_from_db(home_team: str, away_team: str, match_date: str, window: int = 5):
    """Compute the final feature dict exactly matching MODEL_META['features'] order."""
    # query last N matches per team
    home_hist = fetch_recent_team_matches(home_team, cutoff_date=match_date, limit=20)  # get enough history
    away_hist = fetch_recent_team_matches(away_team, cutoff_date=match_date, limit=20)

    # they are ordered DESC; compute using that
    home_feats = compute_rolling_features_for_team(home_hist, home_team, window=window)
    away_feats = compute_rolling_features_for_team(away_hist, away_team, window=window)

    feat = {
        # raw last-match numbers (we also keep current aggregated names)
        "home_shots": home_feats.get("shots_avg"),
        "away_shots": away_feats.get("shots_avg"),
        "home_shots_on_target": home_feats.get("shots_on_target_avg"),
        "away_shots_on_target": away_feats.get("shots_on_target_avg"),
        "home_corners": home_feats.get("corners_avg"),
        "away_corners": away_feats.get("corners_avg"),
        "home_is_win_avg_5": home_feats.get("is_win_avg"),
        "home_is_win_form_3": home_feats.get("is_win_form_3"),
        "home_goal_diff_avg_5": home_feats.get("goal_diff_avg"),
        "home_shots_avg_5": home_feats.get("shots_avg"),
        "home_shots_on_target_avg_5": home_feats.get("shots_on_target_avg"),
        "home_corners_avg_5": home_feats.get("corners_avg"),
        "home_attack_strength_5": home_feats.get("attack_strength"),
        "home_defense_strength_5": home_feats.get("defense_strength"),
        "away_is_win_avg_5": away_feats.get("is_win_avg"),
        "away_is_win_form_3": away_feats.get("is_win_form_3"),
        "away_goal_diff_avg_5": away_feats.get("goal_diff_avg"),
        "away_shots_avg_5": away_feats.get("shots_avg"),
        "away_shots_on_target_avg_5": away_feats.get("shots_on_target_avg"),
        "away_corners_avg_5": away_feats.get("corners_avg"),
        "away_attack_strength_5": away_feats.get("attack_strength"),
        "away_defense_strength_5": away_feats.get("defense_strength"),
    }

    # Add implied odds: try to read latest odds from DB if exist between the two teams on the date,
    # otherwise leave NaN and user may supply them.
    engine = create_engine(DATABASE_URL)
    sql = text("""
       SELECT b365_home_odds, b365_draw_odds, b365_away_odds
       FROM public.matches
       WHERE date < :date
         AND ((home_team = :home AND away_team = :away) OR (home_team = :away AND away_team = :home))
       ORDER BY date DESC
       LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"date": match_date, "home": home_team, "away": away_team}).fetchone()
    if row and row[0] is not None:
        bh, bd, ba = row
        feat["home_prob_implied"] = float(1.0 / bh) if bh else np.nan
        feat["draw_prob_implied"] = float(1.0 / bd) if bd else np.nan
        feat["away_prob_implied"] = float(1.0 / ba) if ba else np.nan
        # normalize
        s = feat["home_prob_implied"] + feat["draw_prob_implied"] + feat["away_prob_implied"]
        if s and s > 0:
            feat["home_prob_implied"] /= s
            feat["draw_prob_implied"] /= s
            feat["away_prob_implied"] /= s
    else:
        feat["home_prob_implied"] = np.nan
        feat["draw_prob_implied"] = np.nan
        feat["away_prob_implied"] = np.nan

    # Derived features used in training
    feat["form_diff"] = (feat.get("home_is_win_avg_5") or 0) - (feat.get("away_is_win_avg_5") or 0)
    feat["strength_diff"] = (feat.get("home_attack_strength_5") or 0) - (feat.get("away_defense_strength_5") or 0)
    feat["total_avg_goals"] = (feat.get("home_attack_strength_5") or 0) + (feat.get("away_attack_strength_5") or 0)

    # Reorder according to MODEL_META.features
    order = MODEL_META["features"]
    final = {k: float(feat.get(k)) if feat.get(k) is not None else np.nan for k in order}
    return final

# ---- core predict helper ----

def predict_from_features_dict(features_dict: Dict[str, float]):
    """
    Accepts a dict with keys exactly equal to MODEL_META['features'] (or at least covering them).
    Returns probability vector and other metadata.
    """
    # Build DataFrame for single row
    feature_names = MODEL_META["features"]
    X_df = pd.DataFrame([features_dict], columns=feature_names)

    # Preprocess (impute/scale) using PREPROCESSOR
    X_prep = PREPROCESSOR.transform(X_df)  # numpy array

    if MODEL_META.get("model_type") == "sklearn_xgb":
        model = SKLEARN_MODEL
        probs = model.predict_proba(X_prep)[0]
    else:
        # booster: use DMatrix created from DataFrame with proper column names
        dmat = xgb.DMatrix(pd.DataFrame(X_prep, columns=feature_names), feature_names=feature_names)
        probs = BOOSTER.predict(dmat)[0]

    # Build implied odds and edge (if implied present)
    implied = {k: float(features_dict.get(k)) for k in ["home_prob_implied", "draw_prob_implied", "away_prob_implied"]}
    # if implied are NaN, set None
    implied = { "home": implied.get("home_prob_implied") or None,
                "draw": implied.get("draw_prob_implied") or None,
                "away": implied.get("away_prob_implied") or None }

    p = {"home": float(probs[0]), "draw": float(probs[1]), "away": float(probs[2])}
    edge = {}
    for k in ("home","draw","away"):
        if implied[k] is None or implied[k] != implied[k]:  # NaN check
            edge[k] = None
        else:
            edge[k] = float(p[k] - implied[k])

    return p, implied, edge

# ---- endpoints ----

@router.post("/match", response_model=PredictResponse)
def predict_match(req: MatchRequest = Body(...)):
    """
    Predict endpoint.
    - mode=auto or server: API computes features from DB then predicts.
    - mode=features: caller supplies features dict (must match MODEL_META['features']).
    """
    if MODEL_META is None or PREPROCESSOR is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    mode = req.mode.lower()
    if mode in ("auto", "server"):
        # compute features server-side
        features_dict = build_feature_vector_from_db(req.home_team, req.away_team, req.match_date)
    elif mode == "features":
        if not req.features:
            raise HTTPException(status_code=400, detail="features payload required when mode='features'")
        # ensure keys map to numeric type
        features_dict = {k: float(v) for k, v in req.features.items()}
        # if keys are not in meta features, attempt to subset or error
        missing = [f for f in MODEL_META["features"] if f not in features_dict]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required feature keys: {missing}")
    else:
        raise HTTPException(status_code=400, detail="mode must be one of: auto|server|features")

    # predict
    probs, implied, edge = predict_from_features_dict(features_dict)

    match_id = f"{req.match_date}_{req.home_team.replace(' ','')}_{req.away_team.replace(' ','')}"
    resp = {
        "match_id": match_id,
        "features": features_dict,
        "probabilities": probs,
        "implied_odds": {"home": implied["home"], "draw": implied["draw"], "away": implied["away"]},
        "edge": edge,
        "model_meta": {
            "created_at": MODEL_META.get("created_at"),
            "model_type": MODEL_META.get("model_type"),
            "n_train": MODEL_META.get("n_train"),
            "n_test": MODEL_META.get("n_test")
        }
    }
    return resp
