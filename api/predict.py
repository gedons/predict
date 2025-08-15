# app/api/predict.py
import json
import os
import tempfile
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from fastapi import APIRouter, Depends, HTTPException, Body, Path, Request
import requests
from pathlib import Path
from pydantic import BaseModel, Field
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from app.db.database import get_db

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "artifacts/20250814_141549.json")  

# module-level globals populated at startup
MODEL = None
PREPROCESSOR = None
MODEL_META = None
BOOSTER = None
SKLEARN_MODEL = None

router = APIRouter()

class MatchRequest(BaseModel):
    home_team: str = Field(..., example="Man United") # type: ignore
    away_team: str = Field(..., example="Fulham") # type: ignore
    match_date: str = Field(..., example="2025-08-16")  # type: ignore # YYYY-MM-DD
    mode: str = Field("auto", description="auto | server | features", example="auto") # type: ignore
    # if mode == "features", supply feature dict below
    features: Dict[str, float] = None # type: ignore

class PredictResponse(BaseModel):  # type: ignore
    match_id: str
    features: Dict[str, float]
    probabilities: Dict[str, float]
    implied_odds: Dict[str, float]
    edge: Dict[str, float]
    model_meta: Dict[str, Any]

def _download_to_tmp(url: str) -> str:
    """Download a http(s) URL to a temp file and return local path."""    
    parsed = urlparse(url)
    path_for_suffix = parsed.path
    suffix = Path(path_for_suffix).suffix or ".bin"

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(32 * 1024):
            if chunk:
                f.write(chunk)
    return tmp_path

def _resolve_artifact_path(artifact_path: str, meta_dir: str = None) -> str: # type: ignore
    """
    Resolve artifact path:
      - If HTTP/S URL -> download and return local tmp file path
      - If file:// URI -> strip scheme and return local path
      - If relative path -> if meta_dir given, join meta_dir; else join project root
      - If absolute path -> return as is
    """
    if not artifact_path:
        raise RuntimeError("Empty artifact_path")
    parsed = urlparse(artifact_path)
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https"):
        return _download_to_tmp(artifact_path)
    if scheme == "file":
        return parsed.path
    # treat as local path (absolute or relative)
    p = Path(artifact_path)
    if p.is_absolute() and p.exists():
        return str(p)
    # if relative and meta_dir given, try meta_dir/artifact_path
    if meta_dir:
        candidate = Path(meta_dir) / artifact_path
        if candidate.exists():
            return str(candidate)
    # else try project root relative
    root_candidate = Path.cwd() / artifact_path
    if root_candidate.exists():
        return str(root_candidate)
    # not found on disk
    raise RuntimeError(f"Artifact not found or unsupported URI: {artifact_path}")

def load_model_at_startup():
    """
    Robust loader:
      Priority:
        1) Explicit MODEL_META_PATH if set (but prefer artifact_url / preprocessor_url inside it)
        2) DB active model_registry row (artifact_path may be a URL)
        3) Local artifacts/model_meta_*.json fallback
    """
    global MODEL_META, PREPROCESSOR, BOOSTER, SKLEARN_MODEL, MODEL

    MODEL_META = None
    meta_path = None
    meta_dir = None

    # 1) explicit file
    meta_path_env = os.getenv("MODEL_META_PATH")
    if meta_path_env:
        try:
            if Path(meta_path_env).exists():
                with open(meta_path_env, "r", encoding="utf-8") as f:
                    MODEL_META = json.load(f)
                meta_path = str(Path(meta_path_env))
                meta_dir = str(Path(meta_path_env).parent)
                print(f"Loaded model_meta from explicit MODEL_META_PATH: {meta_path}")
            else:
                print(f"MODEL_META_PATH set but file not found: {meta_path_env}")
        except Exception as e:
            print(f"Warning: failed to read MODEL_META_PATH {meta_path_env}: {e}")

    # 2) DB active model (enrich or fill MODEL_META)
    if MODEL_META is None or not any(k in MODEL_META for k in ("artifact_url", "preprocessor_url", "model_path", "artifact_path", "artifact_filename")):
        DATABASE_URL = os.getenv("DATABASE_URL")
        if DATABASE_URL:
            try:
                engine = create_engine(DATABASE_URL)
                with engine.connect() as conn:
                    row = conn.execute(text(
                        "SELECT id, model_name, version, artifact_path, metadata FROM model_registry WHERE is_active = true ORDER BY created_at DESC LIMIT 1"
                    )).fetchone()
                if row:
                    artifact_path_db = row["artifact_path"] if "artifact_path" in row.keys() else row[3] # type: ignore
                    metadata_db = row["metadata"] if "metadata" in row.keys() else row[4] # type: ignore
                    if isinstance(metadata_db, str):
                        try:
                            metadata_db = json.loads(metadata_db)
                        except Exception:
                            metadata_db = {}
                    if MODEL_META is None:
                        MODEL_META = metadata_db or {}
                    else:
                        for k, v in (metadata_db or {}).items():
                            MODEL_META.setdefault(k, v)
                    MODEL_META.setdefault("artifact_path_source", artifact_path_db)
                    print(f"Found active model in DB: artifact_path={artifact_path_db}")
            except Exception as e:
                print(f"Warning: failed to query DB for active model: {e}")

    # 3) local artifacts fallback
    if MODEL_META is None:
        candidates = sorted(Path("app/artifacts").glob("model_meta_*.json")) + sorted(Path("artifacts").glob("model_meta_*.json"))
        if candidates:
            meta_path = str(candidates[-1])
            meta_dir = str(Path(meta_path).parent)
            with open(meta_path, "r", encoding="utf-8") as f:
                MODEL_META = json.load(f)
            print(f"Loaded local model_meta: {meta_path}")
        else:
            raise RuntimeError("No model metadata available from MODEL_META_PATH, DB, or local artifacts.")

    # Ensure dict
    if not isinstance(MODEL_META, dict):
        raise RuntimeError("MODEL_META content is not a JSON object.")

    # Candidate extraction (prefer URLs)
    preproc_url = MODEL_META.get("preprocessor_url") or MODEL_META.get("preprocessorURI") or MODEL_META.get("preprocessorPath")
    preproc_path_raw = MODEL_META.get("preprocessor_path") or MODEL_META.get("preprocessorPath")

    artifact_url = MODEL_META.get("artifact_url") or MODEL_META.get("artifactURL") or MODEL_META.get("artifactPath") or MODEL_META.get("artifact_path_source")
    if not artifact_url:
        artifact_url = MODEL_META.get("model_path") or MODEL_META.get("artifact_path") or MODEL_META.get("artifact_filename")

    resolved_preproc = None
    resolved_model = None

    # Try preprocessor_url then preprocessor_path
    if preproc_url:
        try:
            resolved_preproc = _resolve_artifact_path(preproc_url, meta_dir=meta_dir) # type: ignore
            print(f"Resolved preprocessor from URL: {preproc_url} -> {resolved_preproc}")
        except Exception as e:
            print(f"Warning: failed to resolve preprocessor_url {preproc_url}: {e}")

    if resolved_preproc is None and preproc_path_raw:
        try:
            resolved_preproc = _resolve_artifact_path(preproc_path_raw, meta_dir=meta_dir) # type: ignore
            print(f"Resolved preprocessor from path: {preproc_path_raw} -> {resolved_preproc}")
        except Exception as e:
            print(f"Warning: failed to resolve preprocessor path {preproc_path_raw}: {e}")

    # Try artifact_url first
    if artifact_url:
        try:
            resolved_model = _resolve_artifact_path(artifact_url, meta_dir=meta_dir) # type: ignore
            print(f"Resolved model artifact from: {artifact_url} -> {resolved_model}")
        except Exception as e:
            print(f"Warning: failed to resolve artifact {artifact_url}: {e}")
            resolved_model = None

    # If not resolved, search meta_dir but do NOT match model_meta_*.json
    if resolved_model is None and meta_dir:
        model_patterns = [
            "xgb_booster_*.joblib", "xgb_booster_*.json",
            "xgb_model_*.joblib", "xgb_model_*.json",
            "model_*.joblib", "model_*.json"
        ]
        for patt in model_patterns:
            for candidate in Path(meta_dir).glob(patt):
                # Skip meta files (defensive) and ensure file actually exists
                if "model_meta_" in candidate.name.lower():
                    continue
                if candidate.exists():
                    resolved_model = str(candidate)
                    print(f"Found model artifact next to meta: {resolved_model}")
                    break
            if resolved_model:
                break

    # Also try project-level models/ folder
    if resolved_model is None:
        for candidate in Path("models").glob("*.json"):
            if "model_meta_" in candidate.name.lower():
                continue
            resolved_model = str(candidate)
            print(f"Found model in models/: {resolved_model}")
            break

    # final preprocessor fallback: look next to meta (we did earlier)
    if resolved_preproc is None and meta_dir:
        for candidate in Path(meta_dir).glob("preprocessor_*.joblib"):
            resolved_preproc = str(candidate)
            print(f"Found preprocessor next to meta: {resolved_preproc}")
            break

    # load preprocessor if found
    if resolved_preproc:
        try:
            PREPROCESSOR = joblib.load(resolved_preproc)
            print(f"Loaded preprocessor from: {resolved_preproc}")
        except Exception as e:
            print(f"Warning: failed to load preprocessor {resolved_preproc}: {e}")
            PREPROCESSOR = None
    else:
        print("Preprocessor not resolved/loaded - continuing without it (inference may require it).")

    if not resolved_model:
        raise RuntimeError("Model artifact could not be resolved. Aborting model load.")

    # finally load model
    model_type = MODEL_META.get("model_type", "xgb_booster")
    try:
        if model_type == "sklearn_xgb" or str(resolved_model).endswith(".joblib"):
            SKLEARN_MODEL = joblib.load(resolved_model)
            MODEL = SKLEARN_MODEL
            BOOSTER = None
        else:
            BOOSTER = xgb.Booster()
            BOOSTER.load_model(resolved_model)
            MODEL = BOOSTER
            SKLEARN_MODEL = None
        print(f"Loaded model artifact: {resolved_model} (type={model_type})")
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifact {resolved_model}: {e}")

def fetch_recent_team_matches(team: str, cutoff_date: str, limit: int = 10):
    """Fetch last N matches for team before cutoff_date from database (both home and away)."""
    engine = create_engine(DATABASE_URL) # type: ignore
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
    engine = create_engine(DATABASE_URL) # type: ignore # type: ignore
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
    order = MODEL_META["features"] # type: ignore
    final = {k: float(feat.get(k)) if feat.get(k) is not None else np.nan for k in order} # type: ignore
    return final

def log_prediction(db_conn, resp: dict, model_id: Optional[int] = None, client_ip: Optional[str] = None, user_id: Optional[str] = None): # type: ignore
    """
    Insert a prediction log into prediction_logs table using an existing DB connection.
    db_conn: a SQLAlchemy Connection or Session that supports execute()
    resp: the response dict we return to the client
    """
    try:
        insert_sql = text("""
            INSERT INTO prediction_logs
              (match_id, features, probabilities, implied_odds, edge, model_id, user_id, client_ip)
            VALUES
              (:match_id, :features, :probs, :implied, :edge, :model_id, :user_id, :client_ip)
        """)
        db_conn.execute(insert_sql, {
            "match_id": resp.get("match_id"),
            "features": json.dumps(resp.get("features", {})),
            "probs": json.dumps(resp.get("probabilities", {})),
            "implied": json.dumps(resp.get("implied_odds", {})),
            "edge": json.dumps(resp.get("edge", {})),
            "model_id": model_id,
            "user_id": user_id,
            "client_ip": client_ip
        })
        # commit if using Connection
        try:
            db_conn.commit()
        except Exception:
            # if db_conn is a raw Connection in SQLAlchemy, commit may be on the transaction context;
            pass
    except Exception as e:
        # never block prediction on logging failure; log locally
        print("Warning: failed to write prediction log:", e)
        
def predict_from_features_dict(features_dict: Dict[str, float]):
    """
    Accepts a dict with keys exactly equal to MODEL_META['features'] (or at least covering them).
    Returns probability vector and other metadata.
    """
    # Build DataFrame for single row
    feature_names = MODEL_META["features"] # pyright: ignore[reportOptionalSubscript]
    X_df = pd.DataFrame([features_dict], columns=feature_names)

    # Preprocess (impute/scale) using PREPROCESSOR
    X_prep = PREPROCESSOR.transform(X_df)  # type: ignore 

    if MODEL_META.get("model_type") == "sklearn_xgb":  # type: ignore 
        model = SKLEARN_MODEL
        probs = model.predict_proba(X_prep)[0]  # type: ignore 
    else:
        # booster: use DMatrix created from DataFrame with proper column names
        dmat = xgb.DMatrix(pd.DataFrame(X_prep, columns=feature_names), feature_names=feature_names)
        probs = BOOSTER.predict(dmat)[0]  # type: ignore 

    # Build implied odds and edge (if implied present)
    implied = {k: float(features_dict.get(k)) for k in ["home_prob_implied", "draw_prob_implied", "away_prob_implied"]}  # type: ignore 
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
            edge[k] = float(p[k] - implied[k])  # type: ignore 

    return p, implied, edge


# ---- endpoints ----

@router.post("/match", response_model=PredictResponse)
def predict_match(req: MatchRequest = Body(...), request: Request = None, db = Depends(get_db)): # type: ignore
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

        # try to find active model id in DB (optional)
    model_id = None
    try:
        row = db.execute(text("SELECT id FROM model_registry WHERE is_active = true ORDER BY created_at DESC LIMIT 1")).fetchone()
        if row:
            # row may be tuple-like or mapping
            try:
                model_id = int(row['id'])
            except Exception:
                model_id = int(row[0])
    except Exception as e:
        # ignore DB read failure (we still want to return prediction)
        print("Warning: cannot read active model from DB:", e)

    # client IP
    client_ip = None
    try:
        client_ip = request.client.host if request and request.client else None
    except Exception:
        client_ip = None

    # optional: user_id if you have authentication. We'll leave None for now.
    user_id = None

    # Log prediction (best-effort)
    try:
        log_prediction(db, resp, model_id=model_id, client_ip=client_ip, user_id=user_id) # type: ignore
    except Exception as e:
        print("Warning: log_prediction raised:", e)

    return resp
