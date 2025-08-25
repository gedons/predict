# app/api/predict.py
import json
import math
import os
import tempfile
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from fastapi import APIRouter, Depends, HTTPException, Body, Path, Request
import requests
import traceback
from pathlib import Path
from pydantic import BaseModel, Field
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from app.db.database import get_db
from fastapi import Depends, Request
from app.middleware.rate_limiter import limiter
from slowapi.util import get_remote_address
from app.core.auth import get_current_user  
rate_limit_decorator = limiter
from app.core.quota import quota_dependency

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_META_PATH = os.getenv("MODEL_META_PATH")

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
    features: Optional[Dict[str, Optional[float]]] = None

class PredictResponse(BaseModel):
    match_id: str
    # features may contain None for missing values
    features: Dict[str, Optional[float]]
    # probabilities are always floats produced by model
    probabilities: Dict[str, float]
    # implied odds may be missing -> Optional[float]
    implied_odds: Dict[str, Optional[float]]
    # edge may be computed or None when implied missing
    edge: Dict[str, Optional[float]]
    model_meta: Dict[str, Any]


def sanitize_for_json(obj):
    """
    Recursively convert values to JSON-safe Python types:
      - numpy/pandas NaN/NA -> None
      - numpy ints/floats -> int/float
      - other problematic types -> str()
    Returns a new object (doesn't mutate original).
    """
    # None
    if obj is None:
        return None

    # numpy/pandas scalars & floats
    if isinstance(obj, (np.floating, float)):
        if math.isnan(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    # Pandas NA / NA-like
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # dict
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]

    # bytes -> decode
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)

    # datetimes -> iso
    try:
        import datetime
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
    except Exception:
        pass

    # booleans
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    # other numpy numbers
    try:
        if isinstance(obj, np.number):
            return float(obj)
    except Exception:
        pass

    if isinstance(obj, (str, bool)):
        return obj

    # fallback
    try:
        return float(obj) if isinstance(obj, (int, float)) else obj
    except Exception:
        return str(obj)


def safe_float(val, default=0.0):
    """Return float(val) or default if NaN/None/uncoercible."""
    try:
        if val is None:
            return float(default)
        if isinstance(val, (np.floating, float)):
            if math.isnan(float(val)):
                return float(default)
            return float(val)
        return float(val)
    except Exception:
        return float(default)


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


def _resolve_artifact_path(artifact_path: str, meta_dir: str = None) -> str:  # type: ignore
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
                    )).mappings().fetchone()
                if row:
                    artifact_path_db = row.get("artifact_path") or row.get("artifact_path", None)
                    metadata_db = row.get("metadata", None)
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
            resolved_preproc = _resolve_artifact_path(preproc_url, meta_dir=meta_dir)  # type: ignore
            print(f"Resolved preprocessor from URL: {preproc_url} -> {resolved_preproc}")
        except Exception as e:
            print(f"Warning: failed to resolve preprocessor_url {preproc_url}: {e}")

    if resolved_preproc is None and preproc_path_raw:
        try:
            resolved_preproc = _resolve_artifact_path(preproc_path_raw, meta_dir=meta_dir)  # type: ignore
            print(f"Resolved preprocessor from path: {preproc_path_raw} -> {resolved_preproc}")
        except Exception as e:
            print(f"Warning: failed to resolve preprocessor path {preproc_path_raw}: {e}")

    # Try artifact_url first
    if artifact_url:
        try:
            resolved_model = _resolve_artifact_path(artifact_url, meta_dir=meta_dir)  # type: ignore
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


def load_model_by_id(model_id: int):
    """
    Load model by id from model_registry (DB). This will:
      - query model_registry for the record (using mappings() so columns accessible by name),
      - resolve artifact_url / preprocessor_url or find local files,
      - set MODEL_META and load PREPROCESSOR and MODEL into global state.
    Returns True on success, raises RuntimeError on failure.
    """
    global MODEL_META, PREPROCESSOR, BOOSTER, SKLEARN_MODEL, MODEL

    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured, cannot load model by id")

    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, model_name, artifact_path, metadata FROM model_registry WHERE id = :id"),
            {"id": model_id}
        ).mappings().fetchone()

    if not row:
        raise RuntimeError(f"Model id {model_id} not found in registry")

    artifact_path = row.get("artifact_path")
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    MODEL_META = metadata or {}
    MODEL_META["artifact_path_source"] = artifact_path

    meta_dir = None
    preproc_path_raw = MODEL_META.get("preprocessor_path") or MODEL_META.get("preprocessor_url")
    artifact_raw = MODEL_META.get("artifact_url") or MODEL_META.get("artifact_path_source") or MODEL_META.get("model_path")

    resolved_preproc = None
    resolved_model = None

    # try resolve preprocessor URL/path first
    if preproc_path_raw:
        try:
            resolved_preproc = _resolve_artifact_path(preproc_path_raw, meta_dir=meta_dir)  # type: ignore
        except Exception as e:
            print(f"load_model_by_id: failed to resolve preprocessor: {e}")

    # try resolve artifact_raw as URL/path
    if artifact_raw:
        try:
            resolved_model = _resolve_artifact_path(artifact_raw, meta_dir=meta_dir)  # type: ignore
        except Exception as e:
            print(f"load_model_by_id: failed to resolve model artifact: {e}")
            resolved_model = None

    # fallback: try to find preprocessor next to local artifacts
    if resolved_preproc is None:
        for candidate in Path("app/artifacts").glob("preprocessor_*.joblib"):
            resolved_preproc = str(candidate)
            break

    if resolved_preproc:
        try:
            PREPROCESSOR = joblib.load(resolved_preproc)
            print(f"Loaded preprocessor for model_id {model_id} from {resolved_preproc}")
        except Exception as e:
            print(f"load_model_by_id: failed to load preprocessor {resolved_preproc}: {e}")
            PREPROCESSOR = None

    # fallback: search project for model files if not resolved
    if not resolved_model:
        candidates = list(Path("app/models").glob("*")) + list(Path("models").glob("*")) + list(Path("app/artifacts").glob("*"))
        for c in candidates:
            name = c.name.lower()
            if "model_meta_" in name:
                continue
            if c.exists() and c.is_file():
                if name.endswith(".json") or name.endswith(".joblib") or name.endswith(".bin"):
                    resolved_model = str(c)
                    break

    if not resolved_model:
        raise RuntimeError(f"Could not resolve model artifact for model_id {model_id} (artifact_path={artifact_path})")

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
        print(f"Loaded model id={model_id} from {resolved_model} (type={model_type})")
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifact {resolved_model}: {e}")


def fetch_recent_team_matches(team: str, cutoff_date: str, limit: int = 10):
    """Fetch last N matches for team before cutoff_date from database (both home and away)."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    engine = create_engine(DATABASE_URL)  # type: ignore
    sql = text("""
        SELECT date, home_team, away_team,
               full_time_home_goals, full_time_away_goals,
               home_shots, away_shots, home_shots_on_target, away_shots_on_target,
               home_corners, away_corners, full_time_result,
               b365_home_odds, b365_draw_odds, b365_away_odds
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
    """
    Compute aggregates used in training for a single team from its past matches.
    team_df expected ordered DESC (most recent first).
    Returns Python-native floats or None (no np.nan).
    """
    if team_df is None or team_df.empty:
        return {
            "is_win_avg": None,
            "is_win_form_3": None,
            "goal_diff_avg": None,
            "shots_avg": None,
            "shots_on_target_avg": None,
            "corners_avg": None,
            "attack_strength": None,
            "defense_strength": None
        }

    df = team_df.copy()

    def row_perspective(row):
        if row.get("home_team") == team_name:
            gf = row.get("full_time_home_goals")
            ga = row.get("full_time_away_goals")
            shots = row.get("home_shots")
            sots = row.get("home_shots_on_target")
            corners = row.get("home_corners")
            result = row.get("full_time_result")
            is_win = 1 if result == "H" else 0
        else:
            gf = row.get("full_time_away_goals")
            ga = row.get("full_time_home_goals")
            shots = row.get("away_shots")
            sots = row.get("away_shots_on_target")
            corners = row.get("away_corners")
            result = row.get("full_time_result")
            is_win = 1 if result == "A" else 0
        gf = pd.to_numeric(gf, errors="coerce")
        ga = pd.to_numeric(ga, errors="coerce")
        shots = pd.to_numeric(shots, errors="coerce")
        sots = pd.to_numeric(sots, errors="coerce")
        corners = pd.to_numeric(corners, errors="coerce")
        return {
            "gf": gf,
            "ga": ga,
            "goal_diff": (gf - ga) if (pd.notna(gf) and pd.notna(ga)) else np.nan,
            "shots": shots,
            "sots": sots,
            "corners": corners,
            "is_win": is_win
        }

    stats_df = df.apply(lambda r: pd.Series(row_perspective(r)), axis=1)
    lastk = stats_df.head(window)

    def col_mean_safe(series):
        if series is None or series.empty:
            return None
        val = series.mean(skipna=True)
        if pd.isna(val):
            return None
        return float(val)

    res = {
        "is_win_avg": col_mean_safe(lastk["is_win"]),
        "is_win_form_3": col_mean_safe(lastk["is_win"].head(3)),
        "goal_diff_avg": col_mean_safe(lastk["goal_diff"]),
        "shots_avg": col_mean_safe(lastk["shots"]),
        "shots_on_target_avg": col_mean_safe(lastk["sots"]),
        "corners_avg": col_mean_safe(lastk["corners"]),
        "attack_strength": col_mean_safe(lastk["gf"]) if "gf" in lastk.columns else None,
        "defense_strength": col_mean_safe(lastk["ga"]) if "ga" in lastk.columns else None
    }
    return res


def build_feature_vector_from_db(home_team: str, away_team: str, match_date: str, window: int = 5):
    """
    Compute the final feature dict exactly matching MODEL_META['features'] order.
    Returns a dict of Python floats or None (no numpy NaN).
    """
    home_hist = fetch_recent_team_matches(home_team, cutoff_date=match_date, limit=20)
    away_hist = fetch_recent_team_matches(away_team, cutoff_date=match_date, limit=20)

    home_feats = compute_rolling_features_for_team(home_hist, home_team, window=window)
    away_feats = compute_rolling_features_for_team(away_hist, away_team, window=window)

    feat = {
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

    # implied odds retrieval
    bh = bd = ba = None
    if DATABASE_URL:
        engine = create_engine(DATABASE_URL)  # type: ignore
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
        if row:
            try:
                bh = row["b365_home_odds"] # type: ignore
                bd = row["b365_draw_odds"] # type: ignore
                ba = row["b365_away_odds"] # type: ignore
            except Exception:
                try:
                    bh, bd, ba = row[0], row[1], row[2]
                except Exception:
                    bh = bd = ba = None

    def odds_to_prob(x):
        try:
            if x is None:
                return None
            xv = float(x)
            if xv == 0:
                return None
            return float(1.0 / xv)
        except Exception:
            return None

    feat["home_prob_implied"] = odds_to_prob(bh)
    feat["draw_prob_implied"] = odds_to_prob(bd)
    feat["away_prob_implied"] = odds_to_prob(ba)

    ph = feat.get("home_prob_implied")
    pd_raw = feat.get("draw_prob_implied")
    pa = feat.get("away_prob_implied")
    if ph is not None and pd_raw is not None and pa is not None:
        s = ph + pd_raw + pa
        if s > 0:
            feat["home_prob_implied"] = ph / s
            feat["draw_prob_implied"] = pd_raw / s
            feat["away_prob_implied"] = pa / s

    # Derived features
    feat["form_diff"] = None
    if feat.get("home_is_win_avg_5") is not None or feat.get("away_is_win_avg_5") is not None:
        h = feat.get("home_is_win_avg_5") or 0.0
        a = feat.get("away_is_win_avg_5") or 0.0
        feat["form_diff"] = float(h - a)

    feat["strength_diff"] = None
    if feat.get("home_attack_strength_5") is not None or feat.get("away_defense_strength_5") is not None:
        ha = feat.get("home_attack_strength_5") or 0.0
        ad = feat.get("away_defense_strength_5") or 0.0
        feat["strength_diff"] = float(ha - ad)

    feat["total_avg_goals"] = None
    if feat.get("home_attack_strength_5") is not None or feat.get("away_attack_strength_5") is not None:
        ha2 = feat.get("home_attack_strength_5") or 0.0
        aa2 = feat.get("away_attack_strength_5") or 0.0
        feat["total_avg_goals"] = float(ha2 + aa2)

    # Reorder according to MODEL_META.features (ensure keys exist)
    order = MODEL_META.get("features") if MODEL_META else list(feat.keys())
    final = {}
    for k in order:  # type: ignore
        v = feat.get(k)
        if v is None:
            final[k] = None
        else:
            try:
                final[k] = float(v)
            except Exception:
                final[k] = None

    final_safe = sanitize_for_json(final)
    return final_safe


def log_prediction(db_conn, resp: dict, model_id: Optional[int] = None, client_ip: Optional[str] = None, user_id: Optional[str] = None):
    """
    Insert a prediction log into prediction_logs table using an existing DB connection.
    db_conn: SQLAlchemy Session or Connection from get_db()
    resp: the response dict we return to the client
    """
    try:
        features_safe = sanitize_for_json(resp.get("features", {}))
        probs_safe = sanitize_for_json(resp.get("probabilities", {}))
        implied_safe = sanitize_for_json(resp.get("implied_odds", {}))
        edge_safe = sanitize_for_json(resp.get("edge", {}))

        features_json = json.dumps(features_safe)
        probs_json = json.dumps(probs_safe)
        implied_json = json.dumps(implied_safe)
        edge_json = json.dumps(edge_safe)

        insert_sql = text("""
            INSERT INTO prediction_logs
              (match_id, features, probabilities, implied_odds, edge, model_id, user_id, client_ip)
            VALUES
              (:match_id, :features, :probs, :implied, :edge, :model_id, :user_id, :client_ip)
        """)

        res = db_conn.execute(insert_sql, {
            "match_id": resp.get("match_id"),
            "features": features_json,
            "probs": probs_json,
            "implied": implied_json,
            "edge": edge_json,
            "model_id": model_id,
            "user_id": user_id,
            "client_ip": client_ip
        })

        try:
            db_conn.commit()
        except Exception:
            pass

    except Exception as e:
        print("Warning: failed to write prediction log:", e)


def predict_from_features_dict(features_dict: Dict[str, float]):
    """
    Accepts a dict with keys exactly equal to MODEL_META['features'] (or at least covering them).
    Returns probability dict (home,draw,away), implied dict and edge dict (all sanitized).
    """
    feature_names = MODEL_META["features"]  # type: ignore
    row = {fn: features_dict.get(fn) for fn in feature_names}
    X_df = pd.DataFrame([row], columns=feature_names)

    X_prep = PREPROCESSOR.transform(X_df)  # type: ignore

    if MODEL_META.get("model_type") == "sklearn_xgb":  # type: ignore
        model = SKLEARN_MODEL
        probs_arr = model.predict_proba(X_prep)[0]  # type: ignore
    else:
        dmat = xgb.DMatrix(pd.DataFrame(X_prep, columns=feature_names), feature_names=feature_names)
        probs_arr = BOOSTER.predict(dmat)[0]  # type: ignore

    probs = {"home": float(probs_arr[0]), "draw": float(probs_arr[1]), "away": float(probs_arr[2])}

    implied_home = features_dict.get("home_prob_implied")
    implied_draw = features_dict.get("draw_prob_implied")
    implied_away = features_dict.get("away_prob_implied")

    implied = {
        "home": sanitize_for_json(implied_home),
        "draw": sanitize_for_json(implied_draw),
        "away": sanitize_for_json(implied_away)
    }

    # compute edge: if implied is None, produce None for edge
    def compute_edge(p_val, implied_val):
        if implied_val is None:
            return None
        return float(p_val - float(implied_val))

    edge_calc = {
        "home": compute_edge(probs["home"], implied["home"]),
        "draw": compute_edge(probs["draw"], implied["draw"]),
        "away": compute_edge(probs["away"], implied["away"])
    }

    probs_safe = sanitize_for_json(probs)
    implied_safe = sanitize_for_json(implied)
    edge_safe = sanitize_for_json(edge_calc)

    return probs_safe, implied_safe, edge_safe


# ---- endpoints ----
@router.post("/match", response_model=PredictResponse)
@rate_limit_decorator.limit("120/minute")  # type: ignore
def predict_match(req: MatchRequest = Body(...), request: Request = None, current_user: Dict[str, Any] = Depends(get_current_user), db=Depends(get_db), _quota = Depends(quota_dependency("predict_match"))):  # type: ignore
    """
    Predict endpoint.
    - mode=auto or server: API computes features from DB then predicts.
    - mode=features: caller supplies features dict (must match MODEL_META['features']).
    """
    if MODEL_META is None or PREPROCESSOR is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    mode = (req.mode or "auto").lower()
    if mode in ("auto", "server"):
        features_dict = build_feature_vector_from_db(req.home_team, req.away_team, req.match_date)
    elif mode == "features":
        if not req.features:
            raise HTTPException(status_code=400, detail="features payload required when mode='features'")
        # allow optional numeric values, coerce where possible
        features_dict = {k: (None if v is None else float(v)) for k, v in req.features.items()}
        missing = [f for f in MODEL_META["features"] if f not in features_dict]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required feature keys: {missing}")
    else:
        raise HTTPException(status_code=400, detail="mode must be one of: auto|server|features")

    probs, implied, edge = predict_from_features_dict(features_dict)  # type: ignore

    match_id = f"{req.match_date}_{req.home_team.replace(' ', '')}_{req.away_team.replace(' ', '')}"
    # sanitize the pieces before returning
    features_safe = sanitize_for_json(features_dict)
    implied_safe = sanitize_for_json(implied)
    edge_safe = sanitize_for_json(edge)
    probs_safe = sanitize_for_json(probs)

    resp = {
        "match_id": match_id,
        "features": features_safe,
        "probabilities": probs_safe,
        "implied_odds": {"home": implied_safe.get("home"), "draw": implied_safe.get("draw"), "away": implied_safe.get("away")}, # type: ignore
        "edge": {"home": edge_safe.get("home"), "draw": edge_safe.get("draw"), "away": edge_safe.get("away")}, # type: ignore
        "model_meta": {
            "created_at": MODEL_META.get("created_at"),
            "model_type": MODEL_META.get("model_type"),
            "n_train": MODEL_META.get("n_train"),
            "n_test": MODEL_META.get("n_test")
        }
    }

    # find active model id (best-effort)
    model_id = None
    try:
        row = db.execute(text("SELECT id FROM model_registry WHERE is_active = true ORDER BY created_at DESC LIMIT 1")).fetchone()
        if row:
            try:
                model_id = int(row["id"])
            except Exception:
                model_id = int(row[0])
    except Exception as e:
        print("Warning: cannot read active model from DB:", e)

    client_ip = None
    try:
        client_ip = request.client.host if request and request.client else None
    except Exception:
        client_ip = None

    user_id = None

    try:
        log_prediction(db, resp, model_id=model_id, client_ip=client_ip, user_id=user_id)  # type: ignore
    except Exception as e:
        print("Warning: log_prediction raised:", e)

    return resp
