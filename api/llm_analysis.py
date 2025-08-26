# app/api/llm_analysis.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Dict, Any, Optional, Union, Tuple
from starlette.requests import Request as StarletteRequest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.db.database import get_db
from app.services.context_builder import build_match_context
from app.services.llm_service import call_gemini, LLMServiceError
import json
import time
import traceback
import logging
from pydantic import BaseModel, Field
from datetime import datetime
from app.core.auth import get_current_user, admin_required
from app.middleware.rate_limiter import limiter as rate_limiter  # use limiter instance
from app.core.quota import quota_dependency
from app.core.analytics import capture_event


import joblib
import xgboost as xgb
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict_enriched"])


class MatchRequest(BaseModel):
    home_team: str = Field(..., min_length=1, max_length=100)
    away_team: str = Field(..., min_length=1, max_length=100)
    date: Optional[str] = Field(None, description="Match date in YYYY-MM-DD format")
    league: Optional[str] = Field(None, max_length=100)
    venue: Optional[str] = Field(None, max_length=100)
    match_id: Optional[str] = Field(None, max_length=200)
    extra_context: Optional[str] = Field(None, max_length=2000)
    mode: Optional[str] = Field("auto", description="Prediction mode: auto, server, or features")
    features: Optional[Dict[str, Any]] = Field(None, description="Feature vector when mode=features")
    match_date: Optional[str] = Field(None, description="Alias for date field for compatibility")


class PredictionResponse(BaseModel):
    model: Dict[str, Any]
    llm_raw: str
    llm_parsed: Dict[str, Any]
    cached: bool
    processing_time_ms: int
    error: Optional[str] = None


# Try to import existing predict function with better error handling
predict_match_helper = None
try:
    from app.api.predict import predict_match as predict_match_helper  # type: ignore
    logger.info("Successfully imported predict_match helper")
except ImportError as e:
    logger.warning(f"Could not import predict_match helper: {e}")
except Exception as e:
    logger.error(f"Unexpected error importing predict_match helper: {e}")


def _get_base_prediction(match_request: Union[Dict[str, Any], MatchRequest], request: Request = None) -> Dict[str, Any]: # type: ignore
    """
    Get base prediction without going through the rate-limited endpoint.
    This directly calls the prediction logic to avoid dependency injection issues.
    """
    try:
        # Create compatible object if a dict was passed
        if isinstance(match_request, dict):
            request_obj = MatchRequest(
                home_team=match_request.get("home_team", ""),
                away_team=match_request.get("away_team", ""),
                date=match_request.get("date"),
                league=match_request.get("league"),
                venue=match_request.get("venue"),
                match_id=match_request.get("match_id"),
                extra_context=match_request.get("extra_context"),
                mode=match_request.get("mode", "auto"),
                features=match_request.get("features"),
                match_date=match_request.get("match_date") or match_request.get("date")
            )
        else:
            request_obj = match_request
            if not request_obj.match_date:
                request_obj.match_date = request_obj.date

        # Instead of calling the endpoint, let's import and use the underlying prediction logic
        try:
            # Import the necessary components from the predict module
            from app.api.predict import (
                MODEL_META, PREPROCESSOR, MODEL, 
                build_feature_vector_from_db, 
                predict_from_features_dict,
                sanitize_for_json
            )
            
            if MODEL_META is None or PREPROCESSOR is None or MODEL is None:
                raise HTTPException(status_code=503, detail="Model not loaded yet")

            # Handle different modes
            mode = (request_obj.mode or "auto").lower()
            if mode in ("auto", "server"):
                features_dict = build_feature_vector_from_db(
                    request_obj.home_team, 
                    request_obj.away_team, 
                    request_obj.match_date # type: ignore
                )
            elif mode == "features":
                if not request_obj.features:
                    raise HTTPException(status_code=400, detail="features payload required when mode='features'")
                features_dict = {k: (None if v is None else float(v)) for k, v in request_obj.features.items()}
                missing = [f for f in MODEL_META["features"] if f not in features_dict]
                if missing:
                    raise HTTPException(status_code=400, detail=f"Missing required feature keys: {missing}")
            else:
                raise HTTPException(status_code=400, detail="mode must be one of: auto|server|features")

            # Get predictions
            probs, implied, edge = predict_from_features_dict(features_dict) # type: ignore
            
            # Create match_id
            match_id = f"{request_obj.match_date}_{request_obj.home_team.replace(' ', '')}_{request_obj.away_team.replace(' ', '')}"
            
            # Sanitize outputs
            features_safe = sanitize_for_json(features_dict)
            implied_safe = sanitize_for_json(implied)
            edge_safe = sanitize_for_json(edge)
            probs_safe = sanitize_for_json(probs)

            # Build response in the same format as the endpoint
            base = {
                "match_id": match_id,
                "features": features_safe,
                "probabilities": probs_safe,
                "implied_odds": {
                    "home": implied_safe.get("home"), # type: ignore
                    "draw": implied_safe.get("draw"), # type: ignore
                    "away": implied_safe.get("away") # type: ignore
                },
                "edge": {
                    "home": edge_safe.get("home"), # type: ignore
                    "draw": edge_safe.get("draw"), # type: ignore
                    "away": edge_safe.get("away") # type: ignore
                },
                "model_meta": {
                    "created_at": MODEL_META.get("created_at"),
                    "model_type": MODEL_META.get("model_type"),
                    "n_train": MODEL_META.get("n_train"),
                    "n_test": MODEL_META.get("n_test")
                }
            }

            logger.info("Successfully generated prediction using direct model access")
            return base

        except ImportError as e:
            logger.error(f"Failed to import prediction components: {e}")
            # Fall back to mock data if imports fail
            logger.warning("Using mock prediction data due to import failure")
            return {
                "features": {
                    "home_shots_avg_5": 12.5,
                    "away_shots_avg_5": 10.2,
                    "form_diff": 0.3,
                    "home_goals_avg_5": 1.8,
                    "away_goals_avg_5": 1.2
                },
                "probabilities": {
                    "home": 0.45,
                    "draw": 0.30,
                    "away": 0.25
                },
                "model_meta": {
                    "model_version": "1.0",
                    "features_used": 15,
                    "confidence_score": 0.78
                }
            }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



def _validate_base_prediction(base: Dict[str, Any]) -> None:
    if not isinstance(base, dict):
        raise HTTPException(status_code=500, detail="Prediction must return a dictionary")

    probs = base.get("probabilities", {})
    if not probs:
        raise HTTPException(status_code=500, detail="Prediction returned no probabilities")

    required_outcomes = ["home", "draw", "away"]
    for outcome in required_outcomes:
        if outcome not in probs:
            raise HTTPException(status_code=500, detail=f"Missing probability for outcome: {outcome}")
        if not isinstance(probs[outcome], (int, float)) or probs[outcome] < 0 or probs[outcome] > 1:
            raise HTTPException(status_code=500, detail=f"Invalid probability for {outcome}: {probs[outcome]}")

    total_prob = sum(probs.values())
    if abs(total_prob - 1.0) > 0.1:
        logger.warning(f"Probabilities don't sum to 1.0: {total_prob}")


def _generate_match_id(match_request: Union[Dict[str, Any], MatchRequest]) -> str:
    if isinstance(match_request, MatchRequest):
        match_dict = match_request.model_dump()
    else:
        match_dict = match_request

    if match_dict.get("match_id"):
        return str(match_dict["match_id"])

    date = match_dict.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
    home = match_dict.get("home_team", "unknown")
    away = match_dict.get("away_team", "unknown")

    home_clean = "".join(c for c in home if c.isalnum())[:20]
    away_clean = "".join(c for c in away if c.isalnum())[:20]

    return f"{date}_{home_clean}_{away_clean}"


def _create_cache_table(db) -> bool:
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS llm_analysis_cache (
                id SERIAL PRIMARY KEY,
                match_id TEXT UNIQUE,
                model_meta JSONB,
                model_probs JSONB,
                llm_raw TEXT,
                llm_parsed JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_llm_cache_match_id ON llm_analysis_cache(match_id)"))
        db.commit()
        return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to create cache table: {e}")
        db.rollback()
        return False


def _get_cached_analysis(db, match_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        query = text("""
            SELECT llm_raw, llm_parsed
            FROM llm_analysis_cache
            WHERE match_id = :mid
            ORDER BY created_at DESC
            LIMIT 1
        """)
        result = db.execute(query, {"mid": match_id}).fetchone()
        if result and result[0]:
            llm_raw = result[0]
            llm_parsed = result[1] if result[1] else {"text": llm_raw}
            logger.info(f"Retrieved cached analysis for match_id: {match_id}")
            return llm_raw, llm_parsed
        return None
    except SQLAlchemyError as e:
        logger.error(f"Failed to retrieve cached analysis: {e}")
        return None


def _save_to_cache(db, match_id: str, base: Dict[str, Any], llm_raw: str, llm_parsed: Dict[str, Any]) -> bool:
    try:
        upsert_query = text("""
            INSERT INTO llm_analysis_cache (match_id, model_meta, model_probs, llm_raw, llm_parsed, updated_at)
            VALUES (:mid, :meta, :probs, :raw, :parsed, CURRENT_TIMESTAMP)
            ON CONFLICT (match_id)
            DO UPDATE SET
                model_meta = EXCLUDED.model_meta,
                model_probs = EXCLUDED.model_probs,
                llm_raw = EXCLUDED.llm_raw,
                llm_parsed = EXCLUDED.llm_parsed,
                updated_at = CURRENT_TIMESTAMP
        """)
        db.execute(upsert_query, {
            "mid": match_id,
            "meta": json.dumps(base.get("model_meta", {})),
            "probs": json.dumps(base.get("probabilities", {})),
            "raw": llm_raw,
            "parsed": json.dumps(llm_parsed if isinstance(llm_parsed, dict) else {"text": str(llm_parsed)})
        })
        db.commit()
        logger.info(f"Saved analysis to cache for match_id: {match_id}")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to save to cache: {e}")
        db.rollback()
        return False


@router.post("/enriched", response_model=PredictionResponse)
@rate_limiter.limit("60/minute")  # type: ignore
def predict_enriched(
    request: Request,
    match_request: MatchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db = Depends(get_db),
    use_cache: bool = True,
    _quota = Depends(quota_dependency("predict_enriched")),
    force_refresh: bool = False
):
    """
    Enhanced endpoint that combines model predictions with LLM analysis.
    - `request` is required for slowapi rate-limiter.
    - `current_user` is DB-validated user dict.
    - `db` is the DB session/connection.
    """
    start_time = datetime.utcnow()
    try:
        logger.info(f"Getting base prediction for {match_request.home_team} vs {match_request.away_team}")
        base = _get_base_prediction(match_request, request)
        _validate_base_prediction(base)

        match_id = _generate_match_id(match_request)

        match_info = {
            "home_team": match_request.home_team,
            "away_team": match_request.away_team,
            "date": match_request.date or datetime.utcnow().strftime("%Y-%m-%d"),
            "league": match_request.league,
            "venue": match_request.venue
        }

        # Cache handling
        cached_result = None
        if use_cache and not force_refresh and db is not None:
            _create_cache_table(db)
            cached_result = _get_cached_analysis(db, match_id)
            if cached_result:
                llm_raw, llm_parsed = cached_result
                processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                return PredictionResponse(
                    model=base,
                    llm_raw=llm_raw,
                    llm_parsed=llm_parsed,
                    cached=True,
                    processing_time_ms=processing_time
                )

        # Build prompt
        recent_stats = {}
        features = base.get("features", {})
        if features:
            stat_mappings = {
                "home_shots_avg_5": "Home shots (last 5)",
                "away_shots_avg_5": "Away shots (last 5)",
                "home_goals_avg_5": "Home goals (last 5)",
                "away_goals_avg_5": "Away goals (last 5)",
                "form_diff": "Form difference",
                "home_win_rate": "Home win rate",
                "away_win_rate": "Away win rate"
            }
            for feature_key, display_name in stat_mappings.items():
                if feature_key in features and features[feature_key] is not None:
                    recent_stats[display_name] = features[feature_key]

        prompt = build_match_context(
            model_probs=base["probabilities"],
            match_info=match_info,
            recent_stats=recent_stats,
            extra_context=match_request.extra_context
        )

        try:
            llm_raw, llm_parsed = call_gemini(prompt)
            logger.info("Successfully generated LLM analysis")
             # analytics success
            capture_event(
                event="llm_enrichment_completed",
                properties={
                    "match_id": match_id,
                    "home_team": match_request.home_team,
                    "away_team": match_request.away_team,
                    "prob_home": float(base["probabilities"]["home"]),
                    "prob_draw": float(base["probabilities"]["draw"]),
                    "prob_away": float(base["probabilities"]["away"]),
                    # "llm_latency_ms": llm_latency_ms,
                    "used_cache": bool(cached_result),
                },
                request=request
            )
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            capture_event(
                event="llm_enrichment_failed",
                properties={
                    "match_id": match_id,
                    "home_team": match_request.home_team,
                    "away_team": match_request.away_team,
                    # "llm_latency_ms": llm_latency_ms,
                    "error_short": str(e)[:200],
                },
                request=request
            )
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return PredictionResponse(
                model=base,
                llm_raw="LLM analysis unavailable",
                llm_parsed={"error": f"LLM service failed: {str(e)}"},
                cached=False,
                processing_time_ms=processing_time,
                error=f"LLM analysis failed: {str(e)}"
            )

        if use_cache and db is not None:
            _save_to_cache(db, match_id, base, llm_raw, llm_parsed)

        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return PredictionResponse(
            model=base,
            llm_raw=llm_raw,
            llm_parsed=llm_parsed,
            cached=False,
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict_enriched: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


@router.get("/cache/stats")
@rate_limiter.limit("200/minute")  # type: ignore
def get_cache_stats(request: Request, current_user: Dict[str, Any] = Depends(get_current_user), db = Depends(get_db)):
    """Get cache statistics; request param required for rate limiter."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        table_check = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'llm_analysis_cache'
            )
        """)
        table_exists = db.execute(table_check).scalar()

        if not table_exists:
            return {"cache_enabled": False, "total_entries": 0}

        stats_query = text("""
            SELECT 
                COUNT(*) as total_entries,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as entries_last_24h,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as entries_last_7d,
                MAX(created_at) as last_entry
            FROM llm_analysis_cache
        """)

        result = db.execute(stats_query).fetchone()

        return {
            "cache_enabled": True,
            "total_entries": result[0] if result else 0,
            "entries_last_24h": result[1] if result else 0,
            "entries_last_7d": result[2] if result else 0,
            "last_entry": result[3].isoformat() if result and result[3] else None
        }

    except SQLAlchemyError as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")


@router.delete("/cache/{match_id}")
@rate_limiter.limit("30/minute")  # type: ignore
def clear_cache_entry(request: Request, match_id: str, current_user: Dict[str, Any] = Depends(get_current_user), db = Depends(get_db), admin_user = Depends(admin_required)):
    """Clear a specific cache entry (admin only)."""

    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        delete_query = text("DELETE FROM llm_analysis_cache WHERE match_id = :mid")
        result = db.execute(delete_query, {"mid": match_id})
        db.commit()

        if getattr(result, "rowcount", 0) > 0:
            logger.info(f"Cleared cache entry for match_id: {match_id}")
            return {"success": True, "message": f"Cleared cache for match_id: {match_id}"}
        else:
            return {"success": False, "message": "No cache entry found for this match_id"}

    except SQLAlchemyError as e:
        logger.error(f"Failed to clear cache entry: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to clear cache entry")