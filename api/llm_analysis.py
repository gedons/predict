# app/api/llm_analysis.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from sqlalchemy import text
from app.db.database import get_db
from app.services.context_builder import build_match_context
from app.services.llm_service import call_gemini
import json
import traceback

router = APIRouter(prefix="/predict", tags=["predict_enriched"])

try:

    from app.api.predict import predict_match as predict_match_helper  # type: ignore
except Exception:
    predict_match_helper = None

@router.post("/enriched")
def predict_enriched(match_request: Dict[str, Any], db = Depends(get_db), use_cache: bool = True):
    """
    Accepts a match payload with at least:
      - home_team
      - away_team
      - date (optional)
      - league (optional)
    Returns:
      - base model output
      - llm_analysis (parsed JSON or raw text)
    """

    # 1) Get base model output (try to reuse your predict module)
    if predict_match_helper:
        try:
            base = predict_match_helper(match_request, raw_output=True) # type: ignore
            # expected structure:
            # { "features": {...}, "probabilities": {"home":..,"draw":..,"away":..}, "model_meta": {...} }
        except TypeError:
            # maybe helper signature is different (no raw_output). Try basic call
            base = predict_match_helper(match_request) # type: ignore
    else:
        raise HTTPException(status_code=500, detail="Prediction helper not found in app.api.predict. Please expose `predict_match(match_request, raw_output=True)`.")

    # minimal validation
    probs = base.get("probabilities") or {}
    if not probs:
        raise HTTPException(status_code=500, detail="Prediction returned no probability output")

    match_id = match_request.get("match_id") or f"{match_request.get('date','')}_{match_request.get('home_team','')}_{match_request.get('away_team','')}"
    match_info = {
        "home_team": match_request.get("home_team"),
        "away_team": match_request.get("away_team"),
        "date": match_request.get("date"),
        "league": match_request.get("league"),
        "venue": match_request.get("venue")
    }

    # 2) Check DB cache (optional)
    llm_result = None
    if use_cache and db is not None:
        try:
            # create cache table if not exists (idempotent)
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS IF NOT EXISTS llm_analysis_cache (
                    id SERIAL PRIMARY KEY,
                    match_id TEXT UNIQUE,
                    model_meta JSONB,
                    model_probs JSONB,
                    llm_raw TEXT,
                    llm_parsed JSONB,
                    created_at TIMESTAMP DEFAULT now()
                )"""))
            db.commit()
        except Exception:
            # ignore DB create errors - server might not have permissions
            db.rollback()

        try:
            q = text("SELECT llm_raw, llm_parsed FROM llm_analysis_cache WHERE match_id = :mid LIMIT 1")
            row = db.execute(q, {"mid": match_id}).fetchone()
            if row and row[0]:
                # If stored, return cached analysis
                llm_raw = row[0]
                llm_parsed = row[1] if len(row) > 1 else None
                return {"model": base, "llm_raw": llm_raw, "llm_parsed": llm_parsed, "cached": True}
        except Exception:
            # if cache query fails, continue to call LLM
            pass

    # 3) Build prompt and call Gemini
    # Optionally pass `recent_stats` from `base["features"]` if available
    recent_stats = {
        # you can extend this mapping depending on what your features contain
        "home_shots_avg_5": base.get("features", {}).get("home_shots_avg_5"),
        "away_shots_avg_5": base.get("features", {}).get("away_shots_avg_5"),
        "form_diff": base.get("features", {}).get("form_diff")
    }

    prompt = build_match_context(probs, match_info, recent_stats=recent_stats, extra_context=match_request.get("extra_context"))

    try:
        raw_text, parsed = call_gemini(prompt)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}\n{tb}")

    # 4) Save to cache if DB available
    if use_cache and db is not None:
        try:
            insert = text("""
                INSERT INTO llm_analysis_cache (match_id, model_meta, model_probs, llm_raw, llm_parsed)
                VALUES (:mid, :meta, :probs, :raw, :parsed)
                ON CONFLICT (match_id) DO UPDATE SET llm_raw = EXCLUDED.llm_raw, llm_parsed = EXCLUDED.llm_parsed, model_meta = EXCLUDED.model_meta, model_probs = EXCLUDED.model_probs, created_at = now()
            """)
            db.execute(insert, {
                "mid": match_id,
                "meta": json.dumps(base.get("model_meta", {})),
                "probs": json.dumps(probs),
                "raw": raw_text,
                "parsed": json.dumps(parsed if isinstance(parsed, dict) else {"text": parsed})
            })
            db.commit()
        except Exception:
            db.rollback()

    return {"model": base, "llm_raw": raw_text, "llm_parsed": parsed, "cached": False}
