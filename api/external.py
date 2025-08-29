# app/api/external.py
import os
import json
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
import httpx

CACHE_DIR = Path("app/tmp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/external", tags=["external"])

SPORTRADAR_KEY = os.getenv("SPORTRADAR_API_KEY") or os.getenv("VITE_SPORTRADAR_API_KEY") or os.getenv("SPORTRADAR_API")
SPORTRADAR_BASE = "https://api.sportradar.com/soccer/trial/v4/en"

# Cache TTL in seconds (configurable via env)
CACHE_TTL_SECONDS = int(os.getenv("SPORTRADAR_CACHE_TTL", "600"))  # 10 minutes default


def _cache_path(competition_id: str):
    safe = competition_id.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"sportradar_{safe}.json"


async def _fetch_sportradar(url: str, params: dict = None) -> dict: # type: ignore
    timeout = httpx.Timeout(15.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


@router.get("/sportradar/schedules")
async def get_sportradar_schedules(
    competition_id: str = Query("sr:competition:17", description="Sportradar competition id, default EPL sr:competition:17"),
    force_refresh: bool = Query(False, description="Force refresh ignoring cache"),
):
    """
    Proxy endpoint to fetch schedules from Sportradar for a competition.
    Example: GET /external/sportradar/schedules?competition_id=sr:competition:17
    This endpoint caches results (default TTL configurable via SPORTRADAR_CACHE_TTL).
    """
    if not SPORTRADAR_KEY:
        raise HTTPException(status_code=500, detail="Sportradar API key not configured on server (SPORTRADAR_API_KEY)")

    cache_file = _cache_path(competition_id)

    # return cache if valid and not forced
    if cache_file.exists() and not force_refresh:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ts = payload.get("_fetched_at")
            if ts:
                fetched_at = datetime.fromisoformat(ts)
                if datetime.utcnow() - fetched_at < timedelta(seconds=CACHE_TTL_SECONDS):
                    # return cached content (strip internal metadata)
                    res = payload.get("data", payload)
                    return {"cached": True, "fetched_at": ts, "data": res}
        except Exception:
            # ignore cache errors and re-fetch
            pass

    # Build Sportradar URL
    url = f"{SPORTRADAR_BASE}/competitions/{competition_id}/schedules.json"
    params = {"api_key": SPORTRADAR_KEY}

    try:
        data = await _fetch_sportradar(url, params=params)
    except httpx.HTTPStatusError as exc:
        detail = f"Sportradar returned {exc.response.status_code}: {exc.response.text[:300]}"
        raise HTTPException(status_code=502, detail=detail)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to contact Sportradar: {str(exc)}")

    # Save to cache with fetched timestamp
    try:
        wrapped = {"_fetched_at": datetime.utcnow().isoformat(), "data": data}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(wrapped, f)
    except Exception:
        # ignore cache write errors
        pass

    return {"cached": False, "fetched_at": datetime.utcnow().isoformat(), "data": data}
