# app/api/external_sportradar.py
import os
import requests
import json
from urllib.parse import quote
from fastapi import APIRouter, Query, HTTPException
from typing import Any, Dict, List, Tuple

router = APIRouter(prefix="/external/sportradar", tags=["external_sportradar"])

SPORTRADAR_KEY = os.getenv("SPORTRADAR_KEY")
SPORTRADAR_BASE = os.getenv("SPORTRADAR_BASE", "/soccer/trial/v4/en")
SPORTRADAR_HOST = os.getenv("SPORTRADAR_HOST", "https://api.sportradar.com")
SPORTRADAR_SEASON_ID = os.getenv("SPORTRADAR_SEASON_ID") 

def _http_get(url: str, params: dict = None, timeout: int = 10) -> Tuple[int, Any, str]: # type: ignore
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        text = r.text
        try:
            data = r.json()
        except Exception:
            data = text
        return r.status_code, data, text
    except requests.RequestException as e:
        return 0, None, str(e)


def _dict_contains_value(obj: Any, value: str) -> bool:
    if obj is None:
        return False
    if isinstance(obj, str):
        return value in obj
    if isinstance(obj, dict):
        for v in obj.values():
            if _dict_contains_value(v, value):
                return True
        return False
    if isinstance(obj, list):
        for item in obj:
            if _dict_contains_value(item, value):
                return True
        return False
    return False


def _pull_possible_matches(obj: Any) -> List[Dict[str, Any]]:
    found = []

    def _walk(x):
        if isinstance(x, dict):
            keys = set(k.lower() for k in x.keys())
            if ("home" in keys and "away" in keys) or "competitors" in keys or "sport_event" in keys or "scheduled" in keys or "start_time" in keys:
                found.append(x)
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for it in x:
                _walk(it)
        else:
            return

    _walk(obj)
    return found


def _normalize_match_obj(raw: Dict[str, Any]) -> Dict[str, Any]:
    mid = raw.get("id") or raw.get("match_id") or raw.get("sport_event_id") or raw.get("game_id")
    scheduled = raw.get("scheduled") or raw.get("start_time") or raw.get("start")
    home = ""
    away = ""
    if "home" in raw and "away" in raw:
        h = raw.get("home"); a = raw.get("away")
        home = (h.get("name") if isinstance(h, dict) else h) or ""
        away = (a.get("name") if isinstance(a, dict) else a) or ""
    elif "competitors" in raw and isinstance(raw["competitors"], list):
        comps = raw["competitors"]
        if len(comps) > 0:
            home = comps[0].get("name") or comps[0].get("id") or ""
        if len(comps) > 1:
            away = comps[1].get("name") or comps[1].get("id") or ""
    elif "sport_event" in raw and isinstance(raw["sport_event"], dict):
        se = raw["sport_event"]
        if "competitors" in se and isinstance(se["competitors"], list):
            comps = se["competitors"]
            if len(comps) > 0:
                home = comps[0].get("name") or comps[0].get("id") or ""
            if len(comps) > 1:
                away = comps[1].get("name") or comps[1].get("id") or ""
        home = home or se.get("home", "") or se.get("home_team", "")
        away = away or se.get("away", "") or se.get("away_team", "")
    else:
        home = raw.get("home_team") or raw.get("home_name") or ""
        away = raw.get("away_team") or raw.get("away_name") or ""

    try:
        home = (home.get("name") if isinstance(home, dict) else home) or ""
    except Exception:
        home = str(home or "")
    try:
        away = (away.get("name") if isinstance(away, dict) else away) or ""
    except Exception:
        away = str(away or "")

    return {"id": mid, "home_team": home, "away_team": away, "scheduled": scheduled, "raw": raw}


@router.get("/season_schedules")
def season_schedules(
    season_id: str = Query(None, description="Season id, e.g. sr:season:118689"),
    competition_id: str = Query("sr:competition:17", description="Competition id to filter by (EPL default)")
):
    """
    Fetch season schedules and return only matches belonging to the requested competition_id.
    If season_id is not provided, will fall back to SPORTRADAR_SEASON_ID env variable.
    """
    if not SPORTRADAR_KEY:
        raise HTTPException(status_code=500, detail="SPORTRADAR_KEY not configured on server.")

    # If no season_id in query, try env fallback
    if not season_id:
        season_id = "sr:season:118689"   # type: ignore

    if not season_id:
        raise HTTPException(status_code=400, detail={
            "message": "season_id query parameter is required (or set SPORTRADAR_SEASON_ID env var).",
            "example": "/external/sportradar/season_schedules?season_id=sr:season:118689&competition_id=sr:competition:17"
        })

    season_id_enc = quote(season_id, safe="")
    base = SPORTRADAR_HOST.rstrip("/") + SPORTRADAR_BASE.rstrip("/")
    url = f"{base}/seasons/{season_id_enc}/schedules.json"
    params = {"api_key": SPORTRADAR_KEY}

    status, data, raw = _http_get(url, params=params)
    if status != 200:
        raise HTTPException(status_code=max(status or 500, 400), detail={
            "message": f"Sportradar returned {status}: {raw[:500] if raw else ''}",
            "url": url
        })

    candidate_matches = _pull_possible_matches(data)
    filtered = [cm for cm in candidate_matches if _dict_contains_value(cm, competition_id)]

    if not filtered:
        # fallback: find schedule blocks containing the competition_id and collect matches from them
        def find_blocks_with_comp(obj):
            results = []
            if isinstance(obj, dict):
                if _dict_contains_value(obj, competition_id):
                    results.append(obj)
                for v in obj.values():
                    results.extend(find_blocks_with_comp(v))
            elif isinstance(obj, list):
                for it in obj:
                    results.extend(find_blocks_with_comp(it))
            return results

        matches_from_blocks = []
        blocks = find_blocks_with_comp(data)
        for block in blocks:
            matches_from_blocks.extend(_pull_possible_matches(block))
        seen = set()
        for m in matches_from_blocks:
            mid = m.get("id") or m.get("match_id") or json.dumps(m)[:80]
            if mid not in seen:
                filtered.append(m)
                seen.add(mid)

    normalized = [_normalize_match_obj(m) for m in filtered]
    normalized_sorted = sorted(normalized, key=lambda x: x.get("scheduled") or "")

    return {"season_id": season_id, "competition_id": competition_id, "count": len(normalized_sorted), "matches": normalized_sorted}
