# app/api/admin_analytics.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import os
import requests
from core.auth import admin_required

router = APIRouter(prefix="/admin/analytics", tags=["admin_analytics"])

POSTHOG_HOST = os.getenv("POSTHOG_HOST", "https://app.posthog.com").rstrip("/")
POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")  # project API key or personal API key with read access

def _call_posthog_trend(event_name: str, days: int = 7) -> Dict[str, Any]:
    if not POSTHOG_API_KEY:
        raise RuntimeError("POSTHOG_API_KEY not configured")
    # Use the insights/trend endpoint (may differ by PostHog versions). We'll call the /api/insight/trend
    url = f"{POSTHOG_HOST}/api/insight/trend/"
    payload = {
        "events": [{"id": event_name, "type": "events"}],
        "date_from": f"-{days}d",
        "interval": "day",
    }
    headers = {"Authorization": f"Bearer {POSTHOG_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()

@router.get("/posthog-summary", dependencies=[Depends(admin_required)])
def posthog_summary(days: int = 7):
    """
    Return a small summary:
      - total prediction_requested count in last `days`
      - total llm_enrichment_completed count
      - avg llm_latency_ms (calculated client-side if present)
    """
    try:
        # Attempt to query trend for prediction_requested
        pred_trend = _call_posthog_trend("prediction_requested", days=days)
        llm_trend = _call_posthog_trend("llm_enrichment_completed", days=days)
        return {"prediction_trend": pred_trend, "llm_trend": llm_trend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PostHog query failed: {e}")
