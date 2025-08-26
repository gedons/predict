# app/core/analytics.py
import os
import logging
from typing import Any, Dict, Optional

from fastapi import Request

# Defensive import: try canonical name, else fallback to module import
try:
    from posthog import Posthog  # type: ignore # canonical
except Exception:
    try:
        import posthog as _ph
        Posthog = getattr(_ph, "Posthog", None)
    except Exception:
        Posthog = None  # analytics disabled if import fails

logger = logging.getLogger(__name__)

POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")  # server-side project API key
POSTHOG_HOST = os.getenv("POSTHOG_HOST", "https://app.posthog.com")
POSTHOG_ENABLED = os.getenv("POSTHOG_ENABLED", "true").lower() not in ("0", "false", "no")

_client: Optional[Posthog] = None

def _init_client() -> Optional[Posthog]:
    global _client
    if not POSTHOG_ENABLED:
        return None
    if _client is not None:
        return _client
    if not Posthog:
        logger.warning("posthog library not available; analytics disabled.")
        return None
    if not POSTHOG_API_KEY:
        logger.warning("POSTHOG_API_KEY not set; analytics disabled.")
        return None
    try:
        _client = Posthog(POSTHOG_API_KEY, host=POSTHOG_HOST)
        logger.info("PostHog client initialized")
        return _client
    except Exception as e:
        logger.exception("Failed to initialize PostHog client: %s", e)
        _client = None
        return None

def _extract_distinct_id_from_request(request: Optional[Request]) -> str:
    """
    Best-effort extraction: attempt to decode JWT and return 'sub'. If failure -> 'anonymous'.
    We call decode_jwt in a try/except to avoid raising.
    """
    if not request:
        return "anonymous"
    try:
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if not auth:
            return "anonymous"
        parts = auth.split()
        if len(parts) != 2:
            return "anonymous"
        token = parts[1]
        # Import here to avoid cycle at module import time
        from app.core.auth import decode_jwt
        payload = decode_jwt(token)
        sub = payload.get("sub") or payload.get("user_id") or payload.get("id")
        return str(sub) if sub is not None else "anonymous"
    except Exception:
        return "anonymous"

def capture_event(
    event: str,
    properties: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
    distinct_id: Optional[str] = None,
) -> None:
    """
    Fire an event to PostHog. Non-blocking: errors are caught and logged.
    """
    client = _init_client()
    if not client:
        return
    try:
        d_id = distinct_id or _extract_distinct_id_from_request(request) or "anonymous"
        props = properties.copy() if properties else {}
        # Add safe defaults
        props.setdefault("service", "predict-back")
        # Don't include any raw jwt or PII here.
        client.capture(d_id, event, properties=props)
    except Exception as e:
        logger.exception("PostHog capture failed for %s: %s", event, e)
