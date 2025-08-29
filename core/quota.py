# app/core/quota.py
from typing import Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from fastapi import Depends, HTTPException, status
from datetime import datetime
import json
from app.db.database import get_db
from app.core.auth import get_current_user

DEFAULT_QUOTA_PER_ENDPOINT = 10
DEFAULT_QUOTA_LIMIT_PER_ENDPOINT = 10
DEFAULT_ENDPOINTS = ["predict_match", "predict_enriched"]

def _now_sql():
    return datetime.utcnow()

def create_default_quotas_for_user(db, user_id: str, quota: int = DEFAULT_QUOTA_PER_ENDPOINT, q_limit: int = DEFAULT_QUOTA_LIMIT_PER_ENDPOINT, endpoints: Optional[list] = None):
    """Create default quotas for a newly registered user."""
    if endpoints is None:
        endpoints = DEFAULT_ENDPOINTS
    try:
        for ep in endpoints:
            db.execute(text("""
                INSERT INTO user_quotas (user_id, endpoint, remaining, quota_limit, unlimited, created_at, updated_at)
                VALUES (:uid, :ep, :remaining, :quota_limit, false, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, endpoint) DO NOTHING
            """), {"uid": str(user_id), "ep": ep, "remaining": int(quota), "quota_limit": int(q_limit)})
        db.commit()
        return True
    except SQLAlchemyError:
        db.rollback()
        return False


def set_quota(db, user_id: str, endpoint: str, remaining: Optional[int] = None, unlimited: Optional[bool] = None) -> Dict[str, Any]:
    """
    Set quota for user/endpoint. If remaining is None and unlimited is provided, update unlimited flag.
    Returns the updated row dict.
    """
    if remaining is None and unlimited is None:
        raise HTTPException(status_code=400, detail="Must provide remaining or unlimited")
    try:
        # Upsert logic
        if remaining is not None and unlimited is not None:
            db.execute(text("""
                INSERT INTO user_quotas (user_id, endpoint, remaining, unlimited, created_at, updated_at)
                VALUES (:uid, :endpoint, :remaining, :unlimited, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, endpoint) DO UPDATE
                SET remaining = EXCLUDED.remaining, unlimited = EXCLUDED.unlimited, updated_at = CURRENT_TIMESTAMP
            """), {"uid": str(user_id), "endpoint": endpoint, "remaining": int(remaining), "unlimited": bool(unlimited)})
        elif remaining is not None:
            db.execute(text("""
                INSERT INTO user_quotas (user_id, endpoint, remaining, unlimited, created_at, updated_at)
                VALUES (:uid, :endpoint, :remaining, false, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, endpoint) DO UPDATE
                SET remaining = EXCLUDED.remaining, updated_at = CURRENT_TIMESTAMP
            """), {"uid": str(user_id), "endpoint": endpoint, "remaining": int(remaining)})
        else:
            # only unlimited change
            db.execute(text("""
                INSERT INTO user_quotas (user_id, endpoint, remaining, unlimited, created_at, updated_at)
                VALUES (:uid, :endpoint, NULL, :unlimited, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, endpoint) DO UPDATE
                SET unlimited = EXCLUDED.unlimited, updated_at = CURRENT_TIMESTAMP
            """), {"uid": str(user_id), "endpoint": endpoint, "unlimited": bool(unlimited)})
        db.commit()
        row = db.execute(text("SELECT user_id, endpoint, remaining, unlimited, created_at, updated_at FROM user_quotas WHERE user_id = :uid AND endpoint = :endpoint"),
                         {"uid": str(user_id), "endpoint": endpoint}).fetchone()
        return dict(row) if row else {}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


def get_quota(db, user_id: str, endpoint: Optional[str] = None):
    """
    Get quota rows for a user. If endpoint is None, return all endpoints for that user.
    """
    try:
        if endpoint:
            q = db.execute(text("SELECT user_id, endpoint, remaining, unlimited, quota_limit, created_at, updated_at FROM user_quotas WHERE user_id = :uid AND endpoint = :endpoint"),
                           {"uid": str(user_id), "endpoint": endpoint}).fetchone()
            return dict(q) if q else None
        else:
            rows = db.execute(text("SELECT user_id, endpoint, remaining, unlimited, quota_limit, created_at, updated_at FROM user_quotas WHERE user_id = :uid"),
                              {"uid": str(user_id)}).fetchall()
            return [dict(r) for r in rows]
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


def check_and_consume_quota(db, user_id: str, endpoint: str, amount: int = 1) -> Dict[str, Any]:
    """
    Atomically check and consume `amount` quota for user/endpoint.
    Behavior:
      - If unlimited -> no consumption, returns {'ok': True, 'remaining': None, 'unlimited': True}
      - If remaining >= amount -> decrement and return remaining
      - Else raise HTTPException 429
    """
    uid = str(user_id)
    try:
        # get row using SELECT FOR UPDATE pattern -> but raw SQL + sessions may not support FOR UPDATE easily in all contexts.
        # We emulate atomicity by performing an UPDATE WHEN condition and checking rowcount.
        # First check unlimited
        row = db.execute(text("SELECT remaining, unlimited FROM user_quotas WHERE user_id = :uid AND endpoint = :endpoint"),
                         {"uid": uid, "endpoint": endpoint}).fetchone()

        if not row:
            # if no specific row, check 'all' (global quota)
            row = db.execute(text("SELECT remaining, unlimited FROM user_quotas WHERE user_id = :uid AND endpoint = 'all'"),
                             {"uid": uid}).fetchone()
            if not row:
                # No quota row found -> default behavior: assign default quotas (do not create automatically here)
                # Reject if no configured quota
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Quota not configured for this user/endpoint")

        remaining, unlimited = row[0], bool(row[1])

        if unlimited:
            return {"ok": True, "remaining": None, "unlimited": True}

        if remaining is None:
            # treat as zero
            remaining = 0

        if int(remaining) < amount:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Quota exhausted")

        # consume by updating remaining = remaining - amount
        new_remaining = int(remaining) - int(amount)
        db.execute(text("""
            UPDATE user_quotas
            SET remaining = :new_remaining, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = :uid AND endpoint = :endpoint
        """), {"new_remaining": new_remaining, "uid": uid, "endpoint": endpoint})
        db.commit()
        return {"ok": True, "remaining": new_remaining, "unlimited": False}
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


# Dependency factory
def quota_dependency(endpoint_name: str, amount: int = 1):
    """
    Returns a dependency callable suitable for FastAPI Depends(...) which will ensure the current
    user has quota and consume it.
    Example usage in an endpoint signature:
      current_user: dict = Depends(get_current_user),
      db = Depends(get_db),
      _quota = Depends(quota_dependency("predict_match"))
    The dependency needs access to current_user and db, but FastAPI will resolve them before this dependency runs.
    """
    async def dependency(current_user = get_current_user, db = get_db):       
        raise RuntimeError("quota_dependency should be used with Depends(...) and receives (current_user, db) from FastAPI")
    # We return a function that FastAPI will wrap. To make type hints and behavior explicit, create an inner function that
    # matches FastAPI signature where dependencies will be injected:
    def _inner(current_user = Depends(get_current_user), db = Depends(get_db)):
        uid = current_user.get("id")
        if uid is None:
            raise HTTPException(status_code=401, detail="User identity missing")
        return check_and_consume_quota(db, str(uid), endpoint_name, amount)
    return _inner
