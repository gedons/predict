# app/api/quotas.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from core.auth import get_current_user
from db.database import get_db

router = APIRouter(prefix="/me", tags=["me"])


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert a SQLAlchemy RowMapping or tuple row to a dict safely."""
    if row is None:
        return {}
    # If already mapping-like
    try:
        return dict(row)
    except Exception:
        # Fallback: convert sequence of pairs -> dict (rare)
        try:
            return {k: v for k, v in row}
        except Exception:
            # Last fallback: string representation
            return {"raw": str(row)}


@router.get("/quotas", summary="Get current user's quotas")
def me_quotas(current_user: Dict[str, Any] = Depends(get_current_user), db = Depends(get_db)):
    """
    Return all quota rows for the authenticated user.
    """
    if not current_user or "id" not in current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    uid = str(current_user["id"])
    try:
        sql = text("""
            SELECT user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at
            FROM user_quotas
            WHERE user_id = :uid
            ORDER BY endpoint
        """)
        # Use .mappings() to ensure we get dict-like rows
        rows = db.execute(sql, {"uid": uid}).mappings().fetchall()
        quotas = [dict(r) for r in rows]
        return {"user_id": uid, "quotas": quotas}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.get("/quotas/{endpoint}", summary="Get quota for an endpoint for current user")
def me_quota_for_endpoint(endpoint: str, current_user: Dict[str, Any] = Depends(get_current_user), db = Depends(get_db)):
    """
    Return single quota row for the authenticated user and endpoint.
    """
    if not current_user or "id" not in current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    uid = str(current_user["id"])
    try:
        sql = text("""
            SELECT user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at
            FROM user_quotas
            WHERE user_id = :uid AND endpoint = :endpoint
            LIMIT 1
        """)
        row = db.execute(sql, {"uid": uid, "endpoint": endpoint}).mappings().fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No quota configured for endpoint '{endpoint}'")
        return {"user_id": uid, "endpoint": endpoint, "quota": dict(row)}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
