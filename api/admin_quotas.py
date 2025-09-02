# app/api/admin_quotas.py
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import json

from core.auth import admin_required
from db.database import get_db

router = APIRouter(prefix="/admin/quotas", tags=["admin_quotas"])


class SetQuotaRequest(BaseModel):
    endpoint: str = Field(..., min_length=1)
    limit: Optional[int] = Field(None, ge=0, description="Configured limit (null means unspecified)")
    remaining: Optional[int] = Field(None, ge=0, description="Remaining calls (null means unspecified)")
    unlimited: Optional[bool] = Field(False, description="True to mark unlimited usage")


class GrantDefaultsRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="If not provided and apply_to_all is False, error")
    apply_to_all: Optional[bool] = Field(False, description="If true, apply quotas to all users")
    quotas: List[SetQuotaRequest] = Field(..., description="List of quotas to grant")


def _ensure_quota_table(db) -> None:
    """
    Ensure the user_quotas table exists (idempotent).
    Uses `quota_limit` column name to avoid SQL reserved keywords.
    """
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS user_quotas (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                quota_limit INTEGER,
                remaining INTEGER,
                unlimited BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, endpoint)
            )
        """))
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_user_quotas_userid ON user_quotas(user_id)"))
        try:
            db.commit()
        except Exception:
            pass
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass


async def _read_json_like(request: Request) -> Dict[str, Any]:
    """
    Robustly parse incoming request bodies:
      - prefer JSON object
      - accept form data (x-www-form-urlencoded) where values might be JSON strings
      - accept raw JSON strings in body
    Returns a dict
    """
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")
        if isinstance(body, dict):
            return body
        return {"__raw_body__": body}

    try:
        form = await request.form()
        if form:
            out: Dict[str, Any] = {}
            for k, v in form.multi_items():
                if isinstance(v, str):
                    s = v.strip()
                    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                        try:
                            out[k] = json.loads(s)
                            continue
                        except Exception:
                            pass
                out[k] = v
            if out:
                return out
    except Exception:
        pass

    try:
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty request body")
        raw_s = raw.decode("utf-8").strip()
        parsed = json.loads(raw_s)
        if isinstance(parsed, dict):
            return parsed
        return {"__raw_body__": parsed}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Unable to parse request body as JSON")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Request body contains invalid UTF-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse request body: {e}")


def _normalize_quota_row(row_mapping) -> Dict[str, Any]:
    """
    Convert ResultMapping row to simple dict and safe isoformat for datetimes.
    Map `quota_limit` -> `limit` in output for frontend compatibility.
    """
    if not row_mapping:
        return {}
    d = dict(row_mapping)
    # map quota_limit to limit so frontend does not need to change
    if "quota_limit" in d:
        d["limit"] = d.pop("quota_limit")
    for k, v in list(d.items()):
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    d.setdefault("limit", None)
    d.setdefault("remaining", None)
    d.setdefault("unlimited", False)
    return d


@router.get("/", dependencies=[Depends(admin_required)])
def list_quotas(limit: int = 200, offset: int = 0, db = Depends(get_db)):
    _ensure_quota_table(db)
    try:
        sql = text("""
            SELECT user_id, endpoint, quota_limit AS limit, remaining, unlimited, created_at, updated_at
            FROM user_quotas
            ORDER BY user_id, endpoint
            LIMIT :limit OFFSET :offset
        """)
        rows = db.execute(sql, {"limit": limit, "offset": offset}).mappings().fetchall()
        quotas = [_normalize_quota_row(r) for r in rows]
        defaults = { q["endpoint"]: q["limit"] for q in quotas if q.get("user_id") == "default" }
        sample_user_quotas = [q for q in quotas if q.get("user_id") != "default"][:20]
        return {"count": len(quotas), "quotas": quotas, "defaults": defaults, "user_quotas_sample": sample_user_quotas}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.get("/user/{user_id}", dependencies=[Depends(admin_required)])
def get_user_quotas(user_id: str, db = Depends(get_db)):
    _ensure_quota_table(db)
    try:
        rows = db.execute(
            text("""
                SELECT user_id, endpoint, quota_limit AS limit, remaining, unlimited, created_at, updated_at
                FROM user_quotas
                WHERE user_id = :uid
                ORDER BY endpoint
            """),
            {"uid": str(user_id)}
        ).mappings().fetchall()
        quotas = [_normalize_quota_row(r) for r in rows]
        return {"user_id": str(user_id), "quotas": quotas}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.post("/user/{user_id}/set", dependencies=[Depends(admin_required)])
async def set_quota_for_user(user_id: str, request: Request, db = Depends(get_db)):
    _ensure_quota_table(db)
    payload = await _read_json_like(request)

    try:
        if "endpoint" in payload:
            parsed = SetQuotaRequest.parse_obj(payload)
        else:
            if isinstance(payload, dict) and len(payload) >= 1:
                if all(isinstance(k, str) for k in payload.keys()):
                    ep, lim = next(iter(payload.items()))
                    parsed = SetQuotaRequest.parse_obj({"endpoint": ep, "limit": lim})
                else:
                    parsed = SetQuotaRequest.parse_obj(payload)
            else:
                parsed = SetQuotaRequest.parse_obj(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"validation_errors": json.loads(e.json())})

    upsert_sql = text("""
        INSERT INTO user_quotas (user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at)
        VALUES (:uid, :endpoint, :limit, :remaining, :unlimited, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, endpoint)
        DO UPDATE SET
          quota_limit = COALESCE(EXCLUDED.quota_limit, user_quotas.quota_limit),
          remaining = COALESCE(EXCLUDED.remaining, user_quotas.remaining),
          unlimited = EXCLUDED.unlimited,
          updated_at = CURRENT_TIMESTAMP
        RETURNING user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at
    """)
    try:
        res = db.execute(upsert_sql, {
            "uid": str(user_id),
            "endpoint": parsed.endpoint,
            "limit": parsed.limit,
            "remaining": parsed.remaining,
            "unlimited": bool(parsed.unlimited)
        })
        row = res.mappings().fetchone()
        try:
            db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
        return {"status": "ok", "quota": _normalize_quota_row(row) if row else None}
    except SQLAlchemyError as e:
        try:
            db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.post("/grant-default", dependencies=[Depends(admin_required)])
async def grant_default_quotas(request: Request, apply_to_all: bool = Query(False), db = Depends(get_db)):
    """
    Grant a list of quotas to one user or to ALL users.

    Accepts inputs like:
    {
      "user_id": "<uuid>",
      "apply_to_all": false,
      "quotas": [
         {"endpoint": "predict_match", "limit": 10},
         {"endpoint": "predict_enriched", "limit": 10}
      ]
    }

    Also accepts a simple mapping form:
    { "predict_match": 20, "predict_enriched": 10 }
    """
    _ensure_quota_table(db)
    payload = await _read_json_like(request)

    # Parse quotas into list of tuples: (endpoint, limit, remaining, unlimited)
    quotas_list = []

    # If top-level contains "quotas"
    if isinstance(payload, dict) and "quotas" in payload:
        q = payload["quotas"]
        if isinstance(q, list):
            for item in q:
                if isinstance(item, dict) and "endpoint" in item:
                    quotas_list.append((
                        item["endpoint"],
                        item.get("limit"),
                        item.get("remaining"),
                        bool(item.get("unlimited", False))
                    ))
        elif isinstance(q, dict):
            # allow {"quotas": {"predict_match": 20}}
            for ep, lim in q.items():
                quotas_list.append((ep, lim, None, False))
        else:
            raise HTTPException(status_code=400, detail="Invalid 'quotas' structure")
    else:
        # allow mapping form { "predict_match": 20 } OR a single quota object { "endpoint": "...", "limit": X }
        if isinstance(payload, dict):
            # mapping-like: all keys strings and values numbers
            mapping_like = all(isinstance(k, str) for k in payload.keys()) and not ("endpoint" in payload and "limit" in payload)
            if mapping_like:
                for ep, lim in payload.items():
                    quotas_list.append((ep, lim, None, False))
            elif "endpoint" in payload:
                quotas_list.append((
                    payload["endpoint"],
                    payload.get("limit"),
                    payload.get("remaining"),
                    bool(payload.get("unlimited", False))
                ))
            else:
                raise HTTPException(status_code=400, detail="Couldn't parse quotas payload")
        else:
            raise HTTPException(status_code=400, detail="Invalid payload for grant-default")

    if not quotas_list:
        raise HTTPException(status_code=400, detail="No quotas parsed from payload")

    try:
        now = datetime.utcnow()
        created = []
        # Use `db.begin()` transaction but execute via the session (`db.execute`) — do NOT use conn.execute on the SessionTransaction object.
        with db.begin():
            # insert/update default quotas
            upsert_default_sql = text("""
                INSERT INTO user_quotas (user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at)
                VALUES ('default', :endpoint, :limit, :remaining, :unlimited, :now, :now)
                ON CONFLICT (user_id, endpoint)
                DO UPDATE SET quota_limit = EXCLUDED.quota_limit, remaining = EXCLUDED.remaining, unlimited = EXCLUDED.unlimited, updated_at = EXCLUDED.updated_at
                RETURNING user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at
            """)
            for endpoint, limit, remaining, unlimited in quotas_list:
                res = db.execute(upsert_default_sql, {
                    "endpoint": endpoint,
                    "limit": limit,
                    "remaining": remaining,
                    "unlimited": unlimited,
                    "now": now
                })
                row = res.mappings().fetchone()
                created.append(_normalize_quota_row(row) if row else {"user_id": "default", "endpoint": endpoint})

            # Optionally apply to all users
            if apply_to_all:
                # Insert missing quotas for all users (do not override existing)
                for endpoint, limit, remaining, unlimited in quotas_list:
                    db.execute(text("""
                        INSERT INTO user_quotas (user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at)
                        SELECT u.id::text, :endpoint, :limit, :remaining, :unlimited, :now, :now
                        FROM users u
                        WHERE NOT EXISTS (
                            SELECT 1 FROM user_quotas uq WHERE uq.user_id = u.id::text AND uq.endpoint = :endpoint
                        )
                    """), {"endpoint": endpoint, "limit": limit, "remaining": remaining, "unlimited": unlimited, "now": now})

        # transaction committed on exit
        return {"status": "ok", "created_defaults": created}
    except SQLAlchemyError as e:
        # if using a Session, the context manager will rollback on exception, but keep safe handling
        try:
            db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.post("/user/{user_id}/update", dependencies=[Depends(admin_required)])
async def update_user_quota(user_id: str, request: Request, db = Depends(get_db)):
    _ensure_quota_table(db)
    payload = await _read_json_like(request)
    try:
        parsed = SetQuotaRequest.parse_obj(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"validation_errors": json.loads(e.json())})

    upsert_sql = text("""
        INSERT INTO user_quotas (user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at)
        VALUES (:uid, :endpoint, :limit, :remaining, :unlimited, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, endpoint)
        DO UPDATE SET 
            quota_limit = COALESCE(EXCLUDED.quota_limit, user_quotas.quota_limit),
            remaining = COALESCE(EXCLUDED.remaining, user_quotas.remaining),
            unlimited = EXCLUDED.unlimited,
            updated_at = CURRENT_TIMESTAMP
        RETURNING user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at
    """)
    try:
        res = db.execute(upsert_sql, {
            "uid": str(user_id),
            "endpoint": parsed.endpoint,
            "limit": parsed.limit,
            "remaining": parsed.remaining,
            "unlimited": bool(parsed.unlimited)
        })
        row = res.mappings().fetchone()
        try:
            db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
        return {"status": "ok", "quota": _normalize_quota_row(row) if row else None}
    except SQLAlchemyError as e:
        try:
            db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.delete("/user/{user_id}/endpoint/{endpoint}", dependencies=[Depends(admin_required)])
def delete_user_quota(user_id: str, endpoint: str, db = Depends(get_db)):
    _ensure_quota_table(db)
    try:
        res = db.execute(
            text("DELETE FROM user_quotas WHERE user_id = :uid AND endpoint = :endpoint RETURNING user_id, endpoint"),
            {"uid": str(user_id), "endpoint": endpoint}
        )
        row = res.mappings().fetchone()
        try:
            db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass

        if not row:
            raise HTTPException(status_code=404, detail="Quota not found")
        return {"status": "deleted", "user_id": row["user_id"], "endpoint": row["endpoint"]}
    except SQLAlchemyError as e:
        try:
            db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.post("/user/{user_id}/reset", dependencies=[Depends(admin_required)])
def reset_user_quotas(user_id: str, endpoint: Optional[str] = None, db = Depends(get_db)):
    _ensure_quota_table(db)
    try:
        if endpoint:
            res = db.execute(
                text("""
                  UPDATE user_quotas
                  SET remaining = quota_limit,
                      updated_at = CURRENT_TIMESTAMP
                  WHERE user_id = :uid AND endpoint = :endpoint
                  RETURNING user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at
                """), {"uid": str(user_id), "endpoint": endpoint}
            )
            row = res.mappings().fetchone()
            try:
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
            if not row:
                raise HTTPException(status_code=404, detail="Quota not found")
            return {"status": "ok", "quota": _normalize_quota_row(row)}
        else:
            res = db.execute(text("""
                UPDATE user_quotas
                SET remaining = quota_limit,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = :uid
                RETURNING user_id, endpoint, quota_limit, remaining, unlimited, created_at, updated_at
            """), {"uid": str(user_id)})
            rows = res.mappings().fetchall()
            try:
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
            return {"status": "ok", "count": len(rows), "quotas": [_normalize_quota_row(r) for r in rows]}
    except SQLAlchemyError as e:
        try:
            db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.get("/users/", dependencies=[Depends(admin_required)])
def list_users(search: Optional[str] = Query(None), limit: int = Query(50), offset: int = Query(0), db=Depends(get_db)):
    try:
        if search:
            q = f"%{search.strip()}%"
            sql = text("""
                SELECT id, email, username
                FROM users
                WHERE email ILIKE :q OR username ILIKE :q OR id::text ILIKE :q
                ORDER BY email NULLS LAST
                LIMIT :limit OFFSET :offset
            """)
            res = db.execute(sql, {"q": q, "limit": limit, "offset": offset})
        else:
            sql = text("SELECT id, email, username FROM users ORDER BY email NULLS LAST LIMIT :limit OFFSET :offset")
            res = db.execute(sql, {"limit": limit, "offset": offset})

        rows = res.mappings().fetchall()
        users = [{"id": r["id"], "email": r.get("email"), "username": r.get("username")} for r in rows]
        return {"count": len(users), "users": users}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error listing users: {e}")
