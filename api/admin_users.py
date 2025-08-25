# app/api/admin_users.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Any, Optional
from sqlalchemy import text
import json

from app.core.auth import admin_required
from app.db.database import get_db

router = APIRouter(prefix="/admin/users", tags=["admin_users"])


@router.get("/", dependencies=[Depends(admin_required)])
def list_users(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None, description="Search email or username"),
    db = Depends(get_db)
):
    """
    List users for admin UI. Returns id, email, username, is_admin, created_at
    Optional `search` filters by email or username (ILIKE).
    """
    try:
        if search:
            sql = text("""
                SELECT id, email, username, is_admin, created_at
                FROM users
                WHERE email ILIKE :q OR username ILIKE :q
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
            q = f"%{search}%"
            rows = db.execute(sql, {"q": q, "limit": limit, "offset": offset}).mappings().fetchall()
        else:
            sql = text("""
                SELECT id, email, username, is_admin, created_at
                FROM users
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
            rows = db.execute(sql, {"limit": limit, "offset": offset}).mappings().fetchall()

        users = []
        for r in rows:
            users.append({
                "id": r.get("id"),
                "email": r.get("email"),
                "username": r.get("username"),
                "is_admin": bool(r.get("is_admin", False)),
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None
            })
        return {"count": len(users), "users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list users: {e}")


@router.get("/{user_id}", dependencies=[Depends(admin_required)])
def get_user(user_id: str, db = Depends(get_db)):
    """
    Get a single user by id (UUID or int depending on your schema).
    """
    try:
        sql = text("""
            SELECT id, email, username, is_admin, created_at
            FROM users
            WHERE id = :id
            LIMIT 1
        """)
        row = db.execute(sql, {"id": user_id}).mappings().fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return {
            "id": row.get("id"),
            "email": row.get("email"),
            "username": row.get("username"),
            "is_admin": bool(row.get("is_admin", False)),
            "created_at": row.get("created_at").isoformat() if row.get("created_at") else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user: {e}")
