# app/api/admin_models.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
import os
import json
from app.core.auth import admin_required
from app.db.database import get_db

router = APIRouter(prefix="/admin/models", tags=["admin_models"])

DATABASE_URL = os.getenv("DATABASE_URL")

def _engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return create_engine(DATABASE_URL)

@router.get("/", dependencies=[Depends(admin_required)])
def list_models(limit: int = 100, offset: int = 0):
    engine = _engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, model_name, version, artifact_path, metadata, created_by, created_at, is_active FROM model_registry ORDER BY created_at DESC LIMIT :limit OFFSET :offset"), {"limit": limit, "offset": offset}).fetchall()
    result = []
    for r in rows:
        meta = r["metadata"] if r["metadata"] is not None else {} # type: ignore
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        result.append({
            "id": r["id"], # type: ignore
            "model_name": r["model_name"], # type: ignore
            "version": r["version"], # type: ignore
            "artifact_path": r["artifact_path"], # type: ignore
            "metadata": meta,
            "created_by": r["created_by"], # type: ignore
            "created_at": r["created_at"].isoformat() if r["created_at"] else None, # type: ignore
            "is_active": bool(r["is_active"]) # type: ignore
        })
    return {"count": len(result), "models": result}

@router.get("/{model_id}", dependencies=[Depends(admin_required)])
def get_model(model_id: int):
    engine = _engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, model_name, version, artifact_path, metadata, created_by, created_at, is_active FROM model_registry WHERE id = :id"), {"id": model_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    meta = row["metadata"] or {} # type: ignore
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except:
            meta = {}
    return {
        "id": row["id"], # type: ignore
        "model_name": row["model_name"], # type: ignore
        "version": row["version"], # type: ignore
        "artifact_path": row["artifact_path"], # type: ignore
        "metadata": meta,
        "created_by": row["created_by"], # type: ignore
        "created_at": row["created_at"].isoformat() if row["created_at"] else None, # type: ignore
        "is_active": bool(row["is_active"]) # type: ignore
    }

@router.post("/{model_id}/activate", dependencies=[Depends(admin_required)])
def activate_model(model_id: int, reload: bool = True):
    """
    Activate this model (set is_active=true; others false).
    If reload=True, call the running server's loader to load into memory.
    """
    engine = _engine()
    with engine.begin() as conn:
        # deactivate others, activate this model
        conn.execute(text("UPDATE model_registry SET is_active = false WHERE is_active = true"))
        res = conn.execute(text("UPDATE model_registry SET is_active = true WHERE id = :id RETURNING id, artifact_path, metadata"), {"id": model_id})
        row = res.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")

        artifact_path = row["artifact_path"] # type: ignore
        metadata = row["metadata"] or {} # type: ignore
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

    # optionally reload into memory immediately
    if reload:
        try:
            # call the loader that reads DB-active model
            from app.api.predict import load_model_at_startup
            load_model_at_startup()
        except Exception as e:
            # activation succeeded in DB but reload failed
            return {"status": "activated_in_db", "reload": "failed", "error": str(e)}

    return {"status": "activated", "model_id": model_id, "artifact_path": artifact_path, "metadata": metadata}

@router.post("/{model_id}/deactivate", dependencies=[Depends(admin_required)])
def deactivate_model(model_id: int):
    engine = _engine()
    with engine.begin() as conn:
        res = conn.execute(text("UPDATE model_registry SET is_active = false WHERE id = :id RETURNING id"), {"id": model_id})
        row = res.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deactivated", "model_id": model_id}

@router.post("/{model_id}/reload", dependencies=[Depends(admin_required)])
def reload_model(model_id: int):
    """
    Force the running process to load the artifact for the given model id.
    (Useful if you want to reload without changing DB active flag.)
    """
    engine = _engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, artifact_path, metadata FROM model_registry WHERE id = :id"), {"id": model_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    # If the artifact_path is stored as URL in DB metadata or artifact_path column, we still rely on load_model_at_startup() to handle download.
    try:
        from app.api.predict import load_model_at_startup
        load_model_at_startup()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
    return {"status": "reloaded", "model_id": model_id}
