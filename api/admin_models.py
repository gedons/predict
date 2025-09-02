# app/api/admin_models.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from sqlalchemy import text
import json

from core.auth import admin_required
from db.database import get_db
from core.reload_pubsub import publish_model_reload


router = APIRouter(prefix="/admin/models", tags=["admin_models"])


def _parse_metadata(raw_meta: Optional[Any]) -> Dict[str, Any]:
    """Safely parse metadata stored as JSONB or string in the DB."""
    if not raw_meta:
        return {}
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, str):
        try:
            return json.loads(raw_meta)
        except Exception:
            return {"raw": raw_meta}
    try:
        return dict(raw_meta)
    except Exception:
        return {"raw": str(raw_meta)}


@router.get("/", dependencies=[Depends(admin_required)])
def list_models(limit: int = 100, offset: int = 0, db=Depends(get_db)):
    """
    List models in model_registry (paginated).
    """
    sql = text("""
        SELECT id, model_name, version, artifact_path, metadata, created_by, created_at, is_active
        FROM model_registry
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """)
    rows = db.execute(sql, {"limit": limit, "offset": offset}).mappings().all()

    models = []
    for r in rows:
        meta = _parse_metadata(r["metadata"])
        models.append({
            "id": r["id"],
            "model_name": r["model_name"],
            "version": r["version"],
            "artifact_path": r["artifact_path"],
            "metadata": meta,
            "created_by": r["created_by"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "is_active": bool(r["is_active"])
        })

    return {"count": len(models), "models": models}


@router.get("/{model_id}", dependencies=[Depends(admin_required)])
def get_model(model_id: int, db=Depends(get_db)):
    sql = text("""
        SELECT id, model_name, version, artifact_path, metadata, created_by, created_at, is_active
        FROM model_registry
        WHERE id = :id
    """)
    row = db.execute(sql, {"id": model_id}).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    meta = _parse_metadata(row["metadata"])
    return {
        "id": row["id"],
        "model_name": row["model_name"],
        "version": row["version"],
        "artifact_path": row["artifact_path"],
        "metadata": meta,
        "created_by": row["created_by"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "is_active": bool(row["is_active"])
    }


@router.post("/{model_id}/activate", dependencies=[Depends(admin_required)])
def activate_model(model_id: int, reload: bool = True, current_user: Dict = Depends(admin_required), db=Depends(get_db)):
    """
    Activate this model (set is_active=true; others false).
    If reload=True, attempt to load the model locally and publish a reload event for other workers.
    """
    update_sql_deactivate = text("UPDATE model_registry SET is_active = false WHERE is_active = true")
    update_sql_activate = text("""
        UPDATE model_registry
        SET is_active = true
        WHERE id = :id
        RETURNING id, artifact_path, metadata, model_name, created_by, created_at
    """)

    try:
        db.execute(update_sql_deactivate)
        res = db.execute(update_sql_activate, {"id": model_id}).mappings()
        row = res.fetchone()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB update failed: {e}")

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    artifact_path = row["artifact_path"]
    metadata = _parse_metadata(row["metadata"])

    reload_result = {"reloaded_here": False, "error": None}
    if reload:
        try:
            from app.api.predict import load_model_by_id
            load_model_by_id(model_id)
            reload_result["reloaded_here"] = True
        except Exception as e:
            reload_result["error"] = str(e)

    try:
        publish_model_reload(model_id)
        published = True
    except Exception as e:
        published = False
        print(f"Warning: failed to publish model reload for model_id={model_id}: {e}")

    return {
        "status": "activated",
        "model_id": model_id,
        "artifact_path": artifact_path,
        "metadata": metadata,
        "activated_by": current_user.get("email"),
        "reloaded_here": reload_result["reloaded_here"],
        "reload_error": reload_result["error"],
        "published_to_pubsub": published
    }


@router.post("/{model_id}/deactivate", dependencies=[Depends(admin_required)])
def deactivate_model(model_id: int, current_user: Dict = Depends(admin_required), db=Depends(get_db)):
    """
    Deactivate a model in registry (set is_active=false).
    """
    sql = text("UPDATE model_registry SET is_active = false WHERE id = :id RETURNING id")
    try:
        res = db.execute(sql, {"id": model_id}).mappings()
        row = res.fetchone()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB update failed: {e}")

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    return {"status": "deactivated", "model_id": model_id, "deactivated_by": current_user.get("email")}

@router.post("/{model_id}/reload", dependencies=[Depends(admin_required)])
def reload_model(model_id: int, db=Depends(get_db)):
    """
    Force the running worker to load the artifact for the given model id.
    Does not change DB active flags.
    """
    row = db.execute(
        text("SELECT id FROM model_registry WHERE id = :id"), {"id": model_id}
    ).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    try:
        from app.api.predict import load_model_by_id
        load_model_by_id(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")

    try:
        publish_model_reload(model_id)
    except Exception as e:
        print(f"Warning: publish_model_reload failed after reload_model: {e}")

    return {"status": "reloaded", "model_id": model_id}


    """
    Grant default quotas to a user (useful after registering).
    Payload: {"user_id": "...", "quota": 10 }
    """
    user_id = payload.get("user_id")
    quota = int(payload.get("quota", 10))
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    ok = create_default_quotas_for_user(db, user_id=user_id, quota=quota)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create default quotas")
    return {"status": "ok", "user_id": user_id, "granted_quota": quota}