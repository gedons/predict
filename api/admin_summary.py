from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Any
from datetime import datetime, timedelta

from core.auth import admin_required
from db.database import get_db

router = APIRouter(prefix="/admin", tags=["admin_summary"])


@router.get("/summary", dependencies=[Depends(admin_required)])
def get_admin_summary(db = Depends(get_db)) -> Dict[str, Any]:
    """
    Return a small admin summary:
      - total_users
      - total_models, active_model (id/name/version)
      - total_quotas, quotas_with_remaining_null
      - llm_cache stats (total, last_24h)
      - recent_predictions_last_24h (if predictions_log exists)
      - server_time
    """
    try:
        # total users
        total_users = db.execute(text("SELECT COUNT(*) FROM users")).scalar() or 0

        # model counts + active model
        total_models = db.execute(text("SELECT COUNT(*) FROM model_registry")).scalar() or 0
        active_row = db.execute(text(
            "SELECT id, model_name, version, metadata, created_at FROM model_registry WHERE is_active = true ORDER BY created_at DESC LIMIT 1"
        )).mappings().fetchone()
        active_model = None
        if active_row:
            active_model = {
                "id": active_row["id"],
                "model_name": active_row.get("model_name"),
                "version": active_row.get("version"),
                "created_at": active_row.get("created_at").isoformat() if active_row.get("created_at") else None,
            }

        # quota counts (if user_quotas table exists)
        quotas_count = 0
        quotas_remaining_null = 0
        try:
            quotas_count = db.execute(text("SELECT COUNT(*) FROM user_quotas")).scalar() or 0
            quotas_remaining_null = db.execute(text("SELECT COUNT(*) FROM user_quotas WHERE remaining IS NULL")).scalar() or 0
        except Exception:
            # table may not exist; ignore
            quotas_count = 0
            quotas_remaining_null = 0

        # LLM cache stats (if llm_analysis_cache exists)
        llm_total = 0
        llm_last_24h = 0
        llm_last_entry = None
        try:
            llm_total = db.execute(text("SELECT COUNT(*) FROM llm_analysis_cache")).scalar() or 0
            llm_last_24h = db.execute(text("SELECT COUNT(*) FROM llm_analysis_cache WHERE created_at > now() - interval '24 hours'")).scalar() or 0
            last = db.execute(text("SELECT MAX(created_at) FROM llm_analysis_cache")).scalar()
            if last:
                try:
                    llm_last_entry = last.isoformat()
                except Exception:
                    llm_last_entry = str(last)
        except Exception:
            llm_total = 0
            llm_last_24h = 0
            llm_last_entry = None

        # recent predictions count (if predictions_log exists)
        recent_predictions_24h = 0
        try:
            recent_predictions_24h = db.execute(text("SELECT COUNT(*) FROM predictions_log WHERE created_at > now() - interval '24 hours'")).scalar() or 0
        except Exception:
            recent_predictions_24h = 0

        # optionally include some sample rows
        sample_quotas = []
        try:
            sample_quotas_rows = db.execute(text("SELECT user_id, endpoint, remaining, unlimited FROM user_quotas ORDER BY updated_at DESC LIMIT 6")).mappings().fetchall()
            sample_quotas = [dict(r) for r in sample_quotas_rows]
        except Exception:
            sample_quotas = []

        # optional models list (recent)
        recent_models = []
        try:
            recent_models_rows = db.execute(text("SELECT id, model_name, version, is_active, created_at FROM model_registry ORDER BY created_at DESC LIMIT 6")).mappings().fetchall()
            recent_models = [dict(r) for r in recent_models_rows]
            for m in recent_models:
                if m.get("created_at"):
                    try:
                        m["created_at"] = m["created_at"].isoformat()
                    except Exception:
                        pass
        except Exception:
            recent_models = []

        summary = {
            "server_time": datetime.utcnow().isoformat(),
            "users": {"total": int(total_users)},
            "models": {"total": int(total_models), "active_model": active_model, "recent_models": recent_models},
            "quotas": {"total": int(quotas_count), "remaining_null": int(quotas_remaining_null), "sample": sample_quotas},
            "llm_cache": {"total": int(llm_total), "last_24h": int(llm_last_24h), "last_entry": llm_last_entry},
            "predictions": {"recent_24h": int(recent_predictions_24h)}
        }
        return summary
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
