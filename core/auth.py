# app/core/auth.py
import os
from typing import Dict, Any

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import text

from db.database import get_db

# The tokenUrl below must match the token endpoint path in app/api/auth.py
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

JWT_SECRET = os.getenv("JWT_SECRET", "changeme")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def decode_jwt(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT. Raises 401 on failure."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    payload = decode_jwt(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload (missing sub)")

    try:
        row = db.execute(
            text("SELECT id, email, username, is_admin FROM users WHERE id = :uid LIMIT 1"),
            {"uid": user_id}
        ).mappings().first()
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Failed to fetch user: {e}")

    if not row:
        raise HTTPException(status_code=401, detail="User not found")

    return {
        "id": row["id"],
        "email": row.get("email") or row.get("username"),
        "is_admin": bool(row.get("is_admin", False))
    }

def admin_required(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency that enforces the user must be an admin.
    Use as: dependencies=[Depends(admin_required)] or as a path param to obtain user.
    """
    if not user.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return user
    