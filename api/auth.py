# app/api/auth.py
from datetime import datetime, timedelta
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy import text

from app.db.database import get_db
from app.core.auth import admin_required, get_current_user  

router = APIRouter(prefix="/auth", tags=["auth"])

# JWT / token settings (override via .env)
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: Optional[str]) -> bool:
    """Verify password using bcrypt (passlib). Fallback to plaintext compare for legacy/dev."""
    if not hashed_password:
        return False
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return plain_password == hashed_password


def get_user_by_username_or_email(db, username_or_email: str):
    """Get user row by email or username. Adjust SQL to fit your schema if different."""
    with db.begin() as conn:
        row = conn.execute(
            text(
                "SELECT id, email, username, password_hash, password, is_admin "
                "FROM users WHERE email = :u OR username = :u LIMIT 1"
            ),
            {"u": username_or_email}
        ).fetchone()
    return row


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload.update({"exp": expire})
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


@router.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db = Depends(get_db)):
    """
    Exchange username & password for JWT access token.
    - Form fields: username (or email), password
    """
    user_row = get_user_by_username_or_email(db, form_data.username)
    if not user_row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

    # find hashed/password column
    stored_hash = None
    for k in ("password_hash", "password"):
        if k in user_row.keys() and user_row[k]:
            stored_hash = user_row[k]
            break

    if not verify_password(form_data.password, stored_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

    user_id = user_row["id"]
    email = user_row.get("email") or user_row.get("username")
    payload = {
        "sub": str(user_id),
        "email": email,
        "is_admin": bool(user_row.get("is_admin", False))
    }
    access_token = create_access_token(payload)
    return {"access_token": access_token, "token_type": "bearer"}


# Useful helper endpoints for testing
@router.get("/me")
def read_current_user(user = Depends(get_current_user)):
    """Return decoded current user from token (DB-validated)."""
    return {"user": user}


@router.get("/admin-check")
def read_admin_check(admin_user = Depends(admin_required)):
    """Endpoint that only admins can call (use to validate admin token)."""
    return {"ok": True, "admin": admin_user}
