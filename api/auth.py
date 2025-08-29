# app/api/auth.py
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy import text

from app.db.database import get_db
from app.core.auth import admin_required, get_current_user, JWT_SECRET, JWT_ALGORITHM
from app.core.quota import create_default_quotas_for_user

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
    """Get user row by email or username. Works with SQLAlchemy ORM Session or raw SQL."""
    row = db.execute(
        text(
            "SELECT id, email, username, password_hash, is_admin "
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
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_db)
):
    """
    Exchange username & password for JWT access token.
    - Form fields: username (or email), password
    """
    user_row = get_user_by_username_or_email(db, form_data.username)
    if not user_row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    # Convert SQLAlchemy Row to a dict
    user_dict = dict(user_row._mapping)

    # find hashed/password column
    stored_hash = None
    for k in ("password_hash", "password"):
        if k in user_dict and user_dict[k]:
            stored_hash = user_dict[k]
            break

    if not stored_hash or not verify_password(form_data.password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    user_id = user_dict["id"]
    email = user_dict.get("email") or user_dict.get("username")
    payload = {
        "sub": str(user_id),
        "email": email,
        "is_admin": bool(user_dict.get("is_admin", False))
    }
    access_token = create_access_token(payload)
    return {"access_token": access_token, "token_type": "bearer"}

class RegisterRequest(dict):
    # simple container if you want to pydantic later
    pass

@router.post("/register", status_code=201)
def register_user(request_body: Dict[str, Any], db = Depends(get_db)):
    """
    Register a new user.
    JSON body: {"username": "...", "email": "...", "password": "..."}
    Note: this endpoint creates non-admin users only. To create admin users, use database or an admin-only route.
    """
    username = (request_body.get("username") or request_body.get("email") or "").strip()
    email = (request_body.get("email") or request_body.get("username") or "").strip()
    password = request_body.get("password")

    if not password or not (username or email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="username/email and password required")

    # check for existing user
    existing = db.execute(
        text("SELECT id FROM users WHERE email = :email OR username = :username LIMIT 1"),
        {"email": email, "username": username}
    ).fetchone()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User with that email or username already exists")

    # hash password
    hashed = pwd_context.hash(password)

    # Insert user (adjust columns to match your DB schema)
    insert_sql = text(
        "INSERT INTO users (username, email, password_hash, is_admin, created_at) VALUES (:username, :email, :password_hash, false, now()) RETURNING id"
    )
    res = db.execute(insert_sql, {"username": username, "email": email, "password_hash": hashed})
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {e}")

    row = res.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="User creation failed")
    user_id = row[0]

    create_default_quotas_for_user(db, user_id, quota=10, q_limit=10)

    # return a token for convenience
    payload = {"sub": str(user_id), "email": email, "is_admin": False}
    token = create_access_token(payload)
    return {"id": user_id, "access_token": token, "token_type": "bearer"}

# Useful helper endpoints for testing
@router.get("/me")
def read_current_user(user = Depends(get_current_user)):
    """Return decoded current user from token (DB-validated)."""
    return {"user": user}


@router.get("/admin-check")
def read_admin_check(admin_user = Depends(admin_required)):
    """Endpoint that only admins can call (use to validate admin token)."""
    return {"ok": True, "admin": admin_user}
