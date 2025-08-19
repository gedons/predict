# app/middleware/auth_middleware.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
from app.core.auth import decode_jwt   
import os

class TokenPayloadMiddleware(BaseHTTPMiddleware):
    """
    Middleware: if Authorization: Bearer <token> present, decode and attach payload
    to request.state.current_user_payload so other pieces (rate limiter) can use it.
    Keep it tolerant: don't raise if the token is invalid — the dependency-based
    auth will still block when required.
    """
    async def dispatch(self, request: Request, call_next: Callable):
        auth = request.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
            try:
                payload = decode_jwt(token)
                # keep minimal payload on request.state
                request.state.current_user_payload = payload
            except Exception:
                # Ignore decoding errors here (dependency will enforce auth)
                request.state.current_user_payload = None
        else:
            request.state.current_user_payload = None

        response = await call_next(request)
        return response
