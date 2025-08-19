# app/middleware/rate_limiter.py
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

REDIS_URL = os.getenv("REDIS_URL")

# Custom key func: prefer user id from request.state, otherwise IP
def user_or_ip_key(request: Request):
    # request.state.current_user_payload is set by auth middleware (if token provided)
    user_payload = getattr(request.state, "current_user_payload", None)
    if user_payload and user_payload.get("sub"):
        return f"user:{user_payload.get('sub')}"
    return get_remote_address(request)

# Create limiter using chosen key function and storage backend
limiter = Limiter(key_func=user_or_ip_key, storage_uri=REDIS_URL)

def init_rate_limiter(app: FastAPI):
    """
    Call this early when constructing the FastAPI app.
    It registers the exception handler for rate limit errors.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # type: ignore
