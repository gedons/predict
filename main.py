from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from middleware.rate_limiter import init_rate_limiter
from middleware.auth_middleware import TokenPayloadMiddleware
from api.predict import router as predict_router, load_model_at_startup
from api.admin_models import router as admin_models_router
from api.auth import router as auth_router
from api.llm_analysis import router as llm_router
from api.quotas import router as quotas_router
from api.admin_quotas import router as admin_quotas_router
from api.admin_users import router as admin_users_router
from api.admin_summary import router as admin_summary_router
from api.admin_analytics import router as admin_analytics_router
from api.external import router as external_router



# from app.db.database import get_db

app = FastAPI(
    title="Football Match Prediction API",
    description="API for match prediction project",
    version="1.0.0"
)

# token payload middleware
app.add_middleware(TokenPayloadMiddleware)

# init limiter
init_rate_limiter(app)

# Allow CORS in dev (lock down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "API is running!"}

# Load model + preprocessor on startup
@app.on_event("startup")
def startup_event():
    load_model_at_startup() 

app.include_router(predict_router, prefix="/predict", tags=["predict"])
app.include_router(admin_models_router)
app.include_router(auth_router)
app.include_router(llm_router)
app.include_router(quotas_router)
app.include_router(admin_quotas_router)
app.include_router(admin_users_router)
app.include_router(admin_summary_router)
app.include_router(admin_analytics_router)
app.include_router(external_router)