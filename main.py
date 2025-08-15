from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from app.api.predict import router as predict_router, load_model_at_startup
from app.api.admin_models import router as admin_models_router
from app.api.auth import router as auth_router

from app.db.database import get_db

app = FastAPI(
    title="Football Match Prediction API",
    description="API for match prediction project",
    version="1.0.0"
)

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
