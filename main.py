from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import get_db

app = FastAPI(
    title="Football Match Prediction API",
    description="API for match prediction project",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.get("/test-db")
def test_database(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT 1")).scalar()
        if result == 1:
            return {"status": "success", "message": "Database connection successful ✅"}
        else:
            return {"status": "error", "message": "Unexpected DB response ⚠️"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
