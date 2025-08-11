from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv() 

app = FastAPI(title="Match Prediction API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to Match Prediction API"}
