# scripts/fetch_data.py
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set DATABASE_URL in .env")

engine = create_engine(DATABASE_URL, future=True)

def fetch_matches(min_date=None):
    sql = "SELECT * FROM public.matches WHERE date IS NOT NULL ORDER BY date"
    params = {}
    if min_date:
        sql = "SELECT * FROM public.matches WHERE date >= :min_date ORDER BY date"
        params["min_date"] = min_date
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df

if __name__ == "__main__":
    df = fetch_matches()
    print(f"Rows fetched: {len(df)}")
    print(df.columns.tolist())
