#!/usr/bin/env python3
"""
scripts/ingest_processed_to_db_nested.py

Ingest processed CSVs living in:
  data/processed/<league>/<season>/*.csv

For each CSV:
 - normalize columns into the expected column set (fill missing with NULL)
 - add `source_file` column
 - COPY into a temp table
 - upsert into public.matches using match_id UNIQUE constraint
 - on failure -> log and optionally move file to data/failed_ingestion/
"""

import os
import csv
import shutil
from pathlib import Path
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import psycopg2

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
PROCESSED_DIR = Path("data/processed")
FAILED_DIR = Path("data/failed_ingestion")
TMP_DIR = Path("data/tmp_prepared")
TMP_DIR.mkdir(parents=True, exist_ok=True)
FAILED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# canonical columns we will COPY into tmp_matches (order matters)
COLUMNS = [
    "match_id", "season", "league", "date",
    "home_team", "away_team",
    "full_time_home_goals", "full_time_away_goals", "full_time_result",
    "half_time_home_goals", "half_time_away_goals", "half_time_result",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_fouls", "away_fouls", "home_corners", "away_corners",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "b365_home_odds", "b365_draw_odds", "b365_away_odds",
    "source_file"
]

# common alternative headers mapping (lowercased)
ALT_HEADER_MAP = {
    "hometeam": "home_team",
    "awayteam": "away_team",
    "fthg": "full_time_home_goals",
    "ftag": "full_time_away_goals",
    "ftr": "full_time_result",
    "hthg": "half_time_home_goals",
    "htag": "half_time_away_goals",
    "htr": "half_time_result",
    "hs": "home_shots",
    "as": "away_shots",
    "hst": "home_shots_on_target",
    "ast": "away_shots_on_target",
    "hf": "home_fouls",
    "af": "away_fouls",
    "hc": "home_corners",
    "ac": "away_corners",
    "hy": "home_yellow",
    "ay": "away_yellow",
    "hr": "home_red",
    "ar": "away_red",
    "b365h": "b365_home_odds",
    "b365d": "b365_draw_odds",
    "b365a": "b365_away_odds",
    "season": "season",
    "league": "league",
    "date": "date",
    "match_id": "match_id"
}

def ensure_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn

def find_all_csvs():
    return sorted(PROCESSED_DIR.glob("**/*.csv"))

def infer_league_season_from_path(path: Path):
    """
    Expect path like data/processed/<league>/<season>/<file>.csv
    Returns (league, season) or (None, None)
    """
    parts = path.resolve().parts
    # find "data", "processed" indices
    try:
        idx = parts.index("processed")
        # league is next, season is next
        league = parts[idx + 1]
        season = parts[idx + 2]
        return league, season
    except Exception:
        return None, None

def prepare_csv(original_path: Path):
    """
    Produce a temp CSV with exactly the COLUMNS ordering.
    Returns path to prepared CSV (Path) or raises on unrecoverable error.
    """
    league_from_path, season_from_path = infer_league_season_from_path(original_path)
    temp_path = TMP_DIR / (original_path.stem + ".prepared.csv")

    encodings = ["utf-8", "latin1", "cp1252"]
    reader = None
    inf = None
    last_exc = None

    for enc in encodings:
        try:
            # Read a sample to detect delimiter
            with open(original_path, "r", encoding=enc, errors="replace") as f:
                sample = f.read(4096)
                if not sample or not sample.strip():
                    raise RuntimeError("Empty or whitespace-only file")
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    delim = dialect.delimiter
                except Exception:
                    delim = ","
            # Re-open for actual DictReader
            inf = open(original_path, "r", encoding=enc, errors="replace", newline="")
            reader = csv.DictReader(inf, delimiter=delim)
            # If DictReader couldn't detect headers, fallback to pandas to detect columns
            if reader.fieldnames is None:
                try:
                    import pandas as pd
                    # read only header row with python engine to auto-detect delimiter
                    df = pd.read_csv(original_path, encoding=enc, sep=None, engine="python", nrows=0)
                    fieldnames = list(df.columns)
                    inf.close()
                    inf = open(original_path, "r", encoding=enc, errors="replace", newline="")
                    reader = csv.DictReader(inf, fieldnames=fieldnames)
                except Exception as e:
                    # pandas failed, try next encoding
                    last_exc = e
                    try:
                        inf.close()
                    except Exception:
                        pass
                    reader = None
                    continue
            # At this point we have a reader (maybe with headers or with pandas-provided fieldnames)
            break
        except Exception as e:
            last_exc = e
            # ensure file handle closed
            try:
                if inf:
                    inf.close()
            except Exception:
                pass
            reader = None
            continue

    if reader is None:
        raise RuntimeError(f"Could not open/parse CSV {original_path}: {last_exc}")

    # Normalize header mapping
    input_fieldnames = [h.strip() for h in (reader.fieldnames or [])]
    mapped_names = {}
    for h in input_fieldnames:
        key = h.lower().strip()
        if key in ALT_HEADER_MAP:
            mapped_names[h] = ALT_HEADER_MAP[key]
        else:
            k2 = key.replace(" ", "_")
            mapped_names[h] = ALT_HEADER_MAP.get(k2, k2)

    # Write prepared file with canonical columns
    with open(temp_path, "w", encoding="utf-8", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=COLUMNS)
        writer.writeheader()
        # iterate rows via the csv.DictReader we prepared
        for row in reader:
            outrow = {}
            for col in COLUMNS:
                if col == "source_file":
                    outrow[col] = str(original_path)
                    continue
                # direct match
                if col in row and row.get(col) is not None:
                    v = row.get(col)
                    outrow[col] = v.strip() if isinstance(v, str) else v
                    continue
                # mapped header match
                found = False
                for orig_h, mapped in mapped_names.items():
                    if mapped == col:
                        outrow[col] = (row.get(orig_h) or "").strip()
                        found = True
                        break
                if found:
                    continue
                # fallback to path-inferred season/league
                if col == "season":
                    outrow[col] = season_from_path or ""
                elif col == "league":
                    outrow[col] = league_from_path or ""
                else:
                    outrow[col] = ""
            writer.writerow(outrow)

    # close the original file handle if still open
    try:
        if inf:
            inf.close()
    except Exception:
        pass

    return temp_path

def create_temp_table(cur):
    # temp table like public.matches (we select only the columns we inserted)
    cur.execute("CREATE TEMP TABLE tmp_matches (LIKE public.matches INCLUDING DEFAULTS) ON COMMIT DROP;")

def copy_csv_to_temp(cur, prepared_csv_path: Path):
    cols_sql = ", ".join(COLUMNS)
    sql = f"COPY tmp_matches ({cols_sql}) FROM STDIN WITH CSV HEADER DELIMITER ',' NULL ''"
    with open(prepared_csv_path, "r", encoding="utf-8") as f:
        cur.copy_expert(sql, f)

def upsert_from_temp(cur):
    # Build insert/upsert SQL (same logic as earlier script)
    target_cols = [c for c in COLUMNS if c != "source_file"] + ["source_file"]
    insert_cols_sql = ", ".join(target_cols)
    select_cols_sql = ", ".join(target_cols)

    update_cols = [c for c in target_cols if c not in ("match_id",)]
    update_sql = ", ".join([f"{c}=EXCLUDED.{c}" for c in update_cols])

    sql = f"""
    INSERT INTO public.matches ({insert_cols_sql})
    SELECT {select_cols_sql} FROM tmp_matches
    ON CONFLICT (match_id) DO UPDATE SET {update_sql};
    """
    cur.execute(sql)

def process_single(conn, original_csv: Path):
    logging.info(f"Start processing: {original_csv}")
    prepared = None
    cur = conn.cursor()
    try:
        prepared = prepare_csv(original_csv)
        create_temp_table(cur)
        copy_csv_to_temp(cur, prepared)
        upsert_from_temp(cur)
        conn.commit()
        logging.info(f"Upsert successful: {original_csv}")
        return True
    except Exception as e:
        conn.rollback()
        logging.exception(f"Failed to ingest {original_csv}: {e}")
        # move failed file to FAILED_DIR for inspection
        dest = FAILED_DIR / original_csv.name
        try:
            shutil.copy2(original_csv, dest)
            logging.info(f"Copied failed file to {dest}")
        except Exception:
            logging.warning("Failed to copy failed file to failed_ingestion dir.")
        return False
    finally:
        try:
            cur.close()
        except Exception:
            pass
        # cleanup prepared file
        if prepared and prepared.exists():
            try:
                prepared.unlink()
            except Exception:
                pass

def main():
    files = find_all_csvs()
    if not files:
        logging.info("No processed CSV files found under data/processed/")
        return
    conn = ensure_conn()
    successes = 0
    failures = 0
    try:
        for csv_path in tqdm(files, desc="Processing CSVs"):
            ok = process_single(conn, csv_path)
            if ok:
                successes += 1
            else:
                failures += 1
        logging.info(f"Ingestion complete. success={successes}, failed={failures}, total={len(files)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
