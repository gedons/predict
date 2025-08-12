#!/usr/bin/env python3
"""
scripts/clean_football_data.py
Clean and standardize downloaded football data CSVs.
Processes data from data/raw/<league>/<season>/*.csv 
and saves to data/processed/<league>/<season>/*.csv
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Ensure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Column mapping for standardization
COLUMN_RENAME_MAP = {
    # Basic match info
    "div": "division",
    "date": "date",
    "time": "kick_off_time",
    "hometeam": "home_team",
    "awayteam": "away_team",
    "referee": "referee",
    
    # Full time results
    "fthg": "full_time_home_goals",
    "ftag": "full_time_away_goals",
    "ftr": "full_time_result",
    
    # Half time results
    "hthg": "half_time_home_goals",
    "htag": "half_time_away_goals",
    "htr": "half_time_result",
    
    # Match statistics
    "hs": "home_shots",
    "as": "away_shots",
    "hst": "home_shots_on_target",
    "ast": "away_shots_on_target",
    "hf": "home_fouls",
    "af": "away_fouls",
    "hc": "home_corners",
    "ac": "away_corners",
    "hy": "home_yellow_cards",
    "ay": "away_yellow_cards",
    "hr": "home_red_cards",
    "ar": "away_red_cards",
    
    # Bet365 1X2 odds
    "b365h": "b365_home_odds",
    "b365d": "b365_draw_odds",
    "b365a": "b365_away_odds",
    
    # Other bookmaker 1X2 odds
    "bfdh": "betfred_home_odds",
    "bfdd": "betfred_draw_odds",
    "bfda": "betfred_away_odds",
    "bmgmh": "bmgm_home_odds",
    "bmgmd": "bmgm_draw_odds",
    "bmgma": "bmgm_away_odds",
    "bvh": "betvictor_home_odds",
    "bvd": "betvictor_draw_odds",
    "bva": "betvictor_away_odds",
    "bwh": "betway_home_odds",
    "bwd": "betway_draw_odds",
    "bwa": "betway_away_odds",
    "clh": "coral_home_odds",
    "cld": "coral_draw_odds",
    "cla": "coral_away_odds",
    "lbh": "ladbrokes_home_odds",
    "lbd": "ladbrokes_draw_odds",
    "lba": "ladbrokes_away_odds",
    "psh": "pinnacle_home_odds",
    "psd": "pinnacle_draw_odds",
    "psa": "pinnacle_away_odds",
    
    # Market maximum/average odds
    "maxh": "max_home_odds",
    "maxd": "max_draw_odds",
    "maxa": "max_away_odds",
    "avgh": "avg_home_odds",
    "avgd": "avg_draw_odds",
    "avga": "avg_away_odds",
    
    # Betfair exchange odds
    "bfeh": "betfair_ex_home_odds",
    "bfed": "betfair_ex_draw_odds",
    "bfea": "betfair_ex_away_odds",
    
    # Total goals over/under 2.5
    "b365>2.5": "b365_over_2_5",
    "b365<2.5": "b365_under_2_5",
    "p>2.5": "pinnacle_over_2_5",
    "p<2.5": "pinnacle_under_2_5",
    "max>2.5": "max_over_2_5",
    "max<2.5": "max_under_2_5",
    "avg>2.5": "avg_over_2_5",
    "avg<2.5": "avg_under_2_5",
    "bfe>2.5": "betfair_ex_over_2_5",
    "bfe<2.5": "betfair_ex_under_2_5",
    
    # Asian handicap
    "ahh": "asian_handicap_home",
    "b365ahh": "b365_ah_home_odds",
    "b365aha": "b365_ah_away_odds",
    "pahh": "pinnacle_ah_home_odds",
    "paha": "pinnacle_ah_away_odds",
    "maxahh": "max_ah_home_odds",
    "maxaha": "max_ah_away_odds",
    "avgahh": "avg_ah_home_odds",
    "avgaha": "avg_ah_away_odds",
    "bfeahh": "betfair_ex_ah_home_odds",
    "bfeaha": "betfair_ex_ah_away_odds",
    
    # Closing odds (1X2)
    "b365ch": "b365_closing_home_odds",
    "b365cd": "b365_closing_draw_odds",
    "b365ca": "b365_closing_away_odds",
    "bfdch": "betfred_closing_home_odds",
    "bfdcd": "betfred_closing_draw_odds",
    "bfdca": "betfred_closing_away_odds",
    "bmgmch": "bmgm_closing_home_odds",
    "bmgmcd": "bmgm_closing_draw_odds",
    "bmgmca": "bmgm_closing_away_odds",
    "bvch": "betvictor_closing_home_odds",
    "bvcd": "betvictor_closing_draw_odds",
    "bvca": "betvictor_closing_away_odds",
    "bwch": "betway_closing_home_odds",
    "bwcd": "betway_closing_draw_odds",
    "bwca": "betway_closing_away_odds",
    "clch": "coral_closing_home_odds",
    "clcd": "coral_closing_draw_odds",
    "clca": "coral_closing_away_odds",
    "lbch": "ladbrokes_closing_home_odds",
    "lbcd": "ladbrokes_closing_draw_odds",
    "lbca": "ladbrokes_closing_away_odds",
    "psch": "pinnacle_closing_home_odds",
    "pscd": "pinnacle_closing_draw_odds",
    "psca": "pinnacle_closing_away_odds",
    "maxch": "max_closing_home_odds",
    "maxcd": "max_closing_draw_odds",
    "maxca": "max_closing_away_odds",
    "avgch": "avg_closing_home_odds",
    "avgcd": "avg_closing_draw_odds",
    "avgca": "avg_closing_away_odds",
    "bfech": "betfair_ex_closing_home_odds",
    "bfecd": "betfair_ex_closing_draw_odds",
    "bfeca": "betfair_ex_closing_away_odds",
    
    # Closing total goals over/under 2.5
    "b365c>2.5": "b365_closing_over_2_5",
    "b365c<2.5": "b365_closing_under_2_5",
    "pc>2.5": "pinnacle_closing_over_2_5",
    "pc<2.5": "pinnacle_closing_under_2_5",
    "maxc>2.5": "max_closing_over_2_5",
    "maxc<2.5": "max_closing_under_2_5",
    "avgc>2.5": "avg_closing_over_2_5",
    "avgc<2.5": "avg_closing_under_2_5",
    "bfec>2.5": "betfair_ex_closing_over_2_5",
    "bfec<2.5": "betfair_ex_closing_under_2_5",
    
    # Closing Asian handicap
    "ahch": "closing_asian_handicap_home",
    "b365cahh": "b365_closing_ah_home_odds",
    "b365caha": "b365_closing_ah_away_odds",
    "pcahh": "pinnacle_closing_ah_home_odds",
    "pcaha": "pinnacle_closing_ah_away_odds",
    "maxcahh": "max_closing_ah_home_odds",
    "maxcaha": "max_closing_ah_away_odds",
    "avgcahh": "avg_closing_ah_home_odds",
    "avgcaha": "avg_closing_ah_away_odds",
    "bfecahh": "betfair_ex_closing_ah_home_odds",
    "bfecaha": "betfair_ex_closing_ah_away_odds",
}

# Priority columns to keep (in order of preference)
PRIORITY_COLUMNS = [
    # Essential match info
    "division", "date", "kick_off_time", "home_team", "away_team", "referee",
    
    # Match results
    "full_time_home_goals", "full_time_away_goals", "full_time_result",
    "half_time_home_goals", "half_time_away_goals", "half_time_result",
    
    # Match statistics
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_fouls", "away_fouls", "home_corners", "away_corners",
    "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards",
    
    # Main bookmaker odds (Bet365)
    "b365_home_odds", "b365_draw_odds", "b365_away_odds",
    
    # Market summary odds
    "max_home_odds", "max_draw_odds", "max_away_odds",
    "avg_home_odds", "avg_draw_odds", "avg_away_odds",
    
    # Over/Under 2.5 goals
    "b365_over_2_5", "b365_under_2_5",
    "avg_over_2_5", "avg_under_2_5",
    
    # Asian handicap
    "asian_handicap_home", "b365_ah_home_odds", "b365_ah_away_odds",
    
    # Other popular bookmakers
    "pinnacle_home_odds", "pinnacle_draw_odds", "pinnacle_away_odds",
    "betfair_ex_home_odds", "betfair_ex_draw_odds", "betfair_ex_away_odds",
    
    # Metadata (added by script)
    "season", "league", "match_id"
]


def clean_csv(file_path: Path, league: str, season: str) -> Optional[pd.DataFrame]:
    """
    Clean a single CSV file and return the processed DataFrame.
    
    Args:
        file_path: Path to the raw CSV file
        league: League name (e.g., 'premier_league')
        season: Season string (e.g., '2023_2024')
    
    Returns:
        Cleaned DataFrame or None if processing failed
    """
    try:
        logger.info(f"Processing: {file_path}")
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        
        if df.empty:
            logger.warning(f"Empty CSV file: {file_path}")
            return None

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "")
        
        # Rename columns according to our mapping
        df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
        
        # Keep only columns that exist in our dataframe
        available_cols = [col for col in PRIORITY_COLUMNS if col in df.columns or col in ["season", "league", "match_id"]]
        
        # Add metadata columns
        df["season"] = season
        df["league"] = league
        
        # Convert date to standard format
        if "date" in df.columns:
            try:
                # Try different date formats
                df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                # Drop rows with invalid dates
                df = df.dropna(subset=["date"])
            except Exception as e:
                logger.warning(f"Date conversion failed for {file_path}: {e}")
        
        # Create unique match_id
        if all(col in df.columns for col in ["season", "league", "date", "home_team", "away_team"]):
            df["match_id"] = (
                df["season"].astype(str) + "_" +
                df["league"].astype(str) + "_" +
                df["date"].astype(str) + "_" +
                df["home_team"].str.replace(r"[^a-zA-Z0-9]", "", regex=True) + "_" +
                df["away_team"].str.replace(r"[^a-zA-Z0-9]", "", regex=True)
            )
        
        # Select and reorder columns
        final_cols = [col for col in available_cols if col in df.columns]
        df = df[final_cols]
        
        # Basic data quality checks
        if df.empty:
            logger.warning(f"No data remaining after cleaning: {file_path}")
            return None
            
        logger.info(f"Successfully processed {len(df)} rows from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None


def save_cleaned_data(df: pd.DataFrame, league: str, season: str, original_filename: str) -> bool:
    """
    Save cleaned DataFrame to the processed directory structure.
    
    Args:
        df: Cleaned DataFrame
        league: League name
        season: Season string  
        original_filename: Original filename for reference
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create output directory structure: data/processed/<league>/<season>/
        output_dir = PROCESSED_DIR / league / season
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use original filename but ensure it ends with .csv
        output_filename = Path(original_filename).stem + "_cleaned.csv"
        output_path = output_dir / output_filename
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save data for {league}/{season}: {e}")
        return False


def process_all_data() -> Dict[str, Any]:
    """
    Process all raw CSV files and generate summary statistics.
    
    Returns:
        Dictionary with processing summary
    """
    summary = {
        "total_files": 0,
        "processed_files": 0,
        "failed_files": 0,
        "total_matches": 0,
        "leagues": set(),
        "seasons": set(),
        "failed_files_list": []
    }
    
    if not RAW_DIR.exists():
        logger.error(f"Raw data directory not found: {RAW_DIR}")
        return summary
    
    # Process each league directory
    for league_dir in RAW_DIR.iterdir():
        if not league_dir.is_dir():
            continue
            
        league_name = league_dir.name
        summary["leagues"].add(league_name)
        logger.info(f"Processing league: {league_name}")
        
        # Process each season directory within the league
        for season_dir in league_dir.iterdir():
            if not season_dir.is_dir():
                continue
                
            season_name = season_dir.name
            summary["seasons"].add(season_name)
            logger.info(f"Processing season: {league_name}/{season_name}")
            
            # Process all CSV files in the season directory
            csv_files = list(season_dir.glob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in: {season_dir}")
                continue
                
            for csv_file in csv_files:
                summary["total_files"] += 1
                
                # Clean the CSV file
                cleaned_df = clean_csv(csv_file, league_name, season_name)
                
                if cleaned_df is not None:
                    # Save the cleaned data
                    if save_cleaned_data(cleaned_df, league_name, season_name, csv_file.name):
                        summary["processed_files"] += 1
                        summary["total_matches"] += len(cleaned_df)
                    else:
                        summary["failed_files"] += 1
                        summary["failed_files_list"].append(str(csv_file))
                else:
                    summary["failed_files"] += 1
                    summary["failed_files_list"].append(str(csv_file))
    
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print processing summary."""
    print("\n" + "="*50)
    print("FOOTBALL DATA CLEANING SUMMARY")
    print("="*50)
    print(f"Total files processed: {summary['total_files']}")
    print(f"Successfully cleaned: {summary['processed_files']}")
    print(f"Failed to process: {summary['failed_files']}")
    print(f"Total matches processed: {summary['total_matches']:,}")
    print(f"Leagues found: {len(summary['leagues'])}")
    print(f"Seasons found: {len(summary['seasons'])}")
    
    if summary['leagues']:
        print(f"Leagues: {', '.join(sorted(summary['leagues']))}")
    if summary['seasons']:
        print(f"Seasons: {', '.join(sorted(summary['seasons']))}")
        
    if summary['failed_files_list']:
        print(f"\nFailed files:")
        for failed_file in summary['failed_files_list']:
            print(f"  - {failed_file}")
    
    print("="*50)


def main():
    """Main execution function."""
    logger.info("Starting football data cleaning process...")
    
    if not RAW_DIR.exists():
        logger.error(f"Raw data directory not found: {RAW_DIR}")
        print(f"Please ensure the raw data directory exists: {RAW_DIR}")
        return
    
    # Process all data
    summary = process_all_data()
    
    # Print summary
    print_summary(summary)
    
    if summary['processed_files'] > 0:
        logger.info("Data cleaning completed successfully!")
        print(f"\nCleaned data is available in: {PROCESSED_DIR}")
    else:
        logger.warning("No files were successfully processed!")


if __name__ == "__main__":
    main()