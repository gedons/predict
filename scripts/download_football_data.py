#!/usr/bin/env python3
"""
scripts/download_football_data.py
Download CSVs listed from football-data.co.uk england page into data/raw/<league>/<season>/
"""

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm


BASE_HEADERS = {
    "User-Agent": "match-prediction-scraper/1.0 (+https://example.com)"
}

SEASON_RE = re.compile(r'(19|20)\d{2}[\/\-_]\d{2,4}')  # e.g., 2023/2024 or 2019-20
LEAGUE_KEYWORDS = [
    "Premier League", "Championship", "League 1", "League 2", "Conference",
    "Premier", "Championship", "League One", "League Two"
]


def slugify(text: Optional[str]) -> str:
    """Convert text to a safe filename slug."""
    if not text:
        return "unknown"
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s/\\]+', '_', text)
    return text.lower() or "unknown"


def fetch_soup(url: str, timeout: int = 15) -> BeautifulSoup:
    """Fetch and parse HTML from URL."""
    r = requests.get(url, headers=BASE_HEADERS, timeout=timeout)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def find_csv_links(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """Find all CSV links on the page and extract metadata."""
    results = []
    
    for a in soup.find_all("a", href=True):
        # Type guard to ensure we have a Tag object
        if not isinstance(a, Tag):
            continue
            
        href = a.get("href")
        if not href or not isinstance(href, str):
            continue
            
        href = href.strip()
        if ".csv" not in href.lower():  # quick filter
            continue
            
        full_url = urljoin(base_url, href)
        text = a.get_text(" ", strip=True)
        
        # Find season by searching nearby text
        season = None
        season_match = a.find_previous(string=SEASON_RE)
        if season_match:
            m = SEASON_RE.search(str(season_match))
            if m:
                season = m.group(0).replace("/", "_").replace("-", "_")
                
        # Find league by looking at previous headings/strings
        league = None
        header = a.find_previous(['h1', 'h2', 'h3', 'h4', 'strong', 'b', 'p', 'div'])
        iter_count = 0
        
        while header and iter_count < 10:
            if isinstance(header, Tag):
                txt = header.get_text(" ", strip=True)
                for key in LEAGUE_KEYWORDS:
                    if key.lower() in txt.lower():
                        league = key
                        break
                if league:
                    break
            header = header.find_previous(['h1', 'h2', 'h3', 'h4', 'strong', 'b', 'p', 'div'])
            iter_count += 1

        # Ensure we have a valid filename - always return a string
        filename = os.path.basename(urlparse(full_url).path)
        if not filename:
            filename = text.replace(" ", "_") if text else "unknown_file"
        if not filename:
            filename = "unknown_file"
        
        results.append({
            "url": full_url,
            "anchor_text": text or "",
            "season": season or "unknown",
            "league": league or "unknown", 
            "filename": filename
        })
    
    return results


def download_file(
    url: str, 
    out_path: Path, 
    chunk_size: int = 1024, 
    retries: int = 3, 
    sleep_between: float = 1.0
) -> bool:
    """Download a file from URL to local path with progress bar and retries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".part")
    
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, headers=BASE_HEADERS, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                
                with open(temp_path, "wb") as f, tqdm(
                    total=total, 
                    unit="B", 
                    unit_scale=True,
                    desc=f"Downloading {out_path.name}", 
                    leave=False
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            temp_path.rename(out_path)
            return True
            
        except Exception as e:
            print(f"[warn] download failed (attempt {attempt + 1}/{retries}) for {url}: {e}")
            if attempt < retries - 1:  
                time.sleep(sleep_between * (attempt + 1))
    
    return False


def main(page_url: str, out_dir: str, sleep: float = 0.5, overwrite: bool = False) -> None:
    """Main function to download all CSV files from the page."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching index page: {page_url}")
    soup = fetch_soup(page_url)
    entries = find_csv_links(soup, page_url)
    print(f"Found {len(entries)} CSV links.")
    
    summary = {"downloaded": 0, "skipped": 0, "failed": 0}
    
    for i, entry in enumerate(entries, 1):
        print(f"Processing {i}/{len(entries)}: {entry['filename']}")
        
        # All values are guaranteed to be strings now
        league_slug = slugify(entry["league"])
        season_slug = slugify(entry["season"])
        filename = entry["filename"]
        
        # Create safe filename
        safe_name = slugify(filename)
        out_path = out_base / league_slug / season_slug / f"{safe_name}.csv"
        
        # Check if file already exists
        if out_path.exists() and not overwrite:
            print(f"[skip] File already exists: {out_path}")
            summary["skipped"] += 1
            continue
        
        # Download the file
        print(f"[download] {entry['url']} -> {out_path}")
        success = download_file(entry["url"], out_path)
        
        if success:
            print(f"[ok] Successfully saved to: {out_path}")
            summary["downloaded"] += 1
        else:
            print(f"[fail] Failed to download: {entry['url']}")
            summary["failed"] += 1
        
        # Sleep between downloads to be respectful
        if i < len(entries):  # Don't sleep after the last download
            time.sleep(sleep)
    
    print(f"\nDownload complete!")
    print(f"Summary: {summary}")
    print(f"Total files processed: {sum(summary.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CSVs from football-data.co.uk (England page)."
    )
    parser.add_argument(
        "--url", 
        default="https://www.football-data.co.uk/englandm.php", 
        help="Page URL to parse"
    )
    parser.add_argument(
        "--out", 
        default="data/raw", 
        help="Output directory base"
    )
    parser.add_argument(
        "--sleep", 
        type=float, 
        default=0.5, 
        help="Seconds between downloads"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    try:
        main(args.url, args.out, sleep=args.sleep, overwrite=args.overwrite)
    except KeyboardInterrupt:
        print("\n[interrupted] Download cancelled by user.")
    except Exception as e:
        print(f"[error] Script failed: {e}")
        raise