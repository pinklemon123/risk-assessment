from pathlib import Path
import json
from typing import List, Dict, Any

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def write_raw(batch_id: str, rows: List[Dict[str, Any]]):
    p = RAW_DIR / f"{batch_id}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_clean(batch_id: str, rows: List[Dict[str, Any]]):
    p = CLEAN_DIR / f"{batch_id}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
