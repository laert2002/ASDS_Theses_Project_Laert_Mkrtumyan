"""
Config loading and result saving.

Every experiment writes machine-readable results into thesis_lab/results/:
  * JSON for nested data (full provenance: config + raw rows),
  * CSV for flat tables (easy to open in Excel / pandas / plotting).
Keeping raw rows (not only averages) lets you recompute any statistic later
without re-running expensive GPU jobs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

LAB_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = LAB_ROOT / "configs" / "lab.yaml"
RESULTS_DIR = LAB_ROOT / "results"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    print(f"[saved] {path}")
    return path


def save_csv(rows: list[dict], name: str) -> Path:
    """Write a list of dicts; the header is the union of all keys (in first-seen order)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")
    return path
