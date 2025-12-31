from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_market_db_path() -> Path:
    return project_root() / "data" / "tenbagger_analysis_market.sqlite"

