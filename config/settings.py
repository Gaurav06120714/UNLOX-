"""
config/settings.py
─────────────────
Central configuration loaded from .env.
Every module imports from here — never hard-code API keys.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ── API credentials ───────────────────────────────────────
YOUTUBE_API_KEY     = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_CHANNEL_ID  = os.getenv("YOUTUBE_CHANNEL_ID", "")

# ── Paths ─────────────────────────────────────────────────
DATA_RAW_DIR        = ROOT / "data" / "raw"
DATA_PROCESSED_DIR  = ROOT / "data" / "processed"
DATA_EXPORTS_DIR    = ROOT / "data" / "exports"
MODELS_DIR          = ROOT / "models"
LOGS_DIR            = ROOT / "logs"
REPORTS_DIR         = ROOT / "reports"

DB_PATH             = ROOT / os.getenv("DB_PATH", "data/social_engagement.db")

# ── Logging ───────────────────────────────────────────────
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE            = LOGS_DIR / "app.log"

# ── YouTube API limits ────────────────────────────────────
YT_MAX_RESULTS      = 50          # max per page (API cap)
YT_COMMENT_PAGES    = 3           # pages of comments to fetch per video

# ── ML hyperparameters (defaults) ────────────────────────
RANDOM_STATE        = 42
TEST_SIZE           = 0.2

# ── Viral score weights ───────────────────────────────────
VIRAL_WEIGHTS = {
    "views":        0.30,
    "likes":        0.25,
    "comments":     0.20,
    "like_ratio":   0.15,
    "engagement":   0.10,
}

# Ensure critical directories exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EXPORTS_DIR,
             MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
