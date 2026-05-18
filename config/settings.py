import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

YOUTUBE_API_KEY    = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID", "")

DATA_RAW_DIR       = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR         = ROOT / "models"

DB_PATH      = ROOT / os.getenv("DB_PATH", "data/social_engagement.db")
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")

YT_MAX_RESULTS   = 50
YT_COMMENT_PAGES = 3

RANDOM_STATE = 42
TEST_SIZE    = 0.2

VIRAL_WEIGHTS = {
    "views":      0.30,
    "likes":      0.25,
    "comments":   0.20,
    "like_ratio": 0.15,
    "engagement": 0.10,
}

for folder in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
