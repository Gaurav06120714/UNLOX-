"""
src/api/youtube_client.py
─────────────────────────
YouTube Data API v3 wrapper — uses `requests` directly (no httplib2).

Responsibilities
  • Fetch channel statistics.
  • Fetch video list for a channel.
  • Fetch per-video statistics (views, likes, comments).
  • Fetch comment threads for a video.

All functions return plain Python dicts / lists of dicts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import requests

from config.settings import YOUTUBE_API_KEY, YOUTUBE_CHANNEL_ID, YT_MAX_RESULTS, YT_COMMENT_PAGES
from src.utils.logger import get_logger

log = get_logger(__name__)

_BASE = "https://www.googleapis.com/youtube/v3"
_TIMEOUT = 30  # seconds per request


def _get(endpoint: str, params: dict) -> dict:
    """Make a GET request to the YouTube API and return the JSON response."""
    params["key"] = YOUTUBE_API_KEY
    resp = requests.get(f"{_BASE}/{endpoint}", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── Channel info ──────────────────────────────────────────────────────────────

def fetch_channel_stats(channel_id: Optional[str] = None) -> dict:
    channel_id = channel_id or YOUTUBE_CHANNEL_ID
    log.info(f"Fetching channel stats for {channel_id}")
    data = _get("channels", {"part": "snippet,statistics", "id": channel_id})
    item  = data["items"][0]
    stats = item["statistics"]
    return {
        "channel_id":       channel_id,
        "title":            item["snippet"]["title"],
        "subscriber_count": int(stats.get("subscriberCount", 0)),
        "view_count":       int(stats.get("viewCount", 0)),
        "video_count":      int(stats.get("videoCount", 0)),
    }


# ── Video list ────────────────────────────────────────────────────────────────

def fetch_video_ids(channel_id: Optional[str] = None, max_videos: int = 200) -> list[str]:
    channel_id = channel_id or YOUTUBE_CHANNEL_ID
    log.info(f"Fetching video IDs for channel {channel_id}")

    video_ids: list[str] = []
    next_page_token: Optional[str] = None

    while len(video_ids) < max_videos:
        params: dict = {
            "part":       "id",
            "channelId":  channel_id,
            "maxResults": min(YT_MAX_RESULTS, max_videos - len(video_ids)),
            "order":      "date",
            "type":       "video",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            data = _get("search", params)
        except requests.RequestException as e:
            log.error(f"Error fetching video IDs: {e}")
            break

        for item in data.get("items", []):
            video_ids.append(item["id"]["videoId"])

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    log.info(f"Found {len(video_ids)} video IDs.")
    return video_ids


# ── Video details ─────────────────────────────────────────────────────────────

def fetch_video_details(video_ids: list[str]) -> list[dict]:
    log.info(f"Fetching details for {len(video_ids)} videos.")
    results: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        try:
            data = _get("videos", {
                "part": "snippet,statistics,contentDetails",
                "id":   ",".join(chunk),
            })
        except requests.RequestException as e:
            log.error(f"Error fetching video details chunk {i}: {e}")
            continue

        for item in data.get("items", []):
            snip  = item.get("snippet", {})
            stats = item.get("statistics", {})
            cont  = item.get("contentDetails", {})
            results.append({
                "video_id":      item["id"],
                "title":         snip.get("title", ""),
                "description":   snip.get("description", "")[:500],
                "published_at":  snip.get("publishedAt", ""),
                "channel_id":    snip.get("channelId", ""),
                "channel_title": snip.get("channelTitle", ""),
                "view_count":    int(stats.get("viewCount",    0)),
                "like_count":    int(stats.get("likeCount",    0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration":      cont.get("duration", ""),
                "tags":          "|".join(snip.get("tags", [])),
                "thumbnail_url": snip.get("thumbnails", {}).get("high", {}).get("url", ""),
                "fetched_at":    now,
            })

    log.info(f"Retrieved details for {len(results)} videos.")
    return results


# ── Comments ──────────────────────────────────────────────────────────────────

def fetch_comments(video_id: str, max_pages: int = YT_COMMENT_PAGES) -> list[dict]:
    log.info(f"Fetching comments for video {video_id}")
    comments: list[dict] = []
    next_page_token: Optional[str] = None

    for _ in range(max_pages):
        params: dict = {
            "part":       "snippet",
            "videoId":    video_id,
            "maxResults": 100,
            "order":      "relevance",
            "textFormat": "plainText",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            data = _get("commentThreads", params)
        except requests.RequestException as e:
            log.warning(f"Cannot fetch comments for {video_id}: {e}")
            break

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id":      item["id"],
                "video_id":        video_id,
                "author":          top.get("authorDisplayName", ""),
                "text":            top.get("textOriginal", ""),
                "like_count":      int(top.get("likeCount", 0)),
                "published_at":    top.get("publishedAt", ""),
                "sentiment_label": None,
                "sentiment_score": None,
            })

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    log.info(f"Fetched {len(comments)} comments for {video_id}.")
    return comments
