import sys
from pathlib import Path
from datetime import datetime, timezone

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import YOUTUBE_API_KEY, YOUTUBE_CHANNEL_ID, YT_MAX_RESULTS, YT_COMMENT_PAGES
from src.utils.logger import get_logger

log = get_logger(__name__)

BASE_URL = "https://www.googleapis.com/youtube/v3"


def call_api(endpoint, params):
    params["key"] = YOUTUBE_API_KEY
    resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_channel_stats(channel_id=None):
    channel_id = channel_id or YOUTUBE_CHANNEL_ID
    log.info(f"Getting channel stats for {channel_id}")
    data  = call_api("channels", {"part": "snippet,statistics", "id": channel_id})
    item  = data["items"][0]
    stats = item["statistics"]
    return {
        "channel_id":       channel_id,
        "title":            item["snippet"]["title"],
        "subscriber_count": int(stats.get("subscriberCount", 0)),
        "view_count":       int(stats.get("viewCount", 0)),
        "video_count":      int(stats.get("videoCount", 0)),
    }


def fetch_video_ids(channel_id=None, max_videos=200):
    channel_id = channel_id or YOUTUBE_CHANNEL_ID
    log.info(f"Fetching video IDs for channel {channel_id}")

    video_ids = []
    next_page = None

    while len(video_ids) < max_videos:
        params = {
            "part":       "id",
            "channelId":  channel_id,
            "maxResults": min(YT_MAX_RESULTS, max_videos - len(video_ids)),
            "order":      "date",
            "type":       "video",
        }
        if next_page:
            params["pageToken"] = next_page

        try:
            data = call_api("search", params)
        except requests.RequestException as e:
            log.error(f"Failed to fetch video IDs: {e}")
            break

        for item in data.get("items", []):
            video_ids.append(item["id"]["videoId"])

        next_page = data.get("nextPageToken")
        if not next_page:
            break

    log.info(f"Found {len(video_ids)} videos")
    return video_ids


def fetch_video_details(video_ids):
    log.info(f"Fetching details for {len(video_ids)} videos")
    results = []
    now = datetime.now(timezone.utc).isoformat()

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i: i + 50]
        try:
            data = call_api("videos", {
                "part": "snippet,statistics,contentDetails",
                "id":   ",".join(chunk),
            })
        except requests.RequestException as e:
            log.error(f"Error fetching video details: {e}")
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

    log.info(f"Got details for {len(results)} videos")
    return results


def fetch_comments(video_id, max_pages=YT_COMMENT_PAGES):
    log.info(f"Fetching comments for video {video_id}")
    comments  = []
    next_page = None

    for _ in range(max_pages):
        params = {
            "part":       "snippet",
            "videoId":    video_id,
            "maxResults": 100,
            "order":      "relevance",
            "textFormat": "plainText",
        }
        if next_page:
            params["pageToken"] = next_page

        try:
            data = call_api("commentThreads", params)
        except requests.RequestException as e:
            log.warning(f"Could not fetch comments for {video_id}: {e}")
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

        next_page = data.get("nextPageToken")
        if not next_page:
            break

    log.info(f"Got {len(comments)} comments for video {video_id}")
    return comments
