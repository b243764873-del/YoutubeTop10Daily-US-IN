import os
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import gspread
from google.oauth2.service_account import Credentials


# =========================
# Config
# =========================
REGIONS = ["US", "IN"]

MAX_DURATION_SEC = 20
PUBLISHED_WITHIN_DAYS = 3
TOP_N = 10

# Multi-round search to gather enough candidates, then filter ourselves.
SEARCH_ROUNDS = [
    ("#shorts", "viewCount"),
    ("shorts", "viewCount"),
    ("#shorts", "date"),
    ("shorts", "date"),
]

TARGET_CANDIDATE_IDS = 200
MAX_TOTAL_CANDIDATE_IDS = 400  # hard cap (quota/time guard)


# =========================
# Helpers
# =========================
def iso8601_to_seconds(duration: str) -> int | None:
    """
    Parse ISO 8601 duration like PT15S, PT1M2S, PT1H2M3S -> seconds
    """
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", duration or "")
    if not m:
        return None
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + s


def probable_shorts(title: str, desc: str) -> bool:
    """
    Heuristic to improve Shorts accuracy.
    If you feel results are too few, you can relax this function later.
    """
    t = (title or "").lower()
    d = (desc or "").lower()
    return ("#shorts" in t) or ("#shorts" in d) or ("shorts" in t)


def utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def parse_published_at(published_at: str) -> datetime:
    return datetime.fromisoformat(published_at.replace("Z", "+00:00"))


# =========================
# YouTube API
# =========================
def yt_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def search_video_ids(youtube, region: str) -> List[str]:
    """
    Collect candidate video IDs using multiple search rounds.
    search.list can't filter <=20s; it only supports videoDuration=short (<4min),
    so we gather candidates, then filter by videos.list duration <=20s.
    """
    seen = set()
    ids: List[str] = []

    for q, order in SEARCH_ROUNDS:
        if len(ids) >= TARGET_CANDIDATE_IDS or len(ids) >= MAX_TOTAL_CANDIDATE_IDS:
            break

        try:
            res = youtube.search().list(
                part="id",
                q=q,
                type="video",
                regionCode=region,
                order=order,
                videoDuration="short",  # <4 minutes
                maxResults=50,
            ).execute()
        except HttpError as e:
            print(f"[WARN] search.list failed for region={region}, q={q}, order={order}: {e}")
            break

        for it in res.get("items", []):
            vid = it.get("id", {}).get("videoId")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            ids.append(vid)
            if len(ids) >= MAX_TOTAL_CANDIDATE_IDS:
                break

    return ids


def fetch_videos_details(youtube, video_ids: List[str]) -> List[Dict]:
    """
    Fetch videos.list details in chunks of 50.
    """
    out: List[Dict] = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        try:
            res = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(chunk),
            ).execute()
        except HttpError as e:
            print(f"[WARN] videos.list failed: {e}")
            continue

        out.extend(res.get("items", []))
    return out


def build_top_list(youtube, region: str) -> List[Dict]:
    """
    Build Top list for a region with filters:
    - duration <= 20s
    - published within last 3 days
    - probable shorts (heuristic)
    Sort by views desc and return up to TOP_N.
    If fewer than TOP_N, we return what we have (your choice A).
    """
    ids = search_video_ids(youtube, region=region)
    if not ids:
        return []

    items = fetch_videos_details(youtube, ids)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=PUBLISHED_WITHIN_DAYS)

    rows: List[Dict] = []
    for it in items:
        sn = it.get("snippet") or {}
        cd = it.get("contentDetails") or {}
        st = it.get("statistics") or {}

        dur = iso8601_to_seconds(cd.get("duration", ""))
        if dur is None or dur > MAX_DURATION_SEC:
            continue

        published_at = sn.get("publishedAt", "")
        if not published_at:
            continue
        published_dt = parse_published_at(published_at)
        if published_dt < cutoff:
            continue

        title = sn.get("title", "")
        desc = sn.get("description", "")

        # Heuristic (optional)
        if not probable_shorts(title, desc):
            continue

        views = int(st.get("viewCount", 0))
        vid = it.get("id")
        if not vid:
            continue

        rows.append(
            {
                "video_id": vid,
                "title": title,
                "channel_title": sn.get("channelTitle", ""),
                "published_at": published_at,
                "views": views,
                "duration_sec": dur,
                "video_url": f"https://www.youtube.com/watch?v={vid}",
            }
        )

    rows.sort(key=lambda x: x["views"], reverse=True)
    result = rows[:TOP_N]

    if len(result) < TOP_N:
        print(f"[INFO] region={region} only found {len(result)}/{TOP_N} videos matching filters (<=20s, last 3 days).")

    return result


# =========================
# Google Sheets
# =========================
HEADERS = [
    "date",
    "region",
    "rank",
    "video_id",
    "title",
    "channel_title",
    "views",
    "published_at",
    "duration_sec",
    "is_new",
    "video_url",
    "ai_status",
    "ai_hook",
    "ai_script_template",
    "ai_prompt_pack_json",
]


def gsheet_client_from_sa_json(sa_json_str: str):
    sa_info = json.loads(sa_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def get_or_create_worksheet(sh, title: str):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=2000, cols=len(HEADERS))
        return ws


def ensure_headers(ws):
    first_row = ws.row_values(1)
    if first_row != HEADERS:
        ws.update(f"A1:{chr(ord('A') + len(HEADERS) - 1)}1", [HEADERS])


def already_written_today(ws, today_str: str, region: str) -> bool:
    """
    Prevent duplicate appends when you manually run workflow multiple times.
    """
    records = ws.get_all_records()
    return any(r.get("date") == today_str and r.get("region") == region for r in records)


def load_previous_date_ids(ws, today_str: str, region: str) -> Tuple[set, str | None]:
    """
    Load the most recent previous date (< today_str) for the given region,
    and return its video_id set.
    """
    records = ws.get_all_records()
    dates = sorted({r.get("date") for r in records if r.get("region") == region and r.get("date")})
    if not dates:
        return set(), None

    prev_dates = [d for d in dates if d < today_str]
    if not prev_dates:
        return set(), None

    prev_date = prev_dates[-1]
    prev_ids = {r.get("video_id") for r in records if r.get("date") == prev_date and r.get("region") == region}
    prev_ids.discard(None)
    prev_ids.discard("")
    return prev_ids, prev_date


def append_rows(ws, rows: List[List]):
    if rows:
        ws.append_rows(rows, value_input_option="RAW")


# =========================
# Main
# =========================
def main():
    yt_key = os.environ["YOUTUBE_API_KEY"]
    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]

    youtube = yt_client(yt_key)

    gc = gsheet_client_from_sa_json(sa_json)
    sh = gc.open_by_key(sheet_id)
    ws = get_or_create_worksheet(sh, "daily_rank")
    ensure_headers(ws)

    today_str = utc_today_str()

    all_append_rows: List[List] = []

    for region in REGIONS:
        if already_written_today(ws, today_str, region):
            print(f"[INFO] region={region} already written for {today_str}, skip.")
            continue

        top_list = build_top_list(youtube, region)
        if not top_list:
            print(f"[WARN] No results for region={region}. If this happens often, relax probable_shorts().")
            continue

        prev_ids, prev_date = load_previous_date_ids(ws, today_str, region)
        print(f"[INFO] region={region} prev_date={prev_date} prev_count={len(prev_ids)}")

        for idx, v in enumerate(top_list, start=1):
            is_new = (v["video_id"] not in prev_ids)
            ai_status = "pending" if is_new else ""

            all_append_rows.append(
                [
                    today_str,
                    region,
                    idx,
                    v["video_id"],
                    v["title"],
                    v["channel_title"],
                    v["views"],
                    v["published_at"],
                    v["duration_sec"],
                    "TRUE" if is_new else "FALSE",
                    v["video_url"],
                    ai_status,
                    "",
                    "",
                    "",
                ]
            )

    append_rows(ws, all_append_rows)
    print(f"[DONE] appended rows: {len(all_append_rows)}")


if __name__ == "__main__":
    main()
