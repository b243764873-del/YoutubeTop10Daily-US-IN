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

# You want: <=20 seconds, published within last 3 days, sort by views desc
MAX_DURATION_SEC = 20
PUBLISHED_WITHIN_DAYS = 3
TOP_N = 10

# To avoid "not enough candidates" due to strict filters, we gather more candidates.
# Each search() call returns up to 50. We'll do multiple rounds and de-duplicate.
SEARCH_ROUNDS = [
    # (q, order)
    ("#shorts", "viewCount"),
    ("shorts", "viewCount"),
    ("#shorts", "date"),
    ("shorts", "date"),
]

# How many ids we aim to collect before fetching details
TARGET_CANDIDATE_IDS = 200
MAX_TOTAL_CANDIDATE_IDS = 400  # hard cap to control quota/time


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
    Not perfect, but helps accuracy. You can relax/remove this if you want more results.
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
    Note: search.list can't filter <=20s; it only supports videoDuration=short (<4min),
    so we collect candidates, then filter by videos.list duration <=20s.
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
            # If quota or any error, break early
            print(f"[WARN] search.list failed for region={region}, q={q}, order={order}: {e}")
            break

        items = res.get("items", [])
        for it in items:
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
    Build Top10 list for a region with filters:
    - duration <= 20s
    - published within last 3 days
    - probable shorts (optional heuristic)
    Sort by views desc and return top N.
    """
    ids = search_video_ids(youtube, region=region)
    if not ids:
        return []

    items = fetch_videos_details(youtube, ids)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=PUBLISHED_WITHIN_DAYS)

    rows: List[Dict] = []
    for it in items:
        try:
            sn = it["snippet"]
            cd = it["contentDetails"]
            st = it.get("statistics", {})
        except KeyError:
            continue

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

        # Optional heuristic: keep it for better Shorts accuracy
        if not probable_shorts(title, desc):
            continue

        views = int(st.get("viewCount", 0))
        rows.append(
            {
                "video_id": it.get("id"),
                "title": title,
                "channel_title": sn.get("channelTitle", ""),
                "published_at": published_at,
                "views": views,
                "duration_sec": dur,
                "video_url": f"https://www.youtube.com/watch?v={it.get('id')}",
            }
        )

    rows.sort(key=lambda x: x["views"], reverse=True)
    return rows[:TOP_N]


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
        ws = sh.add_worksheet(title=title, rows=1000, cols=len(HEADERS))
        return ws


def ensure_headers(ws):
    """
    If sheet is empty, write headers. If first row doesn't match, still overwrite first row.
    """
    first_row = ws.row_values(1)
    if first_row != HEADERS:
        ws.update("A1:O1", [HEADERS])


def load_previous_date_ids(ws, today_str: str, region: str) -> Tuple[set, str | None]:
    """
    Load the most recent previous date (< today_str) for the given region,
    and return its video_id set.
    This uses get_all_records() for simplicity.
    """
    records = ws.get_all_records()  # expects headers in row1
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


def append_daily_rank(ws, rows: List[List]):
    """
    Append rows at the end.
    """
    if not rows:
        return
    ws.append_rows(rows, value_input_option="RAW")


# =========================
# Main
# =========================
def main():
    # Required env vars
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
        top_list = build_top_list(youtube, region)
        if not top_list:
            print(f"[WARN] No results for region={region}. Try relaxing probable_shorts() filter.")
            continue

        prev_ids, prev_date = load_previous_date_ids(ws, today_str, region)
        print(f"[INFO] region={region} prev_date={prev_date} prev_count={len(prev_ids)}")

        for idx, v in enumerate(top_list, start=1):
            is_new = (v["video_id"] not in prev_ids)

            # New entries -> pending for AI analysis later
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

    append_daily_rank(ws, all_append_rows)
    print(f"[DONE] appended rows: {len(all_append_rows)}")


if __name__ == "__main__":
    main()
