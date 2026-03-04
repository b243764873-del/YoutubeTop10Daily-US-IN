import os
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# =========================
# Config
# =========================
REGIONS = ["US", "IN"]
MAX_DURATION_SEC = 20
PUBLISHED_WITHIN_DAYS = 3

TOP_N = 100  # V3: keep Top100 by velocity score

# Candidate gathering (search.list costy, do multi rounds but capped)
SEARCH_ROUNDS = [
    ("#shorts", "viewCount"),
    ("shorts", "viewCount"),
    ("#shorts", "date"),
    ("shorts", "date"),
]
TARGET_CANDIDATE_IDS = 300
MAX_TOTAL_CANDIDATE_IDS = 600  # quota guard

SHEET_NAME = "daily_rank"

HEADERS = [
    "date",
    "region",
    "rank",
    "video_id",
    "title",
    "channel_title",
    "views",
    "published_at",
    "hours_since_publish",
    "duration_sec",
    "score_views_per_hour",
    "is_new",
    "video_url",
    # AI pipeline columns
    "ai_status",                 # pending/done/failed
    "ai_category",               # category label
    "ai_generatable",            # TRUE/FALSE
    "ai_hook",                   # 3 hook templates joined
    "ai_script_template",        # beat_sheet JSON
    "ai_prompt_pack_json",       # full analysis JSON
    "ai_variants_json",          # 10 variants JSON
]


def iso8601_to_seconds(duration: str) -> int | None:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", duration or "")
    if not m:
        return None
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + s


def parse_published_at(published_at: str) -> datetime:
    return datetime.fromisoformat(published_at.replace("Z", "+00:00"))


def utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def probable_shorts(title: str, desc: str) -> bool:
    t = (title or "").lower()
    d = (desc or "").lower()
    return ("#shorts" in t) or ("#shorts" in d) or ("shorts" in t)


def yt_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def search_video_ids(youtube, region: str) -> List[str]:
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
                videoDuration="short",  # < 4 minutes
                maxResults=50,
            ).execute()
        except HttpError as e:
            print(f"[WARN] search.list failed region={region} q={q} order={order}: {e}")
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
    out: List[Dict] = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
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


def build_top_velocity(youtube, region: str) -> List[Dict]:
    ids = search_video_ids(youtube, region)
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

        vid = it.get("id")
        if not vid:
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
        if not probable_shorts(title, desc):
            continue

        views = int(st.get("viewCount", 0))

        hours = max((now - published_dt).total_seconds() / 3600.0, 1.0)
        score = views / hours

        rows.append({
            "video_id": vid,
            "title": title,
            "channel_title": sn.get("channelTitle", ""),
            "views": views,
            "published_at": published_at,
            "hours_since_publish": round(hours, 2),
            "duration_sec": dur,
            "score": round(score, 2),
            "video_url": f"https://www.youtube.com/watch?v={vid}",
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    result = rows[:TOP_N]
    if len(result) < TOP_N:
        print(f"[INFO] region={region} only found {len(result)}/{TOP_N} matching filters.")
    return result


# =========================
# Google Sheets
# =========================
def gsheet_client_from_sa_json(sa_json_str: str):
    sa_info = json.loads(sa_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def get_or_create_worksheet(sh, title: str):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=5000, cols=len(HEADERS))


def ensure_headers(ws):
    first_row = ws.row_values(1)
    if first_row != HEADERS:
        # Overwrite header row to V3 schema (doesn't delete data rows)
        end_col = chr(ord("A") + len(HEADERS) - 1) if len(HEADERS) <= 26 else "Z"
        ws.update(f"A1:{end_col}1", [HEADERS])


def already_written_today(ws, today_str: str, region: str) -> bool:
    records = ws.get_all_records()
    return any(r.get("date") == today_str and r.get("region") == region for r in records)


def load_previous_date_ids(ws, today_str: str, region: str) -> Tuple[set, str | None]:
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


def main():
    yt_key = os.environ["YOUTUBE_API_KEY"]
    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]

    youtube = yt_client(yt_key)

    gc = gsheet_client_from_sa_json(sa_json)
    sh = gc.open_by_key(sheet_id)
    ws = get_or_create_worksheet(sh, SHEET_NAME)
    ensure_headers(ws)

    today_str = utc_today_str()

    append_rows: List[List] = []

    for region in REGIONS:
        if already_written_today(ws, today_str, region):
            print(f"[INFO] region={region} already written for {today_str}, skip.")
            continue

        prev_ids, prev_date = load_previous_date_ids(ws, today_str, region)
        print(f"[INFO] region={region} prev_date={prev_date} prev_count={len(prev_ids)}")

        top_list = build_top_velocity(youtube, region)
        if not top_list:
            print(f"[WARN] No results for region={region}.")
            continue

        for idx, v in enumerate(top_list, start=1):
            is_new = (v["video_id"] not in prev_ids)
            ai_status = "pending" if is_new else ""

            append_rows.append([
                today_str,
                region,
                idx,
                v["video_id"],
                v["title"],
                v["channel_title"],
                v["views"],
                v["published_at"],
                v["hours_since_publish"],
                v["duration_sec"],
                v["score"],
                "TRUE" if is_new else "FALSE",
                v["video_url"],
                ai_status,
                "",   # ai_category
                "",   # ai_generatable
                "",   # ai_hook
                "",   # ai_script_template
                "",   # ai_prompt_pack_json
                "",   # ai_variants_json
            ])

    if append_rows:
        ws.append_rows(append_rows, value_input_option="RAW")
    print(f"[DONE] appended rows: {len(append_rows)}")


if __name__ == "__main__":
    main()
