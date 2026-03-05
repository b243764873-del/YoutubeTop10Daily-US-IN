import os
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable


# =========================
# Config
# =========================
REGIONS = [x.strip() for x in os.getenv("REGIONS", "US,IN").split(",") if x.strip()]

MAX_DURATION_SEC = int(os.getenv("MAX_DURATION_SEC", "20"))
PUBLISHED_WITHIN_DAYS = int(os.getenv("PUBLISHED_WITHIN_DAYS", "3"))

TOP_N = int(os.getenv("TOP_N", "100"))  # keep TopN by velocity score

# Candidate gathering (search.list costy, do multi rounds but capped)
SEARCH_ROUNDS = [
    ("#shorts", "viewCount"),
    ("shorts", "viewCount"),
    ("#shorts", "date"),
    ("shorts", "date"),
]
TARGET_CANDIDATE_IDS = int(os.getenv("TARGET_CANDIDATE_IDS", "300"))
MAX_TOTAL_CANDIDATE_IDS = int(os.getenv("MAX_TOTAL_CANDIDATE_IDS", "600"))  # quota guard

SHEET_NAME = os.getenv("SHEET_NAME", "daily_rank")

# Transcript
TRANSCRIPT_COL = os.getenv("TRANSCRIPT_COL", "transcript")
TRANSCRIPT_LANGS = [x.strip() for x in os.getenv("TRANSCRIPT_LANGS", "en").split(",") if x.strip()]
TRANSCRIPT_MAX_CHARS = int(os.getenv("TRANSCRIPT_MAX_CHARS", "9000"))
TRANSCRIPT_SLEEP_SEC = float(os.getenv("TRANSCRIPT_SLEEP_SEC", "0.05"))  # soften burst

# Sheet append pacing
APPEND_SLEEP_SEC = float(os.getenv("APPEND_SLEEP_SEC", "0.15"))

# Full columns to ensure (missing ones will be appended to the right)
FULL_HEADERS = [
    # base ranking
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

    # transcript (for meaning fidelity)
    TRANSCRIPT_COL,

    # AI pipeline columns (analyze_pending.py)
    "ai_status",                 # pending/done/failed
    "ai_category",               # category label
    "ai_generatable",            # TRUE/FALSE
    "ai_hook",                   # 3 hook templates joined
    "ai_script_template",        # beat_sheet JSON
    "ai_prompt_pack_json",       # full analysis JSON
    "ai_variants_json",          # variants JSON

    # generation pipeline columns (generate_meme_video.py)
    "gen_status",                # pending/done/failed
    "gen_video_file",            # output path
    "gen_note",                  # debug note

    # YouTube metadata
    "yt_title",
    "yt_description",
    "yt_hashtags",
    "yt_tags",
]


# =========================
# YouTube helpers
# =========================
def iso8601_to_seconds(duration: str) -> Optional[int]:
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
        chunk = video_ids[i:i + 50]
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
            "score_views_per_hour": round(score, 2),
            "video_url": f"https://www.youtube.com/watch?v={vid}",
        })

    rows.sort(key=lambda x: x["score_views_per_hour"], reverse=True)
    result = rows[:TOP_N]
    if len(result) < TOP_N:
        print(f"[INFO] region={region} only found {len(result)}/{TOP_N} matching filters.")
    return result


# =========================
# Transcript
# =========================
def fetch_transcript_text(video_id: str, preferred_langs: List[str]) -> str:
    """
    Best-effort transcript fetch.
    - Tries preferred languages first
    - Falls back to any available transcript
    - Returns "" if none
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # 1) try preferred languages (manual first, then generated)
        for lang in preferred_langs:
            try:
                t = transcript_list.find_manually_created_transcript([lang])
                parts = t.fetch()
                return " ".join([p.get("text", "") for p in parts]).strip()
            except Exception:
                pass
            try:
                t = transcript_list.find_generated_transcript([lang])
                parts = t.fetch()
                return " ".join([p.get("text", "") for p in parts]).strip()
            except Exception:
                pass

        # 2) fallback: first available transcript
        try:
            t = next(iter(transcript_list))
            parts = t.fetch()
            return " ".join([p.get("text", "") for p in parts]).strip()
        except Exception:
            return ""

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return ""
    except Exception:
        return ""


def normalize_transcript(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > TRANSCRIPT_MAX_CHARS:
        t = t[:TRANSCRIPT_MAX_CHARS]
    return t


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
        # Create with enough columns
        return sh.add_worksheet(title=title, rows=5000, cols=max(len(FULL_HEADERS), 30))


def ensure_headers_append_only(ws, required_headers: List[str]) -> List[str]:
    """
    IMPORTANT: Do NOT overwrite existing headers.
    Only append missing headers to the right.
    Returns the final header list.
    """
    header = ws.row_values(1)
    header = [h.strip() for h in header if str(h).strip()]

    if not header:
        ws.update(f"A1:{col_to_a1(len(required_headers))}1", [required_headers])
        return required_headers

    existing = set(header)
    missing = [h for h in required_headers if h not in existing]
    if not missing:
        return header

    new_header = header + missing
    ws.update(f"A1:{col_to_a1(len(new_header))}1", [new_header])
    print(f"[INFO] Added missing headers: {missing}")
    return new_header


def col_to_a1(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def already_written_today(ws, today_str: str, region: str) -> bool:
    records = ws.get_all_records()
    return any(r.get("date") == today_str and r.get("region") == region for r in records)


def load_previous_date_ids(ws, today_str: str, region: str) -> Tuple[set, Optional[str]]:
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


def build_row_by_header(header: List[str], data: Dict[str, Any]) -> List[Any]:
    """
    Create a row list matching current sheet header order.
    Any missing fields -> "".
    """
    row = []
    for h in header:
        v = data.get(h, "")
        if v is None:
            v = ""
        row.append(v)
    return row


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
    ws = get_or_create_worksheet(sh, SHEET_NAME)

    # Ensure columns (append-only, no destructive overwrite)
    header = ensure_headers_append_only(ws, FULL_HEADERS)

    today_str = utc_today_str()
    append_rows: List[List[Any]] = []

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

        for rank, v in enumerate(top_list, start=1):
            vid = v["video_id"]
            is_new = (vid not in prev_ids)

            transcript_text = ""
            if is_new:
                # only fetch transcript for new items (saves time)
                transcript_text = normalize_transcript(fetch_transcript_text(vid, TRANSCRIPT_LANGS))
                time.sleep(TRANSCRIPT_SLEEP_SEC)

            row_data = {
                "date": today_str,
                "region": region,
                "rank": rank,
                "video_id": vid,
                "title": v.get("title", ""),
                "channel_title": v.get("channel_title", ""),
                "views": v.get("views", 0),
                "published_at": v.get("published_at", ""),
                "hours_since_publish": v.get("hours_since_publish", ""),
                "duration_sec": v.get("duration_sec", ""),
                "score_views_per_hour": v.get("score_views_per_hour", ""),
                "is_new": "TRUE" if is_new else "FALSE",
                "video_url": v.get("video_url", ""),

                TRANSCRIPT_COL: transcript_text,

                # AI pipeline default values
                "ai_status": "pending" if is_new else "",
                "ai_category": "",
                "ai_generatable": "",
                "ai_hook": "",
                "ai_script_template": "",
                "ai_prompt_pack_json": "",
                "ai_variants_json": "",

                # generation pipeline default values
                "gen_status": "",
                "gen_video_file": "",
                "gen_note": "",

                # yt metadata default values
                "yt_title": "",
                "yt_description": "",
                "yt_hashtags": "",
                "yt_tags": "",
            }

            append_rows.append(build_row_by_header(header, row_data))
            time.sleep(APPEND_SLEEP_SEC)

    if append_rows:
        ws.append_rows(append_rows, value_input_option="RAW")
    print(f"[DONE] appended rows: {len(append_rows)}")


if __name__ == "__main__":
    main()
