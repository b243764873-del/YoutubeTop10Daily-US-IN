import os
import json
import time
from typing import List, Tuple, Dict, Any

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI


# =========================
# Config
# =========================
SHEET_NAME = os.getenv("SHEET_NAME", "daily_rank")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Safety knobs (avoid "stuck for 10 minutes with no output")
MAX_PER_RUN = int(os.getenv("MAX_PER_RUN", "5"))          # process at most N pending rows per workflow run
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1200"))
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "75"))  # hard-ish timeout: we implement via retries + backoff
SLEEP_BETWEEN_ROWS_SEC = float(os.getenv("SLEEP_BETWEEN_ROWS_SEC", "0.8"))

# Generate fewer variants to keep fast/stable. Change to 10 later.
VARIANT_COUNT = int(os.getenv("VARIANT_COUNT", "3"))


# =========================
# Helpers
# =========================
def gsheet_client_from_sa_json(sa_json_str: str):
    sa_info = json.loads(sa_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def find_header_indices(header_row: List[str]):
    return {name: (i + 1) for i, name in enumerate(header_row)}  # 1-based


def a1_col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def yt_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def safe_truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "..."


def get_transcript_text(video_id: str, prefer_langs=("en", "hi")) -> Tuple[str, dict]:
    """
    Try to get transcript for a video.
    Returns (text, meta). meta may contain error.
    """
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        # preferred languages first
        for lang in prefer_langs:
            try:
                t = transcripts.find_transcript([lang])
                segs = t.fetch()
                text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
                return text, {"lang": t.language_code, "is_generated": getattr(t, "is_generated", None)}
            except Exception:
                pass

        # fallback: first available
        for t in transcripts:
            segs = t.fetch()
            text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
            return text, {"lang": t.language_code, "is_generated": getattr(t, "is_generated", None)}

    except (TranscriptsDisabled, NoTranscriptFound):
        return "", {"error": "no_transcript"}
    except Exception as e:
        return "", {"error": f"transcript_error: {e}"}

    return "", {"error": "no_transcript"}


def get_top_comments(youtube, video_id: str, max_comments: int = 20) -> List[str]:
    """
    Grabs top comments. This can be slow/blocked; we catch and return [] quickly.
    """
    try:
        res = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            order="relevance",
            textFormat="plainText",
        ).execute()
        out = []
        for it in res.get("items", []):
            sn = it["snippet"]["topLevelComment"]["snippet"]
            text = (sn.get("textDisplay") or "").strip()
            if text:
                out.append(text)
        return out[:max_comments]
    except HttpError:
        return []
    except Exception:
        return []


def parse_retry_delay_seconds(err_text: str) -> int | None:
    """
    If error contains 'Please retry in XXs' return that.
    """
    t = (err_text or "")
    # Example: "Please retry in 53.1947s."
    import re
    m = re.search(r"retry in\s+(\d+)(?:\.\d+)?s", t, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def openai_breakdown(client: OpenAI, title: str, transcript: str, comments: List[str], region: str) -> Dict[str, Any]:
    """
    OpenAI structured breakdown. Must return JSON dict.
    Keep output bounded to avoid slow/hanging.
    """
    transcript_snippet = safe_truncate(transcript, 2500)
    comments_snippet = comments[:20]

    system = f"""
You are a short-form video strategist/editor.
Analyze a trending YouTube Short (<=20s) and extract a REUSABLE FORMAT to create an ORIGINAL video.
Do NOT copy exact wording, unique jokes, names/brands, or identifiable characters.
Clone structure/pacing, not content.

Return VALID JSON ONLY (no markdown).
Optimize for <=20 seconds, vertical 9:16.

Schema:
{{
  "category": "string",
  "ai_generatable": true/false,
  "ai_generatable_reason": "string",
  "hook_patterns": ["3 templates with [VARIABLES]"],
  "beat_sheet": [
    {{
      "start_sec": number,
      "end_sec": number,
      "purpose": "hook/setup/twist/payoff/loop",
      "on_screen_text_template": "string with [VARIABLES]",
      "voiceover_template": "string with [VARIABLES]",
      "visual_template": "string",
      "edit_notes": "string"
    }}
  ],
  "subtitle_style": {{
    "max_chars_per_line": number,
    "lines": number,
    "emphasis_rules": ["..."],
    "placement": "top/middle/bottom"
  }},
  "edit_style": {{
    "avg_shot_len_sec": number,
    "transitions": ["..."],
    "zoom_shake_usage": "string",
    "sfx_cues": ["..."]
  }},
  "music_style": {{
    "bpm_range": "string",
    "mood": "string",
    "instruments": ["..."],
    "reference_keywords": ["..."]
  }},
  "reusable_variables": ["[TOPIC]", "..."],
  "risk_notes": ["..."],
  "generation_prompts": {{
    "voiceover_prompt": "string",
    "video_prompt": "string",
    "subtitle_prompt": "string"
  }},
  "variants": [
    {{
      "variant_title": "string",
      "variables": {{ "KEY": "VALUE" }},
      "voiceover": "18-20s voiceover",
      "on_screen_text": ["subtitle lines in order"],
      "video_prompt": "string",
      "subtitle_prompt": "string"
    }}
  ]
}}

Variants requirements:
- Provide EXACTLY {VARIANT_COUNT} variants.
- Keep each voiceover <= 20 seconds.
- Avoid copying source wording; keep original.
"""

    user = {
        "region": region,
        "title": title,
        "transcript": transcript_snippet if transcript_snippet else "(no transcript available)",
        "top_comments": comments_snippet,
        "constraints": {"max_duration_sec": 20, "aspect_ratio": "9:16"},
    }

    # Responses API
    resp = client.responses.create(
        model=OPENAI_MODEL,
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
    raw = (resp.output_text or "").strip()
    return json.loads(raw)


def openai_with_retry(client: OpenAI, title: str, transcript: str, comments: List[str], region: str) -> Dict[str, Any]:
    """
    Retry wrapper to prevent random transient issues from killing the run.
    """
    attempts = 3
    base_sleep = 6

    last_err = None
    for i in range(1, attempts + 1):
        try:
            return openai_breakdown(client, title, transcript, comments, region)
        except Exception as e:
            last_err = str(e)
            # If provider tells you a retry delay, respect it
            delay = parse_retry_delay_seconds(last_err)
            if delay is None:
                delay = base_sleep * i
            print(f"[WARN] OpenAI attempt {i}/{attempts} failed: {last_err}. Sleep {delay}s", flush=True)
            time.sleep(delay)

    raise RuntimeError(f"OpenAI failed after {attempts} attempts: {last_err}")


# =========================
# Main
# =========================
def main():
    # Required env
    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]
    yt_key = os.environ["YOUTUBE_API_KEY"]

    # Setup clients
    youtube = yt_client(yt_key)
    client = OpenAI()  # reads OPENAI_API_KEY from env

    gc = gsheet_client_from_sa_json(sa_json)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(SHEET_NAME)

    header = ws.row_values(1)
    idx = find_header_indices(header)

    required = [
        "ai_status", "ai_category", "ai_generatable", "ai_hook", "ai_script_template",
        "ai_prompt_pack_json", "ai_variants_json", "video_id", "title", "region"
    ]
    for c in required:
        if c not in idx:
            raise RuntimeError(f"Missing header column: {c}. Please ensure main.py V3 headers are correct.")

    all_rows = ws.get_all_values()
    if len(all_rows) <= 1:
        print("[INFO] No data rows.")
        return

    pending_rnums = []
    for rnum in range(2, len(all_rows) + 1):
        row = all_rows[rnum - 1]
        status = row[idx["ai_status"] - 1].strip() if len(row) >= idx["ai_status"] else ""
        if status == "pending":
            pending_rnums.append(rnum)

    if not pending_rnums:
        print("[INFO] No pending rows.")
        return

    # Limit per run
    total_pending = len(pending_rnums)
    pending_rnums = pending_rnums[:MAX_PER_RUN]
    print(f"[INFO] Pending rows total={total_pending}, will process now={len(pending_rnums)} (MAX_PER_RUN={MAX_PER_RUN})", flush=True)

    # Column indices
    c_status = idx["ai_status"]
    c_cat = idx["ai_category"]
    c_gen = idx["ai_generatable"]
    c_hook = idx["ai_hook"]
    c_script = idx["ai_script_template"]
    c_pack = idx["ai_prompt_pack_json"]
    c_vars = idx["ai_variants_json"]

    start_col = min(c_status, c_cat, c_gen, c_hook, c_script, c_pack, c_vars)
    end_col = max(c_status, c_cat, c_gen, c_hook, c_script, c_pack, c_vars)

    batch_payload = []

    for rnum in pending_rnums:
        row = all_rows[rnum - 1]
        video_id = row[idx["video_id"] - 1].strip()
        title = row[idx["title"] - 1].strip()
        region = row[idx["region"] - 1].strip()

        print(f"[INFO] start row={rnum} video_id={video_id} region={region}", flush=True)

        # Transcript
        prefer_langs = ("en", "hi") if region == "IN" else ("en",)
        transcript, meta = get_transcript_text(video_id, prefer_langs=prefer_langs)
        transcript = safe_truncate(transcript, 4000)

        # Comments
        comments = get_top_comments(youtube, video_id, max_comments=20)

        try:
            # OpenAI breakdown with retry
            data = openai_with_retry(client, title, transcript, comments, region)

            hook_text = " | ".join((data.get("hook_patterns") or [])[:3])
            script_template = json.dumps(data.get("beat_sheet") or [], ensure_ascii=False)
            pack_json = json.dumps(data, ensure_ascii=False)
            variants_json = json.dumps(data.get("variants") or [], ensure_ascii=False)

            ai_category = str(data.get("category", "")).strip()
            ai_generatable = "TRUE" if bool(data.get("ai_generatable", False)) else "FALSE"

            status = "done"
            values = [
                status,
                ai_category,
                ai_generatable,
                hook_text,
                script_template,
                pack_json,
                variants_json,
            ]

            print(
                f"[OK] row={rnum} video_id={video_id} status=done "
                f"transcript={meta.get('error','ok')} comments={len(comments)} variants={len(data.get('variants') or [])}",
                flush=True
            )

        except Exception as e:
            status = "failed"
            err_obj = {"error": str(e)}
            values = [
                status,
                "",
                "",
                "",
                "",
                json.dumps(err_obj, ensure_ascii=False),
                "",
            ]
            print(f"[WARN] row={rnum} video_id={video_id} status=failed err={e}", flush=True)

        rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"
        batch_payload.append({"range": rng, "values": [values]})

        # throttle
        time.sleep(SLEEP_BETWEEN_ROWS_SEC)

    # Batch update (one call)
    ws.batch_update(batch_payload, value_input_option="RAW")
    print(f"[DONE] Updated {len(pending_rnums)} rows. (Left pending={max(total_pending - len(pending_rnums), 0)})", flush=True)


if __name__ == "__main__":
    main()
