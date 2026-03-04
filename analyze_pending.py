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
# Config (env)
# =========================
SHEET_NAME = os.getenv("SHEET_NAME", "daily_rank")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

MAX_PER_RUN = int(os.getenv("MAX_PER_RUN", "5"))
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1800"))
SLEEP_BETWEEN_ROWS_SEC = float(os.getenv("SLEEP_BETWEEN_ROWS_SEC", "0.8"))

VARIANT_COUNT = int(os.getenv("VARIANT_COUNT", "3"))  # increase to 10 after stable


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


def parse_retry_delay_seconds(err_text: str) -> int | None:
    t = err_text or ""
    import re
    m = re.search(r"retry in\s+(\d+)(?:\.\d+)?s", t, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def get_transcript_text(video_id: str, prefer_langs=("en", "hi")) -> Tuple[str, dict]:
    """
    Compatible across youtube-transcript-api versions:
    - If list_transcripts exists => use it (supports language selection + better metadata)
    - Else => fallback to get_transcript
    """
    try:
        # Newer versions support list_transcripts
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
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
                try:
                    segs = t.fetch()
                    text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
                    if text:
                        return text, {"lang": t.language_code, "is_generated": getattr(t, "is_generated", None)}
                except Exception:
                    continue

            return "", {"error": "no_transcript"}

        # Older versions: use get_transcript
        for lang in prefer_langs:
            try:
                segs = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
                if text:
                    return text, {"lang": lang, "is_generated": None}
            except Exception:
                pass

        # last fallback: no language preference
        try:
            segs = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
            if text:
                return text, {"lang": "unknown", "is_generated": None}
        except Exception:
            pass

        return "", {"error": "no_transcript"}

    except (TranscriptsDisabled, NoTranscriptFound):
        return "", {"error": "no_transcript"}
    except Exception as e:
        return "", {"error": f"transcript_error: {e}"}


def get_top_comments(youtube, video_id: str, max_comments: int = 20) -> List[str]:
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


def build_json_schema(variant_count: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {"type": "string"},
            "ai_generatable": {"type": "boolean"},
            "ai_generatable_reason": {"type": "string"},
            "hook_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
            "beat_sheet": {
                "type": "array",
                "minItems": 4,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                        "purpose": {"type": "string"},
                        "on_screen_text_template": {"type": "string"},
                        "voiceover_template": {"type": "string"},
                        "visual_template": {"type": "string"},
                        "edit_notes": {"type": "string"},
                    },
                    "required": [
                        "start_sec",
                        "end_sec",
                        "purpose",
                        "on_screen_text_template",
                        "voiceover_template",
                        "visual_template",
                        "edit_notes",
                    ],
                },
            },
            "subtitle_style": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_chars_per_line": {"type": "number"},
                    "lines": {"type": "number"},
                    "emphasis_rules": {"type": "array", "items": {"type": "string"}},
                    "placement": {"type": "string"},
                },
                "required": ["max_chars_per_line", "lines", "emphasis_rules", "placement"],
            },
            "edit_style": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "avg_shot_len_sec": {"type": "number"},
                    "transitions": {"type": "array", "items": {"type": "string"}},
                    "zoom_shake_usage": {"type": "string"},
                    "sfx_cues": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["avg_shot_len_sec", "transitions", "zoom_shake_usage", "sfx_cues"],
            },
            "music_style": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "bpm_range": {"type": "string"},
                    "mood": {"type": "string"},
                    "instruments": {"type": "array", "items": {"type": "string"}},
                    "reference_keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["bpm_range", "mood", "instruments", "reference_keywords"],
            },
            "reusable_variables": {"type": "array", "items": {"type": "string"}},
            "risk_notes": {"type": "array", "items": {"type": "string"}},
            "generation_prompts": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "voiceover_prompt": {"type": "string"},
                    "video_prompt": {"type": "string"},
                    "subtitle_prompt": {"type": "string"},
                },
                "required": ["voiceover_prompt", "video_prompt", "subtitle_prompt"],
            },
            "variants": {
                "type": "array",
                "minItems": variant_count,
                "maxItems": variant_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "variant_title": {"type": "string"},
                        "voiceover": {"type": "string"},
                        "on_screen_text": {"type": "array", "items": {"type": "string"}},
                        "video_prompt": {"type": "string"},
                        "subtitle_prompt": {"type": "string"},
                    },
                    "required": [
                        "variant_title",
                        "voiceover",
                        "on_screen_text",
                        "video_prompt",
                        "subtitle_prompt",
                    ],
                },
            },
        },  # ✅ close properties
        "required": [
            "category",
            "ai_generatable",
            "ai_generatable_reason",
            "hook_patterns",
            "beat_sheet",
            "subtitle_style",
            "edit_style",
            "music_style",
            "reusable_variables",
            "risk_notes",
            "generation_prompts",
            "variants",
        ],
    }  # ✅ close return dict

def openai_breakdown_jsonschema_chat(
    client: OpenAI,
    title: str,
    transcript: str,
    comments: List[str],
    region: str,
    variant_count: int,
) -> Dict[str, Any]:
    """
    OpenAI Python 2.x compatible: Chat Completions with response_format json_schema.
    """
    schema = build_json_schema(variant_count)

    system = (
        "You are a short-form video strategist/editor. "
        "Extract reusable format from a trending YouTube Short (<=20s). "
        "Do NOT copy exact wording, unique jokes, names/brands, or identifiable characters. "
        "Clone structure/pacing, not content."
    )

    user = {
        "region": region,
        "title": title,
        "transcript": safe_truncate(transcript, 2500) if transcript else "(no transcript available)",
        "top_comments": comments[:20],
        "constraints": {"max_duration_sec": 20, "aspect_ratio": "9:16", "variant_count": variant_count},
    }

    # ✅ Chat Completions structured output
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "shorts_breakdown",
                "schema": schema,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()
    return json.loads(content)


def openai_with_retry(client: OpenAI, title: str, transcript: str, comments: List[str], region: str) -> Dict[str, Any]:
    attempts = 3
    base_sleep = 6
    last_err = None

    for i in range(1, attempts + 1):
        try:
            return openai_breakdown_jsonschema_chat(
                client=client,
                title=title,
                transcript=transcript,
                comments=comments,
                region=region,
                variant_count=VARIANT_COUNT,
            )
        except Exception as e:
            last_err = str(e)
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
    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]
    yt_key = os.environ["YOUTUBE_API_KEY"]

    youtube = yt_client(yt_key)
    client = OpenAI()

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
            raise RuntimeError(f"Missing header column: {c}. Please ensure daily_rank headers are correct.")

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

    total_pending = len(pending_rnums)
    pending_rnums = pending_rnums[:MAX_PER_RUN]
    print(
        f"[INFO] Pending rows total={total_pending}, will process now={len(pending_rnums)} (MAX_PER_RUN={MAX_PER_RUN})",
        flush=True
    )

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

        prefer_langs = ("en", "hi") if region == "IN" else ("en",)
        transcript, meta = get_transcript_text(video_id, prefer_langs=prefer_langs)
        transcript = safe_truncate(transcript, 4000)

        comments = get_top_comments(youtube, video_id, max_comments=20)

        try:
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

        time.sleep(SLEEP_BETWEEN_ROWS_SEC)

    ws.batch_update(batch_payload, value_input_option="RAW")
    print(f"[DONE] Updated {len(pending_rnums)} rows. (Left pending={max(total_pending - len(pending_rnums), 0)})", flush=True)


if __name__ == "__main__":
    main()
