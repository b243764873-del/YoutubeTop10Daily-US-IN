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


SHEET_NAME = "daily_rank"
OPENAI_MODEL = "gpt-4.1-mini"


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


def get_transcript_text(video_id: str, prefer_langs=("en", "hi")) -> Tuple[str, dict]:
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        for lang in prefer_langs:
            try:
                t = transcripts.find_transcript([lang])
                segs = t.fetch()
                text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
                return text, {"lang": t.language_code, "is_generated": getattr(t, "is_generated", None)}
            except Exception:
                pass

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


def openai_breakdown_v3(client: OpenAI, title: str, transcript: str, comments: List[str], region: str) -> Dict[str, Any]:
    transcript_snippet = (transcript or "").strip()
    if len(transcript_snippet) > 3500:
        transcript_snippet = transcript_snippet[:3500] + "..."

    system = """
You are a short-form video strategist/editor.
Analyze a trending YouTube Short (<=20s) and extract a REUSABLE FORMAT to create an ORIGINAL video.
Do NOT copy exact wording, unique jokes, names/brands, or identifiable characters.
Clone structure/pacing, not content.

Return VALID JSON ONLY (no markdown).
Optimize for <=20 seconds, vertical 9:16.

Schema:
{
  "category": "string",
  "ai_generatable": true/false,
  "ai_generatable_reason": "string",
  "hook_patterns": ["3 templates with [VARIABLES]"],
  "beat_sheet": [
    {
      "start_sec": number,
      "end_sec": number,
      "purpose": "hook/setup/twist/payoff/loop",
      "on_screen_text_template": "string with [VARIABLES]",
      "voiceover_template": "string with [VARIABLES]",
      "visual_template": "string",
      "edit_notes": "string"
    }
  ],
  "subtitle_style": { "max_chars_per_line": number, "lines": number, "emphasis_rules": ["..."], "placement": "top/middle/bottom" },
  "edit_style": { "avg_shot_len_sec": number, "transitions": ["..."], "zoom_shake_usage": "string", "sfx_cues": ["..."] },
  "music_style": { "bpm_range": "string", "mood": "string", "instruments": ["..."], "reference_keywords": ["..."] },
  "reusable_variables": ["[TOPIC]", "..."],
  "risk_notes": ["..."],
  "generation_prompts": {
    "voiceover_prompt": "string",
    "video_prompt": "string",
    "subtitle_prompt": "string"
  },
  "variants": [
    {
      "variant_title": "string",
      "variables": { "KEY": "VALUE" },
      "voiceover": "18-20s voiceover",
      "on_screen_text": ["subtitle lines in order"],
      "video_prompt": "string",
      "subtitle_prompt": "string"
    }
  ]
}

Variants:
- Provide EXACTLY 10 variants
- Each voiceover <=20s
- Avoid copying source wording; keep original
"""

    user = {
        "region": region,
        "title": title,
        "transcript": transcript_snippet if transcript_snippet else "(no transcript available)",
        "top_comments": comments[:20],
        "constraints": {"max_duration_sec": 20, "aspect_ratio": "9:16"},
    }

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    raw = resp.output_text.strip()
    return json.loads(raw)


def main():
    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]
    yt_key = os.environ["YOUTUBE_API_KEY"]

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

    print(f"[INFO] Pending rows: {len(pending_rnums)}")

    youtube = yt_client(yt_key)
    client = OpenAI()  # reads OPENAI_API_KEY from env

    # Columns
    c_status = idx["ai_status"]
    c_cat = idx["ai_category"]
    c_gen = idx["ai_generatable"]
    c_hook = idx["ai_hook"]
    c_script = idx["ai_script_template"]
    c_pack = idx["ai_prompt_pack_json"]
    c_vars = idx["ai_variants_json"]

    # ✅ Update only pending rows (no big rectangle), prevents wiping & reduces Sheets writes
    batch_payload = []

    for rnum in pending_rnums:
        row = all_rows[rnum - 1]
        video_id = row[idx["video_id"] - 1].strip()
        title = row[idx["title"] - 1].strip()
        region = row[idx["region"] - 1].strip()

        prefer_langs = ("en", "hi") if region == "IN" else ("en",)
        transcript, meta = get_transcript_text(video_id, prefer_langs=prefer_langs)
        comments = get_top_comments(youtube, video_id, max_comments=20)

        try:
            data = openai_breakdown_v3(client, title, transcript, comments, region)

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
            print(f"[OK] row={rnum} video_id={video_id} transcript={meta.get('error','ok')} comments={len(comments)} done")

        except Exception as e:
            status = "failed"
            values = [
                status,
                "",
                "",
                "",
                "",
                json.dumps({"error": str(e)}, ensure_ascii=False),
                "",
            ]
            print(f"[WARN] row={rnum} video_id={video_id} failed: {e}")

        start_col = min(c_status, c_cat, c_gen, c_hook, c_script, c_pack, c_vars)
        end_col = max(c_status, c_cat, c_gen, c_hook, c_script, c_pack, c_vars)
        rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"

        batch_payload.append({"range": rng, "values": [values]})

        # little throttle to be nice
        time.sleep(0.6)

    ws.batch_update(batch_payload, value_input_option="RAW")
    print(f"[DONE] Updated {len(pending_rnums)} rows.")


if __name__ == "__main__":
    main()
