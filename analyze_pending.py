import os
import json
from typing import List, Tuple

from openai import OpenAI

import gspread
from google.oauth2.service_account import Credentials
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


SHEET_NAME = "daily_rank"


def gsheet_client_from_sa_json(sa_json_str: str):
    sa_info = json.loads(sa_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def find_header_indices(header_row: List[str]):
    # header_row is list of header strings; return 1-based indices
    return {name: (i + 1) for i, name in enumerate(header_row)}


def a1_col(n: int) -> str:
    """1->A, 2->B ... 27->AA"""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


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


def analyze_to_hook_and_templates(client: OpenAI, title: str, transcript: str, region: str) -> Tuple[str, str, str]:
    """
    Use OpenAI to produce a structured 'clone-the-format-not-the-content' breakdown.
    Returns: (hook_text, script_template, prompt_pack_json)
    """
    transcript_snippet = (transcript or "").strip()
    if len(transcript_snippet) > 4000:
        transcript_snippet = transcript_snippet[:4000] + "..."

    instruction = """
You are a short-form video strategist and editor.
Goal: extract REUSABLE FORMAT (structure, pacing, hook patterns) from a trending YouTube Short (<=20s),
so we can generate an ORIGINAL video with a similar format.

Hard rules:
- Do NOT copy unique jokes, exact phrasing, names, brands, or identifiable characters.
- Output MUST be valid JSON only (no markdown).
- Keep everything optimized for <=20 seconds and 9:16.

Return JSON with keys:
- category: short label
- hook_patterns: array of 3 reusable hook templates with [VARIABLES]
- beat_sheet: array of beats with {start_sec, end_sec, purpose, on_screen_text_template, voiceover_template, visual_template, edit_notes}
- subtitle_style: {max_chars_per_line, lines, emphasis_rules, placement}
- edit_style: {avg_shot_len_sec, transitions, zoom_shake_usage, sfx_cues}
- music_style: {bpm_range, mood, instruments, reference_keywords}
- reusable_variables: array of variables to swap (e.g., [TOPIC], [NUMBER], [CONTRAST], [PAYOFF])
- risk_notes: array of risks to avoid (copyright, repetition, sensitive)
- generation_prompts:
  - voiceover_prompt
  - video_prompt
  - subtitle_prompt
"""

    user_input = {
        "region": region,
        "title": title,
        "transcript": transcript_snippet if transcript_snippet else "(no transcript available)",
        "constraints": {
            "max_duration_sec": 20,
            "aspect_ratio": "9:16",
            "language_hint": "English" if region == "US" else "English/Hindi mix ok",
        },
    }

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
        ],
    )

    raw = resp.output_text.strip()
    data = json.loads(raw)

    hook_text = " | ".join(data.get("hook_patterns", [])[:3])
    script_template = json.dumps(data.get("beat_sheet", []), ensure_ascii=False)
    prompt_pack_json = json.dumps(data, ensure_ascii=False)

    return hook_text, script_template, prompt_pack_json


def main():
    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]

    # OpenAI client (reads OPENAI_API_KEY from env)
    client = OpenAI()

    gc = gsheet_client_from_sa_json(sa_json)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(SHEET_NAME)

    header = ws.row_values(1)
    idx = find_header_indices(header)

    required_cols = ["ai_status", "ai_hook", "ai_script_template", "ai_prompt_pack_json", "video_id", "title", "region"]
    for c in required_cols:
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

    print(f"[INFO] Pending rows: {len(pending_rnums)}")

    c_status = idx["ai_status"]
    c_hook = idx["ai_hook"]
    c_script = idx["ai_script_template"]
    c_pack = idx["ai_prompt_pack_json"]

    min_row = min(pending_rnums)
    max_row = max(pending_rnums)
    min_col = min(c_status, c_hook, c_script, c_pack)
    max_col = max(c_status, c_hook, c_script, c_pack)

    updates = {}
    for rnum in pending_rnums:
        row = all_rows[rnum - 1]
        video_id = row[idx["video_id"] - 1].strip()
        title = row[idx["title"] - 1].strip()
        region = row[idx["region"] - 1].strip()

        prefer_langs = ("en", "hi") if region == "IN" else ("en",)
        transcript, meta = get_transcript_text(video_id, prefer_langs=prefer_langs)

        try:
            hook, script_template, prompt_pack_json = analyze_to_hook_and_templates(
                client=client, title=title, transcript=transcript, region=region
            )
            new_status = "done"  # ✅ OpenAI 成功就 done（即使沒 transcript）
        except Exception as e:
            hook = ""
            script_template = ""
            prompt_pack_json = json.dumps({"error": str(e)}, ensure_ascii=False)
            new_status = "failed"

        updates[rnum] = {
            c_status: new_status,
            c_hook: hook,
            c_script: script_template,
            c_pack: prompt_pack_json,
        }
        print(f"[OK] prepared row={rnum} video_id={video_id} transcript={meta.get('error', 'ok')} ai_status={new_status}")

    values = []
    for rnum in range(min_row, max_row + 1):
        row_vals = []
        for c in range(min_col, max_col + 1):
            if rnum in updates and c in updates[rnum]:
                row_vals.append(updates[rnum][c])
            else:
                row_vals.append("")
        values.append(row_vals)

    a1_range = f"{a1_col(min_col)}{min_row}:{a1_col(max_col)}{max_row}"
    ws.update(a1_range, values, value_input_option="RAW")
    print(f"[DONE] Batch updated range {a1_range} for {len(pending_rnums)} pending rows.")


if __name__ == "__main__":
    main()
