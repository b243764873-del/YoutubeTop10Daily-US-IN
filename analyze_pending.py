import os
import json
from typing import List, Tuple

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


def analyze_to_hook_and_templates(title: str, transcript: str):
    """
    Temporary non-AI analyzer (safe & free):
    Later you can replace this with an AI call.
    """
    t = (transcript or "").strip()
    snippet = t[:220] + ("..." if len(t) > 220 else "")

    hook = f"Hook idea: {title.strip()}"
    script_template = (
        "0-2s: [Hook]\n"
        "2-8s: [Context + 1 key point]\n"
        "8-15s: [Twist / payoff]\n"
        "15-20s: [CTA / loop]\n"
        f"\nTranscript snippet: {snippet}"
    )
    prompt_pack = {
        "voiceover_prompt": "Write an 18-20s voiceover using the structure above. Keep it original; do not copy phrases.",
        "video_prompt": "Create a vertical 9:16 short video, fast cuts, big subtitles, 18-20 seconds.",
        "subtitle_prompt": "Generate concise subtitles; emphasize keywords; keep lines short.",
    }
    return hook, script_template, json.dumps(prompt_pack, ensure_ascii=False)


def main():
    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]

    gc = gsheet_client_from_sa_json(sa_json)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(SHEET_NAME)

    header = ws.row_values(1)
    idx = find_header_indices(header)

    required_cols = ["ai_status", "ai_hook", "ai_script_template", "ai_prompt_pack_json", "video_id", "title", "region"]
    for c in required_cols:
        if c not in idx:
            raise RuntimeError(f"Missing header column: {c}. Please ensure daily_rank headers are correct.")

    # Load all values once
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

    # We'll batch update these columns for each pending row:
    # ai_status, ai_hook, ai_script_template, ai_prompt_pack_json
    c_status = idx["ai_status"]
    c_hook = idx["ai_hook"]
    c_script = idx["ai_script_template"]
    c_pack = idx["ai_prompt_pack_json"]

    # Prepare values in row order for a single rectangular range update:
    # From min_row..max_row and from min_col..max_col
    min_row = min(pending_rnums)
    max_row = max(pending_rnums)
    min_col = min(c_status, c_hook, c_script, c_pack)
    max_col = max(c_status, c_hook, c_script, c_pack)

    # Build a map rnum -> (status, hook, script, pack)
    updates = {}
    for rnum in pending_rnums:
        row = all_rows[rnum - 1]
        video_id = row[idx["video_id"] - 1].strip()
        title = row[idx["title"] - 1].strip()
        region = row[idx["region"] - 1].strip()

        prefer_langs = ("en", "hi") if region == "IN" else ("en",)
        transcript, meta = get_transcript_text(video_id, prefer_langs=prefer_langs)

        hook, script_template, prompt_pack_json = analyze_to_hook_and_templates(title, transcript)

        if meta.get("error") == "no_transcript":
            new_status = "no_transcript"
        elif meta.get("error"):
            new_status = "failed"
        else:
            new_status = "done"

        updates[rnum] = {
            c_status: new_status,
            c_hook: hook,
            c_script: script_template,
            c_pack: prompt_pack_json,
        }
        print(f"[OK] prepared row={rnum} video_id={video_id} status={new_status}")

    # Create a 2D array for the whole rectangle [min_row..max_row] x [min_col..max_col]
    values = []
    for rnum in range(min_row, max_row + 1):
        row_vals = []
        for c in range(min_col, max_col + 1):
            if rnum in updates and c in updates[rnum]:
                row_vals.append(updates[rnum][c])
            else:
                row_vals.append("")  # leave blank for non-pending rows inside the rectangle
        values.append(row_vals)

    a1_range = f"{a1_col(min_col)}{min_row}:{a1_col(max_col)}{max_row}"
    ws.update(a1_range, values, value_input_option="RAW")

    print(f"[DONE] Batch updated range {a1_range} for {len(pending_rnums)} pending rows.")


if __name__ == "__main__":
    main()
