import os
import json
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


SHEET_NAME = "daily_rank"
COLS = {
    "date": "A",
    "region": "B",
    "rank": "C",
    "video_id": "D",
    "title": "E",
    "channel_title": "F",
    "views": "G",
    "published_at": "H",
    "duration_sec": "I",
    "is_new": "J",
    "video_url": "K",
    "ai_status": "L",
    "ai_hook": "M",
    "ai_script_template": "N",
    "ai_prompt_pack_json": "O",
}


def gsheet_client_from_sa_json(sa_json_str: str):
    sa_info = json.loads(sa_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def find_header_indices(header_row):
    # header_row is list of header strings
    idx = {name: (i + 1) for i, name in enumerate(header_row)}  # 1-based
    return idx


def get_transcript_text(video_id: str, prefer_langs=("en", "hi")):
    """
    Try to get transcript. For US/IN we try English/Hindi first, then any.
    Returns (text, meta) where meta includes language and is_generated.
    """
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try preferred languages first
        for lang in prefer_langs:
            try:
                t = transcripts.find_transcript([lang])
                segs = t.fetch()
                text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
                return text, {"lang": t.language_code, "is_generated": getattr(t, "is_generated", None)}
            except Exception:
                pass

        # Fallback: pick first available transcript (manual preferred if possible)
        try:
            t = transcripts.find_manually_created_transcript(transcripts._TranscriptList__langs)  # internal; may fail
            segs = t.fetch()
            text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
            return text, {"lang": t.language_code, "is_generated": getattr(t, "is_generated", None)}
        except Exception:
            pass

        # Otherwise pick first transcript
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
    先做一個「不依賴外部AI」的臨時版本：
    - 取 title + transcript 前幾句，給你可手動用的摘要
    之後你選 AI 供應商，我再幫你把這個函式改成真的呼叫 AI。
    """
    t = transcript.strip()
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
        "voiceover_prompt": "Write a 18-20s voiceover script using the structure above. Keep it original, do not copy phrases.",
        "video_prompt": "Create a vertical 9:16 short video, fast cuts, big subtitles, 18-20 seconds.",
        "subtitle_prompt": "Generate concise subtitles, <=12 chars per line where possible, emphasize keywords.",
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

    # Read all rows (simple & reliable for early stage)
    rows = ws.get_all_values()
    if len(rows) <= 1:
        print("[INFO] No data rows.")
        return

    # Identify pending rows
    # ai_status column name is "ai_status" in your headers
    ai_status_col = idx.get("ai_status")
    video_id_col = idx.get("video_id")
    title_col = idx.get("title")
    region_col = idx.get("region")

    if not all([ai_status_col, video_id_col, title_col, region_col]):
        raise RuntimeError("Missing expected headers in row 1. Ensure daily_rank headers match.")

    pending_row_numbers = []
    for rnum in range(2, len(rows) + 1):  # sheet row numbers
        row = rows[rnum - 1]
        status = row[ai_status_col - 1].strip() if len(row) >= ai_status_col else ""
        if status == "pending":
            pending_row_numbers.append(rnum)

    if not pending_row_numbers:
        print("[INFO] No pending rows.")
        return

    print(f"[INFO] Pending rows: {len(pending_row_numbers)}")

    # Process each pending row
    for rnum in pending_row_numbers:
        row = ws.row_values(rnum)
        video_id = row[video_id_col - 1].strip()
        title = row[title_col - 1].strip()
        region = row[region_col - 1].strip()

        # prefer languages by region
        prefer_langs = ("en", "hi") if region == "IN" else ("en",)

        transcript, meta = get_transcript_text(video_id, prefer_langs=prefer_langs)

        hook, script_template, prompt_pack_json = analyze_to_hook_and_templates(title, transcript)

        # Write back to sheet
        # ai_hook (M), ai_script_template (N), ai_prompt_pack_json (O)
        # ai_status -> done / no_transcript / failed
        if meta.get("error") == "no_transcript":
            new_status = "no_transcript"
        elif meta.get("error"):
            new_status = "failed"
        else:
            new_status = "done"

        ws.update_cell(rnum, idx["ai_status"], new_status)
        ws.update_cell(rnum, idx["ai_hook"], hook)
        ws.update_cell(rnum, idx["ai_script_template"], script_template)
        ws.update_cell(rnum, idx["ai_prompt_pack_json"], prompt_pack_json)

        print(f"[OK] row={rnum} video_id={video_id} status={new_status}")

    print("[DONE] analysis complete.")


if __name__ == "__main__":
    main()
