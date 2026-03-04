import os
import json
from typing import List, Tuple

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

from youtube_transcript_api import YouTubeTranscriptApi
from google import genai


SHEET_NAME = "daily_rank"


def gsheet_client_from_sa_json(sa_json_str: str):
    sa_info = json.loads(sa_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def find_header_indices(header_row: List[str]):
    return {name: (i + 1) for i, name in enumerate(header_row)}


def a1_col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except:
        return ""


def analyze_with_gemini(model, title, transcript):

    prompt = f"""
You are a shorts content strategist.

Analyze this short video:

Title:
{title}

Transcript:
{transcript}

Return JSON only with fields:

category
ai_generatable
hook_patterns
beat_sheet
variants (10 scripts)

Each variant must be <=20 seconds.
"""

    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

text = response.text


    try:
        return json.loads(text)
    except:
        return {"error": text}


def main():

    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]
    gemini_key = os.environ["GEMINI_API_KEY"]

    client = genai.Client(api_key=gemini_key)

    gc = gsheet_client_from_sa_json(sa_json)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(SHEET_NAME)

    header = ws.row_values(1)
    idx = find_header_indices(header)

    rows = ws.get_all_values()

    pending = []

    for rnum in range(2, len(rows)+1):
        row = rows[rnum-1]
        if row[idx["ai_status"]-1] == "pending":
            pending.append(rnum)

    if not pending:
        print("no pending rows")
        return

    updates = {}

    for rnum in pending:

        row = rows[rnum-1]

        video_id = row[idx["video_id"]-1]
        title = row[idx["title"]-1]

        transcript = get_transcript(video_id)

        result = analyze_with_gemini(model, title, transcript)

        if "error" in result:
            status = "failed"
        else:
            status = "done"

        updates[rnum] = {
            idx["ai_status"]: status,
            idx["ai_hook"]: " | ".join(result.get("hook_patterns", [])),
            idx["ai_script_template"]: json.dumps(result.get("beat_sheet", [])),
            idx["ai_prompt_pack_json"]: json.dumps(result),
            idx["ai_variants_json"]: json.dumps(result.get("variants", []))
        }

    data = []

    for rnum, colmap in updates.items():

        row_vals = [
            colmap[idx["ai_status"]],
            colmap[idx["ai_hook"]],
            colmap[idx["ai_script_template"]],
            colmap[idx["ai_prompt_pack_json"]],
            colmap[idx["ai_variants_json"]],
        ]

        start = idx["ai_status"]
        end = idx["ai_variants_json"]

        rng = f"{a1_col(start)}{rnum}:{a1_col(end)}{rnum}"

        data.append({
            "range": rng,
            "values": [row_vals]
        })

    ws.batch_update(data, value_input_option="RAW")

    print("analysis done")


if __name__ == "__main__":
    main()
