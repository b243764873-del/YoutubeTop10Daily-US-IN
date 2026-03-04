import os
import json
from typing import List

import gspread
from google.oauth2.service_account import Credentials
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


def get_transcript(video_id: str):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x["text"] for x in transcript])
    except:
        return ""


def analyze_with_gemini(client, title, transcript):

    prompt = f"""
You are a short-form video strategist.

Analyze this YouTube Short.

Title:
{title}

Transcript:
{transcript}

Return JSON ONLY with fields:

category
ai_generatable
hook_patterns
beat_sheet
variants

Rules:
- Shorts <=20 seconds
- Output must be JSON
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        text = response.text.strip()

        return json.loads(text)

    except Exception as e:
        return {"error": str(e)}


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

    for rnum in range(2, len(rows) + 1):
        row = rows[rnum - 1]
        status = row[idx["ai_status"] - 1]

        if status == "pending":
            pending.append(rnum)

    if not pending:
        print("no pending rows")
        return

    updates = []

    for rnum in pending:

        row = rows[rnum - 1]

        video_id = row[idx["video_id"] - 1]
        title = row[idx["title"] - 1]

        transcript = get_transcript(video_id)

        result = analyze_with_gemini(client, title, transcript)

        if "error" in result:
            status = "failed"
        else:
            status = "done"

        hook = " | ".join(result.get("hook_patterns", []))
        beat = json.dumps(result.get("beat_sheet", []))
        pack = json.dumps(result)
        variants = json.dumps(result.get("variants", []))

        row_values = [
            status,
            hook,
            beat,
            pack,
            variants
        ]

        start = idx["ai_status"]
        end = idx["ai_variants_json"]

        rng = f"{a1_col(start)}{rnum}:{a1_col(end)}{rnum}"

        updates.append({
            "range": rng,
            "values": [row_values]
        })

    ws.batch_update(updates, value_input_option="RAW")

    print("analysis done")


if __name__ == "__main__":
    main()
