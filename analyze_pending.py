import os
import json
import time
import re
from typing import List, Dict, Any, Optional

import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

SHEET_NAME = os.getenv("SHEET_NAME", "daily_rank")
MAX_PER_RUN = int(os.getenv("MAX_PER_RUN", "5"))
SLEEP_BETWEEN_ROWS_SEC = float(os.getenv("SLEEP_BETWEEN_ROWS_SEC", "0.8"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2200"))
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "75"))

VARIANT_COUNT = int(os.getenv("VARIANT_COUNT", "3"))

AI_TARGET_SECONDS_LOW = float(os.getenv("AI_TARGET_SECONDS_LOW", "8.0"))
AI_TARGET_SECONDS_HIGH = float(os.getenv("AI_TARGET_SECONDS_HIGH", "12.0"))
VOICEOVER_MAX_CHARS = int(os.getenv("VOICEOVER_MAX_CHARS", "260"))

# Transcript handling
TRANSCRIPT_COL = os.getenv("TRANSCRIPT_COL", "transcript")  # sheet column name
TRANSCRIPT_MAX_CHARS = int(os.getenv("TRANSCRIPT_MAX_CHARS", "9000"))


def gsheet_client_from_sa_json(sa_json_str: str):
    sa_info = json.loads(sa_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def find_header_indices(header_row: List[str]) -> Dict[str, int]:
    return {name: (i + 1) for i, name in enumerate(header_row)}  # 1-based


def a1_col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def sanitize_no_placeholders(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    # remove placeholder-like brackets
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"\{[^}]+\}", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_schema(variant_count: int) -> Dict[str, Any]:
    """
    Keep backward compatibility:
    - We still output: hook_patterns, beat_sheet, generation_prompts, variants[*].{variant_title, voiceover, on_screen_text, video_prompt, subtitle_prompt}
    Add fidelity fields:
    - core_claims, evidence, must_keep, meaning_guardrails
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {"type": "string"},
            "ai_generatable": {"type": "boolean"},
            "ai_generatable_reason": {"type": "string"},

            "core_claims": {
                "type": "array",
                "minItems": 3,
                "maxItems": 7,
                "items": {"type": "string"},
            },
            "evidence": {
                "type": "array",
                "minItems": 2,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim_index": {"type": "number"},
                        "quote": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["claim_index", "quote", "note"],
                },
            },
            "must_keep": {
                "type": "array",
                "minItems": 2,
                "maxItems": 10,
                "items": {"type": "string"},
            },
            "meaning_guardrails": {
                "type": "array",
                "minItems": 3,
                "maxItems": 10,
                "items": {"type": "string"},
            },

            "hook_patterns": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "string"},
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
                        "start_sec", "end_sec", "purpose",
                        "on_screen_text_template", "voiceover_template",
                        "visual_template", "edit_notes"
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
                        "on_screen_text": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 4
                        },
                        "video_prompt": {"type": "string"},
                        "subtitle_prompt": {"type": "string"},
                    },
                    "required": ["variant_title", "voiceover", "on_screen_text", "video_prompt", "subtitle_prompt"],
                },
            },
        },
        "required": [
            "category", "ai_generatable", "ai_generatable_reason",
            "core_claims", "evidence", "must_keep", "meaning_guardrails",
            "hook_patterns", "beat_sheet",
            "subtitle_style", "edit_style", "music_style",
            "reusable_variables", "risk_notes",
            "generation_prompts", "variants"
        ],
    }


def responses_create_structured(
    client: OpenAI,
    *,
    model: str,
    input_messages: List[Dict[str, str]],
    schema: Dict[str, Any],
):
    """
    Compatibility layer:
    - Some SDKs accept response_format=...
    - Some SDKs require text={"format": ...} in Responses API
    """
    try:
        return client.responses.create(
            model=model,
            input=input_messages,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            response_format={
                "type": "json_schema",
                "name": "shorts_breakdown",
                "schema": schema,
                "strict": True,
            },
            timeout=OPENAI_TIMEOUT_SEC,
        )
    except TypeError:
        return client.responses.create(
            model=model,
            input=input_messages,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "shorts_breakdown",
                    "strict": True,
                    "schema": schema,
                }
            },
            timeout=OPENAI_TIMEOUT_SEC,
        )


def analyze_with_openai(client: OpenAI, title: str, region: str, transcript: str) -> Dict[str, Any]:
    schema = build_schema(VARIANT_COUNT)

    instruction = f"""
You are a YouTube Shorts meme strategist.

You MUST base everything on the provided transcript.
Your meme must preserve the original meaning and key event.

Hard rules:
- Do NOT invent facts that are not in the transcript.
- If transcript is unclear, simplify; NEVER guess new details.
- No brands, no real person names, no copyrighted characters.
- No placeholders like [TOPIC] or {{VAR}}; output publishable text only.
- Voiceover must be FINAL script (1–3 short sentences).
- Target spoken length: {AI_TARGET_SECONDS_LOW:.0f}–{AI_TARGET_SECONDS_HIGH:.0f} seconds.
- Voiceover max {VOICEOVER_MAX_CHARS} characters.
- Provide a clear ending beat (last 2 seconds pause/freeze moment).

Output MUST be STRICT JSON matching the schema.
""".strip()

    # Keep transcript compact but meaningful
    tx = (transcript or "").strip()
    if len(tx) > TRANSCRIPT_MAX_CHARS:
        tx = tx[:TRANSCRIPT_MAX_CHARS]

    user_payload = {
        "title": title,
        "region": region,
        "transcript": tx,
        "constraints": {
            "aspect_ratio": "9:16",
            "max_duration_sec": 20,
            "target_voiceover_seconds": [AI_TARGET_SECONDS_LOW, AI_TARGET_SECONDS_HIGH],
        },
    }

    resp = responses_create_structured(
        client,
        model=OPENAI_MODEL,
        input_messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        schema=schema,
    )

    data = json.loads(resp.output_text)

    # Safety sanitize + enforce char limits
    for v in data.get("variants", []):
        v["voiceover"] = sanitize_no_placeholders(v.get("voiceover", ""))[:VOICEOVER_MAX_CHARS]
        v["variant_title"] = sanitize_no_placeholders(v.get("variant_title", ""))
        v["video_prompt"] = sanitize_no_placeholders(v.get("video_prompt", ""))
        v["subtitle_prompt"] = sanitize_no_placeholders(v.get("subtitle_prompt", ""))
        v["on_screen_text"] = [sanitize_no_placeholders(x) for x in (v.get("on_screen_text") or [])]
        v["on_screen_text"] = [x for x in v["on_screen_text"] if x][:4]

    data["hook_patterns"] = [sanitize_no_placeholders(x) for x in (data.get("hook_patterns") or [])][:3]
    data["core_claims"] = [sanitize_no_placeholders(x) for x in (data.get("core_claims") or [])][:7]
    data["must_keep"] = [sanitize_no_placeholders(x) for x in (data.get("must_keep") or [])][:10]
    data["meaning_guardrails"] = [sanitize_no_placeholders(x) for x in (data.get("meaning_guardrails") or [])][:10]

    return data


def _get_cell(row: List[str], idx: Dict[str, int], col: str) -> str:
    if col not in idx:
        return ""
    i = idx[col] - 1
    return (row[i] if len(row) > i else "").strip()


def main():
    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI()

    gc = gsheet_client_from_sa_json(sa_json)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(SHEET_NAME)

    header = ws.row_values(1)
    idx = find_header_indices(header)

    required_cols = [
        "ai_status", "ai_hook", "ai_script_template", "ai_prompt_pack_json", "ai_variants_json",
        "video_id", "title", "region"
    ]
    for c in required_cols:
        if c not in idx:
            raise RuntimeError(f"Missing header column: {c}")

    # transcript column is optional (but strongly recommended)
    has_transcript = TRANSCRIPT_COL in idx

    all_rows = ws.get_all_values()
    if len(all_rows) <= 1:
        print("[INFO] No rows.", flush=True)
        return

    pending = []
    for rnum in range(2, len(all_rows) + 1):
        row = all_rows[rnum - 1]
        st = _get_cell(row, idx, "ai_status")
        if st == "pending":
            pending.append(rnum)

    if not pending:
        print("[INFO] No pending rows.", flush=True)
        return

    to_process = pending[:MAX_PER_RUN]
    print(f"[INFO] Pending rows total={len(pending)}, will process now={len(to_process)} (MAX_PER_RUN={MAX_PER_RUN})", flush=True)
    if not has_transcript:
        print(f"[WARN] Sheet has no '{TRANSCRIPT_COL}' column. AI will rely on title only -> meaning may drift.", flush=True)

    cols = [idx["ai_status"], idx["ai_hook"], idx["ai_script_template"], idx["ai_prompt_pack_json"], idx["ai_variants_json"]]
    start_col = min(cols)
    end_col = max(cols)
    width = end_col - start_col + 1

    payload = []

    for rnum in to_process:
        row = all_rows[rnum - 1]
        title = _get_cell(row, idx, "title")
        region = _get_cell(row, idx, "region")
        transcript = _get_cell(row, idx, TRANSCRIPT_COL) if has_transcript else ""

        status = "failed"
        hook = ""
        script_template = ""
        pack_json = ""
        variants_json = ""

        try:
            out = analyze_with_openai(client, title=title, region=region, transcript=transcript)
            status = "done"
            hook = " | ".join((out.get("hook_patterns") or [])[:3])
            script_template = json.dumps(out.get("beat_sheet") or [], ensure_ascii=False)
            pack_json = json.dumps(out, ensure_ascii=False)
            variants_json = json.dumps(out.get("variants") or [], ensure_ascii=False)
            print(f"[OK] row={rnum} status=done variants={len(out.get('variants') or [])}", flush=True)
        except Exception as e:
            status = "failed"
            pack_json = json.dumps({"error": str(e)}, ensure_ascii=False)
            variants_json = json.dumps([], ensure_ascii=False)
            print(f"[WARN] row={rnum} status=failed err={e}", flush=True)

        rowvals = [""] * width

        def setcol(col_1based: int, val: str):
            rowvals[col_1based - start_col] = val

        setcol(idx["ai_status"], status)
        setcol(idx["ai_hook"], hook)
        setcol(idx["ai_script_template"], script_template)
        setcol(idx["ai_prompt_pack_json"], pack_json)
        setcol(idx["ai_variants_json"], variants_json)

        rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"
        payload.append({"range": rng, "values": [rowvals]})

        time.sleep(SLEEP_BETWEEN_ROWS_SEC)

    ws.batch_update(payload, value_input_option="RAW")
    print(f"[DONE] Updated {len(to_process)} rows.", flush=True)


if __name__ == "__main__":
    main()
