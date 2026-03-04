import os
import json
import time
import re
from typing import List, Tuple, Dict, Any

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from openai import OpenAI


# =========================
# ENV / CONFIG
# =========================
SHEET_NAME = os.getenv("SHEET_NAME", "daily_rank")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1800"))

MAX_PER_RUN = int(os.getenv("MAX_PER_RUN", "5"))
VARIANT_COUNT = int(os.getenv("VARIANT_COUNT", "3"))

SLEEP_BETWEEN_ROWS_SEC = float(os.getenv("SLEEP_BETWEEN_ROWS_SEC", "0.8"))

# --- C2 meta classifier
AI_CLASSIFY_ENABLED = os.getenv("AI_CLASSIFY_ENABLED", "1") == "1"
AI_CLASSIFY_MODEL = os.getenv("AI_CLASSIFY_MODEL", OPENAI_MODEL)
AI_CLASSIFY_MAX_TOKENS = int(os.getenv("AI_CLASSIFY_MAX_TOKENS", "220"))
AI_CLASSIFY_MIN_CONFIDENCE = int(os.getenv("AI_CLASSIFY_MIN_CONFIDENCE", "45"))
AI_CLASSIFY_SLEEP_SEC = float(os.getenv("AI_CLASSIFY_SLEEP_SEC", "0.3"))

# --- C3 visual classifier (thumbnail)
AI_VISUAL_ENABLED = os.getenv("AI_VISUAL_ENABLED", "1") == "1"
AI_VISUAL_MODEL = os.getenv("AI_VISUAL_MODEL", OPENAI_MODEL)
AI_VISUAL_MAX_TOKENS = int(os.getenv("AI_VISUAL_MAX_TOKENS", "220"))
AI_VISUAL_MIN_CONFIDENCE = int(os.getenv("AI_VISUAL_MIN_CONFIDENCE", "55"))

# --- Allowlist keywords: if any appears in title/desc/tags/channel -> bypass classifier and analyze
# Example:
# AI_ALLOWLIST="midjourney,runway,pika,stablediffusion,sdxl,comfyui,kling,sora,ai generated,#aivideo,#aigenerated"
AI_ALLOWLIST = [s.strip().lower() for s in (os.getenv("AI_ALLOWLIST", "")).split(",") if s.strip()]


# =========================
# Google Sheets helpers
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


# =========================
# YouTube helpers
# =========================
def yt_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def safe_truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "..."


def get_video_snippet(youtube, video_id: str) -> Dict[str, Any]:
    """
    Fetch video snippet (title/description/channelTitle/tags/thumbnails).
    """
    try:
        res = youtube.videos().list(part="snippet", id=video_id, maxResults=1).execute()
        items = res.get("items", [])
        if not items:
            return {}
        return items[0].get("snippet", {}) or {}
    except Exception:
        return {}


def get_top_comments(youtube, video_id: str, max_comments: int = 20) -> List[str]:
    """
    Fetch top comments (best-effort). Return [] on any error.
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


def get_transcript_text(video_id: str, prefer_langs=("en", "hi")) -> Tuple[str, dict]:
    """
    Compatible across youtube-transcript-api versions:
    - Newer: list_transcripts()
    - Older: get_transcript()
    """
    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
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
                try:
                    segs = t.fetch()
                    text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
                    if text:
                        return text, {"lang": t.language_code, "is_generated": getattr(t, "is_generated", None)}
                except Exception:
                    continue

            return "", {"error": "no_transcript"}

        # older fallback
        for lang in prefer_langs:
            try:
                segs = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                text = " ".join(s.get("text", "").replace("\n", " ").strip() for s in segs).strip()
                if text:
                    return text, {"lang": lang, "is_generated": None}
            except Exception:
                pass

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


def pick_thumbnail_url(snippet: Dict[str, Any]) -> str:
    thumbs = (snippet or {}).get("thumbnails") or {}
    return (
        (thumbs.get("maxres") or {}).get("url")
        or (thumbs.get("standard") or {}).get("url")
        or (thumbs.get("high") or {}).get("url")
        or (thumbs.get("medium") or {}).get("url")
        or (thumbs.get("default") or {}).get("url")
        or ""
    )


def allowlist_hit(title: str, desc: str, channel_title: str, tags: List[str]) -> Tuple[bool, List[str]]:
    """
    If AI_ALLOWLIST is configured, check if any allowlist term exists in metadata.
    """
    if not AI_ALLOWLIST:
        return False, []
    blob = " ".join(
        [
            (title or "").lower(),
            (desc or "").lower(),
            (channel_title or "").lower(),
            " ".join([(t or "").lower() for t in (tags or [])]),
        ]
    )
    hits = [kw for kw in AI_ALLOWLIST if kw and kw in blob]
    return (len(hits) > 0), hits[:10]


# =========================
# OpenAI: schema builders
# =========================
def build_meta_classifier_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_ai_generated": {"type": "boolean"},
            "confidence": {"type": "number"},  # 0-100
            "reasons": {"type": "array", "items": {"type": "string"}},
            "signals": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["is_ai_generated", "confidence", "reasons", "signals"],
    }


def build_visual_classifier_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_ai_generated": {"type": "boolean"},
            "confidence": {"type": "number"},  # 0-100
            "reasons": {"type": "array", "items": {"type": "string"}},
            "visual_signals": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["is_ai_generated", "confidence", "reasons", "visual_signals"],
    }


def build_analyze_schema(variant_count: int) -> Dict[str, Any]:
    """
    Important: We intentionally do NOT include 'variables' inside variants,
    because strict schemas get brittle. Put placeholders directly into the text.
    """
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
                    "required": ["variant_title", "voiceover", "on_screen_text", "video_prompt", "subtitle_prompt"],
                },
            },
        },
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
    }


# =========================
# OpenAI: calls + retry
# =========================
def parse_retry_delay_seconds(err_text: str) -> int | None:
    if not err_text:
        return None
    m = re.search(r"retry in\s+(\d+)(?:\.\d+)?s", err_text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def with_retry(fn, attempts: int = 3, base_sleep: int = 6):
    last_err = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = str(e)
            delay = parse_retry_delay_seconds(last_err)
            if delay is None:
                delay = base_sleep * i
            print(f"[WARN] attempt {i}/{attempts} failed: {last_err}. Sleep {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Failed after {attempts} attempts: {last_err}")


def openai_meta_classify(
    client: OpenAI,
    title: str,
    description: str,
    channel_title: str,
    tags: List[str],
    top_comments: List[str],
) -> Dict[str, Any]:
    schema = build_meta_classifier_schema()
    sys = (
        "You are a strict classifier. Decide if this YouTube Short is likely AI-generated.\n"
        "Use ONLY metadata signals: title/description/channel/tags/comments.\n"
        "Return JSON only."
    )
    user = {
        "title": safe_truncate(title, 200),
        "description": safe_truncate(description, 900),
        "channel_title": safe_truncate(channel_title, 120),
        "tags": (tags or [])[:30],
        "top_comments": [safe_truncate(c, 160) for c in (top_comments or [])[:10]],
        "notes": {
            "ai_generated_signals": [
                "explicit AI-generated wording",
                "tools: Midjourney/Runway/Pika/Stable Diffusion/ComfyUI/Kling/Sora",
                "hashtags like #aivideo #aigenerated #midjourney #runway #pika #stablediffusion",
            ]
        },
        "output_rules": {"confidence_scale": "0-100", "be_conservative": True},
    }

    resp = client.chat.completions.create(
        model=AI_CLASSIFY_MODEL,
        max_tokens=AI_CLASSIFY_MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ai_meta_classifier", "schema": schema, "strict": True},
        },
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return json.loads(content)


def openai_visual_classify_from_thumbnail(
    client: OpenAI,
    thumbnail_url: str,
    title: str,
    description: str,
) -> Dict[str, Any]:
    schema = build_visual_classifier_schema()
    sys = (
        "You are a strict classifier. Decide if this YouTube Short is likely AI-generated "
        "based on VISUAL style from the thumbnail.\n"
        "Look for AI/diffusion artifacts, uncanny textures, synthetic lighting, strange details, generic AI aesthetics.\n"
        "Return JSON only."
    )

    user_payload = {
        "title": safe_truncate(title, 200),
        "description": safe_truncate(description, 600),
        "task": "Classify AI-generated likelihood from thumbnail image.",
        "output_rules": {"confidence_scale": "0-100", "be_conservative": True},
    }

    resp = client.chat.completions.create(
        model=AI_VISUAL_MODEL,
        max_tokens=AI_VISUAL_MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ai_visual_classifier", "schema": schema, "strict": True},
        },
        messages=[
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": thumbnail_url}},
                ],
            },
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return json.loads(content)


def openai_analyze(
    client: OpenAI,
    title: str,
    transcript: str,
    comments: List[str],
    region: str,
    variant_count: int,
    meta_classifier: Dict[str, Any] | None,
    visual_classifier: Dict[str, Any] | None,
    allowlist_hits: List[str] | None,
) -> Dict[str, Any]:
    schema = build_analyze_schema(variant_count)

    system = (
        "You are a short-form video strategist/editor.\n"
        "Goal: extract REUSABLE FORMAT (structure, pacing, hook patterns) from a trending YouTube Short (<=20s), "
        "so we can generate an ORIGINAL video with a similar format.\n"
        "Hard rules:\n"
        "- Do NOT copy unique jokes, exact phrasing, names, brands, or identifiable characters.\n"
        "- Optimize for <=20 seconds and 9:16.\n"
        "- Put placeholders like [TOPIC], [NUMBER], [CONTRAST], [PAYOFF] directly in text fields.\n"
        "- Output MUST be valid JSON according to the provided schema.\n"
    )

    user = {
        "region": region,
        "title": title,
        "transcript": safe_truncate(transcript, 2500) if transcript else "(no transcript available)",
        "top_comments": (comments or [])[:20],
        "constraints": {"max_duration_sec": 20, "aspect_ratio": "9:16", "variant_count": variant_count},
        "context_for_trace": {
            "meta_classifier": meta_classifier,
            "visual_classifier": visual_classifier,
            "allowlist_hits": allowlist_hits or [],
        },
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "shorts_breakdown", "schema": schema, "strict": True},
        },
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return json.loads(content)


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

    required_cols = [
        "ai_status",
        "ai_category",
        "ai_generatable",
        "ai_hook",
        "ai_script_template",
        "ai_prompt_pack_json",
        "ai_variants_json",
        "video_id",
        "title",
        "region",
    ]
    for c in required_cols:
        if c not in idx:
            raise RuntimeError(f"Missing header column: {c}. Please ensure daily_rank headers are correct.")

    all_rows = ws.get_all_values()
    if len(all_rows) <= 1:
        print("[INFO] No data rows.", flush=True)
        return

    pending_rnums = []
    for rnum in range(2, len(all_rows) + 1):
        row = all_rows[rnum - 1]
        status = row[idx["ai_status"] - 1].strip() if len(row) >= idx["ai_status"] else ""
        if status == "pending":
            pending_rnums.append(rnum)

    if not pending_rnums:
        print("[INFO] No pending rows.", flush=True)
        return

    total_pending = len(pending_rnums)
    to_process = pending_rnums[:MAX_PER_RUN]
    print(
        f"[INFO] Pending rows total={total_pending}, will process now={len(to_process)} (MAX_PER_RUN={MAX_PER_RUN})",
        flush=True,
    )

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
    width = end_col - start_col + 1

    # batch_update payload
    batch_payload = []

    for rnum in to_process:
        row = all_rows[rnum - 1]
        video_id = row[idx["video_id"] - 1].strip()
        title = row[idx["title"] - 1].strip()
        region = row[idx["region"] - 1].strip()

        print(f"[INFO] start row={rnum} video_id={video_id} region={region}", flush=True)

        # Fetch snippet + comments for classifiers
        snippet = get_video_snippet(youtube, video_id)
        sn_title = (snippet.get("title") or title)
        sn_desc = (snippet.get("description") or "")
        sn_chan = (snippet.get("channelTitle") or "")
        sn_tags = snippet.get("tags") or []
        comments = get_top_comments(youtube, video_id, max_comments=20)

        # Allowlist bypass
        hit, hits = allowlist_hit(sn_title, sn_desc, sn_chan, sn_tags)
        meta_out = None
        visual_out = None

        should_analyze = False
        decision_note = ""

        if hit:
            should_analyze = True
            decision_note = f"allowlist:{hits}"
        else:
            # Meta classifier (C2)
            if AI_CLASSIFY_ENABLED:
                def _do_meta():
                    return openai_meta_classify(
                        client=client,
                        title=sn_title,
                        description=sn_desc,
                        channel_title=sn_chan,
                        tags=sn_tags,
                        top_comments=comments,
                    )

                meta_out = with_retry(_do_meta, attempts=3, base_sleep=4)

                is_ai_meta = bool(meta_out.get("is_ai_generated", False))
                conf_meta = float(meta_out.get("confidence", 0))

                if is_ai_meta and conf_meta >= AI_CLASSIFY_MIN_CONFIDENCE:
                    should_analyze = True
                    decision_note = f"meta_pass(conf={conf_meta})"
                else:
                    # Visual classifier (C3)
                    if AI_VISUAL_ENABLED:
                        thumb_url = pick_thumbnail_url(snippet)
                        if thumb_url:
                            def _do_vis():
                                return openai_visual_classify_from_thumbnail(
                                    client=client,
                                    thumbnail_url=thumb_url,
                                    title=sn_title,
                                    description=sn_desc,
                                )
                            try:
                                visual_out = with_retry(_do_vis, attempts=2, base_sleep=4)
                            except Exception as e:
                                visual_out = {"error": str(e), "thumbnail_url": thumb_url}

                    is_ai_vis = bool((visual_out or {}).get("is_ai_generated", False))
                    conf_vis = float((visual_out or {}).get("confidence", 0)) if isinstance(visual_out, dict) else 0.0

                    if is_ai_vis and conf_vis >= AI_VISUAL_MIN_CONFIDENCE:
                        should_analyze = True
                        decision_note = f"visual_pass(conf={conf_vis})"
                    else:
                        should_analyze = False
                        decision_note = f"skip(meta_conf={conf_meta}, vis_conf={conf_vis})"

                time.sleep(AI_CLASSIFY_SLEEP_SEC)
            else:
                # If classifier disabled, always analyze
                should_analyze = True
                decision_note = "classifier_disabled"

        # If not AI -> skip and write skip info
        if not should_analyze:
            status = "skipped_not_ai"
            skip_obj = {
                "skip_reason": "classifier_c3",
                "decision": decision_note,
                "allowlist_hits": hits,
                "meta_classifier": meta_out,
                "visual_classifier": visual_out,
                "thresholds": {
                    "meta_min_confidence": AI_CLASSIFY_MIN_CONFIDENCE,
                    "visual_min_confidence": AI_VISUAL_MIN_CONFIDENCE,
                },
            }

            # Build row payload for the rectangular range
            rowvals = [""] * width
            def setcol(col, val):
                rowvals[col - start_col] = val

            setcol(c_status, status)
            setcol(c_cat, "")
            setcol(c_gen, "")
            setcol(c_hook, "")
            setcol(c_script, "")
            setcol(c_pack, json.dumps(skip_obj, ensure_ascii=False))
            setcol(c_vars, "")

            rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"
            batch_payload.append({"range": rng, "values": [rowvals]})

            print(f"[SKIP] row={rnum} video_id={video_id} {decision_note}", flush=True)
            time.sleep(SLEEP_BETWEEN_ROWS_SEC)
            continue

        # Transcript (best effort)
        prefer_langs = ("en", "hi") if region == "IN" else ("en",)
        transcript, tmeta = get_transcript_text(video_id, prefer_langs=prefer_langs)
        transcript = safe_truncate(transcript, 4000)

        # Full analyze
        try:
            def _do_analyze():
                return openai_analyze(
                    client=client,
                    title=title,
                    transcript=transcript,
                    comments=comments,
                    region=region,
                    variant_count=VARIANT_COUNT,
                    meta_classifier=meta_out,
                    visual_classifier=visual_out,
                    allowlist_hits=hits,
                )

            data = with_retry(_do_analyze, attempts=3, base_sleep=6)

            hook_text = " | ".join((data.get("hook_patterns") or [])[:3])
            script_template = json.dumps(data.get("beat_sheet") or [], ensure_ascii=False)

            # Add trace fields into pack for debugging
            data["_ai_gate_decision"] = decision_note
            if meta_out:
                data["_ai_classifier_meta"] = meta_out
            if visual_out:
                data["_ai_classifier_visual"] = visual_out
            if hits:
                data["_ai_allowlist_hits"] = hits

            pack_json = json.dumps(data, ensure_ascii=False)
            variants_json = json.dumps(data.get("variants") or [], ensure_ascii=False)

            ai_category = str(data.get("category", "")).strip()
            ai_generatable = "TRUE" if bool(data.get("ai_generatable", False)) else "FALSE"

            status = "done"

            rowvals = [""] * width
            def setcol(col, val):
                rowvals[col - start_col] = val

            setcol(c_status, status)
            setcol(c_cat, ai_category)
            setcol(c_gen, ai_generatable)
            setcol(c_hook, hook_text)
            setcol(c_script, script_template)
            setcol(c_pack, pack_json)
            setcol(c_vars, variants_json)

            rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"
            batch_payload.append({"range": rng, "values": [rowvals]})

            print(
                f"[OK] row={rnum} video_id={video_id} status=done "
                f"transcript={tmeta.get('error','ok')} comments={len(comments)} variants={len(data.get('variants') or [])} "
                f"gate={decision_note}",
                flush=True,
            )

        except Exception as e:
            status = "failed"
            err_obj = {"error": str(e), "decision": decision_note, "meta_classifier": meta_out, "visual_classifier": visual_out}

            rowvals = [""] * width
            def setcol(col, val):
                rowvals[col - start_col] = val

            setcol(c_status, status)
            setcol(c_cat, "")
            setcol(c_gen, "")
            setcol(c_hook, "")
            setcol(c_script, "")
            setcol(c_pack, json.dumps(err_obj, ensure_ascii=False))
            setcol(c_vars, "")

            rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"
            batch_payload.append({"range": rng, "values": [rowvals]})

            print(f"[WARN] row={rnum} video_id={video_id} status=failed err={e}", flush=True)

        time.sleep(SLEEP_BETWEEN_ROWS_SEC)

    # Apply batch update once (save quota)
    ws.batch_update(batch_payload, value_input_option="RAW")

    left = max(total_pending - len(to_process), 0)
    print(f"[DONE] Updated {len(to_process)} rows. (Left pending={left})", flush=True)


if __name__ == "__main__":
    main()
