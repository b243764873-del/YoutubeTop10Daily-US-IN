import os
import json
import time
import base64
import subprocess
from typing import List, Dict, Any

import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI


# =========================
# ENV / CONFIG
# =========================
SHEET_NAME = os.getenv("SHEET_NAME", "daily_rank")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

MAX_PER_RUN_GEN = int(os.getenv("MAX_PER_RUN_GEN", "2"))
VARIANT_INDEX = int(os.getenv("GEN_VARIANT_INDEX", "0"))  # which variant to use

RETRY_FAILED = os.getenv("RETRY_FAILED", "0") == "1"

# TTS
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

# Image generation
IMG_MODEL = os.getenv("IMG_MODEL", "gpt-image-1")
IMG_SIZE = os.getenv("IMG_SIZE", "1024x1536")  # vertical-friendly
IMG_QUALITY = os.getenv("IMG_QUALITY", "medium")
IMG_OUTPUT_FORMAT = os.getenv("IMG_OUTPUT_FORMAT", "png")  # png/webp/jpeg

# Video
VIDEO_W = int(os.getenv("VIDEO_W", "1080"))
VIDEO_H = int(os.getenv("VIDEO_H", "1920"))
FPS = int(os.getenv("FPS", "30"))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "20.0"))

# Subtitles (bottom-center)
SUB_MARGIN_V = int(os.getenv("SUB_MARGIN_V", "220"))
SUB_FONT_SIZE = int(os.getenv("SUB_FONT_SIZE", "84"))

# YouTube metadata
YT_MODEL = os.getenv("YT_MODEL", "gpt-4.1-mini")
YT_LANG = os.getenv("YT_LANG", "en")  # en / hi / etc (for now you chose A=English)
YT_MAX_TITLE_CHARS = int(os.getenv("YT_MAX_TITLE_CHARS", "70"))


# =========================
# Utils
# =========================
def run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout.strip()


def ensure_ffmpeg():
    run(["ffmpeg", "-version"])
    run(["ffprobe", "-version"])


def clamp_duration(d: float) -> float:
    if d <= 0:
        return 8.0
    return min(d, MAX_DURATION_SEC)


def safe_load_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def audio_duration_sec(audio_path: str) -> float:
    out = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ])
    try:
        return float(out)
    except Exception:
        return 0.0


# =========================
# Google Sheets helpers
# =========================
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


# =========================
# OpenAI helpers (version-tolerant)
# =========================
def make_tts_mp3(client: OpenAI, text: str, out_path: str):
    """
    Version-tolerant TTS:
    - Prefer response_format="mp3"
    - Fallback to format="mp3"
    - Use write_to_file if available; else bytes fallback
    """
    resp = None
    try:
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="mp3",
        )
    except TypeError:
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            format="mp3",
        )

    if hasattr(resp, "write_to_file"):
        resp.write_to_file(out_path)
        return

    data = None
    if hasattr(resp, "content"):
        data = resp.content
    elif hasattr(resp, "read"):
        data = resp.read()

    if not data:
        raise RuntimeError("TTS response has no audio bytes (no write_to_file/content/read).")

    with open(out_path, "wb") as f:
        f.write(data)


def generate_ai_image(client: OpenAI, prompt: str, out_path: str) -> Dict[str, Any]:
    resp = client.images.generate(
        model=IMG_MODEL,
        prompt=prompt,
        size=IMG_SIZE,
        quality=IMG_QUALITY,
        output_format=IMG_OUTPUT_FORMAT,
    )

    b64 = None
    revised = None
    if getattr(resp, "data", None) and len(resp.data) > 0:
        item = resp.data[0]
        b64 = getattr(item, "b64_json", None) or (item.get("b64_json") if isinstance(item, dict) else None)
        revised = getattr(item, "revised_prompt", None) or (item.get("revised_prompt") if isinstance(item, dict) else None)

    if not b64:
        raise RuntimeError("Image API returned no b64_json.")

    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))

    return {
        "revised_prompt": revised,
        "model": IMG_MODEL,
        "size": IMG_SIZE,
        "quality": IMG_QUALITY,
        "format": IMG_OUTPUT_FORMAT,
    }


def generate_youtube_metadata(client: OpenAI, variant: Dict[str, Any], region: str) -> Dict[str, Any]:
    """
    Return JSON:
    {
      "title": "...",
      "description": "...",
      "hashtags": ["#shorts", ...],
      "tags": ["...", ...]
    }
    """
    voiceover = str(variant.get("voiceover", "")).strip()
    ons = variant.get("on_screen_text", []) or []
    ons = [str(x).strip() for x in ons if str(x).strip()][:3]
    vtitle = str(variant.get("variant_title", "")).strip()

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 6},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 18},
        },
        "required": ["title", "description", "hashtags", "tags"],
    }

    instruction = f"""
You are a YouTube Shorts copywriter.

Write metadata for a MEME-style Shorts video.
Language: {YT_LANG} (keep it natural for US audience).
Max title length: {YT_MAX_TITLE_CHARS} characters.

Rules:
- Title: short, hooky, 1 emoji max, MUST include "#shorts" at the end.
- Do not use copyrighted character names, brands, or real person names.
- Description: 2 short lines + a simple CTA (e.g., "Follow for more.") + hashtags on last line.
- Hashtags: 3-6, include "#shorts" and 1-2 topical hashtags.
- Tags: 6-18 short keywords, no hashtags inside tags.
- Keep it safe and general (no harassment, no slurs).
Return STRICT JSON only.
""".strip()

    user_payload = {
        "region": region,
        "variant_title": vtitle,
        "voiceover": voiceover[:600],
        "on_screen_text": ons,
    }

    # Responses API with JSON schema output
    resp = client.responses.create(
        model=YT_MODEL,
        input=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        max_output_tokens=450,
        response_format={
            "type": "json_schema",
            "name": "yt_shorts_metadata",
            "schema": schema,
            "strict": True,
        },
    )

    text = (resp.output_text or "").strip()
    return json.loads(text)


# =========================
# Subtitle (ASS)
# =========================
def build_ass_subtitles_bottom(lines: List[str], total_dur: float, ass_path: str):
    cleaned = [str(x).strip() for x in (lines or []) if str(x).strip()]
    if not cleaned:
        cleaned = ["WAIT FOR IT...", "😂", ""]

    beats = cleaned[:3]
    n = len(beats)
    seg = max(total_dur / max(n, 1), 1.5)

    def fmt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    header = f"""[Script Info]
Title: MemeSub
ScriptType: v4.00+
PlayResX: {VIDEO_W}
PlayResY: {VIDEO_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,{SUB_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,8,2,2,60,60,{SUB_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    t0 = 0.0
    for txt in beats:
        start = t0
        end = min(total_dur, start + seg)
        t0 = end
        txt = txt.replace("\n", r"\N")
        events.append(f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},Default,,0,0,0,,{txt}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))


# =========================
# Video render (Ken Burns on AI image)
# =========================
def make_kenburns_video_with_subs(image_path: str, audio_path: str, ass_path: str, out_mp4: str, dur: float):
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    frames = int(max(dur * FPS, 1))
    zoom_expr = "min(zoom+0.0012,1.18)"  # gentle zoom

    vf = (
        f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{VIDEO_H},"
        f"zoompan=z='{zoom_expr}':d={frames}:s={VIDEO_W}x{VIDEO_H}:fps={FPS},"
        f"subtitles={ass_path}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-i", audio_path,
        "-t", f"{dur:.2f}",
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-r", str(FPS),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        out_mp4,
    ]
    run(cmd)


# =========================
# Prompt builder (AI meme image)
# =========================
def build_meme_image_prompt(variant: Dict[str, Any], region: str) -> str:
    vtitle = str(variant.get("variant_title", "")).strip()
    voiceover = str(variant.get("voiceover", "")).strip()
    ons = variant.get("on_screen_text", []) or []
    ons = [str(x).strip() for x in ons if str(x).strip()][:3]

    prompt = f"""
Create an ORIGINAL meme-style vertical scene (9:16) that fits a short funny voiceover.

Context:
- Variant theme: {vtitle or "meme reaction"}
- Voiceover (context only): {voiceover[:240]}
- Caption beats: {", ".join(ons) if ons else "none"}
- Region vibe: {region}

Visual requirements:
- One clear subject, exaggerated facial expression, comedic reaction
- Simple background, high contrast, sharp focus
- No logos, no brand names, no copyrighted characters
- Leave empty space near the bottom for subtitles
- Style: photorealistic or high-quality 3D render, meme-friendly
""".strip()
    return prompt


# =========================
# Main
# =========================
def main():
    ensure_ffmpeg()

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

    required = [
        "ai_status", "ai_variants_json", "video_id", "title", "region",
        "gen_status", "gen_video_file", "gen_note",
        "yt_title", "yt_description", "yt_hashtags", "yt_tags",
    ]
    missing = [c for c in required if c not in idx]
    if missing:
        raise RuntimeError(f"Missing columns in header row: {missing}. Please add them to the sheet.")

    all_rows = ws.get_all_values()
    if len(all_rows) <= 1:
        print("[INFO] No data rows.", flush=True)
        return

    candidates = []
    for rnum in range(2, len(all_rows) + 1):
        row = all_rows[rnum - 1]
        ai_status = (row[idx["ai_status"] - 1] if len(row) >= idx["ai_status"] else "").strip()
        gen_status = (row[idx["gen_status"] - 1] if len(row) >= idx["gen_status"] else "").strip()

        ok_gen_status = (gen_status == "" or gen_status == "pending" or (RETRY_FAILED and gen_status == "failed"))
        if ai_status == "done" and ok_gen_status:
            candidates.append(rnum)

    if not candidates:
        print("[INFO] No rows to generate.", flush=True)
        return

    to_process = candidates[:MAX_PER_RUN_GEN]
    print(f"[INFO] Generate candidates={len(candidates)} will_process={len(to_process)} (MAX_PER_RUN_GEN={MAX_PER_RUN_GEN})", flush=True)

    # We'll update these columns as one rectangular batch update to save Sheets quota
    cols_to_update = [
        idx["gen_status"], idx["gen_video_file"], idx["gen_note"],
        idx["yt_title"], idx["yt_description"], idx["yt_hashtags"], idx["yt_tags"],
    ]
    start_col = min(cols_to_update)
    end_col = max(cols_to_update)
    width = end_col - start_col + 1

    batch_payload = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for rnum in to_process:
        row = all_rows[rnum - 1]
        video_id = (row[idx["video_id"] - 1] if len(row) >= idx["video_id"] else "").strip()
        region = (row[idx["region"] - 1] if len(row) >= idx["region"] else "").strip()

        variants_raw = row[idx["ai_variants_json"] - 1] if len(row) >= idx["ai_variants_json"] else ""
        variants = safe_load_json(variants_raw) or []

        gen_status = "failed"
        gen_file = ""
        gen_note = ""
        yt_title = ""
        yt_description = ""
        yt_hashtags = ""
        yt_tags = ""

        if not variants:
            gen_note = "No variants in ai_variants_json"
            print(f"[WARN] row={rnum} no variants.", flush=True)
        else:
            v = variants[min(max(VARIANT_INDEX, 0), len(variants) - 1)]
            voiceover = str(v.get("voiceover", "")).strip()
            on_screen = v.get("on_screen_text", []) or []

            if not voiceover:
                gen_note = "Variant missing voiceover"
                print(f"[WARN] row={rnum} missing voiceover.", flush=True)
            else:
                try:
                    base = f"{video_id}_ai_meme_v{VARIANT_INDEX+1}".replace("/", "_")
                    mp3_path = os.path.join(OUTPUT_DIR, f"{base}.mp3")
                    ass_path = os.path.join(OUTPUT_DIR, f"{base}.ass")
                    img_path = os.path.join(OUTPUT_DIR, f"{base}.{IMG_OUTPUT_FORMAT}")
                    out_mp4 = os.path.join(OUTPUT_DIR, f"{base}.mp4")

                    # 1) TTS
                    make_tts_mp3(client, voiceover, mp3_path)

                    # 2) Duration clamp
                    dur = clamp_duration(audio_duration_sec(mp3_path))

                    # 3) Subs
                    build_ass_subtitles_bottom(on_screen, dur, ass_path)

                    # 4) AI Image
                    img_prompt = build_meme_image_prompt(v, region)
                    img_meta = generate_ai_image(client, img_prompt, img_path)

                    # 5) Render
                    make_kenburns_video_with_subs(img_path, mp3_path, ass_path, out_mp4, dur)

                    # 6) YouTube metadata
                    meta = generate_youtube_metadata(client, v, region)
                    yt_title = (meta.get("title") or "").strip()
                    yt_description = (meta.get("description") or "").strip()
                    hashtags_list = meta.get("hashtags") or []
                    tags_list = meta.get("tags") or []
                    yt_hashtags = " ".join([str(x).strip() for x in hashtags_list if str(x).strip()])
                    yt_tags = ", ".join([str(x).strip() for x in tags_list if str(x).strip()])

                    gen_status = "done"
                    gen_file = out_mp4
                    gen_note = f"ok; dur={dur:.2f}s; img={img_meta.get('model')}/{img_meta.get('size')}/{img_meta.get('quality')}; fmt={img_meta.get('format')}; region={region}"
                    if img_meta.get("revised_prompt"):
                        gen_note += "; revised_prompt=yes"

                    print(f"[OK] row={rnum} video_id={video_id} gen=done yt_title_len={len(yt_title)}", flush=True)

                except Exception as e:
                    gen_status = "failed"
                    gen_file = ""
                    gen_note = f"error: {e}"
                    print(f"[WARN] row={rnum} failed: {e}", flush=True)

        rowvals = [""] * width

        def setcol(col_1based: int, val: str):
            rowvals[col_1based - start_col] = val

        setcol(idx["gen_status"], gen_status)
        setcol(idx["gen_video_file"], gen_file)
        setcol(idx["gen_note"], gen_note)

        setcol(idx["yt_title"], yt_title)
        setcol(idx["yt_description"], yt_description)
        setcol(idx["yt_hashtags"], yt_hashtags)
        setcol(idx["yt_tags"], yt_tags)

        rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"
        batch_payload.append({"range": rng, "values": [rowvals]})

        time.sleep(0.6)

    ws.batch_update(batch_payload, value_input_option="RAW")
    print(f"[DONE] Updated {len(to_process)} rows gen+yt metadata.", flush=True)


if __name__ == "__main__":
    main()
