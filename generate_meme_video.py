import os
import json
import time
import re
import subprocess
from typing import List, Dict, Any, Tuple, Optional

import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI


# =========================
# ENV / CONFIG
# =========================
SHEET_NAME = os.getenv("SHEET_NAME", "daily_rank")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

MAX_PER_RUN_GEN = int(os.getenv("MAX_PER_RUN_GEN", "2"))
VARIANT_INDEX = int(os.getenv("GEN_VARIANT_INDEX", "0"))
RETRY_FAILED = os.getenv("RETRY_FAILED", "0") == "1"

# TTS
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

# Sora / Videos API
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "sora-2")
VIDEO_SIZE_PRIMARY = os.getenv("VIDEO_SIZE_PRIMARY", "720x1280")   # vertical first
VIDEO_SIZE_FALLBACK = os.getenv("VIDEO_SIZE_FALLBACK", "1280x720") # fallback then crop
POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "10"))
VIDEO_CREATE_TIMEOUT_SEC = float(os.getenv("VIDEO_CREATE_TIMEOUT_SEC", "600"))

# Output
OUT_W = int(os.getenv("OUT_W", "1080"))
OUT_H = int(os.getenv("OUT_H", "1920"))
FPS = int(os.getenv("FPS", "30"))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "20.0"))

# Subtitles safe area
SUB_MARGIN_V = int(os.getenv("SUB_MARGIN_V", "280"))
SUB_FONT_SIZE = int(os.getenv("SUB_FONT_SIZE", "68"))
SUB_MAX_CHARS_PER_LINE = int(os.getenv("SUB_MAX_CHARS_PER_LINE", "18"))

# YouTube metadata
YT_MODEL = os.getenv("YT_MODEL", "gpt-4.1-mini")
YT_LANG = os.getenv("YT_LANG", "en")
YT_MAX_TITLE_CHARS = int(os.getenv("YT_MAX_TITLE_CHARS", "70"))

# NEW: CEIL bucket padding to avoid “ending feels cut”
VIDEO_PAD_SEC = float(os.getenv("VIDEO_PAD_SEC", "1.2"))  # recommend 0.8–1.8


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


def clamp_duration(d: float) -> float:
    if d <= 0:
        return 8.0
    return min(d, MAX_DURATION_SEC)


def choose_sora_seconds_ceiling(voiceover_dur: float, pad: float = 1.2) -> int:
    """
    Sora seconds bucket (4/8/12) by CEILING:
    - Use (voiceover_dur + pad) to pick bucket
    - Never pick a bucket shorter than voiceover
    """
    target = max(0.0, float(voiceover_dur)) + max(0.0, float(pad))
    if target <= 4.0:
        return 4
    if target <= 8.0:
        return 8
    return 12


# =========================
# Sheets helpers
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
# Text sanitize / wrap
# =========================
def sanitize_caption(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"\{[^}]+\}", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wrap_caption(text: str, max_chars: int = 18) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    cut = max_chars
    for i in range(max_chars, max(8, max_chars - 7), -1):
        if i < len(text) and text[i] == " ":
            cut = i
            break
    left = text[:cut].strip()
    right = text[cut:].strip()
    if not right:
        return left
    return left + r"\N" + right


# =========================
# OpenAI helpers
# =========================
def make_tts_mp3(client: OpenAI, text: str, out_path: str):
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
        raise RuntimeError("TTS response has no audio bytes.")
    with open(out_path, "wb") as f:
        f.write(data)


def sora_create_and_download_mp4(
    client: OpenAI,
    prompt: str,
    seconds: int,
    size_primary: str,
    size_fallback: str,
    out_raw_mp4: str,
) -> Tuple[str, str]:
    last_err = None
    for size in [size_primary, size_fallback]:
        try:
            start_ts = time.time()
            video = client.videos.create(
                model=VIDEO_MODEL,
                prompt=prompt,
                size=size,
                seconds=str(seconds),
            )
            vid = video.id
            status = getattr(video, "status", None)
            progress = getattr(video, "progress", None)

            while status in ("queued", "in_progress"):
                if time.time() - start_ts > VIDEO_CREATE_TIMEOUT_SEC:
                    raise RuntimeError(f"Sora timeout > {VIDEO_CREATE_TIMEOUT_SEC}s (video_id={vid}, status={status}, progress={progress})")
                time.sleep(POLL_INTERVAL_SEC)
                video = client.videos.retrieve(vid)
                status = getattr(video, "status", None)
                progress = getattr(video, "progress", None)

            if status != "completed":
                msg = getattr(getattr(video, "error", None), "message", None)
                raise RuntimeError(f"Sora failed status={status} msg={msg}")

            content = client.videos.download_content(vid, variant="video")
            content.write_to_file(out_raw_mp4)
            return size, vid

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Sora create/download failed for both sizes. last_err={last_err}")


def build_ass_subtitles_bottom(lines: List[str], total_dur: float, ass_path: str):
    cleaned = []
    for x in (lines or []):
        s = sanitize_caption(x)
        if s:
            cleaned.append(s)

    if not cleaned:
        cleaned = ["wait for it...", "😂", ""]

    beats = cleaned[:3]
    beats = [wrap_caption(b, SUB_MAX_CHARS_PER_LINE) for b in beats]

    n = max(len(beats), 1)
    seg = max(total_dur / n, 1.6)

    def fmt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    header = f"""[Script Info]
Title: MemeSub
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,{SUB_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,8,2,2,80,80,{SUB_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    t0 = 0.0
    for txt in beats:
        start = t0
        end = min(total_dur, start + seg)
        t0 = end
        txt = (txt or "").replace("\n", r"\N")
        events.append(f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},Default,,0,0,0,,{txt}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))


def compose_final_video(raw_video: str, tts_mp3: str, ass_path: str, out_mp4: str, dur: float):
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    vf = (
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},"
        f"fps={FPS},"
        f"subtitles={ass_path}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", raw_video,
        "-i", tts_mp3,
        "-t", f"{dur:.2f}",
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        out_mp4,
    ]
    run(cmd)


def build_meme_video_prompt(variant: Dict[str, Any], region: str) -> str:
    """
    Keep your original behavior, but with a stronger ending guideline so it doesn't feel cut.
    """
    vp = (variant.get("video_prompt") or "").strip()
    if vp:
        return vp + (
            "\n\nExtra constraints:"
            "\n- Vertical 9:16"
            "\n- Fast meme pacing"
            "\n- Quick push-in near punchline"
            "\n- Slight handheld camera shake"
            "\n- 2–3 distinct actions in the scene"
            "\n- High contrast lighting, crisp subject"
            "\n- NO on-screen text (subtitles will be added later)"
            "\n- No logos, no brands, no copyrighted characters, no real people"
            "\n- Leave clean space near bottom for subtitles"
            "\n- IMPORTANT: last 2 seconds must be a clear ending beat (reaction hold / freeze moment), no abrupt cut"
        )

    voiceover = sanitize_caption(variant.get("voiceover", ""))[:240]
    return f"""
Vertical 9:16 meme-style short video.
Style anchor: viral shorts meme, comedic reaction, dynamic motion, cinematic lighting.
Scene: original comedic situation with exaggerated reaction.
Pacing: fast, punchy.
Camera: quick push-in near the punchline, slight handheld shake.
Action: include 2–3 distinct actions.
No logos, no brands, no copyrighted characters, no real people.
NO on-screen text (subtitles added later). Leave bottom space clean.
IMPORTANT: last 2 seconds must be a clear ending beat (reaction hold / freeze moment), no abrupt cut.
Context voiceover (for vibe only): {voiceover}
Region vibe: {region}
""".strip()


def generate_youtube_metadata(client: OpenAI, variant: Dict[str, Any], region: str) -> Dict[str, Any]:
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

    voiceover = sanitize_caption(variant.get("voiceover", ""))[:600]
    ons = variant.get("on_screen_text") or []
    ons = [sanitize_caption(x) for x in ons if sanitize_caption(x)][:3]
    vtitle = sanitize_caption(variant.get("variant_title", ""))

    instruction = f"""
You are a YouTube Shorts copywriter.

Language: {YT_LANG}
Rules:
- Title: <= {YT_MAX_TITLE_CHARS} chars, hooky, 1 emoji max, MUST end with "#shorts"
- No copyrighted characters, no brands, no real person names.
- Description: 2 short lines + CTA line + hashtags line at end
- Hashtags: 3-6, include "#shorts"
- Tags: 6-18 keywords, no hashtags in tags
Return STRICT JSON only.
""".strip()

    payload = {
        "region": region,
        "variant_title": vtitle,
        "voiceover": voiceover,
        "caption_beats": ons,
    }

    resp = client.responses.create(
        model=YT_MODEL,
        input=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        max_output_tokens=450,
        response_format={
            "type": "json_schema",
            "name": "yt_shorts_metadata",
            "schema": schema,
            "strict": True,
        },
    )

    return json.loads(resp.output_text)


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
        raise RuntimeError(f"Missing columns in header row: {missing}")

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

    cols_to_update = [
        idx["gen_status"], idx["gen_video_file"], idx["gen_note"],
        idx["yt_title"], idx["yt_description"], idx["yt_hashtags"], idx["yt_tags"],
    ]
    start_col = min(cols_to_update)
    end_col = max(cols_to_update)
    width = end_col - start_col + 1

    payload = []
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
        yt_desc = ""
        yt_hashtags = ""
        yt_tags = ""

        try:
            if not variants:
                raise RuntimeError("No variants in ai_variants_json")

            v = variants[min(max(VARIANT_INDEX, 0), len(variants) - 1)]
            voiceover = sanitize_caption(v.get("voiceover", "")).strip()
            on_screen = v.get("on_screen_text") or []
            on_screen = [sanitize_caption(x) for x in on_screen if sanitize_caption(x)][:4]

            if not voiceover:
                raise RuntimeError("Variant missing voiceover")

            base = f"{video_id}_sora_v{VARIANT_INDEX+1}".replace("/", "_")
            tts_path = os.path.join(OUTPUT_DIR, f"{base}.mp3")
            ass_path = os.path.join(OUTPUT_DIR, f"{base}.ass")
            raw_path = os.path.join(OUTPUT_DIR, f"{base}_raw.mp4")
            out_mp4 = os.path.join(OUTPUT_DIR, f"{base}.mp4")

            # 1) TTS
            make_tts_mp3(client, voiceover, tts_path)

            # 2) Duration = full TTS duration (cap 20)
            dur = clamp_duration(audio_duration_sec(tts_path))

            # 3) Choose Sora seconds by CEILING (IMPORTANT FIX)
            seconds = choose_sora_seconds_ceiling(dur, pad=VIDEO_PAD_SEC)

            # IMPORTANT:
            # Do NOT cut dur down to seconds.
            # We will:
            # - generate a video that is >= voiceover duration
            # - then final export trims to dur (audio length)
            # So we removed the old line: dur = min(dur, float(seconds))

            # 4) Subtitles
            # Keep your on_screen beats (3 lines) but aligned in total duration
            build_ass_subtitles_bottom(on_screen, dur, ass_path)

            # 5) Sora text-to-video
            prompt = build_meme_video_prompt(v, region)
            used_size, vid = sora_create_and_download_mp4(
                client=client,
                prompt=prompt,
                seconds=seconds,
                size_primary=VIDEO_SIZE_PRIMARY,
                size_fallback=VIDEO_SIZE_FALLBACK,
                out_raw_mp4=raw_path,
            )

            # 6) Compose final 9:16 + subs + replace audio
            # Final output length follows FULL voiceover duration (dur)
            compose_final_video(raw_path, tts_path, ass_path, out_mp4, dur)

            # 7) YouTube metadata
            meta = generate_youtube_metadata(client, v, region)
            yt_title = sanitize_caption(meta.get("title", "")).strip()
            yt_desc = (meta.get("description", "") or "").strip()
            hashtags_list = meta.get("hashtags") or []
            tags_list = meta.get("tags") or []
            yt_hashtags = " ".join([sanitize_caption(x) for x in hashtags_list if sanitize_caption(x)])
            yt_tags = ", ".join([sanitize_caption(x) for x in tags_list if sanitize_caption(x)])

            gen_status = "done"
            gen_file = out_mp4
            gen_note = f"ok; dur={dur:.2f}s; sora={VIDEO_MODEL}; size={used_size}; video_id={vid}; variant={VARIANT_INDEX+1}; seconds={seconds}(ceil); pad={VIDEO_PAD_SEC:.1f}s"

            print(f"[OK] row={rnum} video_id={video_id} status=done variant={VARIANT_INDEX+1} dur={dur:.2f}s seconds={seconds}", flush=True)

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
        setcol(idx["yt_description"], yt_desc)
        setcol(idx["yt_hashtags"], yt_hashtags)
        setcol(idx["yt_tags"], yt_tags)

        rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"
        payload.append({"range": rng, "values": [rowvals]})

        time.sleep(0.6)

    ws.batch_update(payload, value_input_option="RAW")
    print(f"[DONE] Updated {len(to_process)} rows gen+yt metadata.", flush=True)


if __name__ == "__main__":
    main()
