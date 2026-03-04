import os
import json
import time
import random
import subprocess
from typing import List, Dict, Any, Tuple

import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI


SHEET_NAME = os.getenv("SHEET_NAME", "daily_rank")
BROLL_DIR = os.getenv("BROLL_DIR", "assets/broll")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
MAX_PER_RUN_GEN = int(os.getenv("MAX_PER_RUN_GEN", "2"))

TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

VIDEO_W = int(os.getenv("VIDEO_W", "1080"))
VIDEO_H = int(os.getenv("VIDEO_H", "1920"))

# Keep it short for Shorts
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "20.0"))


def run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout.strip()


def ensure_ffmpeg():
    run(["ffmpeg", "-version"])
    run(["ffprobe", "-version"])


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


def safe_load_json(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def pick_broll_file() -> str:
    if not os.path.isdir(BROLL_DIR):
        return ""
    files = []
    for fn in os.listdir(BROLL_DIR):
        if fn.lower().endswith((".mp4", ".mov", ".m4v")):
            files.append(os.path.join(BROLL_DIR, fn))
    if not files:
        return ""
    return random.choice(files)


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


def make_tts_mp3(client: OpenAI, text: str, out_path: str):
    # OpenAI Python SDK supports write_to_file on the response
    resp = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        format="mp3",
    )
    resp.write_to_file(out_path)


def build_ass_subtitles(lines: List[str], total_dur: float, ass_path: str):
    """
    Meme style: white bold-ish text with black outline, centered.
    Split lines into 3 beats (or less) across the duration.
    """
    # Clean lines
    cleaned = [str(x).strip() for x in (lines or []) if str(x).strip()]
    if not cleaned:
        cleaned = ["WAIT FOR IT...", "😂", ""]

    # Use up to 3 beats
    beats = cleaned[:3]
    n = len(beats)
    seg = max(total_dur / max(n, 1), 1.5)

    def fmt_time(t: float) -> str:
        # ASS time: H:MM:SS.cc
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
Style: Default,Arial Black,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,8,2,2,60,60,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    t0 = 0.0
    for i, txt in enumerate(beats):
        start = t0
        end = min(total_dur, start + seg)
        t0 = end
        # center text; allow \N line breaks if user provides
        txt = txt.replace("\n", r"\N")
        events.append(f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},Default,,0,0,0,,{txt}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))


def make_vertical_video_with_subs(broll_path: str, audio_path: str, ass_path: str, out_mp4: str, dur: float):
    """
    - If broll_path exists: scale/crop to 9:16 and loop/cut to duration
    - Else: generate a solid background
    - Burn ASS subtitles
    - Mix audio
    """
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    if broll_path and os.path.exists(broll_path):
        # Scale to cover and crop center
        vf = (
            f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_W}:{VIDEO_H},"
            f"subtitles={ass_path}"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", broll_path,
            "-i", audio_path,
            "-t", f"{dur:.2f}",
            "-vf", vf,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-r", "30",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            out_mp4,
        ]
    else:
        # Solid background
        vf = f"subtitles={ass_path}"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={VIDEO_W}x{VIDEO_H}:d={dur:.2f}",
            "-i", audio_path,
            "-t", f"{dur:.2f}",
            "-vf", vf,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-r", "30",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            out_mp4,
        ]

    run(cmd)


def main():
    ensure_ffmpeg()

    sheet_id = os.environ["GSHEET_ID"]
    sa_json = os.environ["GSHEET_SA_JSON"]
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI()

    gc = gsheet_client_from_sa_json(sa_json)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(SHEET_NAME)

    header = ws.row_values(1)
    idx = find_header_indices(header)

    required = ["ai_status", "ai_variants_json", "video_id", "title", "region", "gen_status", "gen_video_file", "gen_note"]
    for c in required:
        if c not in idx:
            raise RuntimeError(f"Missing column '{c}' in header row. Please add it to the sheet.")

    all_rows = ws.get_all_values()
    if len(all_rows) <= 1:
        print("[INFO] No data rows.")
        return

    candidates = []
    for rnum in range(2, len(all_rows) + 1):
        row = all_rows[rnum - 1]
        ai_status = (row[idx["ai_status"] - 1] if len(row) >= idx["ai_status"] else "").strip()
        gen_status = (row[idx["gen_status"] - 1] if len(row) >= idx["gen_status"] else "").strip()
        if ai_status == "done" and (gen_status == "" or gen_status == "pending"):
            candidates.append(rnum)

    if not candidates:
        print("[INFO] No rows to generate.")
        return

    to_process = candidates[:MAX_PER_RUN_GEN]
    print(f"[INFO] Generate candidates={len(candidates)} will_process={len(to_process)} (MAX_PER_RUN_GEN={MAX_PER_RUN_GEN})")

    # Batch update ranges to save Sheets write quota
    c_gen_status = idx["gen_status"]
    c_gen_file = idx["gen_video_file"]
    c_gen_note = idx["gen_note"]

    start_col = min(c_gen_status, c_gen_file, c_gen_note)
    end_col = max(c_gen_status, c_gen_file, c_gen_note)
    width = end_col - start_col + 1

    batch_payload = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for rnum in to_process:
        row = all_rows[rnum - 1]
        video_id = (row[idx["video_id"] - 1] if len(row) >= idx["video_id"] else "").strip()
        title = (row[idx["title"] - 1] if len(row) >= idx["title"] else "").strip()
        region = (row[idx["region"] - 1] if len(row) >= idx["region"] else "").strip()
        variants_raw = row[idx["ai_variants_json"] - 1] if len(row) >= idx["ai_variants_json"] else ""
        variants = safe_load_json(variants_raw) or []

        if not variants:
            status = "failed"
            note = "No variants in ai_variants_json"
            out_file = ""
            print(f"[WARN] row={rnum} no variants.")
        else:
            v = variants[0]  # pick variant 1
            voiceover = str(v.get("voiceover", "")).strip()
            on_screen = v.get("on_screen_text", []) or []
            if not voiceover:
                status = "failed"
                note = "Variant missing voiceover"
                out_file = ""
                print(f"[WARN] row={rnum} missing voiceover.")
            else:
                try:
                    base = f"{video_id}_meme_v1".replace("/", "_")
                    mp3_path = os.path.join(OUTPUT_DIR, f"{base}.mp3")
                    ass_path = os.path.join(OUTPUT_DIR, f"{base}.ass")
                    out_mp4 = os.path.join(OUTPUT_DIR, f"{base}.mp4")

                    # 1) TTS
                    make_tts_mp3(client, voiceover, mp3_path)

                    # 2) Duration clamp
                    dur = clamp_duration(audio_duration_sec(mp3_path))

                    # 3) Subs
                    build_ass_subtitles(on_screen, dur, ass_path)

                    # 4) B-roll
                    broll = pick_broll_file()

                    # 5) Render
                    make_vertical_video_with_subs(broll, mp3_path, ass_path, out_mp4, dur)

                    status = "done"
                    out_file = out_mp4
                    note = f"ok; dur={dur:.2f}s; broll={'yes' if broll else 'no'}; region={region}"
                    print(f"[OK] row={rnum} generated {out_mp4}")

                except Exception as e:
                    status = "failed"
                    out_file = ""
                    note = f"error: {e}"
                    print(f"[WARN] row={rnum} failed: {e}")

        # Prepare batch update
        rowvals = [""] * width

        def setcol(col, val):
            rowvals[col - start_col] = val

        setcol(c_gen_status, status)
        setcol(c_gen_file, out_file)
        setcol(c_gen_note, note)

        rng = f"{a1_col(start_col)}{rnum}:{a1_col(end_col)}{rnum}"
        batch_payload.append({"range": rng, "values": [rowvals]})

        time.sleep(0.5)

    ws.batch_update(batch_payload, value_input_option="RAW")
    print(f"[DONE] Updated {len(to_process)} rows gen_status.")


if __name__ == "__main__":
    main()
