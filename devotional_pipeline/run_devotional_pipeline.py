#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_THEME = "A short devotional about hope after a hard week"
_WINDOWS_VOICES_CACHE: Optional[List[str]] = None


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def deep_clone(value: Any) -> Any:
    return copy.deepcopy(value)


def queue_prompt(server: str, workflow: Dict[str, Any]) -> str:
    resp = requests.post(f"{server}/prompt", json={"prompt": workflow}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Missing prompt_id in response: {payload}")
    return str(prompt_id)


def wait_for_history(server: str, prompt_id: str, timeout_s: int = 3600) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error: Optional[str] = None
    while time.time() < deadline:
        resp = requests.get(f"{server}/history/{prompt_id}", timeout=60)
        resp.raise_for_status()
        history_all = resp.json()
        history = history_all.get(prompt_id)
        if not history:
            time.sleep(1.0)
            continue
        status = history.get("status", {}).get("status_str")
        if status == "success":
            return history
        if status == "error":
            last_error = json.dumps(history.get("status", {}), ensure_ascii=True)
            break
        time.sleep(1.0)
    if last_error:
        raise RuntimeError(f"Prompt {prompt_id} failed: {last_error}")
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def extract_json_loose(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty planner output")

    if raw.startswith("```"):
        raw = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", raw, count=1)
        raw = re.sub(r"\s*```$", "", raw, count=1)

    try:
        return json.loads(raw)
    except Exception:
        pass

    candidates: List[str] = []
    obj_start = raw.find("{")
    obj_end = raw.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(raw[obj_start:obj_end + 1])

    arr_start = raw.find("[")
    arr_end = raw.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        candidates.append(raw[arr_start:arr_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("Could not parse planner output as JSON")


def sanitize(s: str, fallback: str) -> str:
    if not isinstance(s, str):
        return fallback
    out = re.sub(r"[^A-Za-z0-9_-]", "_", s).strip("_")
    return out or fallback


def find_output_files(history: Dict[str, Any], comfy_output_dir: Path) -> List[Path]:
    files: List[Path] = []
    outputs = history.get("outputs", {})
    if not isinstance(outputs, dict):
        return files

    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in ("images", "audio", "videos", "gifs"):
            items = node_output.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                subfolder = item.get("subfolder", "")
                if not filename:
                    continue
                if subfolder:
                    files.append(comfy_output_dir / subfolder / filename)
                else:
                    files.append(comfy_output_dir / filename)
    return files


def ffprobe_stream(path: Path, stream_selector: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        stream_selector,
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        return {}
    return streams[0]


def mix_audio_layers(layer_entries: List[Dict[str, Any]], out_file: Path) -> Path:
    if not layer_entries:
        raise ValueError("No layer files to mix")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = ["ffmpeg", "-y"]
    filter_parts: List[str] = []
    mix_inputs: List[str] = []
    for idx, layer in enumerate(layer_entries):
        layer_file = Path(layer["file"])
        layer_type = str(layer.get("layer_type") or "")
        cmd.extend(["-i", str(layer_file)])

        if layer_type == "speaker":
            weight = 1.00
        elif layer_type == "ambient":
            weight = 0.25
        elif layer_type == "music":
            weight = 0.20
        elif layer_type == "sfx":
            weight = 0.35
        else:
            weight = 0.35
        filter_parts.append(f"[{idx}:a]volume={weight:.3f}[a{idx}]")
        mix_inputs.append(f"[a{idx}]")

    amix_chain = "".join(mix_inputs) + f"amix=inputs={len(layer_entries)}:duration=longest:normalize=0,alimiter=limit=0.95[aout]"
    filter_complex = ";".join(filter_parts + [amix_chain])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[aout]",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(out_file),
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_file


def mux_video_audio(video_file: Path, audio_file: Path, out_file: Path) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-i",
        str(audio_file),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-shortest",
        str(out_file),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_file


def sec_to_wan_length(seconds: float, fps: int) -> int:
    raw_frames = max(1.0, float(seconds) * float(fps))
    blocks = max(1, int(round((raw_frames - 1.0) / 4.0)))
    return blocks * 4 + 1


def sec_to_ltx_length(seconds: float, fps: int) -> int:
    raw_frames = max(1.0, float(seconds) * float(fps))
    blocks = max(1, int(round((raw_frames - 1.0) / 8.0)))
    return blocks * 8 + 1


def detect_video_backend(video_wf: Dict[str, Any]) -> str:
    if "104" in video_wf and "100" in video_wf:
        return "wan"
    if "92:97" in video_wf and "92:62" in video_wf:
        return "ltx2"
    return "unknown"


def planner_text_from_history(history: Dict[str, Any], preview_node_id: str = "5") -> str:
    outputs = history.get("outputs", {})
    node = outputs.get(preview_node_id, {})
    texts = node.get("text", [])
    if isinstance(texts, list) and texts:
        return str(texts[0])
    raise ValueError(f"No planner text found on node {preview_node_id}")


def latest_file_after(directory: Path, suffix: str, after_ts: float) -> Optional[Path]:
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob(f"*{suffix}") if p.is_file() and p.stat().st_mtime >= after_ts]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def get_windows_tts_voices() -> List[str]:
    global _WINDOWS_VOICES_CACHE
    if _WINDOWS_VOICES_CACHE is not None:
        return _WINDOWS_VOICES_CACHE

    cmd = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }; $s.Dispose()",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    voices = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    _WINDOWS_VOICES_CACHE = voices
    return voices


def pick_windows_tts_voice(voice_hint: str, speaker_idx: int) -> str:
    voices = get_windows_tts_voices()
    if not voices:
        raise RuntimeError("No installed Windows TTS voices found (System.Speech).")

    hint = (voice_hint or "").strip().lower()

    def _find(name_part: str) -> Optional[str]:
        for v in voices:
            if name_part.lower() in v.lower():
                return v
        return None

    if "hedda" in hint or "german" in hint or hint.startswith("de"):
        v = _find("Hedda")
        if v:
            return v
    if "zira" in hint or "female" in hint or hint.startswith("en"):
        v = _find("Zira")
        if v:
            return v
    if "male" in hint:
        for token in ("David", "Mark", "Guy"):
            v = _find(token)
            if v:
                return v

    # Alternate when multiple speakers are present.
    return voices[speaker_idx % len(voices)]


def synthesize_speaker_tts_windows(text: str, voice_hint: str, out_mp3: Path, duration_s: float, speaker_idx: int) -> Path:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_mp3.with_suffix(".wav")

    voice_name = pick_windows_tts_voice(voice_hint, speaker_idx)
    rate = 0
    if "male" in (voice_hint or "").lower():
        rate = -1

    env = os.environ.copy()
    env["TTS_TEXT"] = text
    env["TTS_VOICE"] = voice_name
    env["TTS_RATE"] = str(rate)
    env["TTS_OUT_WAV"] = str(wav_path)

    ps_script = (
        "$ErrorActionPreference='Stop'; "
        "Add-Type -AssemblyName System.Speech; "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$voices=$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }; "
        "if ($voices -contains $env:TTS_VOICE) { $s.SelectVoice($env:TTS_VOICE) }; "
        "$s.Rate=[int]$env:TTS_RATE; "
        "$s.SetOutputToWaveFile($env:TTS_OUT_WAV); "
        "$s.Speak($env:TTS_TEXT); "
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    duration_s = max(0.5, float(duration_s))
    ff_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-af",
        f"apad=pad_dur={duration_s:.3f},atrim=duration={duration_s:.3f}",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "320k",
        str(out_mp3),
    ]
    subprocess.run(ff_cmd, check=True, capture_output=True, text=True)
    if wav_path.exists():
        wav_path.unlink()
    return out_mp3


def build_audio_layer_items(clip: Dict[str, Any], duration_s: float, include_background: bool = True) -> List[Dict[str, Any]]:
    audio = clip.get("audio", {}) if isinstance(clip, dict) else {}
    if not isinstance(audio, dict):
        audio = {}

    items: List[Dict[str, Any]] = []
    bpm = 84
    key = "C major"
    ts = "4"
    lang = "en"
    duration_s = max(0.5, float(duration_s))

    speakers = audio.get("speakers", [])
    speaker_count = 0
    if isinstance(speakers, list):
        for idx, speaker in enumerate(speakers, start=1):
            if not isinstance(speaker, dict):
                continue
            sid = sanitize(str(speaker.get("id") or f"speaker_{idx}"), f"speaker_{idx}")
            text = str(speaker.get("text") or "").strip()
            if not text:
                continue
            voice = str(speaker.get("voice") or "warm_narrator")
            items.append(
                {
                    "id": sid,
                    "layer_type": "speaker",
                    "speech_text": text,
                    "voice_hint": voice,
                    "duration": duration_s,
                }
            )
            speaker_count += 1

    # Guarantee at least one audible spoken layer.
    if speaker_count == 0:
        items.append(
            {
                "id": "speaker_1",
                "layer_type": "speaker",
                "speech_text": "Hope rises with the dawn.",
                "voice_hint": "warm_female",
                "duration": duration_s,
            }
        )

    ambient = audio.get("ambient", [])
    if include_background and isinstance(ambient, list) and ambient:
        parts = []
        for a in ambient:
            if isinstance(a, dict):
                parts.append(str(a.get("type") or "ambient"))
            elif isinstance(a, str):
                parts.append(a)
        ambient_desc = ", ".join(p for p in parts if p).strip() or "nature ambience"
        items.append(
            {
                "id": "ambient",
                "layer_type": "ambient",
                "tags": f"natural ambience only, {ambient_desc}, no speech, no melody, clean background bed",
                "lyrics": "",
                "bpm": bpm,
                "duration": duration_s,
                "timesignature": ts,
                "language": lang,
                "keyscale": key,
            }
        )

    music = audio.get("music", {})
    if include_background and isinstance(music, dict):
        mood = str(music.get("mood") or "gentle worship pads")
        items.append(
            {
                "id": "music",
                "layer_type": "music",
                "tags": f"{mood}, instrumental devotional underscore, no vocals, clean mix, emotional but restrained",
                "lyrics": "",
                "bpm": bpm,
                "duration": duration_s,
                "timesignature": ts,
                "language": lang,
                "keyscale": key,
            }
        )

    sfx = audio.get("sfx", [])
    if include_background and isinstance(sfx, list) and sfx:
        parts = []
        for s in sfx:
            if isinstance(s, dict):
                parts.append(str(s.get("type") or "soft sfx"))
            elif isinstance(s, str):
                parts.append(s)
        sfx_desc = ", ".join(p for p in parts if p).strip() or "soft devotional foley"
        items.append(
            {
                "id": "sfx",
                "layer_type": "sfx",
                "tags": f"subtle sound design accents, {sfx_desc}, sparse events, no music, no speech",
                "lyrics": "",
                "bpm": bpm,
                "duration": duration_s,
                "timesignature": ts,
                "language": lang,
                "keyscale": key,
            }
        )

    if include_background and not items:
        items.append(
            {
                "id": "music",
                "layer_type": "music",
                "tags": "gentle worship pads, instrumental devotional underscore, no vocals",
                "lyrics": "",
                "bpm": bpm,
                "duration": duration_s,
                "timesignature": ts,
                "language": lang,
                "keyscale": key,
            }
        )
    return items


def pick_file(files: List[Path], suffix: str) -> Optional[Path]:
    matches = [f for f in files if f.suffix.lower() == suffix.lower()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return matches[0]


def fallback_plan(theme: str) -> Dict[str, Any]:
    line = "Hope rises with the dawn."
    return {
        "project": "devotional_series",
        "style_preset": "cinematic_devotional",
        "clips": [
            {
                "id": "clip_001",
                "route": "wan22_t2v",
                "video_prompt": f"Cinematic sunrise devotional scene inspired by: {theme}",
                "negative_prompt": "low quality, artifacts, watermark",
                "image_prompt": "serene sunrise valley, soft light",
                "duration_seconds": 2.0,
                "audio": {
                    "speakers": [{"id": "speaker_1", "text": line, "voice": "warm_female"}],
                    "ambient": [{"type": "wind"}, {"type": "birds"}],
                    "music": {"mood": "gentle worship pads", "volume_db": -18},
                    "sfx": [{"type": "soft bell"}],
                },
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local devotional planner -> video -> layered audio pipeline.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI server URL (default: http://127.0.0.1:8188)")
    parser.add_argument("--runs", type=int, default=1, help="How many episodes to generate")
    parser.add_argument("--theme", default=DEFAULT_THEME, help="Theme input for planner LLM")
    parser.add_argument("--max-duration", type=float, default=2.0, help="Max clip seconds for fast testing")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-job timeout in seconds")
    parser.add_argument("--comfy-output", default=r"C:\ComfyUI\output", help="ComfyUI output directory")
    parser.add_argument(
        "--planner-workflow",
        default="01_qwen_devotional_planner_api.json",
        help="Planner workflow file under workflows/",
    )
    parser.add_argument(
        "--video-workflow",
        default="02_wan22_video_from_devotional_iterator_api.json",
        help="Video workflow file under workflows/ (e.g. 02_wan..., 04_ltx2...)",
    )
    parser.add_argument(
        "--audio-workflow",
        default="03_ace15_audio_layer_iterator_api.json",
        help="Audio workflow file under workflows/",
    )
    parser.add_argument(
        "--background-mode",
        choices=["all", "none"],
        default="all",
        help="Use all layers (speaker+ambient+music+sfx) or speaker-only.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    workflows_dir = base_dir / "workflows"
    planner_wf = load_json(workflows_dir / args.planner_workflow)
    video_wf = load_json(workflows_dir / args.video_workflow)
    audio_wf = load_json(workflows_dir / args.audio_workflow)
    video_backend = detect_video_backend(video_wf)
    if video_backend == "unknown":
        raise ValueError("Unsupported video workflow format. Use WAN or LTX2 workflow templates.")

    comfy_output = Path(args.comfy_output).resolve()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = comfy_output / "devotional_pipeline" / "artifacts" / run_stamp
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    planner_dir = comfy_output / "devotional_pipeline" / "plans"
    planner_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "runs": [],
        "server": args.server,
        "theme": args.theme,
        "workflows_dir": str(workflows_dir),
        "planner_workflow": args.planner_workflow,
        "video_workflow": args.video_workflow,
        "audio_workflow": args.audio_workflow,
        "background_mode": args.background_mode,
    }

    for run_idx in range(max(1, int(args.runs))):
        print(f"[run {run_idx + 1}/{args.runs}] planner")
        planner_prompt = str(planner_wf["7"]["inputs"]["prompt"]).replace("__THEME__", args.theme)
        planner_job = deep_clone(planner_wf)
        planner_job["7"]["inputs"]["prompt"] = planner_prompt
        planner_job["7"]["inputs"]["sampling_mode.seed"] = random.randint(1, 2**31 - 1)

        planner_started = time.time()
        planner_prompt_id = queue_prompt(args.server, planner_job)
        planner_history = wait_for_history(args.server, planner_prompt_id, timeout_s=args.timeout)
        try:
            planner_text = planner_text_from_history(planner_history, preview_node_id="5")
        except Exception:
            planner_file = latest_file_after(planner_dir, ".json", planner_started - 1.0)
            if not planner_file:
                raise
            planner_text = planner_file.read_text(encoding="utf-8", errors="ignore")
        try:
            plan_data = extract_json_loose(planner_text)
        except Exception:
            print("[planner] parse failed, using safe fallback clip")
            plan_data = fallback_plan(args.theme)

        if isinstance(plan_data, list):
            clips = plan_data
            project = "devotional_series"
            style_preset = "cinematic_devotional"
        elif isinstance(plan_data, dict):
            clips = plan_data.get("clips", [])
            project = str(plan_data.get("project") or "devotional_series")
            style_preset = str(plan_data.get("style_preset") or "cinematic_devotional")
        else:
            raise ValueError("Planner did not return a dict/list JSON payload")

        if not isinstance(clips, list) or not clips:
            raise ValueError("Planner returned no clips")

        clip = clips[0]
        if not isinstance(clip, dict):
            raise ValueError("Clip payload is not an object")

        clip_id = sanitize(str(clip.get("id") or "clip_001"), "clip_001")
        duration_s = float(clip.get("duration_seconds") or 2.0)
        duration_s = max(0.5, min(duration_s, float(args.max_duration)))
        if video_backend == "wan":
            fps = int(video_wf["100"]["inputs"]["fps"])
            length = sec_to_wan_length(duration_s, fps)
        else:
            fps = int(float(video_wf.get("92:99", {}).get("inputs", {}).get("value", 24)))
            length = sec_to_ltx_length(duration_s, fps)

        clip_payload = {
            "project": project,
            "style_preset": style_preset,
            "clips": [clip],
        }

        print(f"[run {run_idx + 1}/{args.runs}] video clip={clip_id} duration={duration_s:.2f}s length={length}")
        video_job = deep_clone(video_wf)
        video_job["200"]["inputs"]["input_text"] = json.dumps(clip_payload, ensure_ascii=True)
        video_job["200"]["inputs"]["mode"] = "json"
        video_job["200"]["inputs"]["index"] = 0
        video_job["200"]["inputs"]["run_index"] = run_idx
        video_job["200"]["inputs"]["run_total"] = max(1, int(args.runs))
        if video_backend == "wan":
            video_job["104"]["inputs"]["length"] = length
            if "96" in video_job:
                video_job["96"]["inputs"]["noise_seed"] = random.randint(1, 2**31 - 1)
            if "95" in video_job:
                video_job["95"]["inputs"]["noise_seed"] = 0
        else:
            video_job["92:62"]["inputs"]["value"] = length
            if "92:11" in video_job:
                video_job["92:11"]["inputs"]["noise_seed"] = random.randint(1, 2**31 - 1)
            if "92:67" in video_job:
                video_job["92:67"]["inputs"]["noise_seed"] = random.randint(1, 2**31 - 1)

        video_prompt_id = queue_prompt(args.server, video_job)
        video_history = wait_for_history(args.server, video_prompt_id, timeout_s=args.timeout)
        video_files = find_output_files(video_history, comfy_output)
        video_file = pick_file(video_files, ".mp4")
        if not video_file or not video_file.exists():
            raise FileNotFoundError(f"Video file not found in outputs: {video_files}")

        print(f"[run {run_idx + 1}/{args.runs}] audio layers")
        layer_items = build_audio_layer_items(
            clip,
            duration_s=duration_s,
            include_background=(args.background_mode == "all"),
        )
        layer_entries: List[Dict[str, Any]] = []
        layer_reports: List[Dict[str, Any]] = []

        for layer_idx, layer in enumerate(layer_items):
            layer_id = sanitize(str(layer.get("id") or "layer"), "layer")
            layer_type = str(layer.get("layer_type") or "")
            audio_prompt_id: Optional[str] = None

            if layer_type == "speaker":
                out_path = comfy_output / "devotional_pipeline" / "audio" / run_stamp / f"run_{run_idx:03d}_{clip_id}_{layer_id}.mp3"
                text = str(layer.get("speech_text") or "").strip()
                voice_hint = str(layer.get("voice_hint") or "")
                layer_file = synthesize_speaker_tts_windows(
                    text=text,
                    voice_hint=voice_hint,
                    out_mp3=out_path,
                    duration_s=float(layer.get("duration") or duration_s),
                    speaker_idx=layer_idx,
                )
            else:
                audio_job = deep_clone(audio_wf)
                audio_job["110"]["inputs"]["input_text"] = json.dumps([layer], ensure_ascii=True)
                audio_job["110"]["inputs"]["mode"] = "json"
                audio_job["110"]["inputs"]["index"] = 0
                audio_job["107"]["inputs"]["filename_prefix"] = f"devotional_pipeline/audio/{run_stamp}/run_{run_idx:03d}_{clip_id}_{layer_id}"
                seed = random.randint(1, 2**31 - 1)
                audio_job["94"]["inputs"]["seed"] = seed
                audio_job["3"]["inputs"]["seed"] = seed

                audio_prompt_id = queue_prompt(args.server, audio_job)
                audio_history = wait_for_history(args.server, audio_prompt_id, timeout_s=args.timeout)
                out_files = find_output_files(audio_history, comfy_output)
                layer_file = pick_file(out_files, ".mp3")
                if not layer_file or not layer_file.exists():
                    raise FileNotFoundError(f"Layer audio not found for {layer_id}: {out_files}")

            layer_stream = ffprobe_stream(layer_file, "a:0")
            layer_reports.append(
                {
                    "layer_id": layer_id,
                    "layer_type": layer_type or "ace",
                    "file": str(layer_file),
                    "audio_prompt_id": audio_prompt_id,
                    "bit_rate": layer_stream.get("bit_rate"),
                    "sample_rate": layer_stream.get("sample_rate"),
                }
            )
            layer_entries.append({"file": layer_file, "layer_type": layer_type or "ace", "layer_id": layer_id})

        mixed_audio = artifacts_dir / f"run_{run_idx:03d}_{clip_id}_mix.mp3"
        mix_audio_layers(layer_entries, mixed_audio)

        final_video = artifacts_dir / f"run_{run_idx:03d}_{clip_id}_final.mp4"
        mux_video_audio(video_file, mixed_audio, final_video)

        video_stream = ffprobe_stream(final_video, "v:0")
        mixed_audio_stream = ffprobe_stream(mixed_audio, "a:0")

        manifest["runs"].append(
            {
                "run_index": run_idx,
                "clip_id": clip_id,
                "planner_prompt_id": planner_prompt_id,
                "video_prompt_id": video_prompt_id,
                "video_file": str(video_file),
                "mixed_audio_file": str(mixed_audio),
                "final_video_file": str(final_video),
                "video_metrics": {
                    "width": video_stream.get("width"),
                    "height": video_stream.get("height"),
                    "codec_name": video_stream.get("codec_name"),
                },
                "audio_metrics": {
                    "bit_rate": mixed_audio_stream.get("bit_rate"),
                    "sample_rate": mixed_audio_stream.get("sample_rate"),
                    "codec_name": mixed_audio_stream.get("codec_name"),
                },
                "layers": layer_reports,
            }
        )

        print(
            f"[run {run_idx + 1}/{args.runs}] done "
            f"video={final_video.name} {video_stream.get('width')}x{video_stream.get('height')} "
            f"audio_bitrate={mixed_audio_stream.get('bit_rate')}"
        )

    manifest_file = artifacts_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
