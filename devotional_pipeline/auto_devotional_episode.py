#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_LANGUAGE = "de"
DEFAULT_STYLE_PROFILE = "balanced"
DEFAULT_MUSIC_MODE = "ambient"
DEFAULT_TIMEOUT = 7200
CAMERA_MOVES = ["dolly", "orbit", "crane", "push", "pull", "locked-pan"]
SHOT_TYPES = ["close", "medium", "wide", "detail"]
ROUTE = "wan22_t2v"
SEGMENT_SECONDS = 5.0


@dataclass
class EpisodeRequest:
    topic_or_seed_prompt: str
    target_duration_sec: int
    language: str = DEFAULT_LANGUAGE
    voice_id: Optional[str] = None
    style_profile: str = DEFAULT_STYLE_PROFILE
    music_mode: str = DEFAULT_MUSIC_MODE
    output_name: str = "episode"


@dataclass
class SegmentPlan:
    index: int
    start_sec: float
    end_sec: float
    line_text: str
    visual_intent: str
    shot_type: str
    camera_move: str
    subject_motion: str
    environment_motion: str
    wan_prompt: str
    negative_prompt: str
    continuity_tags: List[str] = field(default_factory=list)


@dataclass
class EpisodePlan:
    title: str
    devotional_text: str
    narration_script: str
    target_duration_sec: int
    segments: List[SegmentPlan]


@dataclass
class AudioPlan:
    provider_used: str
    sample_rate: int
    narration_wav: str
    music_wav: Optional[str]
    sfx_wav: Optional[str]
    mix_wav: str


@dataclass
class QualityReport:
    completed: bool
    duration_ok: bool
    clip_count_ok: bool
    semantic_alignment_score: float
    motion_diversity_score: float
    audio_loudness_ok: bool
    fallbacks_used: List[str] = field(default_factory=list)
    blocking_errors: List[str] = field(default_factory=list)


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


def wait_for_history(server: str, prompt_id: str, timeout_s: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
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
    return streams[0] if streams else {}


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    try:
        return float((proc.stdout or "0").strip())
    except Exception:
        return 0.0


def load_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


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
                files.append(comfy_output_dir / subfolder / filename if subfolder else comfy_output_dir / filename)
    return files


def pick_file(files: Iterable[Path], suffix: str) -> Optional[Path]:
    matches = [f for f in files if f.suffix.lower() == suffix.lower()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return matches[0]


def extract_json_loose(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty LLM output")
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
        candidates.append(raw[obj_start : obj_end + 1])
    arr_start = raw.find("[")
    arr_end = raw.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        candidates.append(raw[arr_start : arr_end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("Could not parse output as JSON")

def normalize_duration(duration_sec: int) -> Tuple[int, int]:
    clamped = max(30, min(60, int(duration_sec)))
    seg_count = max(6, min(12, int(round(clamped / SEGMENT_SECONDS))))
    normalized = int(seg_count * SEGMENT_SECONDS)
    return normalized, seg_count


def normalize_request(req: EpisodeRequest) -> EpisodeRequest:
    duration, _ = normalize_duration(req.target_duration_sec)
    return EpisodeRequest(
        topic_or_seed_prompt=(req.topic_or_seed_prompt or "Ein kurzer Andachtsimpuls zu Hoffnung").strip(),
        target_duration_sec=duration,
        language=(req.language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE,
        voice_id=(req.voice_id or "").strip() or None,
        style_profile=(req.style_profile or DEFAULT_STYLE_PROFILE).strip().lower() or DEFAULT_STYLE_PROFILE,
        music_mode=(req.music_mode or DEFAULT_MUSIC_MODE).strip().lower() or DEFAULT_MUSIC_MODE,
        output_name=sanitize(req.output_name or "episode", "episode"),
    )


def workflow_nodes_by_class_type(workflow: Dict[str, Any], class_type: str) -> List[str]:
    out: List[str] = []
    needle = class_type.lower()
    for nid, node in workflow.items():
        ct = str(node.get("class_type") or "").lower()
        if ct == needle:
            out.append(nid)
    return out


def first_node_with_input(workflow: Dict[str, Any], input_key: str) -> Optional[str]:
    for nid, node in workflow.items():
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and input_key in inputs:
            return nid
    return None


def set_all_seed_inputs(workflow: Dict[str, Any], seed: int) -> None:
    for node in workflow.values():
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for k in list(inputs.keys()):
            if "seed" in k.lower():
                v = inputs[k]
                if isinstance(v, (int, float)):
                    inputs[k] = int(seed)


def planner_text_from_history_any(history: Dict[str, Any]) -> str:
    outputs = history.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("Missing outputs in history")
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        texts = node.get("text")
        if isinstance(texts, list) and texts:
            first = texts[0]
            if isinstance(first, str) and first.strip():
                return first
    raise ValueError("No planner text found in history outputs")


def run_text_generate_pass(
    server: str,
    planner_wf: Dict[str, Any],
    prompt: str,
    timeout_s: int,
) -> Tuple[str, str]:
    job = deep_clone(planner_wf)
    text_node = None
    text_nodes = workflow_nodes_by_class_type(job, "TextGenerate")
    if text_nodes:
        text_node = text_nodes[0]
    else:
        text_node = first_node_with_input(job, "prompt")
    if not text_node:
        raise ValueError("Could not find TextGenerate/prompt node in planner workflow")
    job[text_node]["inputs"]["prompt"] = prompt
    set_all_seed_inputs(job, random.randint(1, 2**31 - 1))
    prompt_id = queue_prompt(server, job)
    history = wait_for_history(server, prompt_id, timeout_s=timeout_s)
    text = planner_text_from_history_any(history)
    return prompt_id, text


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = []
    for c in chunks:
        t = c.strip()
        if t:
            out.append(t)
    return out


def fallback_base(req: EpisodeRequest, segment_count: int) -> Tuple[str, str, str]:
    title = "Licht im Alltag"
    devotional = (
        f"Dieses Andachtswort über \"{req.topic_or_seed_prompt}\" erinnert daran, "
        "dass Hoffnung auch im Druck wächst. Wir gehen Schritt für Schritt, bleiben ehrlich "
        "über Schmerz, und öffnen Raum für Vertrauen, Mut und neue Perspektive."
    )
    lines = [
        "Heute halten wir kurz inne.",
        "Du trägst mehr Last, als andere sehen.",
        "Trotzdem bist du nicht allein.",
        "In der Stille wächst wieder Orientierung.",
        "Ein kleiner Schritt kann alles verändern.",
        "Atme, richte dich auf, und geh weiter.",
    ]
    while len(lines) < segment_count:
        lines.append("Hoffnung bleibt, auch wenn es langsam geht.")
    narration = " ".join(lines[:segment_count])
    return title, devotional, narration


def parse_pass_json(raw: str, fallback: Any) -> Any:
    try:
        parsed = extract_json_loose(raw)
        return parsed
    except Exception:
        return fallback


def normalize_camera_move(value: str, idx: int) -> str:
    src = (value or "").strip().lower()
    mapping = {
        "dolly in": "dolly",
        "dolly out": "dolly",
        "orbiting": "orbit",
        "crane up": "crane",
        "truck": "dolly",
        "push in": "push",
        "pull out": "pull",
        "pan": "locked-pan",
    }
    src = mapping.get(src, src)
    if src in CAMERA_MOVES:
        return src
    return CAMERA_MOVES[idx % len(CAMERA_MOVES)]


def normalize_shot_type(value: str, idx: int) -> str:
    src = (value or "").strip().lower()
    if src in SHOT_TYPES:
        return src
    return SHOT_TYPES[idx % len(SHOT_TYPES)]


def join_continuity_tags(tags: List[str]) -> str:
    cleaned = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
    return ", ".join(cleaned[:6]) if cleaned else "same speaker, same wardrobe, consistent lighting, cinematic color grade"


def token_set(text: str) -> set:
    return {t for t in re.findall(r"\b[\wÄÖÜäöüß]{3,}\b", (text or "").lower())}


def lexical_overlap(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa.intersection(sb))
    base = max(1, min(len(sa), len(sb)))
    return inter / float(base)


def ensure_motion_diversity(segments: List[SegmentPlan]) -> None:
    unique_moves = {s.camera_move for s in segments}
    if len(unique_moves) >= 4 or len(segments) < 4:
        return
    for idx, segment in enumerate(segments):
        segment.camera_move = CAMERA_MOVES[idx % len(CAMERA_MOVES)]


def build_prompt_from_segment(segment: SegmentPlan) -> str:
    continuity = join_continuity_tags(segment.continuity_tags)
    return (
        f"Cinematic devotional scene, {segment.shot_type} shot, {segment.camera_move} camera movement, "
        f"subject motion: {segment.subject_motion}, environment motion: {segment.environment_motion}, "
        f"visual intent: {segment.visual_intent}. Continuity: {continuity}. High detail, natural skin, "
        f"realistic light, compelling composition, no text overlays."
    )


def safe_negative_prompt() -> str:
    return (
        "low quality, blurry, jpeg artifacts, watermark, text overlays, deformed hands, bad anatomy, "
        "flicker, frame glitches, oversaturated, distorted face"
    )


def parse_segments_payload(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("segments", "clips", "items", "scenes", "list"):
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []

def build_episode_plan(req: EpisodeRequest, pass_bundle: Dict[str, Any], fallback_notes: List[str]) -> EpisodePlan:
    target_duration, segment_count = normalize_duration(req.target_duration_sec)
    title, devotional, narration = fallback_base(req, segment_count)

    p1 = pass_bundle.get("pass1")
    if isinstance(p1, dict):
        if isinstance(p1.get("title"), str) and p1["title"].strip():
            title = p1["title"].strip()
        if isinstance(p1.get("devotional_text"), str) and p1["devotional_text"].strip():
            devotional = p1["devotional_text"].strip()

    p2 = pass_bundle.get("pass2")
    if isinstance(p2, dict) and isinstance(p2.get("narration_script"), str) and p2["narration_script"].strip():
        narration = p2["narration_script"].strip()

    lines = split_sentences(narration)
    if not lines:
        _, _, fallback_narration = fallback_base(req, segment_count)
        lines = split_sentences(fallback_narration)
        fallback_notes.append("fallback_narration_sentences")

    while len(lines) < segment_count:
        lines.append(lines[-1] if lines else "Hoffnung bleibt.")
    lines = lines[:segment_count]

    pass3_segments = parse_segments_payload(pass_bundle.get("pass3"))
    pass4_segments = parse_segments_payload(pass_bundle.get("pass4"))

    seg_out: List[SegmentPlan] = []
    for idx in range(segment_count):
        start_sec = idx * SEGMENT_SECONDS
        end_sec = start_sec + SEGMENT_SECONDS
        line_text = lines[idx]

        s3 = pass3_segments[idx] if idx < len(pass3_segments) else {}
        s4 = pass4_segments[idx] if idx < len(pass4_segments) else {}

        visual_intent = str(s3.get("visual_intent") or f"Embodied devotional moment for: {line_text}").strip()
        shot_type = normalize_shot_type(str(s3.get("shot_type") or ""), idx)
        camera_move = normalize_camera_move(str(s3.get("camera_move") or ""), idx)
        subject_motion = str(s3.get("subject_motion") or "gentle natural movement, subtle gestures").strip()
        environment_motion = str(s3.get("environment_motion") or "soft wind, drifting light and depth").strip()
        continuity_tags = s3.get("continuity_tags")
        if not isinstance(continuity_tags, list):
            continuity_tags = ["same speaker", "consistent wardrobe", "warm cinematic daylight", "natural skin tones"]

        seg = SegmentPlan(
            index=idx,
            start_sec=start_sec,
            end_sec=end_sec,
            line_text=line_text,
            visual_intent=visual_intent,
            shot_type=shot_type,
            camera_move=camera_move,
            subject_motion=subject_motion,
            environment_motion=environment_motion,
            wan_prompt="",
            negative_prompt="",
            continuity_tags=[str(x) for x in continuity_tags if isinstance(x, str)],
        )

        if isinstance(s4.get("wan_prompt"), str) and s4["wan_prompt"].strip():
            seg.wan_prompt = s4["wan_prompt"].strip()
        else:
            seg.wan_prompt = build_prompt_from_segment(seg)
        if isinstance(s4.get("negative_prompt"), str) and s4["negative_prompt"].strip():
            seg.negative_prompt = s4["negative_prompt"].strip()
        else:
            seg.negative_prompt = safe_negative_prompt()
        seg_out.append(seg)

    ensure_motion_diversity(seg_out)
    for seg in seg_out:
        if not seg.wan_prompt.strip():
            seg.wan_prompt = build_prompt_from_segment(seg)
        if not seg.negative_prompt.strip():
            seg.negative_prompt = safe_negative_prompt()

    return EpisodePlan(
        title=title,
        devotional_text=devotional,
        narration_script=narration,
        target_duration_sec=target_duration,
        segments=seg_out,
    )


def semantic_alignment_score(plan: EpisodePlan) -> float:
    if not plan.segments:
        return 0.0
    total = 0.0
    for s in plan.segments:
        total += lexical_overlap(s.line_text, f"{s.visual_intent} {s.wan_prompt}")
    return round(total / len(plan.segments), 4)


def motion_diversity_score(plan: EpisodePlan) -> float:
    if not plan.segments:
        return 0.0
    unique = {s.camera_move for s in plan.segments}
    return round(min(1.0, len(unique) / 4.0), 4)


def sec_to_wan_length(seconds: float, fps: int) -> int:
    raw_frames = max(1.0, float(seconds) * float(fps))
    blocks = max(1, int(round((raw_frames - 1.0) / 4.0)))
    return blocks * 4 + 1


def build_qwen_prompts(req: EpisodeRequest, target_duration: int, segment_count: int, state: Dict[str, Any]) -> Dict[str, str]:
    lang = "Deutsch" if req.language.startswith("de") else "English"
    base_topic = req.topic_or_seed_prompt
    pass1 = (
        "You are a devotional architect. Return strict JSON only.\n"
        f"Language: {lang}.\n"
        f"Topic seed: {base_topic}\n"
        f"Target duration total: {target_duration} seconds.\n"
        "Return object with keys: title, devotional_text, theological_core, emotional_tone.\n"
        "Keep devotional_text pastoral, clear, and practical."
    )
    pass2 = (
        "You are a voiceover writer. Return strict JSON only.\n"
        f"Language: {lang}.\n"
        "Input devotional data:\n"
        f"{json.dumps(state.get('pass1') or {}, ensure_ascii=False)}\n"
        f"Create narration for {target_duration} seconds.\n"
        "Return object with key narration_script only."
    )
    pass3 = (
        "You are a cinematic shot director. Return strict JSON only.\n"
        f"Language: {lang}. Segment count: {segment_count}, each 5 seconds.\n"
        f"Narration: {json.dumps((state.get('pass2') or {}).get('narration_script', ''), ensure_ascii=False)}\n"
        "Return object with key segments (array). Each segment needs: "
        "index, line_text, visual_intent, shot_type(close|medium|wide|detail), camera_move(dolly|orbit|crane|push|pull|locked-pan), "
        "subject_motion, environment_motion, continuity_tags(array of strings)."
    )
    pass4 = (
        "You are a WAN 2.2 prompt refiner. Return strict JSON only.\n"
        f"Language: {lang}.\n"
        f"Input segments: {json.dumps((state.get('pass3') or {}).get('segments', []), ensure_ascii=False)}\n"
        "Return object with key segments (array), each item has: index, wan_prompt, negative_prompt.\n"
        "Prompts must be cinematic, movement-aware, and continuity-aware."
    )
    return {"pass1": pass1, "pass2": pass2, "pass3": pass3, "pass4": pass4}


def generate_episode_plan(
    req: EpisodeRequest,
    planner_wf: Dict[str, Any],
    server: str,
    timeout: int,
    fallback_notes: List[str],
) -> Tuple[EpisodePlan, Dict[str, Any], List[str]]:
    target_duration, segment_count = normalize_duration(req.target_duration_sec)
    state: Dict[str, Any] = {}
    prompt_ids: List[str] = []

    prompts = build_qwen_prompts(req, target_duration, segment_count, state)

    for pass_name in ("pass1", "pass2", "pass3", "pass4"):
        try:
            prompt_ids.append(f"{pass_name}:start")
            pid, raw = run_text_generate_pass(server, planner_wf, prompts[pass_name], timeout)
            prompt_ids[-1] = f"{pass_name}:{pid}"
            fallback = {} if pass_name in ("pass1", "pass2") else {"segments": []}
            parsed = parse_pass_json(raw, fallback)
            if not isinstance(parsed, (dict, list)):
                parsed = fallback
                fallback_notes.append(f"{pass_name}_non_json_fallback")
            state[pass_name] = parsed
            prompts = build_qwen_prompts(req, target_duration, segment_count, state)
        except Exception:
            state[pass_name] = {} if pass_name in ("pass1", "pass2") else {"segments": []}
            fallback_notes.append(f"{pass_name}_request_failed")

    plan = build_episode_plan(req, state, fallback_notes)
    return plan, state, prompt_ids

def get_windows_tts_voices() -> List[str]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }; $s.Dispose()",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


def pick_windows_tts_voice(voice_hint: str) -> str:
    voices = get_windows_tts_voices()
    if not voices:
        raise RuntimeError("No installed Windows TTS voices found")
    hint = (voice_hint or "").lower()
    for preferred in ("Hedda", "Katja", "Zira", "David", "Mark"):
        if preferred.lower() in hint:
            for v in voices:
                if preferred.lower() in v.lower():
                    return v
    return voices[0]


def synthesize_speaker_tts_windows(text: str, voice_hint: str, out_mp3: Path, target_duration: float) -> Path:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_mp3.with_suffix(".wav")
    voice_name = pick_windows_tts_voice(voice_hint)

    env = dict(**os.environ)
    env["TTS_TEXT"] = text
    env["TTS_VOICE"] = voice_name
    env["TTS_OUT_WAV"] = str(wav_path)
    ps_script = (
        "$ErrorActionPreference='Stop'; "
        "Add-Type -AssemblyName System.Speech; "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$voices=$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }; "
        "if ($voices -contains $env:TTS_VOICE) { $s.SelectVoice($env:TTS_VOICE) }; "
        "$s.Rate=0; "
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

    dur = max(0.5, float(target_duration))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-af",
            f"apad=pad_dur={dur:.3f},atrim=duration={dur:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(out_mp3),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wav_path.unlink(missing_ok=True)
    return out_mp3


def try_elevenlabs_tts(
    text: str,
    out_mp3: Path,
    api_key: str,
    voice_id: Optional[str],
    language: str,
) -> Path:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    voice_defaults = {
        "de": "TxGEqnHWrfWFTfGW9XjX",
        "en": "JBFqnCBsd6RMkjVDRZzb",
    }
    resolved_voice = voice_id or voice_defaults.get(language, voice_defaults["en"])
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.75,
            "style": 0.35,
            "use_speaker_boost": True,
        },
    }
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{resolved_voice}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    out_mp3.write_bytes(resp.content)
    return out_mp3


def create_silence_audio(path: Path, duration: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            f"{max(0.5, duration):.3f}",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def generate_audio_layer_via_workflow(
    server: str,
    audio_wf: Dict[str, Any],
    layer_item: Dict[str, Any],
    filename_prefix: str,
    timeout: int,
    comfy_output: Path,
) -> Optional[Path]:
    job = deep_clone(audio_wf)
    iter_nodes = workflow_nodes_by_class_type(job, "gaistreich_BatchAudioAceIterator")
    if not iter_nodes:
        return None
    iter_node = iter_nodes[0]
    job[iter_node]["inputs"]["input_text"] = json.dumps([layer_item], ensure_ascii=True)
    job[iter_node]["inputs"]["mode"] = "json"
    job[iter_node]["inputs"]["index"] = 0

    save_nodes = workflow_nodes_by_class_type(job, "SaveAudioMP3")
    if save_nodes:
        job[save_nodes[0]]["inputs"]["filename_prefix"] = filename_prefix

    set_all_seed_inputs(job, random.randint(1, 2**31 - 1))
    prompt_id = queue_prompt(server, job)
    history = wait_for_history(server, prompt_id, timeout_s=timeout)
    out_files = find_output_files(history, comfy_output)
    mp3 = pick_file(out_files, ".mp3")
    return mp3 if mp3 and mp3.exists() else None


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
        if layer_type == "narration":
            weight = 1.0
        elif layer_type == "ambient":
            weight = 0.20
        elif layer_type == "music":
            weight = 0.18
        elif layer_type == "sfx":
            weight = 0.30
        else:
            weight = 0.25
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


def measure_audio_loudness_ok(audio_file: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_file),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "NUL",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stderr = proc.stderr or ""
    m = re.search(r"mean_volume:\s*(-?\d+(\.\d+)?)\s*dB", stderr)
    if not m:
        return False
    mean = float(m.group(1))
    return -30.0 <= mean <= -6.0

def concat_videos(video_paths: List[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.with_suffix(".txt")
    list_file.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in video_paths),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        list_file.unlink(missing_ok=True)
    return output_path


def mux_video_audio(video_file: Path, audio_file: Path, out_file: Path) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_file


def fallback_static_video(out_file: Path, duration: float, width: int = 768, height: int = 1280, fps: int = 24) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=#111111:s={width}x{height}:r={fps}:d={duration:.3f}",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_file


def build_video_clip_payload(plan: EpisodePlan, segment: SegmentPlan) -> Dict[str, Any]:
    return {
        "project": sanitize(plan.title, "devotional_project"),
        "style_preset": "cinematic_devotional",
        "clips": [
            {
                "id": f"clip_{segment.index:03d}",
                "route": ROUTE,
                "video_prompt": segment.wan_prompt,
                "negative_prompt": segment.negative_prompt,
                "image_prompt": segment.visual_intent,
                "duration_seconds": SEGMENT_SECONDS,
                "audio": {
                    "speakers": [{"id": "speaker_1", "text": segment.line_text, "voice": "warm_calm"}],
                    "ambient": [{"type": "wind"}],
                    "music": {"mood": "gentle cinematic devotional"},
                    "sfx": [],
                },
            }
        ],
    }


def queue_video_segment(
    server: str,
    video_wf: Dict[str, Any],
    clip_payload: Dict[str, Any],
    run_index: int,
    run_total: int,
    timeout: int,
    comfy_output: Path,
) -> Optional[Path]:
    job = deep_clone(video_wf)

    iter_nodes = workflow_nodes_by_class_type(job, "gaistreich_BatchDevotionalEpisodeIterator")
    if not iter_nodes:
        raise ValueError("Video workflow missing gaistreich_BatchDevotionalEpisodeIterator node")
    iter_node = iter_nodes[0]
    job[iter_node]["inputs"]["input_text"] = json.dumps(clip_payload, ensure_ascii=True)
    job[iter_node]["inputs"]["mode"] = "json"
    job[iter_node]["inputs"]["index"] = 0
    job[iter_node]["inputs"]["run_index"] = int(run_index)
    job[iter_node]["inputs"]["run_total"] = int(run_total)

    create_nodes = workflow_nodes_by_class_type(job, "CreateVideo")
    fps = 24
    if create_nodes:
        try:
            fps = int(job[create_nodes[0]]["inputs"].get("fps", 24))
        except Exception:
            fps = 24
    length = sec_to_wan_length(SEGMENT_SECONDS, fps)
    latent_nodes = workflow_nodes_by_class_type(job, "EmptyHunyuanLatentVideo")
    if latent_nodes:
        job[latent_nodes[0]]["inputs"]["length"] = int(length)

    set_all_seed_inputs(job, random.randint(1, 2**31 - 1))
    prompt_id = queue_prompt(server, job)
    history = wait_for_history(server, prompt_id, timeout_s=timeout)
    out_files = find_output_files(history, comfy_output)
    return pick_file(out_files, ".mp4")


def generate_segment_video_with_fallback(
    server: str,
    video_wf: Dict[str, Any],
    plan: EpisodePlan,
    segment: SegmentPlan,
    output_dir: Path,
    timeout: int,
    comfy_output: Path,
    fallbacks_used: List[str],
) -> Path:
    clip_payload = build_video_clip_payload(plan, segment)
    try:
        video = queue_video_segment(server, video_wf, clip_payload, segment.index, len(plan.segments), timeout, comfy_output)
        if video and video.exists():
            return video
        raise FileNotFoundError("No mp4 from primary video pass")
    except Exception:
        fallbacks_used.append(f"segment_{segment.index}_wan_backup_prompt")

    backup_segment = copy.deepcopy(segment)
    backup_segment.wan_prompt = (
        f"Cinematic devotional portrait, {backup_segment.shot_type} shot, "
        f"{backup_segment.camera_move} motion, calm subject, natural light, continuity maintained."
    )
    backup_segment.negative_prompt = safe_negative_prompt()
    backup_payload = build_video_clip_payload(plan, backup_segment)
    try:
        video = queue_video_segment(server, video_wf, backup_payload, segment.index, len(plan.segments), timeout, comfy_output)
        if video and video.exists():
            return video
        raise FileNotFoundError("No mp4 from backup prompt")
    except Exception:
        fallbacks_used.append(f"segment_{segment.index}_static_video")

    static_out = output_dir / f"segment_{segment.index:03d}_fallback.mp4"
    return fallback_static_video(static_out, SEGMENT_SECONDS)


def generate_audio_plan(
    req: EpisodeRequest,
    plan: EpisodePlan,
    env: Dict[str, str],
    server: str,
    audio_wf: Dict[str, Any],
    output_dir: Path,
    timeout: int,
    comfy_output: Path,
    fallbacks_used: List[str],
) -> AudioPlan:
    narration_path = output_dir / "narration.mp3"
    total_duration = float(plan.target_duration_sec)
    provider = "local"

    eleven_key = env.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY", "")
    if eleven_key:
        try:
            try_elevenlabs_tts(
                text=plan.narration_script,
                out_mp3=narration_path,
                api_key=eleven_key,
                voice_id=req.voice_id,
                language=req.language,
            )
            provider = "elevenlabs"
        except Exception:
            fallbacks_used.append("elevenlabs_failed_local_tts_used")
            synthesize_speaker_tts_windows(
                text=plan.narration_script,
                voice_hint=req.voice_id or req.language,
                out_mp3=narration_path,
                target_duration=total_duration,
            )
    else:
        fallbacks_used.append("elevenlabs_key_missing_local_tts_used")
        synthesize_speaker_tts_windows(
            text=plan.narration_script,
            voice_hint=req.voice_id or req.language,
            out_mp3=narration_path,
            target_duration=total_duration,
        )

    padded_narration = output_dir / "narration_padded.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(narration_path),
            "-af",
            f"apad=pad_dur={total_duration:.3f},atrim=duration={total_duration:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(padded_narration),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    layer_entries: List[Dict[str, Any]] = [{"file": str(padded_narration), "layer_type": "narration"}]
    music_path: Optional[Path] = None
    sfx_path: Optional[Path] = None

    if req.music_mode != "none":
        music_item = {
            "id": "music",
            "layer_type": "music",
            "tags": "cinematic devotional underscore, no vocals, warm organic instrumentation, subtle dynamic lift",
            "lyrics": "",
            "bpm": 84,
            "duration": total_duration,
            "timesignature": "4",
            "language": req.language,
            "keyscale": "C major",
        }
        music_path = generate_audio_layer_via_workflow(
            server=server,
            audio_wf=audio_wf,
            layer_item=music_item,
            filename_prefix=f"devotional_pipeline/auto_audio/{sanitize(req.output_name, 'episode')}_music",
            timeout=timeout,
            comfy_output=comfy_output,
        )
        if music_path and music_path.exists():
            layer_entries.append({"file": str(music_path), "layer_type": "music"})
        else:
            fallbacks_used.append("music_generation_failed")

    if req.music_mode in ("ambient", "cinematic"):
        ambient_item = {
            "id": "ambient",
            "layer_type": "ambient",
            "tags": "soft wind, subtle room tone, gentle natural ambience, no melody, no speech",
            "lyrics": "",
            "bpm": 84,
            "duration": total_duration,
            "timesignature": "4",
            "language": req.language,
            "keyscale": "C major",
        }
        ambient_path = generate_audio_layer_via_workflow(
            server=server,
            audio_wf=audio_wf,
            layer_item=ambient_item,
            filename_prefix=f"devotional_pipeline/auto_audio/{sanitize(req.output_name, 'episode')}_ambient",
            timeout=timeout,
            comfy_output=comfy_output,
        )
        if ambient_path and ambient_path.exists():
            layer_entries.append({"file": str(ambient_path), "layer_type": "ambient"})
        else:
            fallbacks_used.append("ambient_generation_failed")

    sfx_item = {
        "id": "sfx",
        "layer_type": "sfx",
        "tags": "sparse cinematic accents, subtle transitions, no harsh hits, no speech",
        "lyrics": "",
        "bpm": 84,
        "duration": total_duration,
        "timesignature": "4",
        "language": req.language,
        "keyscale": "C major",
    }
    sfx_path = generate_audio_layer_via_workflow(
        server=server,
        audio_wf=audio_wf,
        layer_item=sfx_item,
        filename_prefix=f"devotional_pipeline/auto_audio/{sanitize(req.output_name, 'episode')}_sfx",
        timeout=timeout,
        comfy_output=comfy_output,
    )
    if sfx_path and sfx_path.exists():
        layer_entries.append({"file": str(sfx_path), "layer_type": "sfx"})
    else:
        fallbacks_used.append("sfx_generation_failed")

    mix_path = output_dir / "mix.mp3"
    try:
        mix_audio_layers(layer_entries, mix_path)
    except Exception:
        fallbacks_used.append("mix_failed_using_narration_only")
        mix_path = output_dir / "mix_narration_only.mp3"
        if padded_narration.exists():
            mix_path.write_bytes(padded_narration.read_bytes())
        else:
            create_silence_audio(mix_path, total_duration)

    return AudioPlan(
        provider_used=provider,
        sample_rate=48000,
        narration_wav=str(padded_narration),
        music_wav=str(music_path) if music_path and music_path.exists() else None,
        sfx_wav=str(sfx_path) if sfx_path and sfx_path.exists() else None,
        mix_wav=str(mix_path),
    )

def run_episode(
    req: EpisodeRequest,
    server: str,
    workflows_dir: Path,
    planner_workflow_name: str,
    video_workflow_name: str,
    audio_workflow_name: str,
    timeout: int,
    comfy_output: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    req = normalize_request(req)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = comfy_output / "devotional_pipeline" / "auto_episodes" / f"{run_stamp}_{req.output_name}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    planner_wf = load_json(workflows_dir / planner_workflow_name)
    video_wf = load_json(workflows_dir / video_workflow_name)
    audio_wf = load_json(workflows_dir / audio_workflow_name)

    fallbacks_used: List[str] = []
    blocking_errors: List[str] = []

    plan, pass_bundle, prompt_ids = generate_episode_plan(
        req=req,
        planner_wf=planner_wf,
        server=server,
        timeout=timeout,
        fallback_notes=fallbacks_used,
    )

    segment_clips: List[Path] = []
    segments_dir = artifacts_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    for segment in plan.segments:
        clip = generate_segment_video_with_fallback(
            server=server,
            video_wf=video_wf,
            plan=plan,
            segment=segment,
            output_dir=segments_dir,
            timeout=timeout,
            comfy_output=comfy_output,
            fallbacks_used=fallbacks_used,
        )
        segment_clips.append(clip)

    concat_video = artifacts_dir / "video_concat.mp4"
    try:
        concat_videos(segment_clips, concat_video)
    except Exception as exc:
        blocking_errors.append(f"concat_failed:{exc}")
        concat_video = artifacts_dir / "video_fallback_full.mp4"
        fallback_static_video(
            out_file=concat_video,
            duration=float(plan.target_duration_sec),
            width=768,
            height=1280,
            fps=24,
        )
        fallbacks_used.append("full_video_static_fallback")

    try:
        audio_plan = generate_audio_plan(
            req=req,
            plan=plan,
            env=env,
            server=server,
            audio_wf=audio_wf,
            output_dir=artifacts_dir,
            timeout=timeout,
            comfy_output=comfy_output,
            fallbacks_used=fallbacks_used,
        )
        mixed_audio = Path(audio_plan.mix_wav)
    except Exception as exc:
        blocking_errors.append(f"audio_pipeline_failed:{exc}")
        fallbacks_used.append("audio_silence_fallback")
        mixed_audio = create_silence_audio(artifacts_dir / "mix_silence.mp3", float(plan.target_duration_sec))
        audio_plan = AudioPlan(
            provider_used="local",
            sample_rate=48000,
            narration_wav=str(mixed_audio),
            music_wav=None,
            sfx_wav=None,
            mix_wav=str(mixed_audio),
        )

    final_video = artifacts_dir / "final_episode.mp4"
    try:
        mux_video_audio(concat_video, mixed_audio, final_video)
    except Exception as exc:
        blocking_errors.append(f"mux_failed:{exc}")
        final_video.write_bytes(concat_video.read_bytes())
        fallbacks_used.append("mux_copy_video_fallback")

    duration = ffprobe_duration(final_video) if final_video.exists() else 0.0
    duration_ok = 30.0 <= duration <= 60.0
    clip_count_ok = 6 <= len(plan.segments) <= 12
    sem_score = semantic_alignment_score(plan)
    motion_score = motion_diversity_score(plan)
    audio_ok = measure_audio_loudness_ok(mixed_audio) if mixed_audio.exists() else False
    completed = final_video.exists()

    report = QualityReport(
        completed=completed,
        duration_ok=duration_ok,
        clip_count_ok=clip_count_ok,
        semantic_alignment_score=sem_score,
        motion_diversity_score=motion_score,
        audio_loudness_ok=audio_ok,
        fallbacks_used=fallbacks_used,
        blocking_errors=blocking_errors,
    )

    plan_json = {
        "title": plan.title,
        "devotional_text": plan.devotional_text,
        "narration_script": plan.narration_script,
        "target_duration_sec": plan.target_duration_sec,
        "segments": [asdict(s) for s in plan.segments],
    }
    quality_json = asdict(report)
    audio_json = asdict(audio_plan)

    (artifacts_dir / "episode_plan.json").write_text(json.dumps(plan_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (artifacts_dir / "quality_report.json").write_text(json.dumps(quality_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (artifacts_dir / "audio_plan.json").write_text(json.dumps(audio_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (artifacts_dir / "llm_passes.json").write_text(json.dumps(pass_bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "completed" if completed else "failed",
        "request": asdict(req),
        "final_video_path": str(final_video),
        "mixed_audio_path": str(mixed_audio),
        "quality_report": quality_json,
        "audio_plan": audio_json,
        "episode_plan": plan_json,
        "artifacts_dir": str(artifacts_dir),
        "planner_prompt_ids": prompt_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto devotional episode orchestrator (30-60s, WAN 2.2, hybrid fallback).")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--topic-or-seed-prompt", required=True)
    parser.add_argument("--target-duration-sec", type=int, default=45)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--voice-id", default="")
    parser.add_argument("--style-profile", default=DEFAULT_STYLE_PROFILE)
    parser.add_argument("--music-mode", default=DEFAULT_MUSIC_MODE, choices=["none", "ambient", "cinematic"])
    parser.add_argument("--output-name", default="episode")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--comfy-output", default=r"C:\ComfyUI\output")
    parser.add_argument("--workflows-dir", default=str(Path(__file__).resolve().parent / "workflows"))
    parser.add_argument("--planner-workflow", default="01_qwen_devotional_planner_api.json")
    parser.add_argument("--video-workflow", default="02_wan22_video_from_devotional_iterator_api.json")
    parser.add_argument("--audio-workflow", default="03_ace15_audio_layer_iterator_api.json")
    parser.add_argument("--env-file", default=str(Path(__file__).resolve().parents[1] / ".env"))
    parser.add_argument("--json-output", action="store_true", help="Print final result as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env(Path(args.env_file))

    req = EpisodeRequest(
        topic_or_seed_prompt=args.topic_or_seed_prompt,
        target_duration_sec=args.target_duration_sec,
        language=args.language,
        voice_id=args.voice_id or None,
        style_profile=args.style_profile,
        music_mode=args.music_mode,
        output_name=args.output_name,
    )

    result = run_episode(
        req=req,
        server=args.server,
        workflows_dir=Path(args.workflows_dir),
        planner_workflow_name=args.planner_workflow,
        video_workflow_name=args.video_workflow,
        audio_workflow_name=args.audio_workflow,
        timeout=args.timeout,
        comfy_output=Path(args.comfy_output).resolve(),
        env=env,
    )

    if args.json_output:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Final video: {result['final_video_path']}")
        print(f"Mixed audio: {result['mixed_audio_path']}")
        print(f"Quality report: {json.dumps(result['quality_report'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
