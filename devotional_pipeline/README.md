# Devotional Pipeline (Local ComfyUI)

This folder now contains two runners:

1. `run_devotional_pipeline.py` (legacy single-clip flow).
2. `auto_devotional_episode.py` (new hybrid 30-60s multi-clip flow with fallback guarantee).

The new auto runner implements:
- Multi-pass Qwen planning (devotional -> narration -> shot map -> prompt refinement)
- WAN 2.2 segment generation in 5s units (6-12 clips total)
- Audio fallback chain (ElevenLabs from `.env`, fallback to local Windows TTS)
- Layered mix + hard-cut concat + final mux
- Quality report and artifact bundle per run

## Files

- `workflows/01_qwen_devotional_planner_api.json`
- `workflows/02_wan22_video_from_devotional_iterator_api.json`
- `workflows/03_ace15_audio_layer_iterator_api.json`
- `workflows/04_ltx2_video_from_devotional_iterator_api.json` (alternative video generator)
- `auto_devotional_episode.py` (new orchestrator)
- `run_devotional_pipeline.py`
- `workflows/06_wan22_qwen3_audio_integrated_api.json` (Qwen3 + WAN2.2 + Ace audio in one graph)
- `workflows/07_ltx2_qwen3_audio_integrated_api.json` (Qwen3 + LTX-2 AV graph)

## Run

```powershell
cd C:\ComfyUI\devotional_pipeline
python .\run_devotional_pipeline.py --runs 1 --theme "A devotional about hope in difficult times"
```

Optional flags:

- `--server http://127.0.0.1:8188`
- `--max-duration 2.0` (fast test mode)
- `--timeout 3600`
- `--comfy-output C:\ComfyUI\output`
- `--video-workflow 04_ltx2_video_from_devotional_iterator_api.json` (switch from WAN to LTX2)
- `--background-mode none` (speaker-only, disables ambient/music/sfx generation)

## New Auto Runner (30-60s)

```powershell
cd C:\ComfyUI\devotional_pipeline
python .\auto_devotional_episode.py `
  --topic-or-seed-prompt "Hoffnung unter Druck" `
  --target-duration-sec 45 `
  --language de `
  --music-mode ambient `
  --output-name devotional_hope `
  --json-output
```

Important:
- Duration is clamped to 30-60 seconds.
- Clip unit is fixed to 5 seconds.
- ElevenLabs key is read from `C:\ComfyUI\.env` (`ELEVENLABS_API_KEY`).
- If ElevenLabs fails/unavailable, local Windows TTS is used automatically.

## Expected outputs

- Layered audio files under `C:\ComfyUI\output\devotional_pipeline\audio\...`
- Mixed audio and final muxed video under `C:\ComfyUI\output\devotional_pipeline\artifacts\<timestamp>\`
- `manifest.json` with measured video resolution and audio bitrate
- For auto runner: `C:\ComfyUI\output\devotional_pipeline\auto_episodes\<timestamp>_<name>\`
  - `final_episode.mp4`
  - `mix.mp3` (or fallback mix)
  - `episode_plan.json`
  - `audio_plan.json`
  - `quality_report.json`
  - `llm_passes.json`
