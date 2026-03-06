from pathlib import Path

from devotional_pipeline.auto_devotional_episode import (
    EpisodeRequest,
    build_episode_plan,
    motion_diversity_score,
    normalize_duration,
    semantic_alignment_score,
)


def test_normalize_duration_bounds_and_step():
    duration, segments = normalize_duration(29)
    assert duration == 30
    assert segments == 6

    duration, segments = normalize_duration(61)
    assert duration == 60
    assert segments == 12

    duration, segments = normalize_duration(47)
    assert duration in (45, 50)
    assert 6 <= segments <= 12


def test_build_episode_plan_repairs_segment_count_and_fields():
    req = EpisodeRequest(
        topic_or_seed_prompt="Hoffnung in schwierigen Zeiten",
        target_duration_sec=45,
        language="de",
        output_name="unit",
    )
    pass_bundle = {
        "pass1": {"title": "Test", "devotional_text": "Kurzandacht"},
        "pass2": {"narration_script": "Satz eins. Satz zwei. Satz drei."},
        "pass3": {"segments": [{"index": 0, "line_text": "x"}]},
        "pass4": {"segments": [{"index": 0, "wan_prompt": "", "negative_prompt": ""}]},
    }
    fallback_notes = []
    plan = build_episode_plan(req, pass_bundle, fallback_notes)

    assert plan.title == "Test"
    assert len(plan.segments) == 9  # 45 sec / 5 sec
    assert all(s.wan_prompt for s in plan.segments)
    assert all(s.negative_prompt for s in plan.segments)
    assert all(s.camera_move for s in plan.segments)
    assert all(s.shot_type for s in plan.segments)


def test_quality_scores_are_computable():
    req = EpisodeRequest(
        topic_or_seed_prompt="Frieden",
        target_duration_sec=30,
        language="de",
    )
    pass_bundle = {
        "pass1": {"title": "Frieden", "devotional_text": "Text"},
        "pass2": {"narration_script": "Frieden beginnt im Herzen. Frieden wirkt nach außen."},
        "pass3": {"segments": []},
        "pass4": {"segments": []},
    }
    plan = build_episode_plan(req, pass_bundle, [])

    sem = semantic_alignment_score(plan)
    mov = motion_diversity_score(plan)
    assert 0.0 <= sem <= 1.0
    assert 0.0 <= mov <= 1.0
