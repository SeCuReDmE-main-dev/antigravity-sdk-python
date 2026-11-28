"""Automated pytest suite for `agy-studio-token-meter` (multimodal meter + HUD v8 geometry)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

THIS_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = THIS_DIR.parent
for candidate in (THIS_DIR, PLUGIN_ROOT / "sidecars" / "meter", PLUGIN_ROOT / "src" / "agy_token_meter"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from hud_pill_overlay import (  # noqa: E402
    ANCHOR_DEADZONE_PX,
    PILL_COLOR_CALM_GREY,
    PILL_COLOR_CRIT_RED,
    PILL_COLOR_WARN_YELLOW,
    SEND_LEFT_TO_BOX_RIGHT,
    TOKEN_CRIT_RED_THRESHOLD,
    TOKEN_WARN_YELLOW_THRESHOLD,
    TURBO_SHIELD_OFFSET,
    _apply_anchor_hysteresis,
    _find_send_button,
    _find_toolbar_left_edge,
    _inspect_live_antigravity_bar,
    _is_dark_input_bar_surface,
    _is_send_pixel,
    _resolve_pill_tier_color,
)
import hud_pill_overlay  # noqa: E402
from multimodal_token_meter import (  # noqa: E402
    GEMINI_IMAGE_BASE_TOKENS,
    MultimodalTokenMeter,
    count_image_tokens_from_dims,
    count_text_tokens,
)


def test_is_send_pixel_recognizes_idle_and_accent() -> None:
    assert _is_send_pixel((142, 142, 142)) is True  # idle #8E8E8E disc
    assert _is_send_pixel((204, 204, 204)) is True  # focused #CCCCCC disc
    assert _is_send_pixel((132, 35, 191)) is True  # active #8423BF purple accent disc
    assert _is_send_pixel((32, 32, 32)) is False  # #202020 bar surface


def test_find_send_button_reanchors_overshot_fill_border() -> None:
    # Simulate a Fill bar where y_safe overshot to 1355, while the 28px Send disc sits at x=1302..1329
    def fake_get_rgb(x: int, y: int) -> tuple[int, int, int]:
        if 1302 <= x <= 1329 and y in (681, 683):
            return (132, 35, 191)
        return (32, 32, 32)

    send = _find_send_button(
        fake_get_rgb,
        x_from=1355,
        x_to=1100,
        rows=(681, 683),
    )
    assert send == (1302, 1329)
    reanchored_box_right = send[0] + SEND_LEFT_TO_BOX_RIGHT
    assert reanchored_box_right == 1334
    assert reanchored_box_right - TURBO_SHIELD_OFFSET == 1190


def test_find_toolbar_left_edge_detects_turbo_shield_and_stops_before_pill() -> None:
    surface = (32, 32, 32)

    # Simulate cluster: shield (1189..1199), Turbo (1208..1238), chevron (1249..1252), mic (1275..1283)
    # and our own pill occupying 1070..1176 (so pill_right_x = 1177 is the first free pixel right of our pill)
    def fake_get_rgb(x: int, _y: int) -> tuple[int, int, int]:
        if 1189 <= x <= 1199 or 1208 <= x <= 1238 or 1249 <= x <= 1252 or 1275 <= x <= 1283:
            return (142, 142, 142)
        if 1070 <= x < 1177:
            return (142, 142, 142)  # our own pill
        return surface

    shield_left = _find_toolbar_left_edge(
        fake_get_rgb,
        start_x=1334 - 46,
        stop_x=1177,
        rows=(671, 674, 677),
        surface_rgb=surface,
    )
    assert shield_left == 1189


def test_multimodal_token_counts_and_meter(tmp_path: Path) -> None:
    assert count_text_tokens("Hello Google Antigravity SDK!") > 0
    assert count_image_tokens_from_dims(320, 240) == (GEMINI_IMAGE_BASE_TOKENS, 1)
    assert count_image_tokens_from_dims(1024, 575) == (GEMINI_IMAGE_BASE_TOKENS * 3, 2)

    conv_dir = tmp_path / "conv-1" / ".system_generated" / "logs"
    conv_dir.mkdir(parents=True)
    (conv_dir / "transcript.jsonl").write_text(
        '{"step_index": 0, "content": "Test prompt", "thinking": "Reasoning step"}\n',
        encoding="utf-8",
    )
    meter = MultimodalTokenMeter(brain_root=tmp_path)
    bd = meter.get_breakdown(conversation_id="conv-1")
    assert bd.total_tokens > 0
    assert bd.studio_label.endswith(" tokens")


def test_dark_input_bar_surface_tolerates_voice_mode_blue_halo() -> None:
    # Standard neutral #202020 surface
    assert _is_dark_input_bar_surface((32, 32, 32)) is True
    # Voice Mode blue/cyan halo tint on the bottom input bar (B - R > 8, previously rejected!)
    assert _is_dark_input_bar_surface((26, 38, 62)) is True
    assert _is_dark_input_bar_surface((34, 44, 78)) is True
    # Violet/purple theme tint
    assert _is_dark_input_bar_surface((42, 30, 68)) is True
    # Bright modal / white page overlay must be rejected
    assert _is_dark_input_bar_surface((220, 224, 230)) is False
    assert _is_dark_input_bar_surface((128, 130, 140)) is False


def test_anchor_hysteresis_eliminates_jitter_within_2px_deadzone() -> None:
    assert ANCHOR_DEADZONE_PX == 2
    assert _apply_anchor_hysteresis(1190, None) == 1190
    # +-1 px or +-2 px fluctuation holds the exact previous coordinate (zero visual jump)
    assert _apply_anchor_hysteresis(1191, 1190) == 1190
    assert _apply_anchor_hysteresis(1188, 1190) == 1190
    # Genuine width change (> 2 px) updates immediately
    assert _apply_anchor_hysteresis(1135, 1190) == 1135


def test_inspect_bar_voice_mode_without_send_across_conversation_widths() -> None:
    screen_w, screen_h = 1360, 768
    y_btn = screen_h - 97

    # Test Narrow (980), Default (1120), Wide (1240), Fill (1334) in Voice Mode (blue halo + NO Send disc)
    orig_fg = hud_pill_overlay._is_antigravity_foreground_or_visible
    orig_cap = hud_pill_overlay._capture_screen_strip
    try:
        hud_pill_overlay._is_antigravity_foreground_or_visible = lambda _hwnd: True

        for true_box_right in (980, 1120, 1240, 1334):
            expected_shield_left = true_box_right - TURBO_SHIELD_OFFSET

            def make_strip(box_r: int, shield_l: int):
                def fake_px(x: int, y: int) -> tuple[int, int, int]:
                    # Outside the input box on the right: dark #101010 outer margin
                    if x > box_r:
                        return (16, 16, 16)
                    # Permanent `shield + Turbo + chevron + mic` cluster (no Send button at all!)
                    if y in (y_btn, y_btn + 3, y_btn + 6):
                        if (
                            shield_l <= x <= shield_l + 10
                            or shield_l + 18 <= x <= shield_l + 48
                            or shield_l + 59 <= x <= shield_l + 63
                            or shield_l + 82 <= x <= shield_l + 90
                        ):
                            return (142, 142, 142)
                    # Voice Mode blue-tinted dark surface (#1C2840 -> B - R = 36 > 8)
                    return (28, 40, 64)

                return fake_px

            hud_pill_overlay._capture_screen_strip = (
                lambda _x0, _y0, _w, _h, br=true_box_right, sl=expected_shield_left: make_strip(br, sl)
            )
            bar_visible, is_send_accent, detected_box_right, surface_hex, turbo_left = (
                _inspect_live_antigravity_bar(screen_w, screen_h, own_hwnd=1)
            )
            assert bar_visible is True
            assert is_send_accent is False
            assert surface_hex == "#1C2840"
            assert turbo_left == expected_shield_left
            assert detected_box_right == expected_shield_left + TURBO_SHIELD_OFFSET
    finally:
        hud_pill_overlay._is_antigravity_foreground_or_visible = orig_fg
        hud_pill_overlay._capture_screen_strip = orig_cap


def test_pill_tier_colors_ban_theme_accent_and_enforce_300k_450k_thresholds() -> None:
    assert TOKEN_WARN_YELLOW_THRESHOLD == 300_000
    assert TOKEN_CRIT_RED_THRESHOLD == 450_000
    assert PILL_COLOR_CALM_GREY == "#8E8E8E"
    assert PILL_COLOR_WARN_YELLOW == "#F59E0B"
    assert PILL_COLOR_CRIT_RED == "#EF4444"

    # Below 300k: ALWAYS calm grey (#8E8E8E), even when is_active_accent=True and theme accent is purple (#8423BF)
    for tok in (0, 234_367, 299_999):
        assert _resolve_pill_tier_color(tok, is_active_accent=False, accent_color="#8423BF") == "#8E8E8E"
        assert _resolve_pill_tier_color(tok, is_active_accent=True, accent_color="#8423BF") == "#8E8E8E"

    # 300k <= total < 450k: ALWAYS warning yellow (#F59E0B)
    for tok in (300_000, 449_999):
        assert _resolve_pill_tier_color(tok, is_active_accent=False, accent_color="#8423BF") == "#F59E0B"
        assert _resolve_pill_tier_color(tok, is_active_accent=True, accent_color="#8423BF") == "#F59E0B"

    # >= 450k: ALWAYS critical red (#EF4444)
    for tok in (450_000, 585_112):
        assert _resolve_pill_tier_color(tok, is_active_accent=False, accent_color="#8423BF") == "#EF4444"
        assert _resolve_pill_tier_color(tok, is_active_accent=True, accent_color="#8423BF") == "#EF4444"


# ---------------------------------------------------------------------------
# E1 Regression Tests — Bug #1 (stale active_state), Bug #2 (API telemetry),
# Bug #3 (turbo_left=null & shrink deadlock)
# ---------------------------------------------------------------------------


def test_bug1_resolve_active_conversation_handles_stale_legacy_and_newer_transcript(tmp_path: Path) -> None:
    from multimodal_token_meter import ACTIVE_STATE_MAX_AGE_MS, resolve_active_conversation_id

    assert ACTIVE_STATE_MAX_AGE_MS == 5000.0

    # Two conversations on disk: conv-old (older) and conv-new (newer)
    old_tr = tmp_path / "conv-old" / ".system_generated" / "logs" / "transcript_full.jsonl"
    new_tr = tmp_path / "conv-new" / ".system_generated" / "logs" / "transcript_full.jsonl"
    old_tr.parent.mkdir(parents=True)
    new_tr.parent.mkdir(parents=True)
    old_tr.write_text('{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT","content":"old"}\n', encoding="utf-8")
    new_tr.write_text('{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT","content":"new"}\n', encoding="utf-8")

    now_s = time.time()
    os.utime(old_tr, (now_s - 60.0, now_s - 60.0))
    os.utime(new_tr, (now_s - 1.0, now_s - 1.0))

    # 1) Legacy active_state.json without updated_at_ms -> MUST NOT stay frozen on conv-old
    assert (
        resolve_active_conversation_id("conv-old", None, brain_root=tmp_path, now_ms=now_s * 1000.0)
        == "conv-new"
    )

    # 2) Stale active_state.json (> 5s old) -> MUST switch to conv-new
    assert (
        resolve_active_conversation_id("conv-old", (now_s - 6.0) * 1000.0, brain_root=tmp_path, now_ms=now_s * 1000.0)
        == "conv-new"
    )

    # 3) Fresh active_state.json (0.2s ago) when NO other transcript moved AFTER the ping -> trust sidecar
    assert (
        resolve_active_conversation_id("conv-old", (now_s - 0.2) * 1000.0, brain_root=tmp_path, now_ms=now_s * 1000.0)
        == "conv-old"
    )

    # 4) Fresh active_state.json (3s ago), BUT conv-new moved 1s ago (more recent than ping) -> switch to conv-new
    assert (
        resolve_active_conversation_id("conv-old", (now_s - 3.0) * 1000.0, brain_root=tmp_path, now_ms=now_s * 1000.0)
        == "conv-new"
    )


def test_bug1_main_mjs_records_updated_at_ms_and_uses_resolver() -> None:
    main_mjs = THIS_DIR / "main.mjs"
    if not main_mjs.is_file():
        main_mjs = PLUGIN_ROOT / "sidecars" / "meter" / "main.mjs"
    if not main_mjs.is_file():
        main_mjs = PLUGIN_ROOT / "src" / "agy_token_meter" / "main.mjs"
    src = main_mjs.read_text(encoding="utf-8")
    assert "updated_at_ms" in src
    assert "resolve_active_conversation_id" in src


def test_bug2_api_telemetry_anchor_exact_plus_post_anchor_delta(tmp_path: Path) -> None:
    from multimodal_token_meter import count_conversation_tokens

    conv_dir = tmp_path / "conv-doc4"
    logs_dir = conv_dir / ".system_generated" / "logs"
    uploads_dir = conv_dir / ".user_uploaded"
    logs_dir.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)

    # Create a tiny valid 100x100 PNG header for multimodal delta test
    img_path = tmp_path / "shot.png"
    img_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + (100).to_bytes(4, "big")
        + (100).to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )

    # Pre-anchor upload (already included in API prompt) vs post-anchor upload (not yet sent)
    old_up = uploads_dir / "already_sent.txt"
    old_up.write_text("This was already sent before step 29 " * 50, encoding="utf-8")
    os.utime(old_up, (1795830000.0, 1795830000.0))  # 2026-11-28T01:40:00Z (before step 29)

    steps_anchor_only = [
        {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "created_at": "2026-11-28T01:40:00Z", "content": "Boot"},
        {
            "step_index": 29,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "created_at": "2026-11-28T01:47:22Z",
            "input_tokens": 3290,
            "cache_read_tokens": 64663,
            "output_tokens": 1946,
            "content": "Cloture FROZEN_ACTIVE",
            "thinking": "Long internal thinking already counted in output_tokens",
        },
    ]
    tr = logs_dir / "transcript_full.jsonl"
    tr.write_text("\n".join(json.dumps(s) for s in steps_anchor_only) + "\n", encoding="utf-8")

    # 1) Exact Gemini API anchor with 0 steps after step 29: 3290 + 64663 + 1946 = 69,899 EXACTLY
    bd_exact = count_conversation_tokens("conv-doc4", brain_root=tmp_path)
    assert bd_exact.total_tokens == 69_899

    # 2) Append post-anchor delta (new user turn + tool result + image media + post-anchor upload)
    delta_user = "Corrige les 3 bugs du tokenizer HUD maintenant."
    delta_tool = "Command exited with code 0: 8 passed"
    steps_with_delta = steps_anchor_only + [
        {
            "step_index": 30,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-11-28T03:24:59Z",
            "content": delta_user,
            "media": [{"uri": f"file:///{img_path.as_posix()}"}],
        },
        {
            "step_index": 31,
            "source": "MODEL",
            "type": "GENERIC",
            "created_at": "2026-11-28T03:25:02Z",
            "content": delta_tool,
        },
    ]
    tr.write_text("\n".join(json.dumps(s) for s in steps_with_delta) + "\n", encoding="utf-8")

    new_up = uploads_dir / "new_staged.txt"
    new_up_text = "Brand new upload dropped after step 29."
    new_up.write_text(new_up_text, encoding="utf-8")
    os.utime(new_up, (1795840000.0, 1795840000.0))  # strictly after 2026-11-28T01:47:22Z

    bd_delta = count_conversation_tokens("conv-doc4", brain_root=tmp_path)
    expected_delta_total = (
        69_899
        + count_text_tokens(delta_user)
        + count_text_tokens(delta_tool)
        + GEMINI_IMAGE_BASE_TOKENS
        + count_text_tokens(new_up_text)
    )
    assert bd_delta.total_tokens == expected_delta_total

    # 3) Meter adds draft_text and extra_files strictly on top of anchored total
    extra_note = tmp_path / "extra.md"
    extra_note.write_text("Extra staged document text", encoding="utf-8")
    draft = "Typing a follow-up question"
    meter = MultimodalTokenMeter(brain_root=tmp_path)
    bd_meter = meter.get_breakdown(conversation_id="conv-doc4", extra_files=[extra_note], draft_text=draft)
    assert bd_meter.total_tokens == expected_delta_total + count_text_tokens("Extra staged document text") + count_text_tokens(draft)


def test_bug3_inspect_bar_detects_turbo_left_at_box_right_1294_tighter_offset_and_wider_gap() -> None:
    screen_w, screen_h = 1360, 768
    y_btn = screen_h - 97

    # Reproduce box_right=1294 with Send disc at 1262..1289, wider 34px gap between chevron and mic,
    # and Turbo shield at 1174..1184 (offset = 120px < old min_offset 129px!).
    orig_fg = hud_pill_overlay._is_antigravity_foreground_or_visible
    orig_cap = hud_pill_overlay._capture_screen_strip
    try:
        hud_pill_overlay._is_antigravity_foreground_or_visible = lambda _hwnd: True

        def fake_px(x: int, y: int) -> tuple[int, int, int]:
            if x > 1294:
                return (16, 16, 16)
            if y in (y_btn + 10, y_btn + 12) and 1262 <= x <= 1289:
                return (142, 142, 142)  # Send disc -> send[0]=1262 -> box_right=1294
            if y in (y_btn, y_btn + 3, y_btn + 6):
                # shield 1174..1184, Turbo 1192..1216, chevron 1222..1226, 34px gap -> mic 1260..1261 (wait: left of send[0]-8=1254)
                if 1174 <= x <= 1184 or 1192 <= x <= 1216 or 1250 <= x <= 1253:
                    return (142, 142, 142)
            return (32, 32, 32)

        hud_pill_overlay._capture_screen_strip = lambda _x0, _y0, _w, _h: fake_px
        bar_visible, _btn_active, detected_box_right, _surf, turbo_left = _inspect_live_antigravity_bar(
            screen_w, screen_h, own_hwnd=1, pill_right_x=1150
        )
        assert bar_visible is True
        assert detected_box_right == 1294
        assert turbo_left == 1174
    finally:
        hud_pill_overlay._is_antigravity_foreground_or_visible = orig_fg
        hud_pill_overlay._capture_screen_strip = orig_cap


def test_bug3_anchor_update_clears_stale_turbo_left_on_bar_shrink_deadlock() -> None:
    from hud_pill_overlay import _update_detected_anchors

    # Bar shrinks left from 1332 -> 1294 while pill was sitting at 1186, causing turbo_left=None.
    # Must clear stale turbo_left (1198 -> None) instead of deadlocking at 1198!
    new_box_right, new_turbo_left, cleared = _update_detected_anchors(
        prev_box_right=1332,
        prev_turbo_left=1198,
        box_right=1294,
        turbo_left=None,
    )
    assert new_box_right == 1294
    assert new_turbo_left is None
    assert cleared is True

    # Conversely, if box_right did NOT move (still 1332) and turbo_left momentarily flickers to None,
    # holding the valid in-range turbo_left=1198 prevents flicker.
    same_box_right, held_turbo_left, cleared2 = _update_detected_anchors(
        prev_box_right=1332,
        prev_turbo_left=1198,
        box_right=1332,
        turbo_left=None,
    )
    assert same_box_right == 1332
    assert held_turbo_left == 1198
    assert cleared2 is False


# ---------------------------------------------------------------------------
# E1 Regression Tests — Focus Lock (`last_user_view_time`) & Sub-5ms Tick Cache
# ---------------------------------------------------------------------------


def test_e1_focus_locks_on_last_user_view_time_when_panel_closed_despite_background_writes(tmp_path: Path) -> None:
    """Panneau fermé (state_updated_at_ms=None ou expiré):
    Fil A consulté (`annotations/A.pbtxt` récent) tandis que B et C écrivent plus récemment dans brain/B et brain/C
    -> resolve_active_conversation_id retourne A (jamais B ni C), ignore archived:true et orphelins sans brain/<id>.
    """
    from multimodal_token_meter import resolve_active_conversation_id

    brain_root = tmp_path / "brain"
    ann_root = tmp_path / "annotations"
    ann_root.mkdir(parents=True)

    now_s = time.time()
    for cid, mtime_offset in (("conv-A", -120.0), ("conv-B", -2.0), ("conv-C", -0.5), ("conv-archived", -1.0)):
        tr = brain_root / cid / ".system_generated" / "logs" / "transcript_full.jsonl"
        tr.parent.mkdir(parents=True)
        tr.write_text(
            f'{{"step_index":0,"source":"USER_EXPLICIT","type":"USER_INPUT","content":"{cid}"}}\n',
            encoding="utf-8",
        )
        os.utime(tr, (now_s + mtime_offset, now_s + mtime_offset))

    # User viewed A at T-10s; B was viewed at T-60s; C was never viewed (or viewed at T-300s)
    t_a = int(now_s - 10)
    t_b = int(now_s - 60)
    t_c = int(now_s - 300)
    t_arch = int(now_s - 1)  # newer view time, but archived:true -> MUST BE IGNORED
    t_orphan = int(now_s)    # newest view time, but no brain/conv-orphan dir -> MUST BE IGNORED

    (ann_root / "conv-A.pbtxt").write_text(
        f'title:"Fil A"  last_user_view_time:{{seconds:{t_a}  nanos:500000000}}\n',
        encoding="utf-8",
    )
    (ann_root / "conv-B.pbtxt").write_text(
        f'title:"Fil B"  last_user_view_time:{{seconds:{t_b}  nanos:100000000}}\n',
        encoding="utf-8",
    )
    (ann_root / "conv-C.pbtxt").write_text(
        f'title:"Fil C"  last_user_view_time:{{seconds:{t_c}  nanos:0}}\n',
        encoding="utf-8",
    )
    (ann_root / "conv-archived.pbtxt").write_text(
        f'title:"Archived"  archived:true  last_user_view_time:{{seconds:{t_arch}  nanos:900000000}}\n',
        encoding="utf-8",
    )
    (ann_root / "conv-orphan.pbtxt").write_text(
        f'title:"Orphan"  last_user_view_time:{{seconds:{t_orphan}  nanos:999000000}}\n',
        encoding="utf-8",
    )

    # 1) Panel closed: state_updated_at_ms=None -> MUST return conv-A (never conv-B or conv-C!)
    assert (
        resolve_active_conversation_id(
            "conv-old", None, brain_root=brain_root, annotations_root=ann_root, now_ms=now_s * 1000.0
        )
        == "conv-A"
    )
    # 2) Panel closed: expired state_updated_at_ms (> 5000ms) -> MUST return conv-A
    assert (
        resolve_active_conversation_id(
            "conv-B", (now_s - 15.0) * 1000.0, brain_root=brain_root, annotations_root=ann_root, now_ms=now_s * 1000.0
        )
        == "conv-A"
    )
    # 3) Panel open & fresh (<= 5000ms) for conv-B, and conv-A's view time (T-10s) is OLDER than ping (T-1s) -> trusts fresh sidecar conv-B
    assert (
        resolve_active_conversation_id(
            "conv-B", (now_s - 1.0) * 1000.0, brain_root=brain_root, annotations_root=ann_root, now_ms=now_s * 1000.0
        )
        == "conv-B"
    )
    # 4) Panel open & fresh (T-4s), BUT conv-A received a newer user click at T-2s -> switches to conv-A
    t_a_newer = int(now_s - 2)
    (ann_root / "conv-A.pbtxt").write_text(
        f'title:"Fil A"  last_user_view_time:{{seconds:{t_a_newer}  nanos:800000000}}\n',
        encoding="utf-8",
    )
    assert (
        resolve_active_conversation_id(
            "conv-B", (now_s - 4.0) * 1000.0, brain_root=brain_root, annotations_root=ann_root, now_ms=now_s * 1000.0
        )
        == "conv-A"
    )


def test_e1_immediate_switch_to_b_when_b_receives_higher_last_user_view_time(tmp_path: Path) -> None:
    """Bascule immédiate vers B dès que annotations/B.pbtxt reçoit un last_user_view_time supérieur."""
    from multimodal_token_meter import resolve_active_conversation_id

    brain_root = tmp_path / "brain"
    ann_root = tmp_path / "annotations"
    ann_root.mkdir(parents=True)

    for cid in ("conv-A", "conv-B"):
        tr = brain_root / cid / ".system_generated" / "logs" / "transcript_full.jsonl"
        tr.parent.mkdir(parents=True)
        tr.write_text('{"step_index":0,"content":"hi"}\n', encoding="utf-8")

    now_s = time.time()
    t1 = int(now_s - 20)
    t2 = int(now_s - 5)
    t3 = int(now_s)

    (ann_root / "conv-A.pbtxt").write_text(
        f'title:"Fil A"  last_user_view_time:{{seconds:{t2}  nanos:100000000}}\n',
        encoding="utf-8",
    )
    (ann_root / "conv-B.pbtxt").write_text(
        f'title:"Fil B"  last_user_view_time:{{seconds:{t1}  nanos:100000000}}\n',
        encoding="utf-8",
    )
    assert (
        resolve_active_conversation_id(None, None, brain_root=brain_root, annotations_root=ann_root, now_ms=now_s * 1000.0)
        == "conv-A"
    )

    # User clicks thread B -> annotations/conv-B.pbtxt updated with higher last_user_view_time
    b_pb = ann_root / "conv-B.pbtxt"
    b_pb.write_text(
        f'title:"Fil B"  last_user_view_time:{{seconds:{t3}  nanos:250000000}}\n',
        encoding="utf-8",
    )
    # Bump mtime_ns explicitly to avoid any coarse filesystem timestamp equality
    st_b = b_pb.stat()
    os.utime(b_pb, ns=(st_b.st_atime_ns, st_b.st_mtime_ns + 1_000_000))

    assert (
        resolve_active_conversation_id(None, None, brain_root=brain_root, annotations_root=ann_root, now_ms=now_s * 1000.0)
        == "conv-B"
    )


def test_e1_tick_tokens_cache_second_call_under_5ms_without_reparsing_jsonl(tmp_path: Path, monkeypatch) -> None:
    """Cache _tick_tokens & MultimodalTokenMeter:
    Second appel sans changement de fichier exécuté en < 5 ms sans re-parser le JSONL.
    """
    import multimodal_token_meter as mtm

    brain_root = tmp_path / "brain"
    ann_root = tmp_path / "annotations"
    ann_root.mkdir(parents=True)

    # Create a large transcript (~500 lines) + uploads dir to make any accidental re-parse obvious
    conv_dir = brain_root / "conv-big"
    logs_dir = conv_dir / ".system_generated" / "logs"
    uploads_dir = conv_dir / ".user_uploaded"
    logs_dir.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)

    lines = [
        json.dumps({"step_index": i, "content": f"Step {i} payload " + ("x" * 180)})
        for i in range(400)
    ]
    lines.append(
        json.dumps(
            {
                "step_index": 400,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-11-28T03:00:00Z",
                "input_tokens": 10000,
                "cache_read_tokens": 160000,
                "output_tokens": 2135,
                "content": "Final answer",
            }
        )
    )
    (logs_dir / "transcript_full.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ann_root / "conv-big.pbtxt").write_text(
        f'title:"Big Thread"  last_user_view_time:{{seconds:{int(time.time())}  nanos:123000000}}\n',
        encoding="utf-8",
    )

    parse_calls = 0
    real_count = mtm.count_conversation_tokens

    def spy_count(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return real_count(*args, **kwargs)

    monkeypatch.setattr(mtm, "count_conversation_tokens", spy_count)

    meter = mtm.MultimodalTokenMeter(brain_root=brain_root, annotations_root=ann_root)

    # First call: cold parse
    cid1 = meter.resolve_conversation_id(None, None)
    assert cid1 == "conv-big"
    bd1 = meter.get_breakdown(conversation_id=cid1)
    assert bd1.total_tokens == 172_135
    assert parse_calls == 1

    # Second call with no file change: MUST NOT re-parse JSONL and MUST execute in < 5 ms
    t0 = time.perf_counter()
    cid2 = meter.resolve_conversation_id(None, None)
    bd2 = meter.get_breakdown(conversation_id=cid2)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert cid2 == "conv-big"
    assert bd2.total_tokens == 172_135
    assert parse_calls == 1, f"Expected 0 re-parses on cached second call, got {parse_calls - 1}"
    assert elapsed_ms < 5.0, f"Expected < 5.0 ms on cached second call, took {elapsed_ms:.3f} ms"

    # Also verify exact cache key includes (conversation_id, transcript_mtime_ns, transcript_size, uploads_dir_mtime_ns)
    cache_entry = meter._cache["conv-big"]
    assert isinstance(cache_entry, tuple) and len(cache_entry) == 4
    cached_mtime_ns, cached_size, cached_uploads_mtime_ns, cached_bd = cache_entry
    assert isinstance(cached_mtime_ns, int) and cached_mtime_ns > 0
    assert isinstance(cached_size, int) and cached_size > 0
    assert isinstance(cached_uploads_mtime_ns, int) and cached_uploads_mtime_ns > 0
    assert cached_bd is bd1


def test_e1_hud_tick_tokens_method_second_call_under_5ms_and_skips_redundant_redraw(tmp_path: Path, monkeypatch) -> None:
    """Exercise StudioTokenPillHUD._tick_tokens() directly (stubbing Tk) to prove:
    - Focus locks on annotations/last_user_view_time
    - Second _tick_tokens() executes in < 5 ms with 0 JSONL re-parses and 0 redundant redraws.
    """
    import multimodal_token_meter as mtm

    brain_root = tmp_path / "brain"
    ann_root = tmp_path / "annotations"
    ann_root.mkdir(parents=True)
    logs_dir = brain_root / "conv-hud" / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "transcript_full.jsonl").write_text(
        json.dumps(
            {
                "step_index": 1,
                "source": "MODEL",
                "input_tokens": 12000,
                "cache_read_tokens": 160000,
                "output_tokens": 4357,
                "content": "OK",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (ann_root / "conv-hud.pbtxt").write_text(
        f'title:"HUD Thread"  last_user_view_time:{{seconds:{int(time.time())}  nanos:100000000}}\n',
        encoding="utf-8",
    )

    parse_calls = 0
    real_count = mtm.count_conversation_tokens

    def spy_count(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return real_count(*args, **kwargs)

    monkeypatch.setattr(mtm, "count_conversation_tokens", spy_count)

    class _DummyRoot:
        def after(self, _ms: int, _cb) -> None:
            pass

    hud = object.__new__(hud_pill_overlay.StudioTokenPillHUD)
    hud._script_path = tmp_path / "missing_script.py"
    hud._script_mtime = 0.0
    hud.state_file = tmp_path / "active_state.json"
    hud.bg_color = "#101010"
    hud.fg_color = "#8E8E8E"
    hud.accent_color = "#8423BF"
    hud._bar_surface_color = "#202020"
    hud._last_conv_id = None
    hud._last_step_count = -1
    hud._last_upload_count = -1
    hud._last_typing_ts = 0.0
    hud._live_draft_chars = 0
    hud._live_pasted_tokens = 0
    hud._base_tokens = 0
    hud._current_label = "0 tokens"
    hud._current_tier_color = PILL_COLOR_CALM_GREY
    hud._is_active_accent = False
    hud._pill_width = 132
    hud._last_drawn_state = None
    hud._shared_state_cache = None
    hud._last_tick_tokens_ms = 0.0
    hud.last_tooltip_text = ""
    hud.root = _DummyRoot()
    hud.meter = mtm.MultimodalTokenMeter(brain_root=brain_root, annotations_root=ann_root)

    redraw_count = 0

    def spy_redraw() -> None:
        nonlocal redraw_count
        total = hud._compute_displayed_total_tokens()
        label = f"{total:,} tokens"
        tier = _resolve_pill_tier_color(total)
        state = (label, tier, hud._bar_surface_color)
        if state == hud._last_drawn_state:
            return
        hud._last_drawn_state = state
        hud._current_label = label
        hud._current_tier_color = tier
        redraw_count += 1

    hud._redraw_pill = spy_redraw  # type: ignore[assignment]

    hud._tick_tokens()
    assert hud._last_conv_id == "conv-hud"
    assert hud._base_tokens == 176_357
    assert parse_calls == 1
    assert redraw_count == 1

    t0 = time.perf_counter()
    hud._tick_tokens()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert hud._base_tokens == 176_357
    assert parse_calls == 1
    assert redraw_count == 1
    assert elapsed_ms < 5.0, f"Second _tick_tokens took {elapsed_ms:.3f} ms"
    assert hud._last_tick_tokens_ms < 5.0


