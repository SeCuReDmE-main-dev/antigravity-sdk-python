"""Automated pytest suite for `agy-studio-token-meter` (multimodal meter + HUD v8 geometry)."""

from __future__ import annotations

from pathlib import Path
import sys

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
