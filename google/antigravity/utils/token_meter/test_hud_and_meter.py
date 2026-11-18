"""Automated pytest suite for `agy-studio-token-meter` (multimodal meter + HUD v8 geometry)."""

from __future__ import annotations

from pathlib import Path
import sys

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PLUGIN_ROOT / "sidecars" / "meter", PLUGIN_ROOT / "src" / "agy_token_meter"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from hud_pill_overlay import (  # noqa: E402
    SEND_LEFT_TO_BOX_RIGHT,
    TURBO_SHIELD_OFFSET,
    _find_send_button,
    _find_toolbar_left_edge,
    _is_send_pixel,
)
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
