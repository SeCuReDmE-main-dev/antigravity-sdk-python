"""Win32/Tkinter Rounded HUD Token Pill anchored inside Antigravity's input bar.

v5 Enhancements:
- 4-Tone Master Frame (`celui maitre`) around the 3rd Accent color:
  1) Outer bevel ring (`#1E1E1E`)
  2) Dark moat (`#141414`)
  3) Active 3rd-color Accent ring (`#8423BF` or user mood color) / Idle ring (`#CCCCCC`)
  4) Inner surface (`#101010`) with indicator dot + token counter text.
- Seamless corner blending into the `#202020` input-bar surface (zero white/aliased corner artifacts).
- Dynamic horizontal tracking across ALL `Conversation Width` modes (`Narrow | Default | Wide | Fill`):
  Scans the safe bottom strip of the input box (`y_safe = sh - 77`, strictly below the pill so the pill
  never occludes the scan!) to find `box_right` in real time and anchors the pill cleanly to the left of `Turbo`.
- Never floats on the Windows Desktop (`Progman`/`WorkerW`) or over File Explorer / Chrome:
  Verifies via Win32 `GetForegroundWindow()` / `WindowFromPoint()` that `Antigravity.exe` is visible.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from multimodal_token_meter import MultimodalTokenMeter, count_text_tokens  # noqa: E402

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TOPMOST = 0x00000008
HWND_TOPMOST = -1
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


_SINGLETON_HANDLE: Any = None

# Turbo shield left edge = box_right - 144 (measured live: box_right 1334, shield 1190 @ 1360x768).
TURBO_SHIELD_OFFSET = 144
# Send disc left edge on rows y_btn+10/+12 -> equivalent box_right
# (measured: 1302 -> 1334 with shield 1190; 1275 -> shield 1162; both = box_right - 144/145).
SEND_LEFT_TO_BOX_RIGHT = 32
ANCHOR_DEADZONE_PX = 2

TOKEN_WARN_YELLOW_THRESHOLD = 300_000
TOKEN_CRIT_RED_THRESHOLD = 450_000
PILL_COLOR_CALM_GREY = "#8E8E8E"
PILL_COLOR_WARN_YELLOW = "#F59E0B"
PILL_COLOR_CRIT_RED = "#EF4444"


def _resolve_pill_tier_color(
    total_tokens: int,
    *,
    is_active_accent: bool = False,
    accent_color: str = "#8423BF",
) -> str:
    """Return the deterministic color for both the indicator dot (`•`) and the token text.

    Theme color #3 (`accent_color` / `is_active_accent`) is permanently banned from the pill so
    typing never turns the dot purple. Color depends strictly on total displayed tokens:
    - `< 300,000`        -> `#8E8E8E` (calm secondary grey, idle and typing alike)
    - `300,000..449,999` -> `#F59E0B` (amber/yellow warning)
    - `>= 450,000`       -> `#EF4444` (critical red)
    """
    _ = (is_active_accent, accent_color)
    if total_tokens >= TOKEN_CRIT_RED_THRESHOLD:
        return PILL_COLOR_CRIT_RED
    if total_tokens >= TOKEN_WARN_YELLOW_THRESHOLD:
        return PILL_COLOR_WARN_YELLOW
    return PILL_COLOR_CALM_GREY


def _is_dark_input_bar_surface(rgb: tuple[int, int, int]) -> bool:
    """Return True when `rgb` at `y_safe` is Antigravity's dark input-bar surface.

    Tolerates the blue/violet glow halo cast by Voice Mode (`B > R`, `B <= 115`) as well as
    theme surface tints, while rejecting bright modals, white overlays, or light backdrops.
    """
    r, g, b = rgb
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if luma > 74.0 or min(r, g, b) < 12 or max(r, g, b) > 115:
        return False
    # Neutral dark grey (#141414 .. #3A3A3A)
    if (max(r, g, b) - min(r, g, b)) <= 18 and max(r, g, b) <= 64:
        return True
    # Voice Mode blue/cyan/violet halo or dark theme accent tint on the input bar (blue-dominant or purple-tinted)
    if b >= r and b >= g - 8 and r <= 68 and g <= 82 and b <= 115:
        return True
    return False


def _apply_anchor_hysteresis(
    candidate_x: int | None,
    previous_x: int | None,
    deadzone_px: int = ANCHOR_DEADZONE_PX,
) -> int | None:
    """Stabilize horizontal anchor coordinate with a deadzone (default 2 px) to prevent 1-2 px jitter."""
    if candidate_x is None:
        return previous_x
    if previous_x is None:
        return candidate_x
    if abs(candidate_x - previous_x) <= deadzone_px:
        return previous_x
    return candidate_x

_LAST_INSPECT_ERROR: str | None = None


def _set_inspect_error(msg: str | None) -> None:
    global _LAST_INSPECT_ERROR
    _LAST_INSPECT_ERROR = msg


def _acquire_singleton(wait_s: float = 3.0) -> bool:
    """Allow exactly ONE pill per Windows session (named mutex).

    Duplicates came from the sidecar respawning while an `os.execv` orphan was still alive;
    two pills then read each other's pixels and drift apart. Waits briefly so a restarting
    predecessor can release the mutex first.
    """
    global _SINGLETON_HANDLE
    if sys.platform != "win32":
        return True
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    deadline = time.monotonic() + wait_s
    while True:
        h = k32.CreateMutexW(None, False, "Local\\AGY_Studio_Token_Pill_Singleton")
        if h and ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
            _SINGLETON_HANDLE = h
            return True
        if h:
            k32.CloseHandle(h)
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _enable_dpi_awareness() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def _read_storage_theme_colors() -> tuple[str, str, str]:
    """Fetch user's configured theme colors from Antigravity's `app_storage.json`.
    Uses muted secondary grey (`#8E8E8E`) matching `Turbo` and `Gemini 4 ...` so the pill
    never shines bright white or distracts the user's peripheral vision.
    """
    bg = "#101010"
    fg = "#8E8E8E"
    accent = "#8423BF"
    storage_path = (
        Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        / "Antigravity"
        / "app_storage.json"
    )
    if storage_path.is_file():
        try:
            data = json.loads(storage_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data.get("hubSurfaceBackgroundColor"), str) and data["hubSurfaceBackgroundColor"].startswith("#"):
                bg = data["hubSurfaceBackgroundColor"]
            for key in ("hubSurfaceForegroundColor", "hubForegroundColor", "foregroundColor"):
                val = data.get(key)
                if isinstance(val, str) and val.startswith("#"):
                    fg = val
            for key in ("hubAccentColor", "hubSurfaceAccentColor", "accentColor"):
                val = data.get(key)
                if isinstance(val, str) and val.startswith("#"):
                    accent = val
        except Exception:
            pass
    return (bg, fg, accent)


def _rgb_from_colorref(color_ref: int) -> tuple[int, int, int]:
    return (color_ref & 0xFF, (color_ref >> 8) & 0xFF, (color_ref >> 16) & 0xFF)


def _get_hwnd_process_name(hwnd: int) -> str:
    """Return lowercase executable name (e.g. 'antigravity.exe') for a window handle."""
    if not hwnd or sys.platform != "win32":
        return ""
    try:
        pid = wintypes.DWORD(0)
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""
        hproc = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hproc:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(512)
            size = wintypes.DWORD(512)
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(size)):
                return Path(buf.value).name.lower()
        finally:
            ctypes.windll.kernel32.CloseHandle(hproc)
    except Exception:
        pass
    return ""


def _is_antigravity_foreground_or_visible(own_hwnd: int) -> bool:
    """Return True ONLY when Antigravity is the active foreground window (or user is hovering the pill).
    Immediately returns False when user is on the Windows Desktop (`explorer.exe`), File Explorer, Chrome, etc.
    """
    if sys.platform != "win32":
        return True
    try:
        user32 = ctypes.windll.user32
        fg_hwnd = int(user32.GetForegroundWindow() or 0)
        if not fg_hwnd:
            # Isolated session fallback
            return True
        if fg_hwnd == own_hwnd:
            return True
        # Our own tooltip / Tk windows (same PID) keep the pill visible.
        fg_pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(fg_hwnd, ctypes.byref(fg_pid))
        if fg_pid.value == os.getpid():
            return True
        proc_name = _get_hwnd_process_name(fg_hwnd)
        # EXACT match: the Agent Manager is `antigravity.exe`; the editor is a separate
        # process `antigravity ide.exe` and must NOT show the pill (substring match leaked it there).
        # Unreadable process names (elevated windows) are treated as foreign.
        if proc_name == "antigravity.exe":
            return not user32.IsIconic(fg_hwnd)
        return False
    except Exception:
        return True


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def _capture_screen_strip(x0: int, y0: int, w: int, h: int) -> Any:
    """Grab a screen rectangle with ONE BitBlt and return `get_rgb(x, y)` in SCREEN coordinates.

    Returns None if the capture fails. Out-of-strip reads return black (0, 0, 0).
    """
    if sys.platform != "win32" or w <= 0 or h <= 0:
        return None
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    user32.GetDC.restype = wintypes.HDC
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.BitBlt.argtypes = [wintypes.HDC] + [ctypes.c_int] * 4 + [wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
    gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]

    hdc_screen = user32.GetDC(None)
    if not hdc_screen:
        return None
    hdc_mem = hbm = old = None
    try:
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbm = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        if not hdc_mem or not hbm:
            return None
        old = gdi32.SelectObject(hdc_mem, hbm)
        if not gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x0, y0, 0x00CC0020):  # SRCCOPY
            return None
        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # top-down rows
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0  # BI_RGB
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.SelectObject(hdc_mem, old)
        old = None
        if gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bmi), 0) != h:
            return None
        raw = buf.raw
    finally:
        if old is not None:
            gdi32.SelectObject(hdc_mem, old)
        if hbm:
            gdi32.DeleteObject(hbm)
        if hdc_mem:
            gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(None, hdc_screen)

    def get_rgb(x: int, y: int) -> tuple[int, int, int]:
        lx, ly = x - x0, y - y0
        if not (0 <= lx < w and 0 <= ly < h):
            return (0, 0, 0)
        i = (ly * w + lx) * 4  # BGRA
        return (raw[i + 2], raw[i + 1], raw[i])

    return get_rgb


def _is_send_pixel(rgb: tuple[int, int, int]) -> bool:
    hi, lo = max(rgb), min(rgb)
    # idle disc is #8E8E8E (unfocused) or #CCCCCC (focused); active disc is the accent colour
    return (hi >= 110 and hi - lo < 25) or (hi - lo >= 45 and hi >= 80)


def _find_send_button(
    get_rgb: Any,
    x_from: int,
    x_to: int,
    rows: tuple[int, ...],
    min_width: int = 18,
    max_hole: int = 3,
) -> tuple[int, int] | None:
    """Rightmost bright/accent run >= `min_width` px (holes <= `max_hole` px for the icon strokes).

    The mic glyph is a 1-px column and toolbar text is #8E8E8E (< 150), so neither qualifies.
    Returns `(left, right)` of the Send disc, or None.
    """
    run_right: int | None = None
    run_left: int | None = None
    hole = 0
    for x in range(x_from, x_to - 1, -1):
        if any(_is_send_pixel(get_rgb(x, y)) for y in rows):
            if run_right is None:
                run_right = x
            run_left = x
            hole = 0
        elif run_right is not None:
            hole += 1
            if hole > max_hole:
                if run_left is not None and run_right - run_left + 1 >= min_width:
                    return (run_left, run_right)
                run_right = run_left = None
                hole = 0
    if run_right is not None and run_left is not None and run_right - run_left + 1 >= min_width:
        return (run_left, run_right)
    return None


def _find_toolbar_left_edge(
    get_rgb: Any,
    start_x: int,
    stop_x: int,
    rows: tuple[int, ...],
    surface_rgb: tuple[int, int, int],
    max_gap: int = 30,
    threshold: int = 24,
) -> int | None:
    """Walk LEFT from `start_x` (just left of Send) across mic, chevron, `Turbo` and its shield icon.

    Returns the x of the leftmost toolbar pixel (the shield's left edge) once a gap wider than
    `max_gap` px of plain surface is seen, or when `stop_x` (our own pill's right edge) is reached.
    Native gaps inside the cluster are 10-23 px, so 30 px safely separates it from anything else.
    """
    leftmost: int | None = None
    x = start_x
    while x >= stop_x:  # stop_x = first free pixel right of our pill: readable
        hit = False
        for y in rows:
            r, g, b = get_rgb(x, y)
            if max(abs(r - surface_rgb[0]), abs(g - surface_rgb[1]), abs(b - surface_rgb[2])) > threshold:
                hit = True
                break
        if hit:
            leftmost = x
        elif leftmost is not None and leftmost - x > max_gap:
            break
        x -= 1
    return leftmost


def _inspect_live_antigravity_bar(
    screen_w: int,
    screen_h: int,
    own_hwnd: int,
    pill_right_x: int | None = None,
) -> tuple[bool, bool, int | None, str, int | None]:
    """Inspect the safe bottom strip of Antigravity's input box (`y_safe = screen_h - 77`)
    strictly BELOW our pill (`y = screen_h - 110 .. screen_h - 82`) so the pill NEVER occludes the scan!

    Returns:
      `(bar_visible, is_send_accent_active, detected_box_right_x, bar_surface_hex, turbo_left_x)`
    """
    if sys.platform != "win32":
        return (True, False, None, "#202020", None)
    if not _is_antigravity_foreground_or_visible(own_hwnd):
        return (False, False, None, "#202020", None)

    try:
        # ONE BitBlt of the bar strip per tick. Per-pixel `GetPixel` on the screen DC costs ~33 ms
        # each under DWM (measured: 400 calls = 13.2 s), which made the old ~100-call scan refresh the
        # position only every ~3 s, so the pill visibly ignored Narrow/Default/Wide/Fill changes.
        y_safe = screen_h - 77  # 5px above the input box bottom frame, BELOW our pill
        y_btn = screen_h - 97
        strip_x0 = int(round(screen_w * 0.3))
        strip_y0 = y_btn - 2
        strip = _capture_screen_strip(strip_x0, strip_y0, screen_w - strip_x0, y_safe - strip_y0 + 1)
        if strip is None:
            return (True, False, None, "#202020", None)
        px = strip
        rm, gm, bm = px(int(round(screen_w * 0.56)), y_safe)
        # Isolated session DC returns pure white (255,255,255)
        if rm == 255 and gm == 255 and bm == 255:
            return (True, False, None, "#202020", None)

        # Verify that the middle of the input bar at y_safe is dark (tolerating Voice Mode blue/violet glow halo)
        if not _is_dark_input_bar_surface((rm, gm, bm)):
            # Input bar is not at the bottom (e.g. bright Settings modal or light overlay covering it)
            return (False, False, None, "#202020", None)

        bar_surface_hex = f"#{rm:02X}{gm:02X}{bm:02X}"

        # Scan horizontally from x = 56% of screen_w to the right edge at y_safe to find exact `box_right`.
        # In Voice Mode, the blue halo creates a gentle gradient across the bar, so we allow either close match
        # to the center pixel OR a smooth local step (<= 4 per 6px step) while staying on the dark bar surface.
        start_x = int(round(screen_w * 0.56))
        box_right = start_x
        prev_r, prev_g, prev_b = rm, gm, bm
        for x in range(start_x, screen_w - 8, 6):
            r, g, b = px(x, y_safe)
            close_to_center = abs(r - rm) <= 10 and abs(g - gm) <= 10 and abs(b - bm) <= 14
            smooth_halo_step = (
                _is_dark_input_bar_surface((r, g, b))
                and abs(r - prev_r) <= 4
                and abs(g - prev_g) <= 4
                and abs(b - prev_b) <= 5
                and (r + g + b) >= 54  # reject outer #101010 background outside the input box
            )
            if close_to_center or smooth_halo_step:
                box_right = x
                prev_r, prev_g, prev_b = r, g, b
            elif x > start_x + 40:
                # Check 12px ahead in case of a tiny artifact; if also outside the bar surface, we hit the right border!
                r2, g2, b2 = px(min(screen_w - 4, x + 12), y_safe)
                if abs(r2 - rm) > 10 or abs(b2 - bm) > 14:
                    break
        # Refine box_right to 1px accuracy
        for x in range(box_right, min(screen_w - 4, box_right + 8)):
            r, g, b = px(x, y_safe)
            if abs(r - prev_r) <= 6 and abs(g - prev_g) <= 6 and abs(b - prev_b) <= 8 and (r + g + b) >= 54:
                box_right = x
            else:
                break

        # Re-anchor on the Send disc when present (bright/accent run >= 18 px on rows y_btn+10/+12).
        send = _find_send_button(
            px,
            x_from=min(screen_w - 2, box_right + 40),
            x_to=max(strip_x0, box_right - 240),
            # Below the text baseline only the ~28 px disc reaches (text/mic/chevron are ~12 px tall).
            rows=(y_btn + 10, y_btn + 12),
        )
        if send is not None:
            box_right = send[0] + SEND_LEFT_TO_BOX_RIGHT

        # Inspect whether the Send disc is lit in the 3rd Accent color.
        x_btn = box_right - 26
        br, bg_c, bb = px(x_btn, y_btn)
        sat = max(br, bg_c, bb) - min(br, bg_c, bb)
        is_send_accent = bool(send is not None and sat >= 45 and max(br, bg_c, bb) >= 80)

        # Locate the real left edge of the permanent `shield + Turbo` cluster on the toolbar row.
        # When the input box is empty or in Voice Mode, the round `Send` disc is absent, so we start
        # the leftward scan from `box_right - 14` (when Send is absent) or `send[0] - 8` (when Send is present),
        # ensuring the permanent `shield + Turbo` cluster is ALWAYS captured in Narrow | Default | Wide | Fill.
        stop_x = max(int(round(screen_w * 0.3)), box_right - 420)
        if pill_right_x is not None and stop_x < pill_right_x < box_right:
            stop_x = pill_right_x  # never read our own pill's pixels
        scan_start_x = (send[0] - 8) if send is not None else (box_right - 14)
        turbo_left = _find_toolbar_left_edge(
            px,
            start_x=min(screen_w - 4, max(stop_x, scan_start_x)),
            stop_x=stop_x,
            rows=(y_btn, y_btn + 3, y_btn + 6),
            surface_rgb=(rm, gm, bm),
        )
        # Plausibility window around the measured shield offset (box_right - 144). When Send is absent
        # (empty bar or Voice Mode), the cluster can sit slightly closer to `box_right` (down to ~85 px)
        # or `box_right` can be slightly wider; when `turbo_left` is valid, normalize `box_right` from
        # the permanent `shield + Turbo` anchor so `box_right` never drifts when Send disappears.
        if turbo_left is not None:
            offset = box_right - turbo_left
            min_offset = (TURBO_SHIELD_OFFSET - 15) if send is not None else (TURBO_SHIELD_OFFSET - 60)
            max_offset = (TURBO_SHIELD_OFFSET + 40) if send is not None else (TURBO_SHIELD_OFFSET + 55)
            if not (min_offset <= offset <= max_offset):
                turbo_left = None
            elif send is None:
                box_right = turbo_left + TURBO_SHIELD_OFFSET
        _set_inspect_error(None)
        return (True, is_send_accent, box_right, bar_surface_hex, turbo_left)
    except Exception:
        import traceback
        _set_inspect_error(traceback.format_exc(limit=4))
        return (True, False, None, "#202020", None)


def _draw_rounded_rect(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    fill: str,
    outline: str,
    width: float = 1.0,
) -> None:
    r = int(min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    if fill:
        canvas.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline="")
        canvas.create_rectangle(x1, y1 + r, x2, y2 - r, fill=fill, outline="")
        canvas.create_arc(x1, y1, x1 + 2 * r, y1 + 2 * r, start=90, extent=90, style="pieslice", fill=fill, outline="")
        canvas.create_arc(x2 - 2 * r, y1, x2, y1 + 2 * r, start=0, extent=90, style="pieslice", fill=fill, outline="")
        canvas.create_arc(x1, y2 - 2 * r, x1 + 2 * r, y2, start=180, extent=90, style="pieslice", fill=fill, outline="")
        canvas.create_arc(x2 - 2 * r, y2 - 2 * r, x2, y2, start=270, extent=90, style="pieslice", fill=fill, outline="")

    if outline and width > 0:
        canvas.create_arc(x1, y1, x1 + 2 * r, y1 + 2 * r, start=90, extent=90, style="arc", outline=outline, width=width)
        canvas.create_arc(x2 - 2 * r, y1, x2, y1 + 2 * r, start=0, extent=90, style="arc", outline=outline, width=width)
        canvas.create_arc(x1, y2 - 2 * r, x1 + 2 * r, y2, start=180, extent=90, style="arc", outline=outline, width=width)
        canvas.create_arc(x2 - 2 * r, y2 - 2 * r, x2, y2, start=270, extent=90, style="arc", outline=outline, width=width)
        canvas.create_line(x1 + r, y1, x2 - r, y1, fill=outline, width=width)
        canvas.create_line(x1 + r, y2, x2 - r, y2, fill=outline, width=width)
        canvas.create_line(x1, y1 + r, x1, y2 - r, fill=outline, width=width)
        canvas.create_line(x2, y1 + r, x2, y2 - r, fill=outline, width=width)


class StudioTokenPillHUD:
    """Borderless Google AI Studio style token counter aligned to Antigravity's Turbo row."""

    PILL_LAYOUT_VERSION = 8  # v8: clicks no longer lock the pill (resets the accidental drag lock from v7)
    PILL_HEIGHT = 22
    TEXT_X = 16  # label starts after the indicator dot
    TEXT_PAD_RIGHT = 2
    GAP_BEFORE_TURBO = 12  # px between label end and Turbo shield (native toolbar spacing is 10-23 px)
    TURBO_FALLBACK_FROM_BOX_RIGHT = TURBO_SHIELD_OFFSET  # used when the shield can't be trusted

    def __init__(self) -> None:
        _enable_dpi_awareness()
        self._script_path = Path(__file__).resolve()
        self._script_mtime = self._script_path.stat().st_mtime if self._script_path.is_file() else 0.0
        self.data_dir = Path(
            os.environ.get(
                "ANTIGRAVITY_EXECUTABLE_DATA_DIR",
                str(Path.home() / ".gemini" / "antigravity" / "sidecar_data" / "agy-studio-token-meter"),
            )
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "active_state.json"
        self.offset_file = self.data_dir / "pill_offset.json"

        # Default fallback offsets; horizontal X dynamically tracks `box_right - GAP_LEFT_OF_TURBO - width`
        self.dx_from_right: int = 335
        self.dy_from_bottom: int = 104
        self._user_custom_dragged: bool = False
        self._detected_box_right: int | None = None
        self._detected_turbo_left: int | None = None
        self._bar_surface_color: str = "#202020"
        self._load_offsets()

        self.bg_color, self.fg_color, self.accent_color = _read_storage_theme_colors()
        self._last_typing_ts: float = 0.0
        self._last_upload_count: int = -1
        self._last_step_count: int = -1
        self._live_draft_chars: int = 0
        self._live_pasted_tokens: int = 0
        self._prev_keys_down: set[int] = set()
        self._base_tokens: int = 0
        self._current_label: str = "0 tokens"
        self._is_active_accent: bool = False
        self._pill_width: int = 132

        self.meter = MultimodalTokenMeter()
        self.root = tk.Tk()
        self.root.title("AGY Token Pill")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=self._bar_surface_color)
        self._label_font = tkfont.Font(root=self.root, family="Segoe UI", size=9)

        self.canvas = tk.Canvas(
            self.root,
            width=self._pill_width,
            height=self.PILL_HEIGHT,
            bg=self._bar_surface_color,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack(fill="both", expand=True)

        self.tooltip: tk.Toplevel | None = None
        self.last_tooltip_text: str = ""
        self._drag_start_x: int = 0
        self._drag_start_y: int = 0
        self._dragging: bool = False

        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<Double-Button-1>", self._on_double_click_reset)

        self.root.update_idletasks()
        self._apply_win32_styles()
        self._redraw_pill()
        self._place_pill(visible=True)

    def _get_own_hwnd(self) -> int:
        if sys.platform != "win32":
            return 0
        return int(ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id())

    def _apply_win32_styles(self) -> None:
        if sys.platform != "win32":
            return
        try:
            hwnd = self._get_own_hwnd()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
            )
            # Suppress Windows 11 DWM automatic white/grey border around overrideredirect windows
            # DWMWA_BORDER_COLOR = 34, DWMWA_COLOR_NONE = 0xFFFFFFFE
            color_none = ctypes.c_uint(0xFFFFFFFE)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                34,
                ctypes.byref(color_none),
                ctypes.sizeof(color_none),
            )
        except Exception:
            pass

    def _load_offsets(self) -> None:
        if self.offset_file.is_file():
            try:
                data = json.loads(self.offset_file.read_text(encoding="utf-8"))
                if int(data.get("version", 1)) == self.PILL_LAYOUT_VERSION:
                    self.dx_from_right = int(data.get("dx_from_right", self.dx_from_right))
                    self.dy_from_bottom = int(data.get("dy_from_bottom", self.dy_from_bottom))
                    self._user_custom_dragged = bool(data.get("user_custom_dragged", False))
                    return
            except Exception:
                pass
        self._save_offsets()

    def _save_offsets(self) -> None:
        try:
            self.offset_file.write_text(
                json.dumps(
                    {
                        "version": self.PILL_LAYOUT_VERSION,
                        "dx_from_right": self.dx_from_right,
                        "dy_from_bottom": self.dy_from_bottom,
                        "user_custom_dragged": self._user_custom_dragged,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _read_shared_state(self) -> tuple[str | None, list[str], str]:
        if not self.state_file.is_file():
            return (None, [], "")
        try:
            data: dict[str, Any] = json.loads(self.state_file.read_text(encoding="utf-8"))
            conv_id = data.get("conversation_id") or None
            extra = [str(p) for p in data.get("staged_files", []) if p]
            sidecar_accent = str(data.get("accent_color") or "")
            return (conv_id, extra, sidecar_accent)
        except Exception:
            return (None, [], "")

    def _poll_live_draft_keys(self) -> bool:
        """Poll key edges so typing or pasting in the bar increments token count in real time BEFORE Send."""
        if sys.platform != "win32":
            return False
        changed = False
        try:
            user32 = ctypes.windll.user32
            curr_down: set[int] = set()
            watched = (0x08, 0x0D, 0x11, 0x20, *range(0x30, 0x5B), *range(0xBA, 0xDF))
            for vk in watched:
                if user32.GetAsyncKeyState(vk) & 0x8000:
                    curr_down.add(vk)

            new_presses = curr_down - self._prev_keys_down
            self._prev_keys_down = curr_down

            if not new_presses:
                return False

            ctrl_held = 0x11 in curr_down
            if 0x0D in new_presses:
                self._live_draft_chars = 0
                self._live_pasted_tokens = 0
                self._last_typing_ts = 0.0
                return True

            if ctrl_held and (0x56 in new_presses):
                self._last_typing_ts = time.monotonic()
                try:
                    clip = self.root.clipboard_get()
                    if isinstance(clip, str) and clip:
                        self._live_pasted_tokens += count_text_tokens(clip)
                        changed = True
                except Exception:
                    self._live_draft_chars += 12
                    changed = True
                return changed

            for vk in new_presses:
                if vk == 0x08:  # Backspace
                    if self._live_draft_chars > 0:
                        self._live_draft_chars = max(0, self._live_draft_chars - 1)
                        self._last_typing_ts = time.monotonic()
                        changed = True
                elif vk != 0x11 and not ctrl_held:
                    self._live_draft_chars += 1
                    self._last_typing_ts = time.monotonic()
                    changed = True
        except Exception:
            pass
        return changed

    def _compute_displayed_total_tokens(self) -> int:
        draft_toks = math.ceil(self._live_draft_chars / 3.8) + self._live_pasted_tokens
        return self._base_tokens + draft_toks

    def _compute_displayed_label(self) -> str:
        total = self._compute_displayed_total_tokens()
        return f"{total:,} tokens"

    def _redraw_pill(self) -> None:
        """Render borderless native toolbar item matching `Turbo` and `+ Gemini 4 ...`:
        - No border (`encadré` removed completely so it never steals visual attention).
        - Background matches the `#202020` input-bar surface seamlessly.
        - Theme color #3 (`self.accent_color`) is strictly banned on the pill.
        - Both the indicator dot (`•`) and the token label (`286,659 tokens`) share the exact
          3-tier token threshold color:
          * `< 300,000`        -> `#8E8E8E` (calm secondary grey)
          * `300,000..449,999` -> `#F59E0B` (warning yellow/amber)
          * `>= 450,000`       -> `#EF4444` (critical red)
        """
        total_tokens = self._compute_displayed_total_tokens()
        self._current_label = f"{total_tokens:,} tokens"
        tier_color = _resolve_pill_tier_color(
            total_tokens,
            is_active_accent=self._is_active_accent,
            accent_color=self.accent_color,
        )
        # Exact rendered width (old `len*7+20` estimate left ~25 px of empty canvas before Turbo).
        text_px = self._label_font.measure(self._current_label)
        self._pill_width = int(self.TEXT_X + text_px + self.TEXT_PAD_RIGHT)
        self.root.configure(bg=self._bar_surface_color)
        self.canvas.configure(width=self._pill_width, height=self.PILL_HEIGHT, bg=self._bar_surface_color)
        self.canvas.delete("all")

        h = self.PILL_HEIGHT
        cy = h / 2.0
        self.canvas.create_oval(4, cy - 3.5, 11, cy + 3.5, fill=tier_color, outline=tier_color)
        self.canvas.create_text(
            self.TEXT_X,
            cy,
            text=self._current_label,
            anchor="w",
            fill=tier_color,
            font=self._label_font,
        )

    DRAG_THRESHOLD_PX = 5

    def _on_drag_start(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._dragging = True
        self._drag_moved = False
        self._press_root = (event.x_root, event.y_root)
        self._drag_start_x = event.x_root - self.root.winfo_x()
        self._drag_start_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        px0, py0 = getattr(self, "_press_root", (event.x_root, event.y_root))
        if not self._drag_moved and max(abs(event.x_root - px0), abs(event.y_root - py0)) <= self.DRAG_THRESHOLD_PX:
            return  # a click with hand jitter is NOT a drag
        self._drag_moved = True
        new_x = event.x_root - self._drag_start_x
        new_y = event.y_root - self._drag_start_y
        self.root.geometry(f"+{new_x}+{new_y}")

    def _on_drag_end(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._dragging = False
        if not getattr(self, "_drag_moved", False):
            # Plain click: previously this froze the pill at a fixed screen offset forever
            # (`user_custom_dragged=True`), so it stopped following sidebar / width changes.
            return
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.dx_from_right = max(40, sw - self.root.winfo_x())
        self.dy_from_bottom = max(25, sh - self.root.winfo_y())
        self._user_custom_dragged = True
        self._save_offsets()

    def _on_double_click_reset(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Double-clicking the pill re-locks it to automatic width tracking next to Turbo."""
        self._user_custom_dragged = False
        self.dy_from_bottom = 104
        self._save_offsets()
        self._place_pill(visible=True)

    def _on_enter(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        if self.tooltip is not None or not self.last_tooltip_text:
            return
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.overrideredirect(True)
        self.tooltip.attributes("-topmost", True)
        self.tooltip.configure(bg=self.bg_color)
        lbl = tk.Label(
            self.tooltip,
            text=self.last_tooltip_text,
            fg=self.fg_color,
            bg=self.bg_color,
            font=("Segoe UI", 8),
            padx=8,
            pady=4,
            highlightbackground=self.accent_color,
            highlightthickness=1,
        )
        lbl.pack()
        x = max(10, self.root.winfo_x() - 140)
        y = max(10, self.root.winfo_y() - 32)
        self.tooltip.geometry(f"+{x}+{y}")

    def _on_leave(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        if self.tooltip is not None:
            self.tooltip.destroy()
            self.tooltip = None

    def _place_pill(self, visible: bool = True) -> None:
        if self._dragging:
            return
        if not visible:
            if self.tooltip is not None:
                self.tooltip.destroy()
                self.tooltip = None
            self.root.withdraw()
            return
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        if self._detected_box_right is not None and not self._user_custom_dragged:
            # Anchor primarily to the left edge of the permanent `shield + Turbo` cluster in Narrow | Default | Wide | Fill.
            turbo_left = self._detected_turbo_left
            if turbo_left is not None:
                # Learn the real shield offset so the fallback lands on the SAME spot (no 2-step jump).
                self._learned_turbo_offset = self._detected_box_right - turbo_left
            else:
                offset = getattr(self, "_learned_turbo_offset", self.TURBO_FALLBACK_FROM_BOX_RIGHT)
                turbo_left = self._detected_box_right - offset
            stable_turbo_left = _apply_anchor_hysteresis(
                turbo_left,
                getattr(self, "_stable_turbo_left", None),
                ANCHOR_DEADZONE_PX,
            )
            self._stable_turbo_left = stable_turbo_left
            anchor_left = stable_turbo_left if stable_turbo_left is not None else turbo_left
            x = max(20, anchor_left - self.GAP_BEFORE_TURBO - self._pill_width)
        else:
            x = max(20, sw - self.dx_from_right)
        y = max(20, sh - self.dy_from_bottom)
        geom = (x, y, self._pill_width)
        now = time.monotonic()
        if (
            geom == getattr(self, "_last_geom", None)
            and self.root.winfo_viewable()
            and now - getattr(self, "_last_topmost_ts", 0.0) < 1.0
        ):
            return  # nothing moved: skip redundant geometry/SetWindowPos (caused redraw stutter)
        if geom != getattr(self, "_last_geom", None):
            self._moved_ts = now  # triggers the fast follow cadence
        self._last_geom = geom
        self._last_topmost_ts = now
        self.root.deiconify()
        self.root.geometry(f"{self._pill_width}x{self.PILL_HEIGHT}+{x}+{y}")
        if sys.platform == "win32":
            try:
                hwnd = self._get_own_hwnd()
                if hwnd:
                    ctypes.windll.user32.SetWindowPos(
                        hwnd,
                        HWND_TOPMOST,
                        x,
                        y,
                        self._pill_width,
                        self.PILL_HEIGHT,
                        SWP_NOACTIVATE | SWP_SHOWWINDOW,
                    )
            except Exception:
                pass

    def _tick_tokens(self) -> None:
        try:
            if self._script_path.is_file() and self._script_path.stat().st_mtime != self._script_mtime:
                # Just exit: the sidecar's `watchFile` (main.mjs) respawns us. `os.execv` on Windows
                # spawned an untracked orphan, so the sidecar's respawn produced a duplicate pill.
                self.root.destroy()
                sys.exit(0)
            bg, fg, acc = _read_storage_theme_colors()
            self.bg_color, self.fg_color = bg, fg
            conv_id, staged_files, sidecar_accent = self._read_shared_state()
            if sidecar_accent.startswith("#"):
                self.accent_color = sidecar_accent
            else:
                self.accent_color = acc

            bd = self.meter.get_breakdown(conversation_id=conv_id, extra_files=staged_files)
            if self._last_step_count >= 0 and bd.step_count > self._last_step_count:
                self._live_draft_chars = 0
                self._live_pasted_tokens = 0
            self._last_step_count = bd.step_count

            curr_uploads = len(bd.attachments)
            if self._last_upload_count >= 0 and curr_uploads > self._last_upload_count:
                self._last_typing_ts = time.monotonic()
            self._last_upload_count = curr_uploads

            self._base_tokens = bd.total_tokens
            self.last_tooltip_text = bd.tooltip_summary
            self._redraw_pill()
        except Exception:
            pass
        self.root.after(450, self._tick_tokens)

    def _tick_visual_and_position(self) -> None:
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            keys_changed = self._poll_live_draft_keys()
            own_hwnd = self._get_own_hwnd()
            pill_right: int | None = None
            if self.root.winfo_viewable():
                pill_right = self.root.winfo_x() + self._pill_width
            bar_visible, btn_active, box_right, bar_surface_hex, turbo_left = _inspect_live_antigravity_bar(
                sw, sh, own_hwnd, pill_right_x=pill_right
            )
            if not bar_visible:
                self._place_pill(visible=False)
            if bar_visible:
                if box_right is not None:
                    self._detected_box_right = _apply_anchor_hysteresis(
                        box_right,
                        self._detected_box_right,
                        ANCHOR_DEADZONE_PX,
                    )
                    self._detected_turbo_left = _apply_anchor_hysteresis(
                        turbo_left,
                        self._detected_turbo_left,
                        ANCHOR_DEADZONE_PX,
                    )
                surface_changed = bar_surface_hex != self._bar_surface_color
                self._bar_surface_color = bar_surface_hex
                has_draft = (
                    self._live_draft_chars > 0
                    or self._live_pasted_tokens > 0
                    or (time.monotonic() - self._last_typing_ts) < 6.0
                )
                new_active = btn_active or has_draft
                if new_active != self._is_active_accent or keys_changed or surface_changed:
                    self._is_active_accent = new_active
                    self._redraw_pill()
                self._place_pill(visible=True)
            if bar_visible:
                self._write_debug(
                    bar_visible=bar_visible, box_right=box_right, turbo_left=turbo_left,
                    surface=bar_surface_hex, pill_right_in=pill_right,
                    pill_x=self.root.winfo_x(), pill_w=self._pill_width,
                    dragged=self._user_custom_dragged, inspect_error=_LAST_INSPECT_ERROR,
                )
        except Exception:
            import traceback
            self._write_debug(tick_error=traceback.format_exc(limit=4))
        # Adaptive cadence: ~60 Hz request (real ~35-50 Hz, one inspection = ~17 ms) for 0.8 s
        # after any movement so sidebar slides / width changes are followed smoothly, 90 ms idle.
        fast = (time.monotonic() - getattr(self, "_moved_ts", 0.0)) < 0.8
        self.root.after(16 if fast else 90, self._tick_visual_and_position)

    def _write_debug(self, **fields: Any) -> None:
        """Throttled (1/s) snapshot of the last placement decision -> `pill_debug.json`."""
        now = time.monotonic()
        if now - getattr(self, "_last_debug_ts", 0.0) < 1.0 and "tick_error" not in fields:
            return
        self._last_debug_ts = now
        try:
            fields["ts"] = time.strftime("%H:%M:%S")
            (self.data_dir / "pill_debug.json").write_text(json.dumps(fields, indent=1), encoding="utf-8")
        except Exception:
            pass

    def run(self) -> None:
        self._tick_tokens()
        self._tick_visual_and_position()
        self.root.mainloop()


if __name__ == "__main__":
    if not _acquire_singleton():
        sys.exit(0)
    StudioTokenPillHUD().run()
