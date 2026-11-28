"""Multimodal Token Counter matching Google AI Studio ('19,433 tokens').

Supports real-time token calculation across:
- Conversation transcripts (`transcript.jsonl`: prompt, response, thinking, tool calls)
- Text / source code / JSON / CSV / Markdown files
- PDF documents (`258 tokens / page` native Gemini vision + extracted text tokens)
- DOCX documents (`word/document.xml` OXML text extraction)
- Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`: `258 tokens` base or `768x768` tiling)
- Videos (`.mp4`, `.mov`, `.m4v`, `.webm`, `.mkv`: `263 tokens / second` via ISO-BMFF `mvhd` parser)
- Audio (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`: `32 tokens / second`)
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import time
import wave
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Official Google AI Studio / Gemini Multimodal Token Rates
GEMINI_IMAGE_BASE_TOKENS: int = 258
GEMINI_IMAGE_SMALL_THRESHOLD_PX: int = 384
GEMINI_IMAGE_TILE_SIZE_PX: int = 768
GEMINI_PDF_PAGE_VISION_TOKENS: int = 258
GEMINI_VIDEO_TOKENS_PER_SECOND: int = 263
GEMINI_AUDIO_TOKENS_PER_SECOND: int = 32
ACTIVE_STATE_MAX_AGE_MS: float = 5000.0
CONVERSATION_FOCUS_CACHE_TTL_S: float = 1.5
_SUBAGENT_SOURCES: frozenset[str] = frozenset({"SUBAGENT", "AGENT", "SYSTEM_SUBAGENT"})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"})
AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"})
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
DOCX_EXTENSIONS: frozenset[str] = frozenset({".docx"})

_WORD_OR_PUNCT_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.UNICODE)
_PDF_COUNT_RE = re.compile(rb"/Count\s+(\d+)")
_PDF_PAGE_ITEM_RE = re.compile(rb"/Type\s*/Page\b")
_PBTXT_QUOTED_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_PBTXT_ARCHIVED_RE = re.compile(r"(?:^|[\s{])archived\s*:\s*true\b", re.IGNORECASE)
_PBTXT_VIEW_TIME_RE = re.compile(r"\blast_user_view_time\s*:?\s*\{([^}]*)\}")
_PBTXT_SECONDS_RE = re.compile(r"\bseconds\s*:\s*(\d+)")
_PBTXT_NANOS_RE = re.compile(r"\bnanos\s*:\s*(\d+)")


@dataclass(frozen=True)
class AttachmentTokenInfo:
    """Token breakdown for a single attached or referenced file."""

    path: str
    name: str
    modality: str  # 'text' | 'pdf' | 'document' | 'image' | 'video' | 'audio'
    tokens: int
    size_bytes: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenBreakdown:
    """Cumulative multimodal token breakdown matching Google AI Studio."""

    conversation_id: str
    step_count: int
    text_tokens: int
    thinking_tokens: int
    tool_tokens: int
    document_tokens: int
    image_tokens: int
    video_tokens: int
    audio_tokens: int
    attachments: list[AttachmentTokenInfo] = field(default_factory=list)

    @property
    def media_and_doc_tokens(self) -> int:
        return self.document_tokens + self.image_tokens + self.video_tokens + self.audio_tokens

    @property
    def total_tokens(self) -> int:
        return (
            self.text_tokens
            + self.thinking_tokens
            + self.tool_tokens
            + self.document_tokens
            + self.image_tokens
            + self.video_tokens
            + self.audio_tokens
        )

    @property
    def studio_label(self) -> str:
        """Exact Google AI Studio label format: '19,433 tokens'."""
        return f"{self.total_tokens:,} tokens"

    @property
    def health_level(self) -> str:
        """Empirical context-window curve calibrated by Jean-Sébastien:
        - 0 .. 300k: 'nominal' (context ramp-up)
        - 300k .. 450k: 'peak' (optimal in-context reasoning & synthesis zone)
        - 450k .. 585k: 'warning' (avoid bulk 20-document loads; attention dilution begins)
        - 585k .. 600k+: 'critical' (585k sweet spot + 15k discharge buffer before 700k-785k cliff)
        """
        total = self.total_tokens
        if total >= 585_000:
            return "critical"
        if total >= 450_000:
            return "warning"
        if total >= 300_000:
            return "peak"
        return "nominal"

    @property
    def tooltip_summary(self) -> str:
        zone_map = {
            "nominal": "Nominal (<300k)",
            "peak": "PEAK (300k-450k)",
            "warning": "Softening (450k-585k)",
            "critical": "DISCHARGE (585k+15k)",
        }
        zone_str = zone_map.get(self.health_level, self.health_level)
        return (
            f"{self.studio_label} [{zone_str}] | "
            f"Text: {self.text_tokens:,} · Think: {self.thinking_tokens:,} · "
            f"Tools: {self.tool_tokens:,} · Docs/PDF: {self.document_tokens:,} · "
            f"Img: {self.image_tokens:,} · Video: {self.video_tokens:,} · "
            f"Audio: {self.audio_tokens:,} ({self.step_count} steps)"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_tokens"] = self.total_tokens
        payload["media_and_doc_tokens"] = self.media_and_doc_tokens
        payload["studio_label"] = self.studio_label
        payload["health_level"] = self.health_level
        payload["tooltip_summary"] = self.tooltip_summary
        return payload


def count_text_tokens(text: str) -> int:
    """Deterministic subword token estimator calibrated to Gemini SentencePiece."""
    if not text:
        return 0
    utf8_len = len(text.encode("utf-8", errors="ignore"))
    if utf8_len == 0:
        return 0
    pieces = _WORD_OR_PUNCT_RE.findall(text)
    subword_count = 0
    for token in pieces:
        t_len = len(token)
        if t_len <= 4:
            subword_count += 1
        else:
            subword_count += math.ceil(t_len / 4.2)
    char_based = math.ceil(utf8_len / 3.85)
    return max(1, int(round((subword_count * 0.65) + (char_based * 0.35))))


def parse_image_dimensions(path: Path) -> tuple[int, int]:
    """Read width and height from PNG, JPEG, GIF, or WebP header without external deps."""
    try:
        with path.open("rb") as fh:
            header = fh.read(64)
            if len(header) < 16:
                return (384, 384)
            # PNG
            if header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR":
                w, h = struct.unpack(">II", header[16:24])
                return (max(1, int(w)), max(1, int(h)))
            # GIF
            if header[:6] in (b"GIF87a", b"GIF89a"):
                w, h = struct.unpack("<HH", header[6:10])
                return (max(1, int(w)), max(1, int(h)))
            # WebP
            if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                chunk_type = header[12:16]
                if chunk_type == b"VP8X" and len(header) >= 30:
                    w = 1 + int.from_bytes(header[24:27], "little")
                    h = 1 + int.from_bytes(header[27:30], "little")
                    return (w, h)
                if chunk_type == b"VP8 " and len(header) >= 30:
                    w, h = struct.unpack("<HH", header[26:30])
                    return (w & 0x3FFF, h & 0x3FFF)
            # JPEG
            if header[:2] == b"\xff\xd8":
                fh.seek(2)
                while True:
                    marker_prefix = fh.read(1)
                    if not marker_prefix:
                        break
                    if marker_prefix != b"\xff":
                        continue
                    marker = fh.read(1)
                    while marker == b"\xff":
                        marker = fh.read(1)
                    if not marker:
                        break
                    m_code = marker[0]
                    if m_code in (0xD8, 0xD9) or (0xD0 <= m_code <= 0xD7):
                        continue
                    seg_len_bytes = fh.read(2)
                    if len(seg_len_bytes) < 2:
                        break
                    seg_len = struct.unpack(">H", seg_len_bytes)[0]
                    if m_code in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        sof_data = fh.read(5)
                        if len(sof_data) == 5:
                            h, w = struct.unpack(">HH", sof_data[1:5])
                            return (max(1, int(w)), max(1, int(h)))
                        break
                    if seg_len < 2:
                        break
                    fh.seek(seg_len - 2, os.SEEK_CUR)
    except OSError:
        return (384, 384)
    return (384, 384)


def count_image_tokens_from_dims(width: int, height: int) -> tuple[int, int]:
    """Calculate Gemini image tokens and tile count matching Google AI Studio."""
    if width <= GEMINI_IMAGE_SMALL_THRESHOLD_PX and height <= GEMINI_IMAGE_SMALL_THRESHOLD_PX:
        return (GEMINI_IMAGE_BASE_TOKENS, 1)
    tiles_x = max(1, math.ceil(width / GEMINI_IMAGE_TILE_SIZE_PX))
    tiles_y = max(1, math.ceil(height / GEMINI_IMAGE_TILE_SIZE_PX))
    tile_count = tiles_x * tiles_y
    if tile_count == 1:
        return (GEMINI_IMAGE_BASE_TOKENS, 1)
    return (GEMINI_IMAGE_BASE_TOKENS * (1 + tile_count), tile_count)


def parse_mp4_duration_seconds(path: Path) -> float:
    """Extract exact video/audio duration in seconds from ISO-BMFF (MP4/MOV/M4A) `mvhd` atom."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as fh:
            # Scan top-level and nested moov -> mvhd boxes
            return _scan_bmff_for_mvhd(fh, 0, file_size)
    except OSError:
        return 0.0


def _scan_bmff_for_mvhd(fh: Any, start: int, end: int) -> float:
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        header = fh.read(8)
        if len(header) < 8:
            break
        box_size, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if box_size == 1:
            ext = fh.read(8)
            if len(ext) < 8:
                break
            box_size = struct.unpack(">Q", ext)[0]
            header_size = 16
        elif box_size == 0:
            box_size = end - pos
        if box_size < header_size or pos + box_size > end:
            break

        if box_type == b"moov":
            found = _scan_bmff_for_mvhd(fh, pos + header_size, pos + box_size)
            if found > 0:
                return found
        elif box_type == b"mvhd":
            fh.seek(pos + header_size)
            mvhd_data = fh.read(min(32, box_size - header_size))
            if len(mvhd_data) >= 20:
                version = mvhd_data[0]
                if version == 0 and len(mvhd_data) >= 20:
                    timescale, duration = struct.unpack(">II", mvhd_data[12:20])
                    if timescale > 0:
                        return float(duration) / float(timescale)
                elif version == 1 and len(mvhd_data) >= 28:
                    timescale, duration = struct.unpack(">IQ", mvhd_data[20:32])
                    if timescale > 0:
                        return float(duration) / float(timescale)
        pos += box_size
    return 0.0


def parse_wav_duration_seconds(path: Path) -> float:
    """Extract exact duration from WAV audio files."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return float(frames) / float(rate)
    except Exception:
        return 0.0
    return 0.0


def parse_pdf_info(path: Path) -> tuple[int, int]:
    """Return (page_count, extracted_text_tokens) for a PDF document."""
    try:
        raw = path.read_bytes()
    except OSError:
        return (1, 0)
    page_items = len(_PDF_PAGE_ITEM_RE.findall(raw))
    if page_items > 0:
        page_count = page_items
    else:
        counts = [int(m.group(1)) for m in _PDF_COUNT_RE.finditer(raw) if int(m.group(1)) > 0]
        page_count = max(counts) if counts else 1

    # Also try pypdf if installed for accurate text extraction alongside page vision
    extracted_text_tokens = 0
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path))
        page_count = max(1, len(reader.pages))
        for page in reader.pages[:50]:
            txt = page.extract_text() or ""
            extracted_text_tokens += count_text_tokens(txt)
    except Exception:
        pass
    return (max(1, page_count), extracted_text_tokens)


def parse_docx_text_tokens(path: Path) -> int:
    """Extract text from a Word .docx archive and count its tokens."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return 0
            xml_bytes = zf.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            texts = [elem.text for elem in root.iter() if elem.text]
            return count_text_tokens(" ".join(texts))
    except Exception:
        return 0


def count_file_tokens(path: str | Path) -> AttachmentTokenInfo:
    """Compute exact multimodal token count for any file (text, PDF, DOCX, image, video, audio)."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return AttachmentTokenInfo(
            path=str(file_path),
            name=file_path.name,
            modality="text",
            tokens=0,
            size_bytes=0,
            detail="missing",
        )

    size_bytes = file_path.stat().st_size
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        w, h = parse_image_dimensions(file_path)
        tokens, tiles = count_image_tokens_from_dims(w, h)
        return AttachmentTokenInfo(
            path=str(file_path),
            name=file_path.name,
            modality="image",
            tokens=tokens,
            size_bytes=size_bytes,
            detail=f"{w}x{h}px ({tiles} tile{'s' if tiles > 1 else ''})",
        )

    if ext in VIDEO_EXTENSIONS:
        duration_s = parse_mp4_duration_seconds(file_path)
        if duration_s <= 0:
            # Fallback estimate from bitrate (~250 KB/s typical compressed video)
            duration_s = max(1.0, size_bytes / 250_000.0)
        tokens = max(GEMINI_VIDEO_TOKENS_PER_SECOND, int(round(duration_s * GEMINI_VIDEO_TOKENS_PER_SECOND)))
        return AttachmentTokenInfo(
            path=str(file_path),
            name=file_path.name,
            modality="video",
            tokens=tokens,
            size_bytes=size_bytes,
            detail=f"{duration_s:.1f}s @ {GEMINI_VIDEO_TOKENS_PER_SECOND} tok/s",
        )

    if ext in AUDIO_EXTENSIONS:
        duration_s = parse_wav_duration_seconds(file_path) if ext == ".wav" else parse_mp4_duration_seconds(file_path)
        if duration_s <= 0:
            # Fallback estimate (~16 KB/s for 128 kbps MP3/AAC)
            duration_s = max(1.0, size_bytes / 16_000.0)
        tokens = max(GEMINI_AUDIO_TOKENS_PER_SECOND, int(round(duration_s * GEMINI_AUDIO_TOKENS_PER_SECOND)))
        return AttachmentTokenInfo(
            path=str(file_path),
            name=file_path.name,
            modality="audio",
            tokens=tokens,
            size_bytes=size_bytes,
            detail=f"{duration_s:.1f}s @ {GEMINI_AUDIO_TOKENS_PER_SECOND} tok/s",
        )

    if ext in PDF_EXTENSIONS:
        pages, text_toks = parse_pdf_info(file_path)
        vision_toks = pages * GEMINI_PDF_PAGE_VISION_TOKENS
        total_pdf_toks = vision_toks + text_toks
        return AttachmentTokenInfo(
            path=str(file_path),
            name=file_path.name,
            modality="pdf",
            tokens=total_pdf_toks,
            size_bytes=size_bytes,
            detail=f"{pages} page{'s' if pages > 1 else ''} ({vision_toks:,} vision + {text_toks:,} text)",
        )

    if ext in DOCX_EXTENSIONS:
        tokens = max(1, parse_docx_text_tokens(file_path))
        return AttachmentTokenInfo(
            path=str(file_path),
            name=file_path.name,
            modality="document",
            tokens=tokens,
            size_bytes=size_bytes,
            detail="DOCX OXML text",
        )

    # Default: text / code / markdown / JSON / CSV
    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        raw_text = ""
    tokens = count_text_tokens(raw_text)
    return AttachmentTokenInfo(
        path=str(file_path),
        name=file_path.name,
        modality="text",
        tokens=tokens,
        size_bytes=size_bytes,
        detail=f"{len(raw_text):,} chars",
    )


def _safe_nonneg_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and math.isfinite(value):
        return max(0, int(value))
    if isinstance(value, str) and value.strip().isdigit():
        return max(0, int(value.strip()))
    return 0


def _parse_iso_epoch(raw_ts: Any) -> float | None:
    if not isinstance(raw_ts, str) or not raw_ts.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw_ts.strip().replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return None


def _is_subagent_transcript(transcript_path: Path) -> bool:
    """Return True only when the transcript explicitly declares a subagent source."""
    try:
        with transcript_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                first = json.loads(stripped)
                if not isinstance(first, dict):
                    return False
                if first.get("is_subagent") is True:
                    return True
                src = str(first.get("source") or "").strip().upper()
                return src in _SUBAGENT_SOURCES
    except (OSError, json.JSONDecodeError):
        return False
    return False


def find_latest_active_conversation_with_mtime(
    brain_root: str | Path | None = None,
) -> tuple[str | None, float]:
    """Locate `(conversation_id, latest_mtime_s)` for the most recently active conversation.

    Sorts by `mtime` descending before reading the first line so only the newest candidate(s)
    are opened on disk (1 file open in the common case instead of hundreds).
    """
    root = Path(brain_root) if brain_root else (Path.home() / ".gemini" / "antigravity" / "brain")
    if not root.is_dir():
        return (None, -1.0)
    candidates: list[tuple[float, str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        transcript_full = child / ".system_generated" / "logs" / "transcript_full.jsonl"
        transcript = (
            transcript_full
            if transcript_full.is_file()
            else (child / ".system_generated" / "logs" / "transcript.jsonl")
        )
        if not transcript.is_file():
            continue
        try:
            mtime = transcript.stat().st_mtime
        except OSError:
            continue
        uploaded_dir = child / ".user_uploaded"
        if uploaded_dir.is_dir():
            for up in uploaded_dir.iterdir():
                if up.is_file():
                    try:
                        mtime = max(mtime, up.stat().st_mtime)
                    except OSError:
                        continue
        candidates.append((mtime, child.name, transcript))
    if not candidates:
        return (None, -1.0)
    candidates.sort(key=lambda item: item[0], reverse=True)
    for mtime, conv_id, transcript in candidates:
        if not _is_subagent_transcript(transcript):
            return (conv_id, mtime)
    top_mtime, top_id, _ = candidates[0]
    return (top_id, top_mtime)


def find_latest_active_conversation(brain_root: str | Path | None = None) -> str | None:
    """Locate the most recently updated conversation ID in `~/.gemini/antigravity/brain`."""
    conv_id, _mtime = find_latest_active_conversation_with_mtime(brain_root)
    return conv_id


def _resolve_brain_and_annotations_roots(
    brain_root: str | Path | None = None,
    annotations_root: str | Path | None = None,
) -> tuple[Path, Path]:
    root = Path(brain_root) if brain_root else (Path.home() / ".gemini" / "antigravity" / "brain")
    ann = Path(annotations_root) if annotations_root else (root.parent / "annotations")
    return (root, ann)


def _parse_annotation_pbtxt(raw_text: str) -> tuple[bool, int | None]:
    """Parse read-only annotation `.pbtxt` content into `(is_archived, last_user_view_time_ns)`."""
    if not raw_text:
        return (False, None)
    # Strip quoted strings (e.g. titles) so keywords inside a title never trigger false positives.
    unquoted = _PBTXT_QUOTED_RE.sub('""', raw_text)
    if _PBTXT_ARCHIVED_RE.search(unquoted):
        return (True, None)
    m_view = _PBTXT_VIEW_TIME_RE.search(unquoted)
    if not m_view:
        return (False, None)
    block = m_view.group(1)
    m_sec = _PBTXT_SECONDS_RE.search(block)
    if not m_sec:
        return (False, None)
    seconds = int(m_sec.group(1))
    if seconds <= 0:
        return (False, None)
    m_nanos = _PBTXT_NANOS_RE.search(block)
    nanos = int(m_nanos.group(1)) if m_nanos else 0
    return (False, seconds * 1_000_000_000 + nanos)


class _AnnotationsFocusScanner:
    """Strictly READ-ONLY incremental scanner over `~/.gemini/antigravity/annotations/*.pbtxt`.

    Uses `os.scandir()` (zero extra stat syscalls on Windows NTFS) and caches each `.pbtxt`
    parse result by `(mtime_ns, size)` so unchanged files are never re-opened.
    """

    def __init__(self) -> None:
        # conv_id -> (mtime_ns, size, is_archived, view_time_ns)
        self._entries: dict[str, tuple[int, int, bool, int | None]] = {}
        self._last_result: tuple[str | None, float] | None = None
        self._last_brain_key: str = ""

    def find_latest_viewed(self, brain_root: Path, ann_root: Path) -> tuple[str | None, float]:
        if not ann_root.is_dir():
            return (None, -1.0)
        brain_key = os.path.normcase(os.path.abspath(brain_root))
        seen_count = 0
        seen_ids: set[str] | None = None
        any_changed = brain_key != self._last_brain_key or self._last_result is None
        try:
            with os.scandir(ann_root) as it:
                for entry in it:
                    name = entry.name
                    if not name.endswith(".pbtxt"):
                        continue
                    conv_id = name[:-6]
                    if not conv_id:
                        continue
                    try:
                        st = entry.stat()
                    except OSError:
                        continue
                    seen_count += 1
                    mtime_ns = st.st_mtime_ns
                    size = st.st_size
                    cached = self._entries.get(conv_id)
                    if cached is None or cached[0] != mtime_ns or cached[1] != size:
                        try:
                            raw = Path(entry.path).read_text(encoding="utf-8", errors="ignore")
                        except OSError:
                            continue
                        is_archived, view_time_ns = _parse_annotation_pbtxt(raw)
                        self._entries[conv_id] = (mtime_ns, size, is_archived, view_time_ns)
                        any_changed = True
                    if seen_ids is not None:
                        seen_ids.add(conv_id)
        except OSError:
            return (None, -1.0)

        if seen_count != len(self._entries):
            # A file was deleted; rebuild seen set and prune
            try:
                live_ids = {
                    e.name[:-6]
                    for e in os.scandir(ann_root)
                    if e.name.endswith(".pbtxt") and len(e.name) > 6
                }
            except OSError:
                live_ids = set()
            for stale_id in list(self._entries):
                if stale_id not in live_ids:
                    del self._entries[stale_id]
                    any_changed = True

        if not any_changed and self._last_result is not None:
            return self._last_result

        candidates = [
            (view_time_ns, mtime_ns, conv_id)
            for conv_id, (mtime_ns, _size, is_archived, view_time_ns) in self._entries.items()
            if not is_archived and view_time_ns is not None
        ]
        self._last_brain_key = brain_key
        if not candidates:
            self._last_result = (None, -1.0)
            return self._last_result

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        for view_time_ns, _mtime_ns, conv_id in candidates:
            if (brain_root / conv_id).is_dir():
                self._last_result = (conv_id, view_time_ns / 1_000_000.0)
                return self._last_result
        self._last_result = (None, -1.0)
        return self._last_result


_ANNOTATION_SCANNERS: dict[str, _AnnotationsFocusScanner] = {}


def find_latest_viewed_conversation(
    brain_root: str | Path | None = None,
    annotations_root: str | Path | None = None,
) -> tuple[str | None, float]:
    """Return `(conversation_id, last_user_view_time_ms)` for the most recently viewed non-archived
    conversation that has a `brain/<id>` directory (strictly READ-ONLY).
    """
    b_root, a_root = _resolve_brain_and_annotations_roots(brain_root, annotations_root)
    if not a_root.is_dir():
        return (None, -1.0)
    key = os.path.normcase(os.path.abspath(a_root))
    scanner = _ANNOTATION_SCANNERS.get(key)
    if scanner is None:
        scanner = _AnnotationsFocusScanner()
        _ANNOTATION_SCANNERS[key] = scanner
    return scanner.find_latest_viewed(b_root, a_root)


def resolve_active_conversation_id(
    state_conv_id: str | None,
    state_updated_at_ms: int | float | None,
    brain_root: str | Path | None = None,
    *,
    annotations_root: str | Path | None = None,
    now_ms: float | None = None,
    max_age_ms: float = ACTIVE_STATE_MAX_AGE_MS,
) -> str | None:
    """Resolve the conversation currently viewed by the user (strictly READ-ONLY).

    Priority order:
    1. `active_state.json` ONLY when `state_updated_at_ms` is present and fresh (`<= max_age_ms`)
       AND no other conversation received a more recent user UI click (`last_user_view_time`).
    2. Primary signal when sidecar panel is closed: the non-archived conversation with the highest
       `last_user_view_time` in `annotations/*.pbtxt` that has a `brain/<id>` directory (immune to
       background agents writing to other transcripts).
    3. Final fallback ONLY when `annotations/` is absent or yields no valid viewed conversation:
       `find_latest_active_conversation_with_mtime(brain_root)`.
    """
    now = float(now_ms) if now_ms is not None else (time.time() * 1000.0)
    b_root, a_root = _resolve_brain_and_annotations_roots(brain_root, annotations_root)

    clean_state_id = str(state_conv_id).strip() if state_conv_id else ""
    updated_ms: float | None = None
    if isinstance(state_updated_at_ms, (int, float)) and not isinstance(state_updated_at_ms, bool):
        if math.isfinite(state_updated_at_ms) and state_updated_at_ms > 0:
            updated_ms = float(state_updated_at_ms)
    is_fresh = bool(clean_state_id) and updated_ms is not None and (now - updated_ms) <= max_age_ms

    viewed_id, viewed_time_ms = find_latest_viewed_conversation(b_root, a_root)

    if is_fresh:
        if viewed_id is not None:
            if viewed_id != clean_state_id and updated_ms is not None and viewed_time_ms > updated_ms:
                return viewed_id
            return clean_state_id
        latest_id, latest_mtime_s = find_latest_active_conversation_with_mtime(b_root)
        if (
            latest_id is not None
            and latest_id != clean_state_id
            and updated_ms is not None
            and latest_mtime_s > (updated_ms / 1000.0)
        ):
            return latest_id
        return clean_state_id

    if viewed_id is not None:
        return viewed_id

    latest_id, _latest_mtime_s = find_latest_active_conversation_with_mtime(b_root)
    return latest_id or (clean_state_id or None)


def count_conversation_tokens(
    conversation_id: str,
    brain_root: str | Path | None = None,
    extra_files: list[str | Path] | None = None,
) -> TokenBreakdown:
    """Calculate the full multimodal token breakdown for an Antigravity conversation.

    When the transcript contains at least one model step with Gemini API telemetry
    (`input_tokens + cache_read_tokens > 0`), uses the last measured
    `input_tokens + cache_read_tokens + output_tokens` as the exact server-measured baseline
    and adds ONLY the post-anchor delta (subsequent steps, newly uploaded files, and `extra_files`).
    Falls back to pure heuristic counting when a conversation has 0 measured model responses.
    """
    root = Path(brain_root) if brain_root else (Path.home() / ".gemini" / "antigravity" / "brain")
    conv_dir = root / conversation_id
    full_path = conv_dir / ".system_generated" / "logs" / "transcript_full.jsonl"
    transcript_path = full_path if full_path.is_file() else (conv_dir / ".system_generated" / "logs" / "transcript.jsonl")
    uploaded_dir = conv_dir / ".user_uploaded"

    text_tokens = 0
    thinking_tokens = 0
    tool_tokens = 0
    step_count = 0
    has_api_anchor = False
    last_anchor_ts: float | None = None
    seen_files: set[str] = set()
    attachments: list[AttachmentTokenInfo] = []

    if transcript_path.is_file():
        try:
            with transcript_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    step_count += 1
                    try:
                        step = json.loads(line)
                    except json.JSONDecodeError:
                        text_tokens += count_text_tokens(line)
                        continue
                    if not isinstance(step, dict):
                        text_tokens += count_text_tokens(json.dumps(step, ensure_ascii=False))
                        continue

                    in_tok = _safe_nonneg_int(step.get("input_tokens"))
                    cache_tok = _safe_nonneg_int(step.get("cache_read_tokens"))
                    out_tok = _safe_nonneg_int(step.get("output_tokens"))
                    if (in_tok + cache_tok) > 0:
                        has_api_anchor = True
                        text_tokens = in_tok + cache_tok + out_tok
                        thinking_tokens = 0
                        tool_tokens = 0
                        attachments = []
                        parsed_ts = _parse_iso_epoch(step.get("created_at"))
                        if parsed_ts is not None:
                            last_anchor_ts = parsed_ts
                        media_list = step.get("media")
                        if isinstance(media_list, list):
                            for m_item in media_list:
                                if isinstance(m_item, dict) and m_item.get("uri"):
                                    uri_str = str(m_item["uri"]).replace("file:///", "").replace("file://", "")
                                    m_path = Path(uri_str)
                                    norm_key = str(m_path.resolve()) if m_path.exists() else uri_str
                                    seen_files.add(norm_key)
                        continue

                    content = step.get("content")
                    if isinstance(content, str) and content:
                        text_tokens += count_text_tokens(content)
                    elif content is not None:
                        text_tokens += count_text_tokens(json.dumps(content, ensure_ascii=False))

                    thinking = step.get("thinking")
                    if isinstance(thinking, str) and thinking:
                        thinking_tokens += count_text_tokens(thinking)

                    tool_calls = step.get("tool_calls")
                    if tool_calls:
                        tool_tokens += count_text_tokens(json.dumps(tool_calls, ensure_ascii=False))

                    media_list = step.get("media")
                    if isinstance(media_list, list):
                        for m_item in media_list:
                            if isinstance(m_item, dict) and m_item.get("uri"):
                                uri_str = str(m_item["uri"]).replace("file:///", "").replace("file://", "")
                                m_path = Path(uri_str)
                                norm_key = str(m_path.resolve()) if m_path.exists() else uri_str
                                if norm_key not in seen_files and m_path.is_file():
                                    seen_files.add(norm_key)
                                    attachments.append(count_file_tokens(m_path))
        except OSError:
            pass

    if uploaded_dir.is_dir():
        anchor_cutoff_ts: float | None = None
        if has_api_anchor:
            if last_anchor_ts is not None:
                anchor_cutoff_ts = last_anchor_ts
            elif transcript_path.is_file():
                try:
                    anchor_cutoff_ts = transcript_path.stat().st_mtime
                except OSError:
                    anchor_cutoff_ts = None
        for up_file in sorted(uploaded_dir.iterdir()):
            if up_file.is_file():
                if anchor_cutoff_ts is not None:
                    try:
                        if up_file.stat().st_mtime <= anchor_cutoff_ts:
                            continue
                    except OSError:
                        continue
                norm_key = str(up_file.resolve())
                if norm_key not in seen_files:
                    seen_files.add(norm_key)
                    attachments.append(count_file_tokens(up_file))

    if extra_files:
        for raw_extra in extra_files:
            ex_path = Path(raw_extra)
            if ex_path.is_file():
                norm_key = str(ex_path.resolve())
                if norm_key not in seen_files:
                    seen_files.add(norm_key)
                    attachments.append(count_file_tokens(ex_path))

    doc_tokens = sum(a.tokens for a in attachments if a.modality in ("pdf", "document", "text"))
    img_tokens = sum(a.tokens for a in attachments if a.modality == "image")
    vid_tokens = sum(a.tokens for a in attachments if a.modality == "video")
    aud_tokens = sum(a.tokens for a in attachments if a.modality == "audio")

    return TokenBreakdown(
        conversation_id=conversation_id,
        step_count=step_count,
        text_tokens=text_tokens,
        thinking_tokens=thinking_tokens,
        tool_tokens=tool_tokens,
        document_tokens=doc_tokens,
        image_tokens=img_tokens,
        video_tokens=vid_tokens,
        audio_tokens=aud_tokens,
        attachments=attachments,
    )


class MultimodalTokenMeter:
    """High-level stateful token meter with ns-mtime/size caching for sub-millisecond UI updates."""

    def __init__(
        self,
        brain_root: str | Path | None = None,
        annotations_root: str | Path | None = None,
        *,
        focus_cache_ttl_s: float = CONVERSATION_FOCUS_CACHE_TTL_S,
    ) -> None:
        self.brain_root, self.annotations_root = _resolve_brain_and_annotations_roots(brain_root, annotations_root)
        self.focus_cache_ttl_s = max(0.0, float(focus_cache_ttl_s))
        # conv_id -> (transcript_mtime_ns, transcript_size, uploads_dir_mtime_ns, TokenBreakdown)
        self._cache: dict[str, tuple[int, int, int, TokenBreakdown]] = {}
        # (monotonic_ts, state_conv_id, state_updated_at_ms, resolved_conv_id)
        self._focus_cache: tuple[float, str | None, float | None, str | None] | None = None

    def resolve_conversation_id(
        self,
        state_conv_id: str | None = None,
        state_updated_at_ms: int | float | None = None,
        *,
        now_ms: float | None = None,
        max_age_ms: float = ACTIVE_STATE_MAX_AGE_MS,
        force_refresh: bool = False,
    ) -> str | None:
        """Resolve the focused conversation ID with a 1.5 s TTL cache to prevent Tk thread stalls."""
        now_mono = time.monotonic()
        norm_state_id = str(state_conv_id).strip() if state_conv_id else None
        norm_updated: float | None = (
            float(state_updated_at_ms)
            if isinstance(state_updated_at_ms, (int, float))
            and not isinstance(state_updated_at_ms, bool)
            and math.isfinite(state_updated_at_ms)
            else None
        )
        if not force_refresh and self._focus_cache is not None and self.focus_cache_ttl_s > 0.0:
            cached_mono, cached_state_id, cached_updated, cached_resolved = self._focus_cache
            if (
                (now_mono - cached_mono) < self.focus_cache_ttl_s
                and cached_state_id == norm_state_id
                and cached_updated == norm_updated
            ):
                return cached_resolved

        resolved = resolve_active_conversation_id(
            norm_state_id,
            norm_updated,
            brain_root=self.brain_root,
            annotations_root=self.annotations_root,
            now_ms=now_ms,
            max_age_ms=max_age_ms,
        )
        self._focus_cache = (now_mono, norm_state_id, norm_updated, resolved)
        return resolved

    def get_breakdown(
        self,
        conversation_id: str | None = None,
        extra_files: list[str | Path] | None = None,
        draft_text: str = "",
    ) -> TokenBreakdown:
        conv_id = conversation_id or self.resolve_conversation_id(None, None) or "none"
        conv_dir = self.brain_root / conv_id
        logs_dir = conv_dir / ".system_generated" / "logs"
        full_path = logs_dir / "transcript_full.jsonl"
        uploaded_dir = conv_dir / ".user_uploaded"

        transcript_mtime_ns = 0
        transcript_size = 0
        try:
            st = full_path.stat()
            transcript_mtime_ns = st.st_mtime_ns
            transcript_size = st.st_size
        except OSError:
            try:
                st = (logs_dir / "transcript.jsonl").stat()
                transcript_mtime_ns = st.st_mtime_ns
                transcript_size = st.st_size
            except OSError:
                pass

        try:
            uploads_dir_mtime_ns = uploaded_dir.stat().st_mtime_ns
        except OSError:
            uploads_dir_mtime_ns = 0

        if not extra_files and not draft_text and conv_id in self._cache:
            cached_mtime_ns, cached_size, cached_uploads_mtime_ns, cached_bd = self._cache[conv_id]
            if (
                cached_mtime_ns == transcript_mtime_ns
                and cached_size == transcript_size
                and cached_uploads_mtime_ns == uploads_dir_mtime_ns
            ):
                return cached_bd

        bd = count_conversation_tokens(conv_id, brain_root=self.brain_root, extra_files=extra_files)
        if draft_text:
            draft_toks = count_text_tokens(draft_text)
            bd = TokenBreakdown(
                conversation_id=bd.conversation_id,
                step_count=bd.step_count,
                text_tokens=bd.text_tokens + draft_toks,
                thinking_tokens=bd.thinking_tokens,
                tool_tokens=bd.tool_tokens,
                document_tokens=bd.document_tokens,
                image_tokens=bd.image_tokens,
                video_tokens=bd.video_tokens,
                audio_tokens=bd.audio_tokens,
                attachments=bd.attachments,
            )

        if not extra_files and not draft_text:
            self._cache[conv_id] = (transcript_mtime_ns, transcript_size, uploads_dir_mtime_ns, bd)
        return bd
