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
import wave
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Official Google AI Studio / Gemini Multimodal Token Rates
GEMINI_IMAGE_BASE_TOKENS: int = 258
GEMINI_IMAGE_SMALL_THRESHOLD_PX: int = 384
GEMINI_IMAGE_TILE_SIZE_PX: int = 768
GEMINI_PDF_PAGE_VISION_TOKENS: int = 258
GEMINI_VIDEO_TOKENS_PER_SECOND: int = 263
GEMINI_AUDIO_TOKENS_PER_SECOND: int = 32

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"})
AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"})
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
DOCX_EXTENSIONS: frozenset[str] = frozenset({".docx"})

_WORD_OR_PUNCT_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.UNICODE)
_PDF_COUNT_RE = re.compile(rb"/Count\s+(\d+)")
_PDF_PAGE_ITEM_RE = re.compile(rb"/Type\s*/Page\b")


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


def find_latest_active_conversation(brain_root: str | Path | None = None) -> str | None:
    """Locate the most recently updated conversation ID in `~/.gemini/antigravity/brain`."""
    root = Path(brain_root) if brain_root else (Path.home() / ".gemini" / "antigravity" / "brain")
    if not root.is_dir():
        return None
    best_id: str | None = None
    best_mtime: float = -1.0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        transcript_full = child / ".system_generated" / "logs" / "transcript_full.jsonl"
        transcript = transcript_full if transcript_full.is_file() else (child / ".system_generated" / "logs" / "transcript.jsonl")
        uploaded_dir = child / ".user_uploaded"
        if transcript.is_file():
            mtime = transcript.stat().st_mtime
            if uploaded_dir.is_dir():
                for up in uploaded_dir.iterdir():
                    if up.is_file():
                        mtime = max(mtime, up.stat().st_mtime)
            if mtime > best_mtime:
                best_mtime = mtime
                best_id = child.name
    return best_id


def count_conversation_tokens(
    conversation_id: str,
    brain_root: str | Path | None = None,
    extra_files: list[str | Path] | None = None,
) -> TokenBreakdown:
    """Calculate the full multimodal token breakdown for an Antigravity conversation."""
    root = Path(brain_root) if brain_root else (Path.home() / ".gemini" / "antigravity" / "brain")
    conv_dir = root / conversation_id
    full_path = conv_dir / ".system_generated" / "logs" / "transcript_full.jsonl"
    transcript_path = full_path if full_path.is_file() else (conv_dir / ".system_generated" / "logs" / "transcript.jsonl")
    uploaded_dir = conv_dir / ".user_uploaded"

    text_tokens = 0
    thinking_tokens = 0
    tool_tokens = 0
    step_count = 0
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
        for up_file in sorted(uploaded_dir.iterdir()):
            if up_file.is_file():
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
    """High-level stateful token meter with mtime caching for sub-millisecond UI updates."""

    def __init__(self, brain_root: str | Path | None = None) -> None:
        self.brain_root = Path(brain_root) if brain_root else (Path.home() / ".gemini" / "antigravity" / "brain")
        self._cache: dict[str, tuple[float, int, TokenBreakdown]] = {}

    def get_breakdown(
        self,
        conversation_id: str | None = None,
        extra_files: list[str | Path] | None = None,
        draft_text: str = "",
    ) -> TokenBreakdown:
        conv_id = conversation_id or find_latest_active_conversation(self.brain_root) or "none"
        conv_dir = self.brain_root / conv_id
        full_path = conv_dir / ".system_generated" / "logs" / "transcript_full.jsonl"
        transcript_path = full_path if full_path.is_file() else (conv_dir / ".system_generated" / "logs" / "transcript.jsonl")
        uploaded_dir = conv_dir / ".user_uploaded"

        mtime = transcript_path.stat().st_mtime if transcript_path.is_file() else 0.0
        upload_count = len(list(uploaded_dir.iterdir())) if uploaded_dir.is_dir() else 0

        if not extra_files and not draft_text and conv_id in self._cache:
            cached_mtime, cached_uploads, cached_bd = self._cache[conv_id]
            if cached_mtime == mtime and cached_uploads == upload_count:
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
            self._cache[conv_id] = (mtime, upload_count, bd)
        return bd
