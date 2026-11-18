"""Open-Source SDK Observability Extension for `google-antigravity/antigravity-sdk-python`.

Bridges the gap in `agent.conversation.total_usage` (which only reports post-hoc API usage)
by providing pre-flight multimodal token counting (text, PDF, DOCX, images, video, audio)
mirroring Google AI Studio's live token counter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .multimodal_token_meter import (
    TokenBreakdown,
    count_conversation_tokens,
    count_file_tokens,
    count_text_tokens,
)


def estimate_multimodal_tokens(
    prompt: str = "",
    attachments: list[str | Path] | None = None,
    conversation_id: str | None = None,
    brain_root: str | Path | None = None,
) -> TokenBreakdown:
    """Compute pre-flight multimodal token count for an SDK prompt + attached files."""
    if conversation_id:
        base = count_conversation_tokens(
            conversation_id=conversation_id,
            brain_root=brain_root,
            extra_files=attachments,
        )
        prompt_toks = count_text_tokens(prompt)
        return TokenBreakdown(
            conversation_id=base.conversation_id,
            step_count=base.step_count,
            text_tokens=base.text_tokens + prompt_toks,
            thinking_tokens=base.thinking_tokens,
            tool_tokens=base.tool_tokens,
            document_tokens=base.document_tokens,
            image_tokens=base.image_tokens,
            video_tokens=base.video_tokens,
            audio_tokens=base.audio_tokens,
            attachments=base.attachments,
        )

    infos = [count_file_tokens(p) for p in (attachments or [])]
    return TokenBreakdown(
        conversation_id="sdk-preflight",
        step_count=0,
        text_tokens=count_text_tokens(prompt),
        thinking_tokens=0,
        tool_tokens=0,
        document_tokens=sum(i.tokens for i in infos if i.modality in ("pdf", "document", "text")),
        image_tokens=sum(i.tokens for i in infos if i.modality == "image"),
        video_tokens=sum(i.tokens for i in infos if i.modality == "video"),
        audio_tokens=sum(i.tokens for i in infos if i.modality == "audio"),
        attachments=infos,
    )


class PreFlightTokenGuardHook:
    """SDK-compatible pre-turn hook enforcing a multimodal token budget before model invocation."""

    def __init__(
        self,
        max_total_tokens: int = 250_000,
        on_update: Callable[[TokenBreakdown], None] | None = None,
    ) -> None:
        if max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        self.max_total_tokens = max_total_tokens
        self.on_update = on_update
        self.last_breakdown: TokenBreakdown | None = None

    def inspect_turn(
        self,
        prompt: str = "",
        attachments: list[str | Path] | None = None,
        conversation_id: str | None = None,
        brain_root: str | Path | None = None,
    ) -> TokenBreakdown:
        breakdown = estimate_multimodal_tokens(
            prompt=prompt,
            attachments=attachments,
            conversation_id=conversation_id,
            brain_root=brain_root,
        )
        self.last_breakdown = breakdown
        if self.on_update is not None:
            self.on_update(breakdown)
        if breakdown.total_tokens > self.max_total_tokens:
            raise RuntimeError(
                f"PreFlightTokenGuard: context size ({breakdown.studio_label}) exceeds "
                f"budget of {self.max_total_tokens:,} tokens."
            )
        return breakdown


def attach_multimodal_counter_to_conversation(conversation: Any, brain_root: str | Path | None = None) -> Any:
    """Dynamically attach `.count_multimodal_tokens()` to an `agent.conversation` instance."""
    conv_id = getattr(conversation, "id", None) or getattr(conversation, "conversation_id", None)

    def _count_multimodal_tokens(prompt: str = "", attachments: list[str | Path] | None = None) -> TokenBreakdown:
        return estimate_multimodal_tokens(
            prompt=prompt,
            attachments=attachments,
            conversation_id=conv_id,
            brain_root=brain_root,
        )

    setattr(conversation, "count_multimodal_tokens", _count_multimodal_tokens)
    return conversation
