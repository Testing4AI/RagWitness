"""Collect complete OpenAI-compatible text streams without retaining reasoning."""

from __future__ import annotations

import time


def usage_counts(usage):
    if usage is None:
        return {}
    return {"prompt_tokens": int(usage.prompt_tokens),
            "completion_tokens": int(usage.completion_tokens),
            "total_tokens": int(usage.total_tokens), "attempts": 1}


def collect_chat_stream(stream, *, kind, timeout_seconds=600):
    parts = []
    usage = None
    model = None
    finish_reason = None
    chunks = 0
    started = last_progress = time.perf_counter()
    try:
        for chunk in stream:
            chunks += 1
            choices = getattr(chunk, "choices", None)
            if not isinstance(choices, list):
                # This is a provider-response failure, not caller argument misuse.
                raise ValueError("Malformed streaming chat-completion chunk")  # noqa: TRY004
            returned_model = getattr(chunk, "model", None)
            if model and returned_model and model != returned_model:
                raise ValueError("Model identity changed within a response stream")
            model = returned_model or model
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for choice in choices:
                if getattr(choice, "index", None) != 0 or getattr(choice, "delta", None) is None:
                    raise ValueError("Unexpected multiple-choice response stream")
                content = getattr(choice.delta, "content", None)
                if content is not None and not isinstance(content, str):
                    raise ValueError("Malformed streaming chat-completion content")
                if content:
                    if finish_reason is not None:
                        raise ValueError("Content received after stream completion")
                    parts.append(content)
                if getattr(choice, "finish_reason", None) is not None:
                    finish_reason = choice.finish_reason
            now = time.perf_counter()
            # The SDK's request timeout does not stop a provider that keeps an
            # SSE connection alive indefinitely.  Treat that case exactly like
            # any other transport timeout and never cache the partial answer.
            if now - started > timeout_seconds:
                raise TimeoutError(
                    f"Streaming response exceeded {timeout_seconds} seconds"
                )
            if now - last_progress >= 20:
                # Counts only: no prompt, answer, hidden reasoning, or credential.
                print("FAITHFUL_STREAM_PROGRESS", kind, "chunks", chunks,
                      "elapsed_seconds", round(now - started, 1), flush=True)
                last_progress = now
    finally:
        stream.close()
    text = "".join(parts).strip()
    if finish_reason is None or not model or not text:
        raise ValueError("Incomplete or empty text response stream")
    return {"text": text, "usage": usage_counts(usage), "returned_model": model,
            "stream_chunks": chunks, "stream_finish_reason": finish_reason,
            "stream_usage_received": usage is not None}
