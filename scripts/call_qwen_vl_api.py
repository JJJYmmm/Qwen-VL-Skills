#!/usr/bin/env python3
"""Call a Qwen-VL OpenAI-compatible endpoint with image/video content."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:"))


def encode_image_source(source: str) -> dict[str, Any]:
    if is_url(source):
        return {"type": "image_url", "image_url": {"url": source}}

    path = Path(source)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def encode_video_source(source: str) -> dict[str, Any]:
    if is_url(source) or source.startswith("file://"):
        return {"type": "video_url", "video_url": {"url": source}}
    return {"type": "video_url", "video_url": {"url": f"file://{Path(source).resolve()}"}}


def load_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.messages:
        return json.loads(args.messages.read_text(encoding="utf-8"))

    content: list[dict[str, Any]] = []
    for image in args.image:
        content.append(encode_image_source(image))
    for video in args.video:
        content.append(encode_video_source(video))
    content.append({"type": "text", "text": args.text})
    return [{"role": "user", "content": content}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.getenv("QWEN_VL_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY")
    parser.add_argument("--text", default="Describe the visual content.")
    parser.add_argument("--image", action="append", default=[], help="Image URL, data URL, or local path. May be repeated.")
    parser.add_argument("--video", action="append", default=[], help="Video URL or path. May be repeated.")
    parser.add_argument("--messages", type=Path, default=None, help="Raw OpenAI-compatible messages JSON file.")
    parser.add_argument("--extra-body", default=None, help="JSON object forwarded as extra_body.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true", help="Print the request payload without calling the endpoint.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON response path.")
    args = parser.parse_args()

    messages = load_messages(args)
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.extra_body:
        kwargs["extra_body"] = json.loads(args.extra_body)

    if args.dry_run:
        payload = {"base_url": args.base_url, "request": kwargs}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI SDK first: pip install openai") from exc

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    response = client.chat.completions.create(**kwargs)
    payload = response.model_dump()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
