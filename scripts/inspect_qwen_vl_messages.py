#!/usr/bin/env python3
"""Inspect Qwen-VL messages with qwen-vl-utils without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PATCH_SIZE_BY_MODEL = {
    "qwen2-vl": 14,
    "qwen2.5-vl": 14,
    "qwen3-vl": 16,
}
METADATA_BY_DEFAULT = {"qwen2.5-vl", "qwen3-vl"}


def load_records(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        return records

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    return [data]


def extract_conversation(record: Any) -> Any:
    if isinstance(record, dict):
        for key in ("messages", "conversations", "conversation"):
            if key in record:
                return record[key]
        if "role" in record and "content" in record:
            return [record]
    return record


def is_message_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], dict)
        and "role" in value[0]
        and "content" in value[0]
    )


def normalize_conversations(records: list[Any]) -> list[Any]:
    if len(records) == 1:
        conv = extract_conversation(records[0])
        if is_message_list(conv):
            return [conv]
        if isinstance(conv, list) and conv and is_message_list(conv[0]):
            return conv

    conversations = []
    for record in records:
        conv = extract_conversation(record)
        if is_message_list(conv):
            conversations.append(conv)
        elif isinstance(conv, list) and conv and is_message_list(conv[0]):
            conversations.extend(conv)
        else:
            raise ValueError("Could not infer Qwen-VL message list from a record.")
    return conversations


def summarize_image(image: Any) -> dict[str, Any]:
    size = getattr(image, "size", None)
    return {"type": type(image).__name__, "size": list(size) if size else None}


def summarize_video(video: Any) -> dict[str, Any]:
    metadata = None
    if isinstance(video, tuple) and len(video) == 2:
        video, metadata = video

    shape = getattr(video, "shape", None)
    summary = {
        "type": type(video).__name__,
        "shape": list(shape) if shape is not None else None,
        "length": len(video) if hasattr(video, "__len__") else None,
    }
    if metadata is not None:
        summary["metadata"] = json_safe(metadata)
    return summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def inspect_conversation(
    conversation: Any,
    *,
    image_patch_size: int,
    return_video_metadata: bool,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    images, videos, video_kwargs = process_vision_info(
        conversation,
        image_patch_size=image_patch_size,
        return_video_kwargs=True,
        return_video_metadata=return_video_metadata,
    )
    return {
        "image_count": len(images or []),
        "video_count": len(videos or []),
        "image_sizes": [summarize_image(image) for image in (images or [])],
        "video_shapes": [summarize_video(video) for video in (videos or [])],
        "video_kwargs": json_safe(video_kwargs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Qwen-VL messages JSON or JSONL file")
    parser.add_argument("--model-family", choices=sorted(PATCH_SIZE_BY_MODEL), default="qwen3-vl")
    parser.add_argument("--image-patch-size", type=int, default=None)
    parser.add_argument("--return-video-metadata", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--out", type=Path, default=None, help="Optional report JSON path")
    args = parser.parse_args()

    image_patch_size = args.image_patch_size or PATCH_SIZE_BY_MODEL[args.model_family]
    if args.return_video_metadata == "auto":
        return_video_metadata = args.model_family in METADATA_BY_DEFAULT
    else:
        return_video_metadata = args.return_video_metadata == "true"

    records = load_records(args.input)
    conversations = normalize_conversations(records)

    report = {
        "input": str(args.input),
        "model_family": args.model_family,
        "image_patch_size": image_patch_size,
        "return_video_metadata": return_video_metadata,
        "conversation_count": len(conversations),
        "items": [],
    }

    failures = 0
    for index, conversation in enumerate(conversations):
        try:
            item = inspect_conversation(
                conversation,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
            )
            item["index"] = index
            item["ok"] = True
        except Exception as exc:  # noqa: BLE001 - report all preprocessing failures.
            failures += 1
            item = {"index": index, "ok": False, "error": repr(exc)}
        report["items"].append(item)

    report["failure_count"] = failures

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
