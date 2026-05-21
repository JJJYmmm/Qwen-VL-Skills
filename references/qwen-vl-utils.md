# qwen-vl-utils

Use this only for exact local preprocessing details.

## Message Shape

```json
[{"role": "user", "content": [
  {"type": "image", "image": "file:///abs/path/image.jpg"},
  {"type": "text", "text": "Describe this image."}
]}]
```

Recognized visual keys include `image`, `image_url`, and `video`.

## Version Differences

| Model family | Local preprocessing | Video metadata in Transformers v5-style processors | 2D grounding coordinates |
| --- | --- | --- | --- |
| Qwen2-VL | Usually `image_patch_size=14` | Not required | Relative `[0, 1000]` |
| Qwen2.5-VL | Usually `image_patch_size=14` | Used to compute `second_per_grid_ts` from sampled FPS | Absolute pixels |
| Qwen3-VL | Usually `image_patch_size=16` | Used to compute frame timestamps in prompts | Relative `[0, 1000]` |

For Qwen2.5-VL and Qwen3-VL local video preprocessing on Transformers v5-style processors, keep video metadata when using preprocessed videos. If `qwen-vl-utils` returns videos as `(video, metadata)` pairs, split them before processor input and pass the metadata as `video_metadata=...`.

## Dynamic Resolution

`qwen-vl-utils` resizes to multiples of `image_patch_size * 2`:

- Qwen2-VL/Qwen2.5-VL: usually `14 * 2 = 28`.
- Qwen3-VL: usually `16 * 2 = 32`.

Choose one mode per visual item:

```json
{"type": "image", "image": "file:///img.jpg", "min_pixels": 65536, "max_pixels": 524288}
```

```json
{"type": "image", "image": "file:///img.jpg", "resized_height": 768, "resized_width": 1024}
```

Rules:

- If both `resized_height` and `resized_width` are present, they are rounded to the resize factor and used as the target size.
- Otherwise, original image size is resized within `min_pixels` and `max_pixels` while preserving aspect ratio.
- `min_pixels`/`max_pixels` are pixel budgets after resizing, not raw token counts.
- Defaults are derived from token-count constants: image min is `4 * factor^2`, image max is `16384 * factor^2`.
- `max_pixels` must be greater than or equal to `min_pixels`.
- Very extreme aspect ratios can fail before resizing.

Practical guidance:

- Increase `max_pixels` for OCR, dense charts, documents, or small text.
- Lower `max_pixels` for throughput, memory, or many-image prompts.
- Prefer `min_pixels`/`max_pixels` for datasets with mixed aspect ratios; use explicit `resized_height`/`resized_width` only when a fixed rendered size is required.

## Dynamic FPS and Frames

For video files, use either `fps` or `nframes`, not both:

```json
{"type": "video", "video": "file:///clip.mp4", "fps": 2, "min_frames": 4, "max_frames": 128}
```

```json
{"type": "video", "video": "file:///clip.mp4", "nframes": 64}
```

Frame-count rules:

- `nframes` is rounded to a multiple of `2`.
- With `fps`, sampled frames are approximately `duration_seconds * fps`.
- `min_frames` and `max_frames` apply only when using `fps`.
- `min_frames` is rounded up to a multiple of `2`; `max_frames` is rounded down to a multiple of `2`.
- The final sampled frame count is clamped by `[min_frames, max_frames]` and by the source video's total frames.
- The final frame count must be between `2` and the available source frame count.
- Defaults are `fps=2.0`, `min_frames=4`, and `max_frames=min(768, total_frames)`.

Clip-window fields:

- `video_start` and `video_end` are seconds.
- Supported video backends clamp the requested time range to the video duration.
- Invalid ranges, such as start after end, should be treated as data errors.

Frame-list inputs:

- If `video` is already a list of image frames, frames are individually processed as images.
- Odd frame counts are padded by repeating the last frame so the count is a multiple of `2`.
- Use `sample_fps` and optionally `raw_fps` only when reconstructing metadata for a frame list.

## Video Resolution Budget

Video spatial resize happens after frame sampling.

Per-frame defaults use the same resize factor:

- frame min pixels: `128 * factor^2`
- frame max pixels: `768 * factor^2`

`total_pixels` controls the total video budget across frames. The effective per-frame `max_pixels` is limited by frame count:

```text
effective_max_pixels =
  max(min(frame_default_max_pixels, total_pixels / nframes * 2), min_pixels * 1.05)
```

Default `total_pixels` comes from `MODEL_SEQ_LEN * factor^2 * 0.9`; `MODEL_SEQ_LEN` defaults to `128000` unless set in the environment. User-provided `max_pixels`, if present, is capped by the effective limit above. This means increasing `fps` or `nframes` can reduce per-frame resolution unless `total_pixels` is also increased.

Practical guidance:

- For temporal reasoning, raise `fps` or `max_frames` first.
- For OCR or fine visual detail in video, raise `max_pixels` and possibly `total_pixels`; otherwise extra frames may make each frame smaller.
- For long videos, cap `max_frames` and set `total_pixels` deliberately.
- For API serving, check whether the server accepts `extra_body.mm_processor_kwargs`; local `qwen-vl-utils` fields and remote server fields are related but not always identical.

## Video Backend

Backend priority in `qwen-vl-utils`: `torchcodec`, then `decord`, then `torchvision`, unless `FORCE_QWENVL_VIDEO_READER` is set.

## Processor Patterns

Qwen2-VL:

```python
images, videos = process_vision_info(messages, image_patch_size=14)
inputs = processor(text=text, images=images, videos=videos, return_tensors="pt")
```

Qwen2.5-VL:

```python
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=14,
    return_video_kwargs=True,
    return_video_metadata=True,
)
if videos is not None and videos and isinstance(videos[0], tuple):
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None
inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    return_tensors="pt",
    **video_kwargs,
)
```

Qwen3-VL:

```python
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None
inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    do_resize=False,
    return_tensors="pt",
    **video_kwargs,
)
```
