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

## Resize

`fetch_image` uses `image_patch_size * 2` as the resize factor:

- Qwen2-VL/Qwen2.5-VL: usually `14 * 2 = 28`.
- Qwen3-VL: usually `16 * 2 = 32`.

If `resized_height` and `resized_width` are present, they are rounded to this factor. Otherwise, resize is controlled by `min_pixels` and `max_pixels`.

## Video

Use either `fps` or `nframes`.

Relevant fields: `min_frames`, `max_frames`, `video_start`, `video_end`, `min_pixels`, `max_pixels`, `total_pixels`.

Backend priority in `qwen-vl-utils`: `torchcodec`, then `decord`, then `torchvision`, unless `FORCE_QWENVL_VIDEO_READER` is set.

## Qwen3-VL Processor Pattern

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
