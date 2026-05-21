---
name: qwen-vl-skills
description: Work with Qwen-VL image/video data and grounding visualizations in a self-contained way. Use for Qwen-VL message preparation, qwen-vl-utils preprocessing, dynamic resolution/FPS settings, Transformers or OpenAI-compatible API inputs, DashScope/Bailian-style calls, and 2D/3D grounding visualization for bbox_2d, point_2d, XML points, and bbox_3d.
---

# Qwen-VL Skills

Use this as a standalone Qwen-VL data and visualization guide. Do not depend on a particular local checkout. Prefer `qwen-vl-utils` for local preprocessing, and use the bundled scripts in this skill for validation or visualization.

Optional pip packages by path:

- Data preprocessing: `qwen-vl-utils`
- API calls: `openai`
- Visualization: `pillow`

## Data Prepare

- Keep messages in Qwen/OpenAI-style chat shape: `role` plus list-valued `content`.
- Local visual items use `{"type": "image", "image": ...}` or `{"type": "video", "video": ...}` for `qwen-vl-utils`.
- OpenAI-compatible API items usually use `image_url` or `video_url`.
- Qwen3-VL local preprocessing normally uses `image_patch_size=16`, `return_video_kwargs=True`, `return_video_metadata=True`, then processor `do_resize=False`.
- Qwen2-VL/Qwen2.5-VL local preprocessing normally uses `image_patch_size=14`.
- Image controls live on visual items: `min_pixels`, `max_pixels`, or both `resized_height` and `resized_width`.
- Video controls: use either `fps` or `nframes`, not both. Optional fields include `min_frames`, `max_frames`, `video_start`, `video_end`, `min_pixels`, `max_pixels`, `total_pixels`.
- vLLM-style OpenAI-compatible video serving may accept `extra_body={"mm_processor_kwargs": {"fps": 2, "do_sample_frames": true}}`.
- DashScope/Bailian-style calls use the OpenAI SDK with `DASHSCOPE_API_KEY` and a compatible base URL such as `https://dashscope.aliyuncs.com/compatible-mode/v1`.

Validate local messages from this skill directory:

```bash
python scripts/inspect_qwen_vl_messages.py data.jsonl --model-family qwen3-vl --out report.json
```

API smoke test or dry-run:

```bash
python scripts/call_qwen_vl_api.py \
  --dry-run \
  --model qwen3-vl-235b-a22b-instruct \
  --image image.jpg \
  --text "Describe this image."
```

Read `references/qwen-vl-utils.md` only when exact preprocessing details are needed.

## Visualization

- Qwen3-VL 2D grounding convention is relative `[0, 1000]` coordinates by default.
- Visualize on the original image unless the coordinates were produced on a resized image.
- 3D grounding requires camera intrinsics from the user, dataset, or camera metadata.
- Qwen-VL cookbook-style 3D boxes use `[x, y, z, x_size, y_size, z_size, pitch, yaw, roll]`; the script defaults to this convention.

Draw 2D/3D overlays from this skill directory:

```bash
python scripts/visualize_qwen_vl_coordinates.py \
  --image image.jpg \
  --annotations response.json \
  --coord-system relative-1000 \
  --out overlay.png
```

For 3D:

```bash
python scripts/visualize_qwen_vl_coordinates.py \
  --image image.jpg \
  --annotations response.json \
  --camera camera.json \
  --out overlay_3d.png
```

Read `references/coordinate-visualization.md` only when coordinate conventions or 3D angle settings are ambiguous.

## Files

- `scripts/inspect_qwen_vl_messages.py`: run `qwen-vl-utils.process_vision_info` on JSON/JSONL messages.
- `scripts/call_qwen_vl_api.py`: call or dry-run DashScope/local OpenAI-compatible Qwen-VL requests.
- `scripts/visualize_qwen_vl_coordinates.py`: draw `bbox_2d`, `point_2d`, XML points, and `bbox_3d`.
