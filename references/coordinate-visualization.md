# Coordinate Visualization

Use this only when grounding coordinates or 3D conventions are unclear.

## 2D

2D grounding coordinate conventions differ by Qwen-VL generation:

| Model family | Coordinate convention |
| --- | --- |
| Qwen2-VL | Relative `[0, 1000]` |
| Qwen2.5-VL | Absolute pixels |
| Qwen3-VL | Relative `[0, 1000]` |

For relative coordinates, convert to pixels with:

```python
x_px = x / 1000 * image_width
y_px = y / 1000 * image_height
```

Typical outputs:

```json
{"bbox_2d": [x1, y1, x2, y2], "label": "object"}
{"point_2d": [x, y], "label": "object"}
```

Use `--coord-system relative-1000` for Qwen2-VL/Qwen3-VL style outputs, and `--coord-system absolute` for Qwen2.5-VL style outputs or datasets that already store pixel coordinates.

## 3D

Qwen-VL cookbook-style 3D visualization unpacks boxes as:

```json
{"bbox_3d": [x, y, z, x_size, y_size, z_size, pitch, yaw, roll], "label": "object"}
```

`visualize_qwen_vl_coordinates.py` follows this convention directly, including the notebook-style angle conversion used by Qwen-VL cookbook examples.

3D overlays need camera intrinsics:

```json
{"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0}
```
