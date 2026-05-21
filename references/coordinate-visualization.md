# Coordinate Visualization

Use this only when grounding coordinates or 3D conventions are unclear.

## 2D

Qwen3-VL 2D grounding commonly uses relative `[0, 1000]` coordinates:

```python
x_px = x / 1000 * image_width
y_px = y / 1000 * image_height
```

Typical outputs:

```json
{"bbox_2d": [x1, y1, x2, y2], "label": "object"}
{"point_2d": [x, y], "label": "object"}
```

Use `--coord-system absolute` only for datasets that already store pixel coordinates.

## 3D

Qwen-VL cookbook-style 3D visualization unpacks boxes as:

```json
{"bbox_3d": [x, y, z, x_size, y_size, z_size, pitch, yaw, roll], "label": "object"}
```

`visualize_qwen_vl_coordinates.py` defaults to this convention:

```bash
--bbox3d-angle-order pitch-yaw-roll --bbox3d-angle-unit cookbook
```

Override with `--bbox3d-angle-unit radians|degrees` or `--bbox3d-angle-order roll-pitch-yaw` only when the source data explicitly uses that convention.

3D overlays need camera intrinsics:

```json
{"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0}
```
