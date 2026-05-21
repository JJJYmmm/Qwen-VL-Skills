#!/usr/bin/env python3
"""Draw Qwen-VL 2D/3D grounding annotations on an image."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from PIL import Image, ImageColor, ImageDraw, ImageFont


EDGES_3D = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def load_annotations(path: Path | None, text: str | None) -> list[dict[str, Any]]:
    if path is None and text is None:
        raise ValueError("Provide --annotations or --text.")
    raw = text if text is not None else path.read_text(encoding="utf-8")
    data = parse_jsonish(raw)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Annotations must parse to a JSON object or list.")
    return [item for item in data if isinstance(item, dict)]


def parse_jsonish(text: str) -> Any:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start_candidates = [idx for idx in (stripped.find("["), stripped.find("{")) if idx >= 0]
    if start_candidates:
        start = min(start_candidates)
        end = max(stripped.rfind("]"), stripped.rfind("}"))
        if end > start:
            return json.loads(stripped[start : end + 1])

    xml_points = parse_xml_points(stripped)
    if xml_points:
        return xml_points
    raise ValueError("Could not parse annotations as JSON or XML points.")


def parse_xml_points(text: str) -> list[dict[str, Any]]:
    points = []
    for match in re.finditer(r"<points\b[^>]*>.*?</points>", text, flags=re.DOTALL):
        try:
            element = ElementTree.fromstring(match.group(0))
        except ElementTree.ParseError:
            continue
        x = element.attrib.get("x") or element.attrib.get("x1")
        y = element.attrib.get("y") or element.attrib.get("y1")
        if x is None or y is None:
            continue
        points.append({"point_2d": [float(x), float(y)], "label": (element.text or "point").strip()})
    return points


def color_cycle() -> list[str]:
    base = [
        "red", "green", "blue", "orange", "purple", "cyan", "magenta",
        "yellow", "lime", "navy", "teal", "coral", "gold",
    ]
    return base + list(ImageColor.colormap)


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("NotoSansCJK-Regular.ttc", "Arial Unicode.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_pixels(values: list[float], width: int, height: int, coord_system: str) -> list[float]:
    if coord_system == "absolute":
        return [float(v) for v in values]
    if coord_system == "relative-1000":
        scaled = []
        for idx, value in enumerate(values):
            denom = width if idx % 2 == 0 else height
            scaled.append(float(value) / 1000.0 * denom)
        return scaled
    raise ValueError(f"Unsupported coordinate system: {coord_system}")


def label_for(item: dict[str, Any], fallback: str) -> str:
    return str(item.get("label") or item.get("name") or item.get("category") or fallback)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], label: str, color: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), label, font=font)
    pad = 2
    bg = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(bg, fill="white")
    draw.text((x, y), label, fill=color, font=font)


def camera_from_args(image: Image.Image, camera_path: Path | None, fov: float) -> dict[str, float]:
    if camera_path is not None:
        return json.loads(camera_path.read_text(encoding="utf-8"))
    width, height = image.size
    fx = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(fov) / 2.0))
    return {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0}


def matmul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[i] * vector[i] for i in range(3)) for row in matrix]


def rotation_matrix(pitch: float, yaw: float, roll: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]
    ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]

    def mm(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    return mm(rz, mm(ry, rx))


def convert_angle(value: float, unit: str) -> float:
    if unit == "radians":
        return value
    if unit == "degrees":
        return math.radians(value)
    if unit == "cookbook":
        return math.radians(value * 180.0)
    raise ValueError(f"Unsupported angle unit: {unit}")


def project_bbox_3d(
    bbox: list[float],
    camera: dict[str, float],
    *,
    angle_order: str,
    angle_unit: str,
) -> list[tuple[float, float]] | None:
    if len(bbox) != 9:
        return None
    x, y, z, sx, sy, sz, a, b, c = [float(v) for v in bbox]
    if angle_order == "pitch-yaw-roll":
        pitch, yaw, roll = a, b, c
    elif angle_order == "roll-pitch-yaw":
        roll, pitch, yaw = a, b, c
    else:
        raise ValueError(f"Unsupported angle order: {angle_order}")
    pitch = convert_angle(pitch, angle_unit)
    yaw = convert_angle(yaw, angle_unit)
    roll = convert_angle(roll, angle_unit)
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    corners = [
        [hx, hy, hz], [hx, hy, -hz], [hx, -hy, hz], [hx, -hy, -hz],
        [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, hz], [-hx, -hy, -hz],
    ]
    rot = rotation_matrix(pitch, yaw, roll)
    projected = []
    for corner in corners:
        rx, ry, rz = matmul(rot, corner)
        px, py, pz = rx + x, ry + y, rz + z
        if pz <= 1e-6:
            return None
        u = camera["fx"] * px / pz + camera["cx"]
        v = camera["fy"] * py / pz + camera["cy"]
        projected.append((u, v))
    return projected


def draw_annotations(
    image: Image.Image,
    annotations: list[dict[str, Any]],
    *,
    coord_system: str,
    camera: dict[str, float],
    bbox3d_angle_order: str,
    bbox3d_angle_unit: str,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = load_font(16)
    colors = color_cycle()
    width, height = output.size

    for index, item in enumerate(annotations):
        color = colors[index % len(colors)]
        label = label_for(item, f"item_{index}")

        if "bbox_2d" in item:
            x1, y1, x2, y2 = to_pixels(item["bbox_2d"], width, height, coord_system)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw_label(draw, (x1, max(0, y1 - 20)), label, color, font)

        if "point_2d" in item:
            x, y = to_pixels(item["point_2d"], width, height, coord_system)
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
            draw_label(draw, (x + 8, y + 8), label, color, font)

        if "bbox_3d" in item:
            points = project_bbox_3d(
                item["bbox_3d"],
                camera,
                angle_order=bbox3d_angle_order,
                angle_unit=bbox3d_angle_unit,
            )
            if points is None:
                continue
            for a, b in EDGES_3D:
                draw.line((points[a], points[b]), fill=color, width=3)
            draw_label(draw, points[0], label, color, font)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--text", default=None, help="Raw JSON/XML annotation text")
    parser.add_argument("--coord-system", choices=("relative-1000", "absolute"), default="relative-1000")
    parser.add_argument("--camera", type=Path, default=None, help="JSON with fx, fy, cx, cy for bbox_3d")
    parser.add_argument("--fov", type=float, default=60.0, help="Fallback FOV when --camera is absent")
    parser.add_argument("--bbox3d-angle-order", choices=("pitch-yaw-roll", "roll-pitch-yaw"), default="pitch-yaw-roll")
    parser.add_argument("--bbox3d-angle-unit", choices=("cookbook", "radians", "degrees"), default="cookbook")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    image = Image.open(args.image)
    annotations = load_annotations(args.annotations, args.text)
    camera = camera_from_args(image, args.camera, args.fov)
    output = draw_annotations(
        image,
        annotations,
        coord_system=args.coord_system,
        camera=camera,
        bbox3d_angle_order=args.bbox3d_angle_order,
        bbox3d_angle_unit=args.bbox3d_angle_unit,
    )
    output.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
