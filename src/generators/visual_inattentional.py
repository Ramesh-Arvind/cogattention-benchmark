"""
Task G: Visual Inattentional Blindness (Multimodal VLM Extension)

Based on Simons & Chabris (1999) gorilla experiment. Subjects focused on a
counting task miss an unexpected stimulus embedded in the scene.

Tests whether VLMs exhibit inattentional blindness: does task-directed focus
cause them to miss anomalous objects in the visual field?

No external datasets — all images generated with PIL, zero data leakage.
"""

import random
import math
import base64
import hashlib
from io import BytesIO
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from .base import TaskInstance, DIFFICULTY_LEVELS

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

ITEMS_PER_DIFFICULTY = 30

# Shape and color pools
SHAPE_TYPES = ["circle", "square", "triangle"]
SHAPE_COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 180, 60),
    "yellow": (220, 200, 40),
    "orange": (230, 130, 30),
    "purple": (140, 40, 180),
}
COLOR_NAMES = list(SHAPE_COLORS.keys())

# Unexpected stimulus types by difficulty
UNEXPECTED_TYPES = {
    "Easy": "star",
    "Medium": "arrow",
    "Hard": "cross",
    "Expert": "dot_pattern",
    "Frontier": "gradient_patch",
}

DIFFICULTY_CONFIG = {
    "Easy": {
        "shape_count": (8, 12),
        "unexpected_size": "large",
        "saliency": "high",
    },
    "Medium": {
        "shape_count": (15, 20),
        "unexpected_size": "medium",
        "saliency": "medium",
    },
    "Hard": {
        "shape_count": (25, 30),
        "unexpected_size": "small",
        "saliency": "low",
    },
    "Expert": {
        "shape_count": (35, 40),
        "unexpected_size": "tiny",
        "saliency": "very_low",
    },
    "Frontier": {
        "shape_count": (45, 50),
        "unexpected_size": "subtle",
        "saliency": "ultra_low",
    },
}

IMG_WIDTH = 600
IMG_HEIGHT = 500


def _draw_shape(draw, shape_type: str, x: int, y: int, size: int, color: Tuple[int, int, int]):
    """Draw a basic shape on the image."""
    if shape_type == "circle":
        draw.ellipse([x, y, x + size, y + size], fill=color)
    elif shape_type == "square":
        draw.rectangle([x, y, x + size, y + size], fill=color)
    elif shape_type == "triangle":
        points = [
            (x + size // 2, y),
            (x, y + size),
            (x + size, y + size),
        ]
        draw.polygon(points, fill=color)


def _draw_star(draw, cx: int, cy: int, size: int, color: Tuple[int, int, int]):
    """Draw a 5-pointed star."""
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = size if i % 2 == 0 else size * 0.4
        px = cx + int(r * math.cos(angle))
        py = cy - int(r * math.sin(angle))
        points.append((px, py))
    draw.polygon(points, fill=color)


def _draw_arrow(draw, cx: int, cy: int, size: int, color: Tuple[int, int, int]):
    """Draw an arrow shape."""
    half = size // 2
    # Arrow body
    draw.rectangle([cx - half, cy - size // 6, cx + half // 2, cy + size // 6], fill=color)
    # Arrow head
    points = [
        (cx + half // 2, cy - half),
        (cx + half, cy),
        (cx + half // 2, cy + half),
    ]
    draw.polygon(points, fill=color)


def _draw_cross(draw, cx: int, cy: int, size: int, color: Tuple[int, int, int]):
    """Draw a cross/plus shape."""
    t = max(size // 4, 2)
    draw.rectangle([cx - t, cy - size, cx + t, cy + size], fill=color)
    draw.rectangle([cx - size, cy - t, cx + size, cy + t], fill=color)


def _draw_dot_pattern(draw, cx: int, cy: int, size: int, color: Tuple[int, int, int], rng: random.Random):
    """Draw a cluster of small dots."""
    for _ in range(8):
        dx = rng.randint(-size, size)
        dy = rng.randint(-size, size)
        r = max(size // 6, 2)
        draw.ellipse([cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r], fill=color)


def _draw_gradient_patch(draw, img: Image.Image, cx: int, cy: int, size: int, color: Tuple[int, int, int]):
    """Draw a subtle gradient patch."""
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            px, py = cx + dx, cy + dy
            if 0 <= px < IMG_WIDTH and 0 <= py < IMG_HEIGHT:
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= size:
                    alpha = 1.0 - (dist / size)
                    bg = img.getpixel((px, py))
                    blended = tuple(
                        int(bg[c] * (1 - alpha * 0.5) + color[c] * alpha * 0.5)
                        for c in range(3)
                    )
                    draw.point((px, py), fill=blended)


def _get_unexpected_color(saliency: str, target_color_name: str, bg_color: Tuple[int, int, int], rng: random.Random) -> Tuple[int, int, int]:
    """Get color for unexpected stimulus based on saliency level."""
    if saliency == "high":
        # Bright, contrasting color
        contrasting = [c for c in COLOR_NAMES if c != target_color_name]
        chosen = rng.choice(contrasting)
        return SHAPE_COLORS[chosen]
    elif saliency == "medium":
        # Similar hue to a random scene color
        base = SHAPE_COLORS[rng.choice(COLOR_NAMES)]
        return tuple(min(255, c + rng.randint(-30, 30)) for c in base)
    elif saliency == "low":
        # Muted, close to background
        return tuple(min(255, bg_color[c] + rng.randint(30, 60)) for c in range(3))
    elif saliency == "very_low":
        # Nearly background colored
        return tuple(min(255, bg_color[c] + rng.randint(15, 35)) for c in range(3))
    else:  # ultra_low
        # Almost invisible
        return tuple(min(255, bg_color[c] + rng.randint(5, 20)) for c in range(3))


def _get_unexpected_size(size_label: str) -> int:
    """Get pixel size for unexpected stimulus."""
    sizes = {
        "large": 40,
        "medium": 28,
        "small": 18,
        "tiny": 10,
        "subtle": 12,
    }
    return sizes.get(size_label, 20)


def _generate_inattentional_image(
    target_color_name: str,
    target_shape: str,
    n_shapes: int,
    unexpected_present: bool,
    unexpected_type: str,
    unexpected_size_label: str,
    saliency: str,
    rng: random.Random,
) -> Tuple[str, int, str]:
    """Generate a scene image with shapes and optionally an unexpected stimulus.

    Returns:
        (base64_png, actual_target_count, unexpected_description)
    """
    if not PIL_AVAILABLE:
        count = rng.randint(3, 8)
        desc = f"{unexpected_type}" if unexpected_present else ""
        return f"[IMAGE_PLACEHOLDER: shapes={n_shapes}]", count, desc

    bg_color = (230, 230, 230)
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(img)

    target_rgb = SHAPE_COLORS[target_color_name]
    target_count = 0

    # Place shapes avoiding too much overlap
    shape_size = max(18, min(30, 500 // int(math.sqrt(n_shapes))))
    positions = []

    for i in range(n_shapes):
        x = rng.randint(10, IMG_WIDTH - shape_size - 10)
        y = rng.randint(10, IMG_HEIGHT - shape_size - 10)
        positions.append((x, y))

        # Decide if this shape is the target type+color
        is_target = rng.random() < 0.25  # ~25% are targets
        if is_target:
            _draw_shape(draw, target_shape, x, y, shape_size, target_rgb)
            target_count += 1
        else:
            # Random non-target (different shape or different color)
            if rng.random() < 0.5:
                # Different color, same shape
                other_colors = [c for c in COLOR_NAMES if c != target_color_name]
                other_color = SHAPE_COLORS[rng.choice(other_colors)]
                _draw_shape(draw, target_shape, x, y, shape_size, other_color)
            else:
                # Different shape, any color
                other_shapes = [s for s in SHAPE_TYPES if s != target_shape]
                other_shape = rng.choice(other_shapes)
                any_color = SHAPE_COLORS[rng.choice(COLOR_NAMES)]
                _draw_shape(draw, other_shape, x, y, shape_size, any_color)

    # Ensure at least 1 target
    if target_count == 0:
        x, y = positions[0] if positions else (IMG_WIDTH // 2, IMG_HEIGHT // 2)
        _draw_shape(draw, target_shape, x, y, shape_size, target_rgb)
        target_count = 1

    # Draw unexpected stimulus if present
    unexpected_description = ""
    if unexpected_present:
        ux_size = _get_unexpected_size(unexpected_size_label)
        ux_color = _get_unexpected_color(saliency, target_color_name, bg_color, rng)
        # Place in a semi-random but visible location
        ux = rng.randint(IMG_WIDTH // 4, 3 * IMG_WIDTH // 4)
        uy = rng.randint(IMG_HEIGHT // 4, 3 * IMG_HEIGHT // 4)

        if unexpected_type == "star":
            _draw_star(draw, ux, uy, ux_size, ux_color)
            unexpected_description = "a star shape"
        elif unexpected_type == "arrow":
            _draw_arrow(draw, ux, uy, ux_size, ux_color)
            unexpected_description = "an arrow shape"
        elif unexpected_type == "cross":
            _draw_cross(draw, ux, uy, ux_size, ux_color)
            unexpected_description = "a cross/plus shape"
        elif unexpected_type == "dot_pattern":
            _draw_dot_pattern(draw, ux, uy, ux_size, ux_color, rng)
            unexpected_description = "a cluster of dots"
        elif unexpected_type == "gradient_patch":
            _draw_gradient_patch(draw, img, ux, uy, ux_size, ux_color)
            unexpected_description = "a subtle gradient patch"

    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")

    return img_b64, target_count, unexpected_description


def generate_visual_inattentional_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    """Generate a single Visual Inattentional Blindness instance."""
    rng = random.Random(seed + instance_idx * 1049 + DIFFICULTY_LEVELS.index(difficulty) * 3907)
    config = DIFFICULTY_CONFIG[difficulty]

    target_color = rng.choice(COLOR_NAMES)
    target_shape = rng.choice(SHAPE_TYPES)
    n_shapes = rng.randint(*config["shape_count"])

    # 50% have unexpected stimulus
    unexpected_present = (instance_idx % 2 == 0)
    unexpected_type = UNEXPECTED_TYPES[difficulty]

    img_b64, target_count, unexpected_desc = _generate_inattentional_image(
        target_color_name=target_color,
        target_shape=target_shape,
        n_shapes=n_shapes,
        unexpected_present=unexpected_present,
        unexpected_type=unexpected_type,
        unexpected_size_label=config["unexpected_size"],
        saliency=config["saliency"],
        rng=rng,
    )

    prompt = (
        f"Look at the image carefully.\n\n"
        f"PRIMARY TASK: Count the number of {target_color} {target_shape}s in the image.\n\n"
        f"SECONDARY TASK: Did you notice anything unusual or out-of-place in the image? "
        f"If so, describe it briefly. If not, say 'nothing unusual'.\n\n"
        f"ANSWER:\n"
        f"1. Count of {target_color} {target_shape}s: [number]\n"
        f"2. Unusual observation: [description or 'nothing unusual']"
    )

    gold_answer = {
        "count": target_count,
        "unexpected_present": unexpected_present,
        "unexpected_description": unexpected_desc if unexpected_present else "",
    }

    task_id = f"visual_inattentional_{difficulty.lower()}_{instance_idx:03d}"

    return TaskInstance(
        task_id=task_id,
        task_type="visual_inattentional",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold_answer,
        metadata={
            "target_color": target_color,
            "target_shape": target_shape,
            "n_shapes": n_shapes,
            "target_count": target_count,
            "unexpected_present": unexpected_present,
            "unexpected_type": unexpected_type if unexpected_present else None,
            "unexpected_description": unexpected_desc,
            "saliency": config["saliency"],
            "image_base64": img_b64,
        },
    )


def generate_visual_inattentional_dataset(seed: int = 2026) -> List[TaskInstance]:
    """Generate full Visual Inattentional Blindness dataset across all difficulty tiers."""
    dataset = []
    idx = 0
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_visual_inattentional_instance(idx, diff, seed)
            dataset.append(inst)
            idx += 1
    return dataset


if __name__ == "__main__":
    if not PIL_AVAILABLE:
        print("PIL/Pillow not installed. Install with: pip install Pillow")
    else:
        dataset = generate_visual_inattentional_dataset(seed=2026)
        print(f"Generated {len(dataset)} Visual Inattentional Blindness instances")
        for diff in DIFFICULTY_LEVELS:
            subset = [d for d in dataset if d.difficulty == diff]
            present = sum(1 for d in subset if d.metadata["unexpected_present"])
            absent = len(subset) - present
            print(f"  {diff}: {len(subset)} items, "
                  f"shapes={subset[0].metadata['n_shapes']}-{subset[-1].metadata['n_shapes']}, "
                  f"unexpected present={present}/absent={absent}")
