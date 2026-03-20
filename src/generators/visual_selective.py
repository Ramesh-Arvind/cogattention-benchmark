"""
Task F: Visual Selective Attention — Visual Stroop (Multimodal VLM Extension)

NOVEL CONTRIBUTION: Procedurally generated Visual Stroop images for VLMs.
Tests whether vision-language models can separate low-level visual features
(ink color) from high-level semantic content (word meaning).

No external datasets — all images generated with PIL, zero data leakage.
"""

import random
import base64
import hashlib
from io import BytesIO
from typing import List
from dataclasses import dataclass, field

from .base import TaskInstance, DIFFICULTY_LEVELS

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

ITEMS_PER_DIFFICULTY = 30

# Color-word Stroop set
COLOR_WORDS = ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE", "PINK", "BROWN"]
COLOR_HEX = {
    "RED": "#FF0000", "BLUE": "#0000FF", "GREEN": "#008000", "YELLOW": "#FFD700",
    "ORANGE": "#FF8C00", "PURPLE": "#800080", "PINK": "#FF69B4", "BROWN": "#8B4513",
}
COLOR_NAMES = list(COLOR_HEX.keys())

DIFFICULTY_CONFIG = {
    "Easy": {"n_items": 1, "font_size": 60, "bg_noise": False, "distractor_shapes": 0},
    "Medium": {"n_items": 2, "font_size": 48, "bg_noise": False, "distractor_shapes": 2},
    "Hard": {"n_items": 3, "font_size": 36, "bg_noise": True, "distractor_shapes": 4},
    "Expert": {"n_items": 4, "font_size": 28, "bg_noise": True, "distractor_shapes": 6},
    "Frontier": {"n_items": 5, "font_size": 24, "bg_noise": True, "distractor_shapes": 8},
}


def _generate_stroop_image(
    word: str,
    ink_color: str,
    font_size: int,
    bg_noise: bool,
    distractor_shapes: int,
    rng: random.Random,
    img_width: int = 500,
    img_height: int = 300,
) -> str:
    """Generate a single Stroop image and return as base64 PNG string.

    Args:
        word: The color word to display (e.g., "RED")
        ink_color: The actual ink color name (e.g., "BLUE")
        font_size: Font size in pixels
        bg_noise: Whether to add background noise shapes
        distractor_shapes: Number of distractor shapes to add
        rng: Random instance for reproducibility

    Returns:
        Base64-encoded PNG string.
    """
    if not PIL_AVAILABLE:
        # Return a placeholder if PIL not installed
        return f"[IMAGE_PLACEHOLDER: word={word} color={ink_color}]"

    bg_color = (240, 240, 240)
    img = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Add distractor shapes
    for _ in range(distractor_shapes):
        shape_type = rng.choice(["rect", "ellipse"])
        x1 = rng.randint(0, img_width - 60)
        y1 = rng.randint(0, img_height - 40)
        x2 = x1 + rng.randint(20, 60)
        y2 = y1 + rng.randint(15, 40)
        shape_color = rng.choice(list(COLOR_HEX.values()))
        if shape_type == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=shape_color, outline=None)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=shape_color, outline=None)

    # Add background noise (random dots)
    if bg_noise:
        for _ in range(50):
            x = rng.randint(0, img_width - 1)
            y = rng.randint(0, img_height - 1)
            noise_color = (rng.randint(180, 230), rng.randint(180, 230), rng.randint(180, 230))
            draw.point((x, y), fill=noise_color)

    # Draw the Stroop word in conflicting ink color
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    ink_hex = COLOR_HEX[ink_color]
    # Center the text
    bbox = draw.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (img_width - text_w) // 2
    y = (img_height - text_h) // 2

    draw.text((x, y), word, fill=ink_hex, font=font)

    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")

    return img_b64


def generate_visual_stroop_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    """Generate a single Visual Stroop instance.

    The model sees an image of a color word printed in a conflicting ink color.
    It must report the INK COLOR, not the word.
    """
    rng = random.Random(seed + instance_idx * 1031 + DIFFICULTY_LEVELS.index(difficulty) * 4201)
    config = DIFFICULTY_CONFIG[difficulty]

    items = []
    gold_answers = {}
    trap_answers = {}
    images_b64 = []

    for i in range(config["n_items"]):
        word = rng.choice(COLOR_WORDS)
        # Ensure ink color conflicts with word
        ink_color = rng.choice([c for c in COLOR_NAMES if c != word])

        img_b64 = _generate_stroop_image(
            word=word,
            ink_color=ink_color,
            font_size=config["font_size"],
            bg_noise=config["bg_noise"],
            distractor_shapes=config["distractor_shapes"],
            rng=rng,
        )

        items.append({"word": word, "ink_color": ink_color.lower(), "index": i + 1})
        gold_answers[str(i + 1)] = ink_color.lower()
        trap_answers[str(i + 1)] = word.lower()
        images_b64.append(img_b64)

    # Build prompt
    if config["n_items"] == 1:
        prompt = (
            "You will see an image containing a color word printed in a specific ink color. "
            "The word may NOT match the ink color. "
            "What is the PHYSICAL INK COLOR of the text? Do NOT read the word — only state the "
            "color of the pixels/ink.\n\n"
            "ANSWER: [color name]"
        )
    else:
        prompt = (
            f"You will see {config['n_items']} images, each containing a color word printed "
            "in a specific ink color. The word may NOT match the ink color. "
            "For each image, state the PHYSICAL INK COLOR of the text. "
            "Do NOT read the word — only state the color of the pixels/ink.\n\n"
            "ANSWER:\n" + "\n".join(f"{i+1}. [color name]" for i in range(config["n_items"]))
        )

    task_id = f"visual_stroop_{difficulty.lower()}_{instance_idx:03d}"

    return TaskInstance(
        task_id=task_id,
        task_type="visual_stroop",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold_answers if config["n_items"] > 1 else gold_answers["1"],
        metadata={
            "items": items,
            "trap_answers": trap_answers,
            "images_base64": images_b64,
            "n_items": config["n_items"],
            "font_size": config["font_size"],
            "bg_noise": config["bg_noise"],
            "distractor_shapes": config["distractor_shapes"],
        },
    )


def generate_visual_stroop_dataset(seed: int = 2026) -> List[TaskInstance]:
    """Generate full Visual Stroop dataset across all difficulty tiers."""
    dataset = []
    idx = 0
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_visual_stroop_instance(idx, diff, seed)
            dataset.append(inst)
            idx += 1
    return dataset


if __name__ == "__main__":
    if not PIL_AVAILABLE:
        print("PIL/Pillow not installed. Install with: pip install Pillow")
    else:
        dataset = generate_visual_stroop_dataset(seed=2026)
        print(f"Generated {len(dataset)} Visual Stroop instances")
        for diff in DIFFICULTY_LEVELS:
            subset = [d for d in dataset if d.difficulty == diff]
            s = subset[0]
            print(f"  {diff}: {s.metadata['n_items']} items, "
                  f"font={s.metadata['font_size']}px, "
                  f"noise={s.metadata['bg_noise']}, "
                  f"distractors={s.metadata['distractor_shapes']}")
            print(f"    Gold: {s.gold_answer}")
            print(f"    Trap: {s.metadata['trap_answers']}")
            # Save sample image
            if s.metadata["images_base64"][0] != "[IMAGE_PLACEHOLDER":
                import os
                os.makedirs("figures", exist_ok=True)
                img_data = base64.b64decode(s.metadata["images_base64"][0])
                with open(f"figures/visual_stroop_sample_{diff.lower()}.png", "wb") as f:
                    f.write(img_data)
                print(f"    Saved sample: figures/visual_stroop_sample_{diff.lower()}.png")
