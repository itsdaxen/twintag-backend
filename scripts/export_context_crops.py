"""Export the clearest full-resolution evidence crop for every detected asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from twintag_backend.fixtures.tags import REFERENCE_TAGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, help="Directory containing sweep face JPGs")
    parser.add_argument("output", type=Path)
    parser.add_argument("--horizontal-margin", type=float, default=1.3)
    parser.add_argument("--vertical-margin", type=float, default=2.5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for tag in REFERENCE_TAGS:
        evidence = max(
            tag.evidence,
            key=lambda item: (item.box.right - item.box.left)
            * (item.box.bottom - item.box.top),
        )
        image_path = args.images / f"{evidence.image_id}.jpg"
        with Image.open(image_path) as image:
            box = evidence.box
            width, height = box.right - box.left, box.bottom - box.top
            crop_box = (
                max(0, int(box.left - width * args.horizontal_margin)),
                max(0, int(box.top - height * args.vertical_margin)),
                min(image.width, int(box.right + width * args.horizontal_margin)),
                min(image.height, int(box.bottom + height * args.vertical_margin)),
            )
            output = args.output / f"{tag.id}-{evidence.image_id}.jpg"
            image.crop(crop_box).convert("RGB").save(output, quality=95)
            print(output)


if __name__ == "__main__":
    main()
