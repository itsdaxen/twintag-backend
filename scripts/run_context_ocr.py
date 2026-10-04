"""Extract structured industrial context from full-resolution asset crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = """Analyze the industrial cabinet around the relay in this image.
Read visible text and classify it by physical source. Return JSON only with this schema:
{
  "official_labels": [{"text": "", "confidence": 0.0}],
  "inspection_markings": [{"text": "", "confidence": 0.0}],
  "field_notes": [{"text": "", "confidence": 0.0}]
}
Official labels are engraved or printed equipment/location plates. Inspection markings are
quality-control or test stickers. Field notes are handwritten or temporary labels. Confidence
must reflect legibility from 0 to 1. Do not infer or translate text. Omit anything unreadable.
"""


def extract_json(text: str) -> dict[str, object]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return JSON: {text}")
    return json.loads(text[start : end + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    results: dict[str, object] = {}
    for image_path in sorted(args.input_dir.glob("*.jpg")):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0]
        results[image_path.stem] = extract_json(answer)
        print(
            json.dumps({image_path.stem: results[image_path.stem]}, ensure_ascii=False)
        )

    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
