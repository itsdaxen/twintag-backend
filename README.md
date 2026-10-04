# TwinTag Backend

Backend services for turning industrial digital-twin scans into reviewed, interactive asset tags.

```bash
uv sync
uv run twintag-backend
```

Run checks with `uv run pytest` and `uv run ruff check .`.

## Context OCR

TwinTag selects the largest full-resolution evidence view for each detected asset, then
uses Qwen3-VL to separate official labels, inspection markings, and unverified field
notes. The checked-in JSON contains genuine precomputed inference for the demo scan.

```bash
uv run python scripts/export_context_crops.py artifacts/veo-reference/images /tmp/context-crops
python -m venv .venv-ocr
.venv-ocr/bin/pip install -r scripts/requirements-ocr.txt
.venv-ocr/bin/python scripts/run_context_ocr.py /tmp/context-crops \
  src/twintag_backend/fixtures/data/context_ocr_qwen3_vl_8b.json
```
