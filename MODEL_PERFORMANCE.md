# REX615 Detector Performance

## Model

- Architecture: YOLO11m
- Input size: 768 px for training; 1280 px for real-scan evaluation
- Training data: 10,000 synthetic images made from four device views and real substation backgrounds: **8,000 training / 2,000 validation**
- The retained [training CSV](docs/model/rex615-training-results.csv) records **10 epochs**
- The saved best checkpoint is from **epoch 8**: its zero-based `epoch` is `7`, and its validation metrics match CSV row 8

## Training Environment

Training ran on **RunPod Secure Cloud with an NVIDIA GeForce RTX 4090**. The original hackathon session retains the successful pod allocation, GPU-query output, and Ultralytics runtime output confirming the device. The recorded runtime used **Python 3.12.3, PyTorch 2.8.0+cu128, and Ultralytics 8.4.172**. See the [sanitized run provenance](docs/model/training-provenance.json).

The old chat summary said training stopped at epoch 9 while the stop was being handled. The completed CSV records ten epochs, and the retained best checkpoint is from epoch eight; the validation scores below refer to that best checkpoint.

## Context Extraction

Pretrained **Qwen/Qwen3-VL-8B-Instruct** processed seven full-resolution evidence crops. This was context/OCR inference, not fine-tuning. Its structured outputs are saved in `src/twintag_backend/fixtures/data/context_ocr_qwen3_vl_8b.json`.

## Results

For the saved best checkpoint, on the held-out synthetic validation set:

- mAP@50: **99.5%**
- mAP@50–95: **99.487%** (**99.49%** rounded)
- Precision: **100%**; recall: **99.995%**

On **108 real VEO sweep images**, the model produced **38 observations** at an **85% confidence threshold**. Multi-view 3D fusion combined these into **7 asset tags**. Recorded detection confidence ranges from **85.12% to 96.75%**.

The 85% setting is a confidence cutoff, not a measured real-site accuracy. The retained scan has no ground-truth detection benchmark. Original run outputs are `artifacts/rex615-run/detections.json` and `artifacts/rex615-run/tags.json`; these scan artifacts and the checkpoint in `models/rex615-yolo11m-best.pt` remain local inputs.

## Recommended Use

- Confidence threshold: **0.80–0.85**
- Combine detections from multiple sweeps before creating a final asset tag
- Keep lower-confidence results for review instead of publishing them automatically

Lower thresholds produced false positives on cabinet details and labels.
