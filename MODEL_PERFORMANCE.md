# REX615 Detector Performance

## Model

- Architecture: YOLO11m
- Input size: 768 px for training; 1280 px for real-scan evaluation
- Training data: 10,000 synthetic images made from four device views and real substation backgrounds
- Training stopped after 9 epochs because the model had already converged

## Results

On the held-out synthetic validation set:

- mAP@50: **99.5%**
- mAP@50–95: **99.46%**

On 108 real VEO sweep images, the model produced 38 observations at the 85% confidence threshold. 3D fusion resolved these into **7 physical devices**, each confirmed from multiple sweeps. Correct detections commonly scored **85–97% confidence**.

## Recommended Use

- Confidence threshold: **0.80–0.85**
- Combine detections from multiple sweeps before creating a final asset tag
- Keep lower-confidence results for review instead of publishing them automatically

Lower thresholds produced false positives on cabinet details and labels. The real-scan results are a visual evaluation, not a manually annotated benchmark.
