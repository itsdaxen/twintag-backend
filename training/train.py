import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=float, default=-1)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    config = yaml.safe_load(dataset.read_text())
    config["path"] = str(dataset.parent)
    runtime_config = dataset.parent / "dataset.runtime.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))

    YOLO(args.model).train(
        data=str(runtime_config),
        epochs=args.epochs,
        imgsz=768,
        batch=args.batch,
        project=str(dataset.parent / "runs"),
        name="device-detector",
        patience=10,
        seed=615,
    )


if __name__ == "__main__":
    run()
