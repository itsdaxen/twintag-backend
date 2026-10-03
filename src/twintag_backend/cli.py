import argparse
from pathlib import Path

from twintag_backend.e57.calibration import E57CalibrationExporter
from twintag_backend.e57.images import E57ImageExtractor
from twintag_backend.e57.potree import PotreeConverter
from twintag_backend.e57.preview import E57PreviewExporter
from twintag_backend.synthetic.exporter import YoloDatasetExporter
from twintag_backend.synthetic.generator import DEFAULT_BACKGROUNDS_DIRECTORY


def run() -> None:
    parser = argparse.ArgumentParser(prog="twintag")
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="Create a browser preview source")
    preview.add_argument("source", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--points", type=int, default=2_000_000)
    preview.add_argument("--potree-output", type=Path)
    preview.add_argument("--converter", type=Path)
    panoramas = commands.add_parser(
        "panoramas", help="Extract registered panorama cube faces"
    )
    panoramas.add_argument("source", type=Path)
    panoramas.add_argument("--output", type=Path, required=True)
    calibration = commands.add_parser(
        "calibrate", help="Verify the E57 image-to-point-cloud registration"
    )
    calibration.add_argument("source", type=Path)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--sweep", type=int, default=0)
    calibration.add_argument("--face", type=int, default=0)
    calibration.add_argument("--points", type=int, default=1_000_000)
    calibration.add_argument("--size", type=int, default=1024)
    calibration.add_argument("--images", type=Path)
    calibration.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Resolve an image-space detection box to a 3D location",
    )
    synthetic = commands.add_parser(
        "synthetic-dataset", help="Export a detector-ready YOLO dataset"
    )
    synthetic.add_argument("sources", type=Path, nargs=4)
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--samples", type=int, default=10_000)
    synthetic.add_argument("--validation", type=float, default=0.2)
    synthetic.add_argument(
        "--backgrounds", type=Path, default=DEFAULT_BACKGROUNDS_DIRECTORY
    )
    synthetic.add_argument("--seed", type=int, default=615)
    synthetic.add_argument("--class-name", default="ABB REX615")
    synthetic.add_argument("--size", type=int, default=768)
    args = parser.parse_args()

    if args.command == "synthetic-dataset":
        result = YoloDatasetExporter().export(
            args.sources,
            args.output,
            count=args.samples,
            validation_fraction=args.validation,
            backgrounds=args.backgrounds,
            seed=args.seed,
            class_name=args.class_name,
            size=args.size,
        )
        print(
            f"Exported {result.training_samples:,} training and "
            f"{result.validation_samples:,} validation samples to {args.output}"
        )
        return

    if args.command == "panoramas":
        images = E57ImageExtractor().extract(args.source, args.output)
        print(f"Extracted {len(images):,} panorama faces to {args.output}")
        return

    if args.command == "calibrate":
        report = E57CalibrationExporter().export(
            args.source,
            args.output,
            sweep_index=args.sweep,
            face_index=args.face,
            max_points=args.points,
            max_size=args.size,
            images_directory=args.images,
            bounding_box=tuple(args.box) if args.box else None,
        )
        print(
            f"Projected {report.projected_points:,} points onto "
            f"{report.image}; {report.visible_pixels:,} visible pixels"
        )
        if report.location:
            point = report.location.position
            print(
                f"Detection mapped to XYZ ({point.x:.4f}, {point.y:.4f}, {point.z:.4f})"
            )
        return

    result = E57PreviewExporter().export(args.source, args.output, args.points)
    print(f"Exported {result.point_count:,} points to {result.path}")
    if args.potree_output:
        metadata = PotreeConverter(args.converter).convert(
            result.path,
            args.potree_output,
        )
        print(f"Built browser point cloud at {metadata}")
