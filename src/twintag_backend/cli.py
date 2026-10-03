import argparse
from pathlib import Path

from twintag_backend.e57.potree import PotreeConverter
from twintag_backend.e57.preview import E57PreviewExporter


def run() -> None:
    parser = argparse.ArgumentParser(prog="twintag")
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="Create a browser preview source")
    preview.add_argument("source", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--points", type=int, default=2_000_000)
    preview.add_argument("--potree-output", type=Path)
    preview.add_argument("--converter", type=Path)
    args = parser.parse_args()

    result = E57PreviewExporter().export(args.source, args.output, args.points)
    print(f"Exported {result.point_count:,} points to {result.path}")
    if args.potree_output:
        metadata = PotreeConverter(args.converter).convert(
            result.path,
            args.potree_output,
        )
        print(f"Built browser point cloud at {metadata}")
