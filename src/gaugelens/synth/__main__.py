"""Command line for the synthetic gauge generator.

    python -m gaugelens.synth download-backgrounds data/backgrounds
    python -m gaugelens.synth generate data/synthetic/dev --split dev --count 1000 \\
        --backgrounds data/backgrounds --workers 8
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from gaugelens.synth.backgrounds import download_polyhaven
from gaugelens.synth.dataset import DatasetConfig, write_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gaugelens.synth")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download-backgrounds", help="fetch CC0 photos")
    download.add_argument("dest")
    download.add_argument("--panoramas", type=int, default=30)
    download.add_argument("--textures", type=int, default=60)

    generate = commands.add_parser("generate", help="write a synthetic dataset")
    generate.add_argument("out")
    generate.add_argument("--split", choices=["train", "dev"], required=True)
    generate.add_argument("--count", type=int, required=True)
    generate.add_argument("--seed-start", type=int, default=None)
    generate.add_argument("--size", type=int, default=640, help="square image side")
    generate.add_argument("--out-of-range", type=float, default=0.1)
    generate.add_argument("--backgrounds", default=None, help="folder of photos")
    generate.add_argument("--workers", type=int, default=1)

    args = parser.parse_args(argv)
    if args.command == "download-backgrounds":
        entries = download_polyhaven(args.dest, args.panoramas, args.textures)
        print(f"{len(entries)} background photos in {args.dest}")
        print("Assets from Poly Haven (https://polyhaven.com), CC0. See manifest.json.")
        return 0

    config = DatasetConfig(
        split=args.split,
        count=args.count,
        seed_start=args.seed_start,
        width=args.size,
        height=args.size,
        out_of_range=args.out_of_range,
        backgrounds=args.backgrounds,
    )
    start = time.perf_counter()
    write_dataset(args.out, config, workers=args.workers)
    seconds = time.perf_counter() - start
    print(f"{config.count} {config.split} scenes in {args.out} ({seconds:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
