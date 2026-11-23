#!/usr/bin/env python
"""Write a synthetic capture session zip in the real data format.

    python tools/make_fake_session.py data/session_fake_train.zip --with-masks
    python tools/make_fake_session.py data/session_fake_test.zip --split test --seed 1 --with-masks
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from unetultra.fake_session import make_fake_session  # noqa: E402
from unetultra.validate import validate_session  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", help="output zip path, e.g. data/session_fake_train.zip")
    parser.add_argument("--n", type=int, default=24, help="number of samples")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="+", default=["airpods_case"], help="class names, without background")
    parser.add_argument("--image-size", type=int, nargs=2, default=[960, 720], metavar=("W", "H"))
    parser.add_argument("--with-masks", action="store_true", help="include true masks, marked accepted")
    args = parser.parse_args()
    path = make_fake_session(
        args.out, n_samples=args.n, split=args.split, classes=["background", *args.classes],
        image_size=tuple(args.image_size), seed=args.seed, with_masks=args.with_masks,
    )
    print(validate_session(path).format())


if __name__ == "__main__":
    main()
