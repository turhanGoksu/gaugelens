"""Find the same photo in two image sources by perceptual hash (D10).

A difference hash (dHash) shrinks the image to 17x16 grey pixels and keeps
one bit per pair of neighbours: is the left one brighter? Re-encoding,
resizing and small colour changes keep almost all 256 bits. Photos of
different gauges flip a quarter or more of them: round white dials look
alike at 9x8 pixels, so the smaller 64-bit hash mistakes them for copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

SIZE = 16
# Bits that may differ for two files to count as the same photo. Copies
# measured 0-1 bits apart; the closest different dial photos, 68.
MAX_BITS = 25


def dhash(path: str | Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    small = cv2.resize(image, (SIZE + 1, SIZE), interpolation=cv2.INTER_AREA)
    small = small.astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicates(
    first: Mapping[str, int], second: Mapping[str, int], max_bits: int = MAX_BITS
) -> list[tuple[str, str, int]]:
    """(id in first, id in second, differing bits) for every close pair."""
    return [
        (a, b, d)
        for a, ha in first.items()
        for b, hb in second.items()
        if (d := distance(ha, hb)) <= max_bits
    ]
