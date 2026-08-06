"""Reference adapter #2 -- landmark-seeded intensity-gradient edge search.

Landmarks locate the nose; the measurement itself comes from the image, by
finding the maximum intensity gradient (the nasofacial groove) inside a narrow
window around the landmark-suggested edge.

Recovers alar width at roughly 1:1 and is invariant to exposure and contrast,
because argmax of gradient magnitude does not move under affine intensity
change. Fails on bridge and tip, where there is no shadow line to find.
"""

from __future__ import annotations

import cv2

from .. import landmarks as LM
from .. import photometric as P
from ..landmarks import scale_ref
from .base import BANDS

NAME = "photometric"


def available() -> bool:
    return True


def measure(path):
    img = cv2.imread(path)
    if img is None:
        return None
    try:
        lm = LM.detect(img)
    except ValueError:
        return None

    gray = P.luma(img, equalize=True)
    ipd = scale_ref(lm.as_array())
    out = {}
    for name, (lo, hi) in BANDS.items():
        w = P.edge_width(img, lm, lo, hi, gray=gray)
        if w is not None:
            out[name] = w / ipd
    return out
