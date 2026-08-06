"""Reference adapter #1 -- MediaPipe FaceMesh landmarks.

The incumbent, and the one already shown to fail on width: its lateral nose
vertices come from the model's shape prior rather than the image. Included as
the control that every other technique has to beat.
"""

from __future__ import annotations

import cv2

from .. import landmarks as LM
from ..landmarks import nose_frame, scale_ref
from ..nose_region import band_width, nose_indices
from .base import BANDS

NAME = "mp_mesh"


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

    pts = lm.as_array()
    idx = nose_indices(pts)
    R, u, v, L = nose_frame(pts)
    ipd = scale_ref(pts)

    out = {}
    for name, (lo, hi) in BANDS.items():
        w = band_width(pts, idx, R, u, v, L, lo, hi)
        if w is not None:
            out[name] = w / ipd
    return out
