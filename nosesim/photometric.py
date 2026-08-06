"""Nose width measured from pixels instead of from landmark positions.

WHY THIS EXISTS
FaceMesh places its lateral nose vertices almost entirely from its shape prior.
Warp a nose 20% narrower and those vertices move ~1%, non-monotonically
(scripts/calibrate.py reproduces this). The nose sidewall is smooth,
low-contrast skin with nothing for a landmark regressor to lock onto, so the
prior wins. Vertical landmarks are fine -- tip and subnasale sit on real edges --
which is why nasal_length behaves and the widths do not.

WHAT WORKS
`alar_width`. The nasofacial groove, where the sidewall meets the cheek, is a
genuine shadow edge. Seeding a narrow search window from the landmarks (for
location only) and taking the maximum intensity gradient inside it tracks a
known warp at roughly 1:1:

    requested  -20.0%  -12.0%  +10.0%
    measured   -18.3%   -9.2%   +9.4%

WHAT DOES NOT
`bridge_width` and `tip_width`. Measured -2.2% / +1.4% / +7.9% against the same
sweep -- no better than the landmarks. High on the nose there is no shadow line
to find: the dorsum blends into the cheek with no edge at all. Two different
criteria (darkest-point and max-gradient) both failed, so this is a property of
the photograph, not of the search.

Options, in the order worth trying: a face-parsing segmentation model with a
dedicated nose class (BiSeNet et al.) so width comes from a mask boundary
rather than an edge; a profile view, where the silhouette is unambiguous; or
hand-annotating four points per training photo, which for ~60 pairs is twenty
minutes and perfectly reliable.

Landmarks still define the nose frame and the search window. They just stop
being the thing that gets measured.
"""

from __future__ import annotations

import cv2
import numpy as np

from .landmarks import nose_frame, scale_ref, to_local
from .nose_region import nose_indices

BANDS = {
    "alar_width": (0.80, 1.00),
    "bridge_width": (0.25, 0.55),
    "tip_width": (0.60, 0.80),
}

# Only alar_width survived validation against a known warp. The others are
# computed but must not be used to fit a signature until a better measurement
# lands -- fitting on them would be fitting on noise.
TRUSTED = ("alar_width",)

SEARCH_WIN = 0.20  # +/- nose-lengths around the landmark-seeded edge location


def luma(image_bgr, equalize=True):
    """Search channel for the gradient hunt.

    CIELAB L* rather than a naive BGR mean: L* is perceptually uniform, so a
    given shadow reads as a similar step regardless of the base tone it sits on.
    CLAHE then normalises local contrast, which is what keeps the nasofacial
    groove findable on darker skin and under flat lighting -- both compress the
    crease into very few code values in plain 8-bit gray.
    """
    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    if equalize:
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    return cv2.GaussianBlur(L.astype(np.float32), (5, 5), 0)


def edge_width(image_bgr, lm, t_lo, t_hi, samples=7, win=SEARCH_WIN, gray=None):
    """Nose width from image gradients, in pixels. None if unmeasurable."""
    if gray is None:
        gray = luma(image_bgr)
    pts = lm.as_array()
    R, u, v, L = nose_frame(pts)

    idx = nose_indices(pts)
    a, b = to_local(pts[idx], R, u, v)
    sel = (a / L >= t_lo) & (a / L <= t_hi)
    if sel.sum() < 2:
        return None
    seeds = (b[sel].min(), b[sel].max())  # landmarks give location, not width

    widths = []
    for t in np.linspace(t_lo, t_hi, samples):
        edges = []
        for seed in seeds:
            bs = np.linspace(seed - win * L, seed + win * L, 121)
            p = R + u * (t * L) + np.outer(bs, v)
            val = cv2.remap(gray,
                            p[:, 0].astype(np.float32).reshape(-1, 1),
                            p[:, 1].astype(np.float32).reshape(-1, 1),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
            edges.append(bs[int(np.argmax(np.abs(np.gradient(val))))])
        widths.append(edges[1] - edges[0])
    return float(np.median(widths))


def measure_widths(image_bgr, lm, trusted_only=True):
    """Photometric width params in IPD units.

    `trusted_only` returns just the measurements that survived validation.
    Pass False to inspect the others; do not fit signatures on them.
    """
    ipd = scale_ref(lm.as_array())
    names = TRUSTED if trusted_only else tuple(BANDS)
    out = {}
    for name in names:
        w = edge_width(image_bgr, lm, *BANDS[name])
        if w is not None:
            out[name] = w / ipd
    return out
