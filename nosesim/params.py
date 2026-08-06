"""Stage 2: Landmarks -> NoseParams.

This is the step that replaces "train a model". Seven scalars, each one a
measurement a surgeon would actually recognise, normalised by interpupillary
distance so they compare across photos, cameras and people.

Averaging seven robust scalars over ~15 before/after pairs is stable. Averaging
60 raw landmark deltas over 15 pairs is mostly averaging landmark noise.
"""

from __future__ import annotations

import numpy as np

from .contracts import Landmarks, NoseParams
from .landmarks import NOSE_TIP, RADIX, nose_frame, scale_ref, to_local
from .nose_region import band_width, nose_indices

# Slices of the nose, as a fraction of radix->subnasale distance.
#
# The subnasale sits at the base of the columella, *below* where the alae flare
# widest, so the widest point of the nose lands near t=0.75 rather than t=1.
# Alar width is therefore the max lateral extent anywhere over the lower nose,
# not the span at t~1 -- that would measure the nostril sills instead.
BRIDGE_BAND = (0.25, 0.55)
ALAR_BAND = (0.55, 1.30)
TIP_BAND = (0.60, 0.88)
TIP_LOBULE_FRAC = 0.60  # of alar half-width; excludes the wings from the tip


def measure(lm: Landmarks) -> NoseParams:
    pts = lm.as_array()
    idx = nose_indices(pts)
    R, u, v, L = nose_frame(pts)
    ipd = scale_ref(pts)

    p = NoseParams(nasal_length=L / ipd)

    if lm.view in ("frontal", "three_quarter"):
        alar = band_width(pts, idx, R, u, v, L, *ALAR_BAND)
        bridge = band_width(pts, idx, R, u, v, L, *BRIDGE_BAND)
        tip = None
        if alar:
            tip = band_width(pts, idx, R, u, v, L, *TIP_BAND,
                             b_limit=TIP_LOBULE_FRAC * alar / 2)
        for name, val in (("alar_width", alar), ("bridge_width", bridge),
                          ("tip_width", tip)):
            if val is not None:
                setattr(p, name, val / ipd)

    if lm.view == "profile":
        p.dorsal_hump = _dorsal_hump(pts, idx, ipd)
        p.tip_projection = _tip_projection(pts, ipd)
        p.tip_rotation_deg = _tip_rotation(pts)

    return p


def _dorsal_hump(pts, idx, ipd):
    """Peak deviation of the dorsum from the straight radix->tip line.

    Positive = convex (a hump), negative = scooped. Meaningless head-on, which
    is why measure() gates it on view.
    """
    R, T = pts[RADIX], pts[NOSE_TIP]
    axis = T - R
    n = np.array([-axis[1], axis[0]]) / (np.linalg.norm(axis) + 1e-9)
    dev = (pts[idx] - R) @ n
    return float(np.abs(dev).max() / ipd) * float(np.sign(dev[np.abs(dev).argmax()]))


def _tip_projection(pts, ipd):
    from .landmarks import SUBNASALE

    return float(np.linalg.norm(pts[NOSE_TIP] - pts[SUBNASALE]) / ipd)


def _tip_rotation(pts):
    """Nasolabial angle proxy: columella direction vs the facial vertical."""
    from .landmarks import SUBNASALE

    c = pts[NOSE_TIP] - pts[SUBNASALE]
    return float(np.degrees(np.arctan2(-c[1], abs(c[0]) + 1e-9)))


def pose_gap_deg(a: Landmarks, b: Landmarks) -> float:
    """Head-pose mismatch between a before/after pair.

    The single highest-value data filter in the pipeline. If the 'after' photo
    was shot at a different angle, the measured delta is mostly parallax and it
    will poison the surgeon's mean. Reject pairs above ~10 degrees.
    """
    return abs(a.yaw_deg - b.yaw_deg)
