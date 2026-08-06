"""Which of the 478 landmarks are actually the nose.

MediaPipe's FACEMESH_NOSE covers the dorsum but skips the alae and nostril sills
— the parts a rhinoplasty changes most visibly from the front. So the curated
indices below are unioned in, and then the whole set is filtered *geometrically*
in the nose frame. Any index that turns out to sit on a cheek or an eyelid gets
dropped by the filter rather than quietly warping the wrong pixels.
"""

from __future__ import annotations

import numpy as np

from .landmarks import nose_frame, to_local

_MIDLINE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
_LEFT = [122, 196, 236, 3, 51, 45, 44, 125, 141, 235, 31, 228, 229, 98, 97,
         129, 49, 48, 64, 219, 218, 237, 220, 115, 131, 134, 102, 60, 20, 79, 166]
_RIGHT = [351, 419, 456, 248, 281, 275, 274, 354, 370, 455, 261, 448, 449, 327,
          326, 358, 279, 278, 294, 439, 438, 457, 440, 344, 360, 363, 331, 290,
          250, 309, 392]

# Geometric gate, in nose-frame units (a/L along the dorsum, b/L lateral).
A_MIN, A_MAX = -0.10, 1.22
B_MAX = 0.62


def _mediapipe_nose():
    try:
        import mediapipe as mp

        conns = mp.solutions.face_mesh_connections.FACEMESH_NOSE
        return {i for c in conns for i in c}
    except Exception:
        return set()


def nose_indices(pts) -> np.ndarray:
    """Candidate union, then reject anything outside the nasal box."""
    cand = sorted(_mediapipe_nose() | set(_MIDLINE) | set(_LEFT) | set(_RIGHT))
    cand = [i for i in cand if i < len(pts)]

    R, u, v, L = nose_frame(pts)
    a, b = to_local(pts[cand], R, u, v)
    keep = (a / L > A_MIN) & (a / L < A_MAX) & (np.abs(b) / L < B_MAX)
    return np.array(cand, dtype=int)[keep]


def anchor_indices(pts, nose_idx, exclusion_ratio=0.30, stride=3) -> np.ndarray:
    """Landmarks held fixed so the rest of the face does not follow the nose.

    A band around the nose is left un-anchored. Anchoring right up to the alar
    crease pins the skin the deformation needs to borrow from, and you get a
    visible pinch along the nasolabial fold.

    `stride` thins the survivors. MLS cost is linear in control-point count and
    all ~380 face landmarks are wildly redundant for "hold the face still" —
    every third one pins it just as well for a third of the compute.
    """
    R, u, v, L = nose_frame(pts)
    nose_pts = pts[nose_idx]
    d = np.linalg.norm(pts[:, None, :] - nose_pts[None, :, :], axis=-1).min(1)
    mask = d > exclusion_ratio * L
    mask[nose_idx] = False
    return np.nonzero(mask)[0][::stride]


def band_width(pts, idx, R, u, v, L, a_lo, a_hi, b_limit=None):
    """Lateral span of the nose over a slice of its length.

    `b_limit` discards points beyond a lateral cutoff, which is how the tip
    lobule gets measured without the alar wings inflating it.
    """
    a, b = to_local(pts[idx], R, u, v)
    t = a / L
    sel = (t >= a_lo) & (t <= a_hi)
    if b_limit is not None:
        sel &= np.abs(b) <= b_limit
    if sel.sum() < 2:
        return None
    return float(b[sel].max() - b[sel].min())
