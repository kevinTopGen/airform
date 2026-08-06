"""Adapter -- InsightFace buffalo_l 106-point 2D landmarks (2d106det.onnx).

A second opinion on the landmark-regression approach. MediaPipe's lateral nose
vertices are dominated by its shape prior; this asks whether a different
regressor, trained differently on a different dataset, has the same problem.

INDEX MAPPING -- determined empirically, not from memory. The 106 points were
projected into the MediaPipe nose frame on bench/normal/baseline.png and drawn
over the image (out/insightface_indices.jpg). Indices 72..86 are the nose, and
they resolve as:

    72, 73, 74      dorsum midline, radix -> supratip
    86              pronasale (the detector's own 5-point nose kp lands on it)
    80              subnasale
    75 / 81         upper sidewall, at nasion level      (t ~ 0.09)
    76 / 82         mid sidewall, supratip               (t ~ 0.61)
    77 / 83         alare -- the widest point            (t ~ 0.83)
    78 / 84         alar base / nostril sill             (t ~ 0.92)
    79 / 85         columella base                       (t ~ 0.95)

t is the height down the nose frame, 0 at the radix (72) and 1 at the subnasale
(80). Verified twice over: geometrically (the five pairs are symmetric about the
dorsum midline and their lateral offsets bracket MediaPipe's alar landmarks to
within 5%) and dynamically (under a +15% alar warp, 77 and 83 are the two
largest movers of all 106 points, and they move outward).

Five lateral samples per side is too sparse to read a band width off directly,
so each side becomes a piecewise-linear contour h(t) -- half-width as a function
of height -- and the nose width w(t) = h_left(t) + h_right(t) is evaluated
wherever a band needs it. Alar width is the maximum of w over the lower nose,
which is what alar width anatomically is; bridge and tip are band averages.
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .base import BANDS

NAME = "insightface_106"

# --- empirically identified index groups (see module docstring) -------------
NOSE = tuple(range(72, 87))
DORSUM = (72, 73, 74)        # midline, radix -> supratip
RADIX_I, SUBNASALE_I = 72, 80
SIDE_A = (75, 76, 77, 78, 79)   # one lateral chain, top -> bottom
SIDE_B = (81, 82, 83, 84, 85)   # the mirrored chain
EYE_A = tuple(range(33, 43))
EYE_B = tuple(range(87, 97))

# Mid-face silhouette, as mirrored index pairs (offset 16 along the contour).
# 9/25 sit at eye level, 12/28 at mid-cheek. Used as the scale reference --
# see _scale_ref for why this is not the interpupillary distance.
FACE_PAIRS = ((9, 25), (10, 26), (11, 27), (12, 28))

# Alar width is the widest point of the lower nose, not the span at any fixed
# height -- the subnasale sits below the flare, so a fixed band would measure
# the nostril sills. Same convention the ground-truth renderer uses.
ALAR_SEARCH = (0.55, 1.05)

_app = None
_lock = threading.Lock()
_fail = None


def _analysis():
    """FaceAnalysis, built once. Preparing it per call costs ~5 s."""
    global _app, _fail
    if _app is not None or _fail is not None:
        return _app
    with _lock:
        if _app is None and _fail is None:
            try:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                    # recognition/genderage are dead weight here and roughly
                    # double the per-image cost.
                    allowed_modules=["detection", "landmark_2d_106"],
                )
                app.prepare(ctx_id=-1, det_size=(640, 640))
                _app = app
            except Exception as e:  # corrupt/absent weights, onnxruntime, ...
                _fail = f"{type(e).__name__}: {e}"
    return _app


def available() -> bool:
    root = os.path.expanduser("~/.insightface/models/buffalo_l")
    needed = ("det_10g.onnx", "2d106det.onnx")
    if not all(os.path.exists(os.path.join(root, f)) for f in needed):
        return False
    return _analysis() is not None


def _landmarks(path):
    app = _analysis()
    if app is None:
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    # Largest face, in case the detector picks up something in the background.
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    p = getattr(f, "landmark_2d_106", None)
    return None if p is None else np.asarray(p, dtype=np.float64)


def _frame(pts):
    """Nose frame from insightface's own points: radix (72) -> subnasale (80).

    +u runs down the dorsum, +v is lateral. Deliberately self-contained -- the
    point of the benchmark is to compare techniques end to end, so this adapter
    never borrows MediaPipe's frame or MediaPipe's scale.
    """
    R = pts[RADIX_I]
    S = pts[SUBNASALE_I]
    axis = S - R
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        return None
    u = axis / L
    v = np.array([-u[1], u[0]])
    return R, u, v, L


def _scale_ref(pts):
    """Mid-face width -- NOT the interpupillary distance, on purpose.

    IPD is the project's usual normaliser, and for this regressor it is a bad
    one. Measured across the alar-warp series, the distance between the two eye
    clusters moves *with* the nose: +0.49% at a +15% alar warp, -0.61% at -20%,
    monotonically. Nothing in the image near the eyes changed, so that is the
    2d106 shape prior leaking nose width into the eyes. Dividing by it cancels
    ~12% of the alar signal.

    The mid-face silhouette pairs do not leak: their drift over all fifteen
    warps stays under 0.16% and shows no trend. They are far from the nose and
    twice the span of the IPD, so pixel noise is a smaller fraction of them.
    Matched mirror pairs rather than max|lateral| -- an extreme-value statistic
    over 33 contour points is the noisier estimator of the same quantity.
    """
    d = float(np.mean([np.linalg.norm(pts[i] - pts[j]) for i, j in FACE_PAIRS]))
    return d if d > 1 else None


def _contour(pts, R, u, v, L):
    """Half-width of each nasal sidewall as a function of height down the nose.

    Returns (t_grid, w) where w is the full width, both sides summed, sampled
    densely enough to average or maximise over any band.
    """
    rel = pts - R
    t_all = (rel @ u) / L
    b_all = (rel @ v) / L

    grid = np.linspace(0.0, 1.10, 221)
    total = np.zeros_like(grid)
    for chain in (SIDE_A, SIDE_B):
        idx = list(chain)
        t = t_all[idx]
        h = np.abs(b_all[idx])          # half-width, sign-agnostic so the two
        order = np.argsort(t)           # chains need no left/right assignment
        # np.interp clamps outside the chain's span, which is what we want:
        # below 75/81 there is no sidewall left to measure.
        total += np.interp(grid, t[order], h[order])
    return grid, total * L              # back to pixels


def measure(path):
    pts = _landmarks(path)
    if pts is None or len(pts) < 106:
        return None
    fr = _frame(pts)
    ipd = _scale_ref(pts)
    if fr is None or ipd is None:
        return None
    R, u, v, L = fr

    grid, w = _contour(pts, R, u, v, L)

    def band_mean(lo, hi):
        sel = (grid >= lo) & (grid <= hi)
        return float(w[sel].mean()) if sel.any() else None

    def band_max(lo, hi):
        sel = (grid >= lo) & (grid <= hi)
        return float(w[sel].max()) if sel.any() else None

    out = {}
    a = band_max(*ALAR_SEARCH)
    if a is not None:
        out["alar_width"] = a / ipd
    for name in ("bridge_width", "tip_width"):
        lo, hi = BANDS[name]
        m = band_mean(lo, hi)
        if m is not None:
            out[name] = m / ipd
    return out


def _overlay(path="bench/normal/baseline.png", dst="out/insightface_indices.jpg"):
    """Draw the index mapping so it can be checked by eye.

        python -m nosesim.measure.insightface_106

    Nose group red and numbered, sidewall chains joined, scale-reference pairs
    joined in white, everything else small and green.
    """
    img = cv2.imread(path)
    pts = _landmarks(path)
    if img is None or pts is None:
        return None
    o = img.copy()
    for i, j in FACE_PAIRS:
        cv2.line(o, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)),
                 (255, 255, 255), 1, cv2.LINE_AA)
    for chain in (SIDE_A, SIDE_B):
        cv2.polylines(o, [pts[list(chain)].astype(np.int32)], False,
                      (0, 200, 255), 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        nose = i in NOSE
        cv2.circle(o, (int(round(x)), int(round(y))), 4 if nose else 2,
                   (0, 0, 255) if nose else (0, 220, 0), -1)
        cv2.putText(o, str(i), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 if nose else 0.3,
                    (0, 0, 255) if nose else (200, 200, 0), 1, cv2.LINE_AA)
    fr = _frame(pts)
    if fr is not None:
        R, u, v, L = fr
        for lo, hi in list(BANDS.values()) + [ALAR_SEARCH]:
            for t in (lo, hi):
                c = R + u * (t * L)
                cv2.line(o, tuple((c - v * 0.75 * L).astype(int)),
                         tuple((c + v * 0.75 * L).astype(int)), (255, 0, 255), 1)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, o, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dst


if __name__ == "__main__":
    print(_overlay(), measure("bench/normal/baseline.png"))
