"""Synthetic profiles with known geometry, for scoring the profile measurement.

The scraped before/afters can only be checked for *direction* -- nobody publishes
the millimetres. This module supplies the other half: a silhouette whose nasion,
pronasale, subnasale and labrale are placed by construction, so the exact value
of every parameter is known before the image exists. Rendering it and measuring
it back gives the same two numbers `scripts/bench.py` reports for the frontal
adapters:

    k   slope of measured against true. 1.0 is a gauge that reports all of a
        change; a stable k below 1 is calibratable, k near 0 is not.
    r   correlation. The primary metric -- a measurement that does not track
        reality cannot be rescued by any correction.

What it does and does not prove: it exercises segmentation, contour extraction,
the hull decomposition and the landmark rules end to end on shapes of known size,
under a light and a dark backdrop. It does not prove anything about skin texture,
hair, jewellery, or head pose, because a filled polygon has none of those. Treat
it as a floor, not a ceiling.

Face proportions below are in units of nasal length (nasion -> subnasale) with
the origin at the nasion, +x anterior and +y down.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .contracts import NoseParams

# (name, x, y). Unnamed points are shape only; named ones are ground truth.
BASE: List[Tuple[str, float, float]] = [
    ("", -0.60, -1.45),      # top of the forehead
    ("", -0.20, -0.70),
    ("", 0.05, -0.26),       # glabella
    ("brow", 0.11, -0.11),
    ("nasion", 0.00, 0.00),
    ("", 0.17, 0.34),        # upper dorsum -- the hump rides here
    ("", 0.275, 0.55),      # both sit exactly on the nasion->tip chord, so the
                            # unmodified dorsum is dead straight and hump=0
    ("pronasale", 0.36, 0.72),
    ("", 0.27, 0.92),        # columella
    ("subnasale", 0.06, 1.00),
    ("", 0.10, 1.14),
    ("labrale", 0.17, 1.30),
    ("", 0.10, 1.42),        # stomion
    ("", 0.18, 1.54),        # lower lip
    ("", 0.02, 1.72),        # mentolabial sulcus
    ("", 0.17, 1.94),        # pogonion
    ("", 0.00, 2.10),
    ("", -0.55, 2.35),       # under the jaw
]
HUMP_AT = 5          # index the hump displaces
DORSUM = (5, 6)      # indices that carry a projection change with the tip


def _catmull_rom(pts: np.ndarray, per_seg: int = 40) -> np.ndarray:
    """Dense smooth curve through every control point."""
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i:i + 4]
        t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2][None, :])
    return np.vstack(out)


def control_points(hump: float = 0.0, proj: float = 0.0,
                   rot: float = 0.0) -> np.ndarray:
    """The face-frame control points with a hump, projection and rotation applied.

    `hump` displaces the upper dorsum anteriorly (units of nasal length; negative
    scoops it). `proj` pushes the tip and the dorsum below it forward together,
    which is what an increase in tip projection actually is. `rot` swings the tip
    and columella up about the subnasale, in degrees -- tip rotation, the
    manoeuvre that opens the nasolabial angle.
    """
    pts = np.array([[x, y] for _, x, y in BASE], dtype=np.float64)
    pts[HUMP_AT, 0] += hump
    pts[HUMP_AT + 1, 0] += hump * 0.55     # spread it, so the hump is a curve
    for i in DORSUM:
        pts[i, 0] += proj * (0.4 if i == DORSUM[0] else 0.7)
    pts[7, 0] += proj                      # pronasale
    pts[8, 0] += proj * 0.6                # columella follows the tip
    if rot:
        t = np.radians(rot)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        piv = pts[9]                       # subnasale
        for i in (6, 7, 8):
            w = {6: 0.4, 7: 1.0, 8: 1.0}[i]
            pts[i] = piv + (R @ (pts[i] - piv)) * w + (pts[i] - piv) * (1 - w)
    return pts


def truth(hump: float = 0.0, proj: float = 0.0, rot: float = 0.0) -> NoseParams:
    """Exact parameters of the synthetic face, from the curve, not from an image."""
    pts = control_points(hump, proj, rot)
    curve = _catmull_rom(pts)
    idx = {n: i for i, (n, _, _) in enumerate(BASE) if n}
    P = {n: pts[i] for n, i in idx.items()}
    scale = np.linalg.norm(P["labrale"] - P["nasion"])

    # Same definitions as profile.measure_profile, evaluated on the exact curve.
    chord = P["pronasale"] - P["nasion"]
    nrm = np.array([chord[1], -chord[0]])
    nrm = nrm / np.linalg.norm(nrm)
    if nrm[0] < 0:
        nrm = -nrm
    lo = int(np.linalg.norm(curve - P["nasion"], axis=1).argmin())
    hi = int(np.linalg.norm(curve - P["pronasale"], axis=1).argmin())
    dev = (curve[lo:hi + 1] - P["nasion"]) @ nrm
    h = float(dev[np.abs(dev).argmax()]) / scale

    base = P["subnasale"] - P["nasion"]
    bn = np.array([base[1], -base[0]])
    bn = bn / np.linalg.norm(bn)
    tp = float(abs((P["pronasale"] - P["nasion"]) @ bn)) / scale

    v1, v2 = P["pronasale"] - P["subnasale"], P["labrale"] - P["subnasale"]
    ang = float(np.degrees(np.arccos(np.clip(
        v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))

    return NoseParams(nasal_length=float(np.linalg.norm(base)) / scale,
                      dorsal_hump=h, tip_projection=tp, tip_rotation_deg=ang)


def render(hump: float = 0.0, proj: float = 0.0, rot: float = 0.0,
           size: int = 1300, tone: str = "light") -> np.ndarray:
    """A filled head silhouette on a plain backdrop, facing image-right."""
    curve = _catmull_rom(control_points(hump, proj, rot))
    px = size / 4.6                             # pixels per nasal length
    origin = np.array([size * 0.55, size * 0.42])
    poly = curve * px + origin
    back = np.array([[origin[0] - 3 * px, poly[-1, 1] + 2 * px],
                     [origin[0] - 3 * px, poly[0, 1] - 2 * px]])
    poly = np.vstack([poly, back]).astype(np.int32)

    bg, skin = ((255, 255, 255), (168, 186, 208)) if tone == "light" \
        else ((6, 6, 6), (74, 88, 106))
    img = np.full((size, size, 3), bg, np.uint8)
    cv2.fillPoly(img, [poly], skin)
    return cv2.GaussianBlur(img, (3, 3), 0)     # anti-alias the edge


def sweep(param: str = "hump", levels=(-0.06, -0.03, 0.0, 0.03, 0.06, 0.09),
          tone: str = "light") -> Dict[str, list]:
    """Render and measure the sweep; returns the true and measured series."""
    from .profile import measure_profile

    out = {"level": [], "true": [], "meas": [], "field": []}
    field = {"hump": "dorsal_hump", "proj": "tip_projection",
             "rot": "tip_rotation_deg"}[param]
    for v in levels:
        kw = {param: v}
        res = measure_profile(render(tone=tone, **kw))
        if not res.ok:
            continue
        out["level"].append(v)
        out["true"].append(getattr(truth(**kw), field))
        out["meas"].append(getattr(res.params, field))
    out["field"] = field
    return out
