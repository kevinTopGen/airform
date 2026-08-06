"""Reference adapter #3 -- iso-shading contours of the dorsal light reflex.

WHY THE INCUMBENT UNDER-REPORTS THE TIP
`photometric` seeds its search window from the lateral nose landmarks and takes
the maximum intensity gradient inside it. Those seeds sit on the *silhouette*
(|b| ~ 0.9 of the alar half-width, where b is the lateral nose-frame
coordinate), so the argmax locks onto the nasofacial groove -- the crease where
the nose meets the cheek.

Measuring the achieved displacement field directly (block-match the baseline
against a -20% tip render, row by row, in the nose frame) shows why that is
fatal:

    |b| / alar-half-width      0.3    0.5    0.7    0.9    1.0
    fraction of the requested
    lateral scale realised     0.80   0.65   0.25   0.05   0.00

A tip warp is a *lobule* warp. Past |b| ~ 0.7 the image barely moves at all, so
a measurement anchored there is structurally incapable of reporting the change
no matter how well it is localised. That is the whole of photometric's
tip_width k = 0.089: high r (the groove does drift a little, monotonically) with
almost no gain. The detector was never the problem -- the search window was.

WHAT THIS ADAPTER DOES INSTEAD
Work inward, not outward. Along each row of the nose frame the lateral profile
of CIELAB L* is a single broad hill: the dorsal light reflex, falling away on
both flanks. Take a contour of that hill -- the lateral positions where L* has
fallen a fixed fraction of the way from the ridge down to a reference minimum.
The contour lands at |b| ~ 0.2-0.5, inside the region the warp actually moves,
which is what recovers the gain.

Two properties make the contour a good measurement rather than a lucky one:

  * It is invariant to affine intensity change. Both the ridge value and the
    reference minimum are transported by any gain/contrast transform, so the
    threshold moves with them and the crossing does not. That is why bridge
    width survives the dark_flat tone here (r 0.68 -> 0.99) where a CLAHE-based
    gradient argmax does not -- CLAHE is local and non-affine, so under low
    contrast it re-ranks competing weak edges and the argmax hops.

  * It is a level crossing on a slope, refined sub-pixel, not an argmax or an
    argmin. It cannot branch-jump between two competing extrema, which is
    exactly the failure mode that makes direct alar-groove valley detection
    noisy (r = 0.94) on the same images.

The reference minimum differs per band because the anatomy does:

  tip_width     ridge -> the alar groove, the crease bounding the tip lobule,
                sought in a window inside the silhouette. Contour at 80% of that
                drop, i.e. just inside the crease.
  bridge_width  there is no crease on the dorsum, so the reference is the
                nasofacial groove and the contour is taken at half maximum --
                the dorsal aesthetic line, the visible width of the bridge.
  alar_width    a genuine shadow edge exists here and the incumbent already
                recovers it near 1:1, so this is `photometric`'s gradient search
                unchanged, on raw L* rather than CLAHE L* for tone stability.

LIMITS
The contour is proportional to the true width only for a fixed cross-section
shape, so the three numbers are three different gauges and their absolute ratios
are not anatomical ratios (they still order alar > tip > bridge). It needs a
directional-enough light to give the dorsum a reflex with two resolvable flanks;
under perfectly flat frontal light the hill loses its shoulders and the crossing
drifts. And it assumes intensity is transported by the deformation, which is
true of a warped photograph and only approximately true of a real post-op face,
where narrowing the lobule also re-shades it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import landmarks as LM
from .. import photometric as P
from ..landmarks import nose_frame, scale_ref, to_local
from ..nose_region import nose_indices

NAME = "photometric_v2"

# Bands in nose-frame units (fraction of radix->subnasale distance).
# tip: the lobule proper, from the supratip break to the nostril apices. The
# 0.60-0.80 band in base.BANDS straddles the supratip, where half the rows are
# still dorsum and the lobule crease has not formed yet.
TIP_BAND = (0.78, 0.95)
BRIDGE_BAND = (0.28, 0.55)
ALAR_BAND = (0.80, 1.00)

N_ROWS = 11          # profile rows per band; the width is their median
N_SAMPLES = 601      # lateral samples per profile
B_SPAN = 1.30        # profile extent, in alar half-widths
RIDGE_WIN = 0.30     # ridge is the brightest point within +-this of the midline

# Where each band looks for the reference minimum, as an offset from the ridge
# (inner) or as an absolute |b| (outer), in alar half-widths.
INNER_WIN = (0.18, 0.70)
OUTER_MIN_B = 0.45
# Contour levels, as the fraction of the ridge->reference drop. Averaged rather
# than taken singly so the answer does not hinge on one arbitrary level.
TIP_FRACS = (0.70, 0.80, 0.90)
BRIDGE_FRACS = (0.40, 0.50, 0.60)


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
    R, u, v, L = nose_frame(pts)
    ipd = scale_ref(pts)
    if not (np.isfinite(L) and L > 1 and np.isfinite(ipd) and ipd > 1):
        return None
    b_nose = to_local(pts[nose_indices(pts)], R, u, v)[1]
    half = float(np.abs(b_nose).max()) if len(b_nose) else 0.0
    if not (np.isfinite(half) and half > 1):
        return None

    gray = _luma(img)
    out = {}

    w = P.edge_width(img, lm, *ALAR_BAND, gray=gray)
    if w:
        out["alar_width"] = w / ipd

    w = _contour_width(gray, R, u, v, L, half, TIP_BAND, TIP_FRACS, inner=True)
    if w:
        out["tip_width"] = w / ipd

    w = _contour_width(gray, R, u, v, L, half, BRIDGE_BAND, BRIDGE_FRACS, inner=False)
    if w:
        out["bridge_width"] = w / ipd

    return out or None


def _luma(image_bgr):
    """Raw CIELAB L*, lightly blurred.

    Deliberately not CLAHE-equalised. Every threshold below is a fraction of a
    measured ridge-to-minimum drop, so an affine change of the whole image
    cancels exactly -- but only if nothing local and non-linear has been applied
    first. Equalisation buys contrast and costs that invariance.
    """
    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    return cv2.GaussianBlur(L, (0, 0), 1.2)


def _row(gray, R, u, v, L, t, bs):
    """L* sampled along the lateral line at depth t down the nose."""
    p = R + u * (t * L) + np.outer(bs, v)
    return cv2.remap(gray,
                     p[:, 0].astype(np.float32).reshape(-1, 1),
                     p[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def _smooth(y, sigma):
    k = int(max(1, round(sigma * 3))) * 2 + 1
    return cv2.GaussianBlur(y.reshape(1, -1), (k, 1), sigma).ravel()


def _crossing(bs, y, i0, sign, level):
    """First outward crossing of `level` from index i0, linearly interpolated."""
    n = len(bs)
    i = i0
    while 0 < i < n - 1:
        j = i + sign
        if y[j] <= level:
            d = y[i] - y[j]
            f = (y[i] - level) / d if d > 1e-9 else 0.0
            return bs[i] + f * (bs[j] - bs[i])
        i = j
    return None


def _contour_width(gray, R, u, v, L, half, band, fracs, inner):
    """Median over the band of the iso-shading contour width, in pixels."""
    bs = np.linspace(-B_SPAN * half, B_SPAN * half, N_SAMPLES)
    step = bs[1] - bs[0]
    sigma = max(1.0, 0.025 * half / step)
    order = np.arange(N_SAMPLES)

    widths = []
    for t in np.linspace(band[0], band[1], N_ROWS):
        y = _smooth(_row(gray, R, u, v, L, t, bs), sigma)
        near = np.abs(bs) < RIDGE_WIN * half
        if near.sum() < 3:
            continue
        i_ridge = int(order[near][int(np.argmax(y[near]))])
        peak = float(y[i_ridge])

        edges = {}
        for sign in (-1, 1):
            if inner:
                lo = i_ridge + sign * int(round(INNER_WIN[0] * half / step))
                hi = i_ridge + sign * int(round(INNER_WIN[1] * half / step))
                sel = order[min(lo, hi):max(lo, hi) + 1]
            else:
                sel = order[(np.sign(bs - bs[i_ridge]) == sign)
                            & (np.abs(bs) > OUTER_MIN_B * half)]
            sel = sel[(sel >= 0) & (sel < N_SAMPLES)]
            if len(sel) < 5:
                break
            floor = float(y[sel].min())
            if peak - floor < 1.0:          # no hill to contour: unmeasurable
                break
            for f in fracs:
                edges[(sign, f)] = _crossing(bs, y, i_ridge, sign,
                                             floor + (1.0 - f) * (peak - floor))

        row = [edges[(1, f)] - edges[(-1, f)] for f in fracs
               if edges.get((1, f)) is not None and edges.get((-1, f)) is not None]
        if row:
            widths.append(float(np.mean(row)))

    if len(widths) < max(3, N_ROWS // 3):
        return None
    return float(np.median(widths))
