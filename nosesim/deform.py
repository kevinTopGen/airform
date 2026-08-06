"""Stage 4: NoseParams -> pixels.

Target parameters become per-landmark displacements in the nose frame, which
become MLS control points, which become the warp. Nothing here knows about
surgeons; it only knows how to make a nose match a set of numbers.
"""

from __future__ import annotations

import numpy as np

from .contracts import Landmarks, NoseParams
from .landmarks import nose_frame, to_global, to_local
from .mls import border_anchors, warp
from .nose_region import anchor_indices, nose_indices
from .params import measure


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def plan(lm: Landmarks, target: NoseParams):
    """Return (src_pts, dst_pts) control points realising `target`.

    `target` is an absolute measurement, not a delta — build it with
    `measure(lm).apply(delta)` so the delta's per-field mode is resolved first.

    Width changes are applied as lateral scale factors that blend across three
    overlapping regions of the nose, so the bridge can narrow without dragging
    the alae and the tip can refine without collapsing the sidewalls.
    """
    pts = lm.as_array()
    cur = measure(lm)
    idx = nose_indices(pts)
    R, u, v, L = nose_frame(pts)

    a, b = to_local(pts[idx], R, u, v)
    t = a / L

    def ratio(field):
        c, g = getattr(cur, field), getattr(target, field)
        return 1.0 if (c is None or g is None or c <= 0) else float(g / c)

    s_bridge, s_tip, s_alar = ratio("bridge_width"), ratio("tip_width"), ratio("alar_width")
    s_len = ratio("nasal_length")

    # Where along the nose each width parameter has authority.
    w_bridge = _smoothstep(t, 0.02, 0.28) * (1 - _smoothstep(t, 0.55, 0.78))
    w_lower = _smoothstep(t, 0.55, 0.80)

    # Across the lower nose, split authority between the alar wings (far from
    # the midline) and the tip lobule (near it).
    half = np.abs(b).max() + 1e-9
    w_alar_lat = _smoothstep(np.abs(b) / half, 0.42, 0.88)
    w_tip_lat = 1.0 - w_alar_lat

    lateral = (1.0
               + w_bridge * (s_bridge - 1.0)
               + w_lower * (w_tip_lat * (s_tip - 1.0) + w_alar_lat * (s_alar - 1.0)))

    dst = to_global(a * s_len, b * lateral, R, u, v)

    anchors = pts[anchor_indices(pts, idx)]
    frame = border_anchors(lm.width, lm.height)
    src_pts = np.vstack([pts[idx], anchors, frame])
    dst_pts = np.vstack([dst, anchors, frame])
    return src_pts, dst_pts


def apply(image, lm: Landmarks, target: NoseParams, **kw):
    src, dst = plan(lm, target)
    return warp(image, src, dst, **kw)


def apply_signature(image, lm: Landmarks, delta: NoseParams, strength=1.0, **kw):
    """Measure this face, apply the surgeon's delta to it, render the result.

    The delta is mixed-mode (see contracts.DELTA_MODES): its width fields are
    fractions of *this* face's baseline, so a -10.7% base reduction is -10.7%
    here regardless of how wide the surgeon's usual patient is. `NoseParams.apply`
    resolves that into an absolute target before anything touches pixels.

    `strength` scales the delta, which is what drives the before/after slider —
    the same code path renders every intermediate frame.
    """
    return apply(image, lm, measure(lm).apply(delta.scaled(strength)), **kw)
