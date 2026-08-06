"""Moving Least Squares image deformation (Schaefer et al., SIGGRAPH 2006).

The rigid variant: control points drag the image around, and every local
neighbourhood is constrained to rotate+translate only. No local shear, no local
scale. That constraint is exactly why warped skin still looks like skin.

Derivation of the inner loop, so nobody has to rediscover it:

    A_i = w_i * [[ p̂ ], [-p̂ᗮ]] · [[ d ], [-dᗮ]]ᵀ   where d = v - p*

collapses to a scaled rotation matrix w_i * [[s, t], [-t, s]] with
s = p̂·d and t = p̂ × d, so q̂_i A_i is just a 2-vector:

    ( q̂x·s - q̂y·t ,  q̂x·t + q̂y·s )

Sanity check: set q = p and the sum becomes μ_s·d, which normalises back to
exactly d, so f(v) = v. Identity in, identity out.
"""

from __future__ import annotations

import cv2
import numpy as np


def _mls_rigid_points(V, P, Q, alpha=1.2, eps=1e-3):
    """Map query points V through the rigid MLS field defined by P -> Q.

    Kept strictly 2-D (M,N) throughout — materialising the (M,N,2) intermediates
    the textbook form implies makes this memory-bandwidth bound and ~10x slower.
    """
    Px, Py = P[None, :, 0], P[None, :, 1]  # (1, N)
    Qx, Qy = Q[None, :, 0], Q[None, :, 1]
    vx, vy = V[:, 0:1], V[:, 1:2]  # (M, 1)

    rx, ry = vx - Px, vy - Py  # (M, N) — v to each control point
    W = 1.0 / np.maximum(rx * rx + ry * ry, eps) ** alpha
    Wsum = W.sum(1, keepdims=True)

    psx = (W * Px).sum(1, keepdims=True) / Wsum  # weighted centroids, (M, 1)
    psy = (W * Py).sum(1, keepdims=True) / Wsum
    qsx = (W * Qx).sum(1, keepdims=True) / Wsum
    qsy = (W * Qy).sum(1, keepdims=True) / Wsum

    ax, ay = Px - psx, Py - psy  # p̂
    qx, qy = Qx - qsx, Qy - qsy  # q̂
    dx, dy = vx - psx, vy - psy  # d = v - p*

    s = ax * dx + ay * dy  # p̂ · d
    t = ax * dy - ay * dx  # p̂ × d

    fx = (W * (qx * s - qy * t)).sum(1)
    fy = (W * (qx * t + qy * s)).sum(1)

    fnorm = np.sqrt(fx * fx + fy * fy) + 1e-12
    dnorm = np.sqrt((dx[:, 0] ** 2 + dy[:, 0] ** 2))
    return np.stack([fx / fnorm * dnorm + qsx[:, 0],
                     fy / fnorm * dnorm + qsy[:, 0]], axis=-1)


def warp(image, src_pts, dst_pts, alpha=1.2, grid_step=4, chunk=40000):
    """Warp `image` so that each src_pts[i] lands on dst_pts[i].

    cv2.remap needs a *backward* map — for each destination pixel, which source
    pixel feeds it — so the MLS field is solved in the reverse direction
    (dst -> src). Solving it forward and inverting is the classic mistake; it
    leaves holes wherever the warp expands.

    The field is smooth, so it is solved on a coarse lattice and bilinearly
    upsampled. `grid_step=4` is visually indistinguishable from per-pixel and
    ~16x cheaper.
    """
    h, w = image.shape[:2]
    P = np.asarray(dst_pts, dtype=np.float32)  # note the swap
    Q = np.asarray(src_pts, dtype=np.float32)

    gx = np.arange(0, w, grid_step, dtype=np.float32)
    gy = np.arange(0, h, grid_step, dtype=np.float32)
    GX, GY = np.meshgrid(gx, gy)
    V = np.stack([GX.ravel(), GY.ravel()], axis=-1)

    out = np.empty_like(V)
    for i in range(0, len(V), chunk):
        out[i : i + chunk] = _mls_rigid_points(V[i : i + chunk], P, Q, alpha)

    # Upsample the DISPLACEMENT, never the absolute coordinates. cv2.resize's
    # pixel-centre convention shifts a coordinate field by a fraction of a
    # pixel, which silently resamples the entire image -- every pixel changes,
    # even where nothing moved. Displacement is exactly zero away from the nose,
    # so resampling it is harmless and identity stays bit-exact.
    dx = cv2.resize((out[:, 0] - V[:, 0]).reshape(GX.shape), (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize((out[:, 1] - V[:, 1]).reshape(GX.shape), (w, h), interpolation=cv2.INTER_CUBIC)

    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(image, X + dx, Y + dy, interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def border_anchors(w, h, n_per_side=6):
    """Fixed control points pinning the frame.

    Without these the whole image drifts toward the moved points. With them the
    background stretches smoothly into whatever space the nose vacates — which
    is what stops a shaved hump from leaving a ghost silhouette behind.
    """
    xs = np.linspace(0, w - 1, n_per_side)
    ys = np.linspace(0, h - 1, n_per_side)
    pts = [(x, 0) for x in xs] + [(x, h - 1) for x in xs]
    pts += [(0, y) for y in ys] + [(w - 1, y) for y in ys]
    return np.array(sorted(set(pts)), dtype=np.float64)
