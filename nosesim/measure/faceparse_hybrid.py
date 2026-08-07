"""Adapter -- FaRL/LaPa nose mask SEEDS the search, image gradients MEASURE it.

THE IDEA
Segmentation and photometry fail in opposite directions. A face-parsing mask
knows *where* the nose is on any lighting, but its boundary is a semantic
decision made at low resolution, so it under-reports how far that boundary
moved: on the known-warp benchmark the LaPa nose mask tracks a +/-20% alar warp
with r=0.997 but a slope of only k=0.5 -- half the real motion. A raw gradient
search knows *how far* an edge moved to sub-pixel precision, but on its own it
needs a wide window and will happily lock onto the orbital shadow or the
nasolabial fold instead of the nose.

So: take the mask's boundary as the seed, then let the image move it, but only
within +/-0.05 nose-lengths. The mask stops the gradient search from wandering;
the gradient recovers the amplitude the mask throws away. On alar width that
combination measures k=0.88 / r=0.999 (normal) and k=0.84 / r=1.000 (dark,
flat), against k=0.51/0.45 for the mask boundary alone -- the hypothesis holds
where a real edge exists.

WHERE IT DOES NOT HOLD, AND WHAT IS DONE INSTEAD
Across the bridge there is no nasofacial crease: the dorsum shades smoothly into
the cheek and the strongest gradient near the mask boundary belongs to the
orbital rim, which does not move when the nose narrows. Refining there is worse
than not refining (r collapses from 0.99 to 0.03 in one tone), and no local
cue -- gradient prominence, mask sharpness, peak isolation -- separates the two
cases; all of them rate the orbital shadow as the *better* edge.

What does exist on the bridge is the pair of dorsal aesthetic lines: the
half-maximum contour of the dorsal highlight. Their separation is the width a
surgeon means by "bridge", it is a shading feature rather than an edge, and a
half-maximum is invariant under affine intensity change -- which is why it holds
its gain across both tone conditions (k=0.42 normal, k=0.43 dark) where the
plain gradient search swings from 0.50 to 0.94.

Hence one rule per band, chosen by which nasal boundary physically exists there:

    alar, tip  ->  mask boundary refined by max |grad L*| in a tight window
    bridge     ->  mask-bounded dorsal-highlight half-max width

ENVIRONMENT
facer/torch only import under .venv-parse, so the mask is produced by a
subprocess running that interpreter and cached on disk by (path, size, mtime).
First call on a new image costs ~10s; afterwards it is free. Set
NOSESIM_PARSE_PY / NOSESIM_FPH_CACHE to override interpreter and cache location.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from .. import landmarks as LM
from .. import photometric as P
from ..landmarks import nose_frame, scale_ref
from .base import BANDS

NAME = "faceparse_hybrid"

ROWS = 9            # profiles per band; the band width is the median over them
REFINE_WIN = 0.05   # gradient search half-window, in nose lengths
HALF_MAX = 0.50     # dorsal highlight contour level (FWHM)
MASK_LEVEL = 0.50   # nose-probability contour taken as the mask boundary
GRAD_BAND = ("alar_width", "tip_width")   # bands with a real crease to refine on

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PARSE_PY = os.environ.get("NOSESIM_PARSE_PY", os.path.join(_ROOT, ".venv-parse", "bin", "python"))
_CACHE = os.environ.get("NOSESIM_FPH_CACHE",
                        os.path.join(tempfile.gettempdir(), "nosesim_faceparse_hybrid"))

# Run inside .venv-parse only. Kept as source text rather than a module so this
# adapter stays a single file and cannot collide with another agent's worker.
_WORKER = r'''
import os, sys
import numpy as np, torch, facer

cache, paths = sys.argv[1], sys.argv[2:]
os.makedirs(cache, exist_ok=True)
device = "cpu"
det = facer.face_detector("retinaface/mobilenet", device=device)
par = facer.face_parser("farl/lapa/448", device=device)
NOSE = 6  # LaPa class index; label order is bg, face, rb, lb, re, le, nose, ...

for spec in paths:
    out, src = spec.split("|", 1)
    if os.path.exists(out):
        continue
    image = facer.hwc2bchw(facer.read_hwc(src)).to(device=device)
    with torch.inference_mode():
        faces = det(image)
        if faces["rects"].shape[0] == 0:
            continue
        faces = par(image, faces)
    nose = faces["seg"]["logits"].softmax(dim=1)[0, NOSE].cpu().numpy()
    np.savez_compressed(out, nose=np.clip(np.rint(nose * 65535.0), 0, 65535).astype(np.uint16))
'''


def available() -> bool:
    """Ready if the parsing interpreter is installed, or masks are already cached."""
    if os.path.exists(_PARSE_PY):
        return True
    return os.path.isdir(_CACHE) and any(f.endswith(".npz") for f in os.listdir(_CACHE))


# ---------------------------------------------------------------- mask access

def _cache_path(path: str) -> str:
    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    return os.path.join(_CACHE, hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz")


def nose_mask(path: str, timeout: float = 600.0):
    """Soft P(nose) over the whole image, from FaRL/LaPa. None if unavailable."""
    dst = _cache_path(path)
    if not os.path.exists(dst):
        os.makedirs(_CACHE, exist_ok=True)
        worker = os.path.join(_CACHE, "_worker.py")
        if not os.path.exists(worker):
            with open(worker, "w") as f:
                f.write(_WORKER)
        try:
            subprocess.run([_PARSE_PY, worker, _CACHE, f"{dst}|{os.path.abspath(path)}"],
                           check=True, timeout=timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001 -- any failure means "no mask"
            print(f"[{NAME}] parse worker failed on {path}: {e}", file=sys.stderr)
            return None
    if not os.path.exists(dst):
        return None
    return np.load(dst)["nose"].astype(np.float32) / 65535.0


# ------------------------------------------------------------- 1-D primitives

def _sample(img, pts):
    return cv2.remap(img,
                     pts[:, 0].astype(np.float32).reshape(-1, 1),
                     pts[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def _first_crossing(x, y, level):
    """First place y falls below `level`, scanning from x[0] outward. Linear
    interpolation between the bracketing samples -- the mask is a soft
    probability, so its 0.5 contour is defined to well under a pixel."""
    below = np.nonzero(y < level)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    j = below[0]
    y0, y1 = y[j - 1], y[j]
    if y0 == y1:
        return float(x[j])
    return float(x[j - 1] + (y0 - level) / (y0 - y1) * (x[j] - x[j - 1]))


def _parabolic(x, y, i):
    """Sub-sample peak position from the three samples around index i."""
    if i <= 0 or i >= len(y) - 1:
        return float(x[i])
    den = y[i - 1] - 2.0 * y[i] + y[i + 1]
    if abs(den) < 1e-12:
        return float(x[i])
    return float(x[i] + 0.5 * (y[i - 1] - y[i + 1]) / den * (x[1] - x[0]))


def _walk_to_level(x, y, level, start, step):
    """From index `start`, walk in `step` until y drops below level; interpolate."""
    i, n = start, len(y)
    while 0 <= i + step < n and y[i + step] >= level:
        i += step
    j = i + step
    if not (0 <= j < n):
        return None
    y0, y1 = y[i], y[j]
    if y0 == y1:
        return float(x[i])
    return float(x[i] + (y0 - level) / (y0 - y1) * (x[j] - x[i]))


# ----------------------------------------------------------------- the method

def _mask_edges(mask, org, v, L):
    """Mask boundary either side of the midline, in nose-frame lateral units."""
    edges = []
    for sgn in (-1, 1):
        bs = np.linspace(0.0, sgn * 0.9 * L, 240)
        p = org + np.outer(bs, v)
        e = _first_crossing(bs, _sample(mask, p), MASK_LEVEL)
        if e is None:
            return None
        edges.append(e)
    return edges


def _refine(gray, org, v, L, seed):
    """Strongest intensity gradient within REFINE_WIN of the mask boundary."""
    w = REFINE_WIN * L
    bs = np.linspace(seed - w, seed + w, 121)
    g = _sample(gray, org + np.outer(bs, v))
    d = np.abs(np.gradient(g, bs))
    return _parabolic(bs, d, int(np.argmax(d)))


def _highlight_width(gray, org, v, L, edges):
    """Half-maximum width of the dorsal highlight, bounded by the mask.

    Affine-invariant by construction: the level is a fixed fraction of the
    highlight's own peak-to-base amplitude, so a gain/contrast change moves the
    level with the profile and the crossings stay put.
    """
    lo, hi = edges
    bs = np.linspace(lo, hi, 241)
    g = _sample(gray, org + np.outer(bs, v))
    g = np.convolve(g, np.ones(3) / 3.0, mode="same")
    mid = 0.5 * (lo + hi)
    central = np.abs(bs - mid) < 0.30 * (hi - lo)          # ridge sits near the midline
    ip = int(np.argmax(np.where(central, g, -np.inf)))
    base = min(g[2], g[-3])                                 # the two flanking creases
    level = base + HALF_MAX * (g[ip] - base)
    left = _walk_to_level(bs, g, level, ip, -1)
    right = _walk_to_level(bs, g, level, ip, +1)
    if left is None or right is None:
        return None
    return right - left


def measure(path):
    img = cv2.imread(path)
    if img is None:
        return None
    try:
        lm = LM.detect(img)
    except ValueError:
        return None
    mask = nose_mask(path)
    if mask is None or mask.shape[:2] != img.shape[:2]:
        return None

    pts = lm.as_array()
    R, u, v, L = nose_frame(pts)
    ipd = scale_ref(pts)
    gray = P.luma(img, equalize=True)

    out = {}
    for name, (t_lo, t_hi) in BANDS.items():
        widths = []
        for t in np.linspace(t_lo, t_hi, ROWS):
            org = R + u * (t * L)
            edges = _mask_edges(mask, org, v, L)
            if edges is None:
                continue
            if name in GRAD_BAND:
                lo = _refine(gray, org, v, L, edges[0])
                hi = _refine(gray, org, v, L, edges[1])
                widths.append(hi - lo)
            else:
                w = _highlight_width(gray, org, v, L, edges)
                if w is not None:
                    widths.append(w)
        if widths:
            out[name] = float(np.median(widths)) / ipd
    return out or None
