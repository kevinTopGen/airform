"""Adapter: nose width from FaRL/LaPa semantic segmentation.

The parser has a dedicated 'nose' class, so the measurement is the position of a
learned *region boundary* rather than a landmark or an intensity edge. That is
the interesting property: a boundary that the network places from per-pixel
evidence should survive a tone change that destroys an edge search, and should
move when the nose actually moves, unlike a shape-prior landmark.

Three things make it work rather than merely run:

1.  Sub-pixel boundary. The worker hands back the decision margin
    m = logit[nose] - max(other logits), not a binary mask. The boundary is the
    m = 0 crossing, found by linear interpolation along a sampling ray. A binary
    mask would quantise every width to whole pixels; on a nose ~125 px wide, the
    smallest bench step (8%) is ~10 px, so quantisation would not be fatal, but
    it costs precision for nothing.

2.  Width profile, not bounding box. The mask is resampled into the nose frame
    (MediaPipe radix -> subnasale axis) and the lateral extent is taken row by
    row, then averaged inside each band from base.BANDS. A bbox over the whole
    mask reports one number -- the alar width -- and throws away the bridge and
    tip entirely.

3.  Rays start at the midline and stop at the first crossing, so a nostril
    shadow that the parser assigned to another class, or a stray blob on the
    cheek, cannot inflate the width the way a per-row min/max would.

MediaPipe is used only to place the frame and to supply the interpupillary
scale, both of which are rigid-face quantities the surgery does not move; every
number that varies between before and after comes from the mask.
"""

from __future__ import annotations

import atexit
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading

import cv2
import numpy as np

from .. import landmarks as LM
from ..landmarks import nose_frame, scale_ref
from .base import BANDS

NAME = "faceparse"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_WORKER = os.path.join(_HERE, "_faceparse_worker.py")
_PARSE_PY = os.environ.get("NOSESIM_PARSE_PY",
                           os.path.join(_ROOT, ".venv-parse", "bin", "python"))
_CACHE = os.environ.get("NOSESIM_FACEPARSE_CACHE",
                        os.path.join(_ROOT, ".cache", "faceparse"))

# --- measurement geometry -------------------------------------------------
# All lengths in units of L = |radix -> subnasale|, so they follow the face.
B_MAX = 0.75          # how far laterally a ray is allowed to look for the edge
STEP = 0.25           # ray sample spacing, pixels
N_ROWS = 21           # rows sampled per band
MIN_VALID = 0.5       # fraction of rows that must find both edges
WIDEST_FRAC = 0.25    # alar width = mean of the widest quarter of its rows

_lock = threading.Lock()
_proc = None


# --------------------------------------------------------------------------
# worker plumbing
# --------------------------------------------------------------------------
def available() -> bool:
    if not (os.path.exists(_PARSE_PY) and os.path.exists(_WORKER)):
        return False
    # facer must be installed in the parse venv; importing it here would blow up
    # the main venv, so just look for it on disk.
    root = os.path.dirname(os.path.dirname(_PARSE_PY))
    return bool(glob.glob(os.path.join(root, "lib", "python*", "site-packages", "facer")))


def _start():
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    os.makedirs(_CACHE, exist_ok=True)
    log = open(os.path.join(_CACHE, "worker.log"), "ab", buffering=0)
    _proc = subprocess.Popen(
        [_PARSE_PY, _WORKER, "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        cwd=_ROOT, text=True, bufsize=1,
    )
    ready = _proc.stdout.readline()          # blocks through the 617 MB load
    if not ready:
        _proc = None
        raise RuntimeError("faceparse worker died during startup; see "
                           + os.path.join(_CACHE, "worker.log"))
    return _proc


def _shutdown():
    global _proc
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            _proc.stdin.flush()
            _proc.wait(timeout=5)
        except Exception:
            _proc.kill()
    _proc = None


atexit.register(_shutdown)


def _request(image_path, out_png):
    global _proc
    for attempt in (0, 1):                   # one free restart if it fell over
        p = _start()
        try:
            p.stdin.write(json.dumps({"image": image_path, "out": out_png}) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            if not line:
                raise BrokenPipeError("no reply")
            return json.loads(line)
        except Exception:
            _proc = None
            if attempt:
                raise
    return None


def _key(path):
    h = hashlib.sha1()
    h.update(b"faceparse-v1\0")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def margin_map(path):
    """Signed nose-vs-rest decision margin for `path`, cached on disk.

    Returns (float32 HxW, meta) or (None, None) if no face was parsed. The cache
    is keyed by file content, so regenerating the benchmark invalidates it and
    reusing an identical image anywhere hits it.
    """
    os.makedirs(_CACHE, exist_ok=True)
    stem = os.path.join(_CACHE, _key(path))
    png = stem + ".png"
    js = png + ".json"          # sidecar name is fixed by the worker

    with _lock:
        if not (os.path.exists(png) and os.path.exists(js)):
            res = _request(os.path.abspath(path), png)
            if not res or not res.get("ok"):
                with open(stem + ".miss", "w") as f:
                    json.dump(res or {}, f)
                return None, None

    with open(js) as f:
        meta = json.load(f)
    raw = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None
    clip = float(meta.get("clip", 20.0))
    m = (raw.astype(np.float32) / 65535.0 - 0.5) * (2.0 * clip)
    return m, meta


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def _band_width(m, R, u, v, L, t_lo, t_hi, widest=False):
    """Lateral extent of the nose mask over t in [t_lo, t_hi].

    One ray per row, walked outward from the dorsum midline in both directions
    until the margin changes sign; the sign change is located by linear
    interpolation between the two straddling samples.

    `widest` averages only the widest WIDEST_FRAC of the rows instead of all of
    them, and is used for alar width alone. params.py explains why: the
    subnasale sits *below* where the alae flare, so the bottom of the alar band
    is the columella base, where the mask closes down to ~30% of its peak. On
    this subject a plain band mean puts alar width below tip width and breaks
    the alar > tip > bridge invariant. Taking the widest quarter is the same
    "widest lateral extent anywhere over the lower nose" rule params.py already
    uses, just averaged over a few rows so it is not one noisy row.
    """
    h, w = m.shape[:2]
    b_max = B_MAX * L
    nb = int(2 * b_max / STEP) + 1
    b = np.linspace(-b_max, b_max, nb)
    ts = np.linspace(t_lo, t_hi, N_ROWS)

    # (rows, samples) sampling grid in image coordinates
    base = R[None, None, :] + (ts * L)[:, None, None] * u[None, None, :]
    xy = base + b[None, :, None] * v[None, None, :]
    mapx = np.ascontiguousarray(xy[:, :, 0], dtype=np.float32)
    mapy = np.ascontiguousarray(xy[:, :, 1], dtype=np.float32)
    if (mapx.min() < 0 or mapy.min() < 0 or mapx.max() > w - 1 or mapy.max() > h - 1):
        mapx = np.clip(mapx, 0, w - 1)
        mapy = np.clip(mapy, 0, h - 1)
    prof = cv2.remap(m, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

    c = nb // 2                                   # index of b = 0
    widths = []
    for row in prof:
        if row[c] <= 0:                           # midline not inside the mask
            continue
        right = np.flatnonzero(row[c:] <= 0)
        left = np.flatnonzero(row[:c + 1][::-1] <= 0)
        if right.size == 0 or left.size == 0:     # edge beyond the search span
            continue
        i = c + int(right[0])
        f = row[i - 1] / (row[i - 1] - row[i])
        b_r = b[i - 1] + f * (b[i] - b[i - 1])
        j = c - int(left[0])
        f = row[j + 1] / (row[j + 1] - row[j])
        b_l = b[j + 1] + f * (b[j] - b[j + 1])
        widths.append(b_r - b_l)

    if len(widths) < MIN_VALID * N_ROWS:
        return None
    widths = np.asarray(widths)
    if widest:
        k = max(1, int(round(WIDEST_FRAC * widths.size)))
        widths = np.sort(widths)[-k:]
    return float(np.mean(widths))


def measure(path):
    img = cv2.imread(path)
    if img is None:
        return None
    try:
        lm = LM.detect(img)
    except ValueError:
        return None

    m, _meta = margin_map(path)
    if m is None:
        return None

    pts = lm.as_array()
    R, u, v, L = nose_frame(pts)
    ipd = scale_ref(pts)

    out = {}
    for name, (lo, hi) in BANDS.items():
        w = _band_width(m, R, u, v, L, lo, hi, widest=(name == "alar_width"))
        if w is not None:
            out[name] = w / ipd
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, measure(p))
