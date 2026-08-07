"""Registry of nose-width measurement adapters.

Adapters are imported defensively: a technique whose weights failed to download
or whose deps conflict should drop out of the benchmark, not crash it.

Three things live here besides the registry itself, all of them the glue the
pipeline needs to *use* an adapter rather than merely score one:

  get(name)            one adapter, memoised
  measure_image(...)   run an adapter on pixels the pipeline already holds
  bench_scores()       what bench.py measured, so a caller can decide whether
                       to believe a number before it uses it

`measure_image` exists because of a deliberate asymmetry in the adapter
contract: adapters take a *path* (see base.py) so the benchmark can address its
ground-truth corpus by filename, while the runtime pipeline holds a decoded BGR
array. Bridging the two costs one lossless round-trip through a temp file.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile

from .base import BANDS, KEYS

MODULES = [
    "mp_mesh",        # MediaPipe FaceMesh landmarks (the incumbent)
    "photometric",    # landmark-seeded intensity-gradient edge search
    "faceparse",      # FaRL/LaPa segmentation, nose class
    "mp_tasks",       # MediaPipe Tasks API Face Landmarker
    "insightface_106",
    "photometric_v2",
    "faceparse_hybrid",
]

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where bench.py writes its scoreboard. Overridable so a caller can score
# against a re-run on their own subject without editing code.
BENCH_RESULTS = os.environ.get("AIRFORM_BENCH_RESULTS",
                               os.path.join(_ROOT, "bench", "results.json"))

_loaded = {}


def load():
    """Return {name: module} for every adapter that imports and reports ready."""
    out = {}
    for m in MODULES:
        try:
            mod = importlib.import_module(f"{__name__}.{m}")
            if mod.available():
                out[mod.NAME] = mod
        except Exception:
            pass
    return out


def get(name):
    """One adapter by NAME, or None if it is unavailable on this machine.

    Memoised, and defensive in the same way `load` is: a missing weight file or
    a conflicting dependency must degrade the caller to its fallback, not raise
    out of a measurement call.
    """
    if name not in _loaded:
        mod = None
        try:
            candidate = importlib.import_module(f"{__name__}.{name}")
            if candidate.available() and candidate.NAME == name:
                mod = candidate
        except Exception:
            mod = None
        _loaded[name] = mod
    return _loaded[name]


def measure_image(name, image_bgr):
    """Run adapter `name` on an in-memory BGR array. -> dict | None.

    The temp file is PNG (lossless -- a JPEG round-trip would perturb exactly
    the shading gradients a photometric adapter measures), it is written to the
    system temp directory and *never* inside the repository tree, and it is
    unlinked before this returns. That matters beyond tidiness: this repo is
    public and the images that flow through here are photographs of faces.
    """
    mod = get(name)
    if mod is None or image_bgr is None:
        return None

    import cv2

    fd, path = tempfile.mkstemp(prefix="airform_", suffix=".png")
    os.close(fd)
    try:
        if not cv2.imwrite(path, image_bgr):
            return None
        return mod.measure(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def bench_scores(path=None):
    """bench.py's scoreboard: {adapter: {"param|tone": {k, r, rms, n}}}.

    Empty dict when it has not been generated (bench/ is gitignored, so a fresh
    clone has none). Callers must treat "no score" as "not calibrated" rather
    than as a licence to assume k=1.
    """
    try:
        with open(path or BENCH_RESULTS) as fh:
            return json.load(fh)
    except Exception:
        return {}


__all__ = ["load", "get", "measure_image", "bench_scores",
           "BANDS", "KEYS", "MODULES", "BENCH_RESULTS"]
