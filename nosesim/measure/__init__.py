"""Registry of nose-width measurement adapters.

Adapters are imported defensively: a technique whose weights failed to download
or whose deps conflict should drop out of the benchmark, not crash it.
"""

from __future__ import annotations

import importlib

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


__all__ = ["load", "BANDS", "KEYS", "MODULES"]
