"""Adapter contract for nose-width measurement techniques.

Every technique -- landmark regressor, segmentation mask, photometric edge
search -- hides behind this one interface so the benchmark can score them on
identical ground truth.

    NAME: str                     short identifier used in reports
    def available() -> bool       are weights/deps present on this machine
    def measure(path) -> dict|None
        keys:  'alar_width', 'bridge_width', 'tip_width'
        units: ANY, as long as they are self-consistent between calls
        omit a key the technique genuinely cannot produce
        return None if no face was found

Units are free because the benchmark only ever looks at ratios: measure a
baseline, measure a known warp, compare the fractional change. Whatever
normalising constant an adapter uses cancels. That also means an adapter is
free to use its own landmarks, its own nose frame and its own scale reference
-- techniques are compared end to end, not spliced together.

Read the scoring rules in scripts/bench.py before optimising anything: r (does
the measurement track the truth at all) matters far more than k (how much gain
correction it needs), because a stable gain is calibratable and noise is not.
"""

from __future__ import annotations

KEYS = ("alar_width", "bridge_width", "tip_width")

# Nose-frame bands, as a fraction of radix->subnasale distance. Adapters that
# work in the nose frame should use these so they measure the same anatomy.
BANDS = {
    "bridge_width": (0.25, 0.55),
    "tip_width": (0.60, 0.80),
    "alar_width": (0.80, 1.00),
}
