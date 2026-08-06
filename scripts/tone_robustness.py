"""Does the photometric width measurement survive low shadow contrast?

`alar_width` is recovered by finding the nasofacial groove -- a shadow edge.
Anything that flattens that shadow threatens it: flat or ring lighting, camera
underexposure, and darker skin, where an 8-bit sRGB frame compresses shadow
detail into very few code values.

This sweeps exposure and contrast down and asks whether a KNOWN warp is still
measured correctly. Getting the absolute width right is not enough -- the
pipeline only ever uses differences, so what has to survive is the delta.

IMPORTANT: reducing luminance is a proxy for one specific failure mode (loss of
shadow contrast), not a substitute for testing on real diverse faces. Melanin
changes spectral reflectance and subsurface scattering, not just gain. Treat
this as a lower bound on the problem and validate on actual photographs.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nosesim import deform, landmarks, params, photometric
from nosesim.contracts import NoseParams
from nosesim.landmarks import scale_ref

TRUE_DELTA = -0.15  # the warp we apply and then try to recover

CONDITIONS = [
    ("baseline", 1.00, 1.00),
    ("exposure -30%", 0.70, 1.00),
    ("exposure -55%", 0.45, 1.00),
    ("flat light", 1.00, 0.45),
    ("dark + flat", 0.50, 0.45),
    ("severe", 0.38, 0.30),
]


def tone(img, gain, contrast):
    f = img.astype(np.float32)
    m = f.mean()
    return np.clip((f - m) * contrast + m * gain, 0, 255).astype(np.uint8)


def alar(img, lm, equalize):
    g = photometric.luma(img, equalize=equalize)
    w = photometric.edge_width(img, lm, *photometric.BANDS["alar_width"], gray=g)
    return None if w is None else w / scale_ref(lm.as_array())


def main():
    img = cv2.imread("data/kevin.jpg")
    lm = landmarks.detect(img)
    base = params.measure(lm)

    target = NoseParams(**base.to_dict())
    target.alar_width = base.alar_width * (1 + TRUE_DELTA)
    warped = deform.apply(img, lm, target)

    print(f"recovering a known {TRUE_DELTA*100:+.0f}% alar change\n")
    print(f"{'condition':<16}{'gray (naive)':>16}{'L* + CLAHE':>16}   detect")
    print("-" * 66)

    for name, gain, contrast in CONDITIONS:
        a, b = tone(img, gain, contrast), tone(warped, gain, contrast)
        try:
            la, lb = landmarks.detect(a), landmarks.detect(b)
        except ValueError:
            print(f"{name:<16}{'--':>16}{'--':>16}   FACE NOT DETECTED")
            continue

        cells = []
        for eq in (False, True):
            w0, w1 = alar(a, la, eq), alar(b, lb, eq)
            cells.append("n/a" if not (w0 and w1) else f"{(w1/w0-1)*100:+.1f}%")
        print(f"{name:<16}{cells[0]:>16}{cells[1]:>16}   ok")

    print(f"\nanything far from {TRUE_DELTA*100:+.0f}% is a silent wrong answer,")
    print("not a failure -- the measurement returns a number either way.")


if __name__ == "__main__":
    main()
