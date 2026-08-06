"""Does the warp only touch the nose, and does it move it by the right amount?

Two checks worth having before anyone trusts a rendered result:

  locality  -- where did pixels actually change? Anything outside the nose box
               means the anchors are leaking and the face is drifting.
  closure   -- re-measure the warped image. If asking for -10% alar width
               yields -10% alar width when measured back, the parameter layer
               and the pixel layer agree.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nosesim import deform, landmarks, params, signatures
from nosesim.landmarks import nose_frame


def main():
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "data/kevin.jpg")
    outdir = "out"
    lm = landmarks.detect(img)
    base = params.measure(lm)
    pts = lm.as_array()
    R, u, v, L = nose_frame(pts)

    print(f"{'surgeon':<28}{'locality':<34}{'closure (asked -> measured)'}")
    print("-" * 100)

    panels = []
    for sig in signatures.ARCHETYPES:
        warped = deform.apply_signature(img, lm, sig.delta)

        d = np.abs(warped.astype(np.int16) - img.astype(np.int16)).sum(2)
        ys, xs = np.nonzero(d > 10)
        # express the changed-pixel box in nose lengths from the radix
        far = np.linalg.norm(np.stack([xs, ys], 1) - R, axis=1).max() / L
        frac = len(xs) / d.size * 100

        got = params.measure(landmarks.detect(warped))
        bits = []
        for f in ("alar_width", "bridge_width", "tip_width", "nasal_length"):
            b, g = getattr(base, f), getattr(got, f)
            want = b + (getattr(sig.delta, f) or 0.0)
            bits.append(f"{f.split('_')[0][:5]} {(want/b-1)*100:+.1f}->{(g/b-1)*100:+.1f}%")
        print(f"{sig.name:<28}{f'{frac:.2f}% of px, max {far:.2f} nose-len':<34}{'  '.join(bits)}")

        hm = cv2.applyColorMap(np.clip(d * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        panels.append(cv2.addWeighted(img, 0.35, hm, 0.65, 0))

    sheet = np.hstack(panels)
    s = 1800 / sheet.shape[1]
    cv2.imwrite(f"{outdir}/locality.jpg", cv2.resize(sheet, None, fx=s, fy=s))
    print(f"\nwrote {outdir}/locality.jpg")


if __name__ == "__main__":
    main()
