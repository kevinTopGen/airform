"""Measure how much of a real geometric change the landmark detector reports.

FaceMesh carries a strong learned prior about what noses look like. Warp a nose
10% narrower and the detector, re-run on the result, reports far less than 10% —
it regularises part of the edit away.

This matters because it biases the pipeline in one specific direction:

  extraction  a surgeon's true change T is recorded as k*T   (k < 1, attenuated)
  rendering   the renderer delivers exactly what it is asked, verified 1:1

so feeding a stored signature straight back in renders k*T -- a fraction of the
real surgery. Left uncorrected, real scraped signatures would render as almost
no visible change, and nothing in the pipeline would look broken.

The fix is a per-parameter gain of 1/k, measured here by sweeping known warps
and regressing what comes back.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nosesim import deform, landmarks, params
from nosesim.contracts import NoseParams

FIELDS = ("alar_width", "bridge_width", "tip_width", "nasal_length")
LEVELS = (-0.20, -0.14, -0.08, 0.08)  # fractional change to request


def main():
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "data/kevin.jpg")
    lm = landmarks.detect(img)
    base = params.measure(lm)

    gains = {}
    for f in FIELDS:
        b = getattr(base, f)
        req, obs = [], []
        for lv in LEVELS:
            target = NoseParams(**{k: v for k, v in base.to_dict().items()})
            setattr(target, f, b * (1 + lv))
            warped = deform.apply(img, lm, target)
            got = getattr(params.measure(landmarks.detect(warped)), f)
            req.append(lv)
            obs.append(got / b - 1)
            print(f"  {f:14s} requested {lv*100:+6.1f}%   observed {(got/b-1)*100:+6.2f}%")

        k = float(np.polyfit(req, obs, 1)[0])  # slope through the sweep
        r = float(np.corrcoef(req, obs)[0, 1])

        # A gain is only meaningful if the response actually tracks the request.
        # Where it does not, 1/k is huge and multiplies pure noise -- which is
        # exactly what the width parameters do here (k~0.005, r~0). Emit None
        # and say so, rather than a confident 20x that quietly destroys a
        # signature fit downstream.
        usable = k > 0.05 and r > 0.9
        gains[f] = round(1.0 / k, 3) if usable else None
        note = "" if usable else "  <-- UNUSABLE: response does not track request"
        print(f"  -> {f}: detector retains k={k:.3f}, r={r:+.2f} => gain {gains[f]}{note}\n")

    with open("out/calibration.json", "w") as fh:
        json.dump({"detector": "mediapipe_facemesh_478", "gain": gains}, fh, indent=2)
    print("gains:", gains)
    print("wrote out/calibration.json")


if __name__ == "__main__":
    main()
