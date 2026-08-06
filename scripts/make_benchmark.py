"""Generate ground truth: warps of known magnitude, for adapters to recover.

The renderer is verified to deliver exactly what it is asked (100% of requested
displacement at control points), so a warp of -14% bridge width IS a -14% bridge
width. That makes it usable as truth, and turns "which measurement technique is
best" into a question with a number for an answer.

Written as PNG. JPEG would add compression noise to precisely the low-contrast
regions the gradient methods depend on.

Single subject -- this measures whether a technique can see a change, not
whether it generalises across faces. Treat every result as an upper bound.
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

PARAMS = ("alar_width", "bridge_width", "tip_width")
LEVELS = (-0.20, -0.14, -0.08, 0.08, 0.15)
TONES = {                    # name: (gain, contrast)
    "normal": (1.00, 1.00),
    "dark_flat": (0.50, 0.45),
}
OUT = "bench"


def tone(img, gain, contrast):
    f = img.astype(np.float32)
    m = f.mean()
    return np.clip((f - m) * contrast + m * gain, 0, 255).astype(np.uint8)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/kevin.jpg"
    img = cv2.imread(src)
    lm = landmarks.detect(img)
    base = params.measure(lm)

    manifest = {"source": src, "params": list(PARAMS), "levels": list(LEVELS),
                "tones": list(TONES), "images": []}

    renders = {"baseline": img}
    for p in PARAMS:
        for lv in LEVELS:
            t = NoseParams(**base.to_dict())
            setattr(t, p, getattr(base, p) * (1 + lv))
            renders[f"{p}_{lv:+.2f}"] = deform.apply(img, lm, t)
            print(f"rendered {p} {lv:+.0%}")

    for tname, (g, c) in TONES.items():
        os.makedirs(f"{OUT}/{tname}", exist_ok=True)
        for key, im in renders.items():
            path = f"{OUT}/{tname}/{key}.png"
            cv2.imwrite(path, tone(im, g, c))
            if key != "baseline":
                p, lv = key.rsplit("_", 1)
                manifest["images"].append({"path": path, "tone": tname,
                                           "param": p, "level": float(lv)})
        manifest.setdefault("baselines", {})[tname] = f"{OUT}/{tname}/baseline.png"

    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(manifest['images'])} test images + {len(TONES)} baselines -> {OUT}/")


if __name__ == "__main__":
    main()
