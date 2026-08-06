"""End-to-end proof of concept: one photo -> five surgeon archetypes.

    python scripts/run_variants.py <image> [outdir]

Also writes a landmark overlay, because the first thing to verify is that the
nose region was actually found before trusting anything downstream.
"""

from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nosesim import deform, landmarks, params, signatures
from nosesim.nose_region import anchor_indices, nose_indices


def debug_overlay(img, lm):
    out = img.copy()
    pts = lm.as_array()
    idx = nose_indices(pts)
    anc = anchor_indices(pts, idx)
    for i in anc:
        cv2.circle(out, tuple(pts[i].astype(int)), 1, (90, 90, 90), -1)
    for i in idx:
        cv2.circle(out, tuple(pts[i].astype(int)), 3, (0, 240, 255), -1)
    R, u, v, L = landmarks.nose_frame(pts)
    cv2.arrowedLine(out, tuple(R.astype(int)), tuple((R + u * L).astype(int)),
                    (0, 0, 255), 2, tipLength=0.08)
    cv2.arrowedLine(out, tuple(R.astype(int)), tuple((R + v * L * 0.5).astype(int)),
                    (0, 255, 0), 2, tipLength=0.15)
    return out


def label(img, title, sub=""):
    out = img.copy()
    h, w = out.shape[:2]
    bar = int(h * 0.13)
    panel = np.full((h + bar, w, 3), 22, np.uint8)
    panel[:h] = out
    cv2.putText(panel, title, (int(w * 0.035), h + int(bar * 0.42)),
                cv2.FONT_HERSHEY_SIMPLEX, w / 1150, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(panel, sub, (int(w * 0.035), h + int(bar * 0.78)),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 1900, (150, 190, 255), 1, cv2.LINE_AA)
    return panel


def describe(delta):
    """Render a mixed-mode delta in its own units, so the mode is visible."""
    bits = []
    for k, v in delta.to_dict().items():
        short = k.split("_")[0]
        bits.append(f"{short} {v*100:+.1f}%" if delta.mode(k) == "proportional"
                    else f"{short} {v:+.4f}ipd")
    return "  ".join(bits)


def crop_nose(img, lm, pad=1.85):
    """Tight crop for the comparison strip — the changes are millimetres."""
    pts = lm.as_array()
    R, u, v, L = landmarks.nose_frame(pts)
    c = R + u * L * 0.6
    r = int(L * pad / 2)
    x0, y0 = max(0, int(c[0] - r)), max(0, int(c[1] - r * 1.05))
    x1, y1 = min(img.shape[1], int(c[0] + r)), min(img.shape[0], int(c[1] + r * 1.05))
    return img[y0:y1, x0:x1]


def main():
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "out"
    os.makedirs(outdir, exist_ok=True)

    img = cv2.imread(src)
    if img is None:
        raise SystemExit(f"could not read {src}")
    if max(img.shape[:2]) > 1400:
        s = 1400 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    t0 = time.time()
    lm = landmarks.detect(img)
    base = params.measure(lm)
    print(f"view={lm.view}  yaw={lm.yaw_deg:+.1f}deg  "
          f"landmarks={len(lm.points)}  ({time.time()-t0:.2f}s)")
    print(f"nose landmarks selected: {len(nose_indices(lm.as_array()))}")
    print("\nmeasured (units = fraction of interpupillary distance)")
    for k, v in base.to_dict().items():
        print(f"  {k:18s} {v:+.4f}")

    cv2.imwrite(f"{outdir}/00_landmarks.jpg", debug_overlay(img, lm))
    cv2.imwrite(f"{outdir}/01_original.jpg", img)

    print("\nsignature deltas: widths are % of YOUR baseline, lengths are absolute IPD")

    strip = [label(crop_nose(img, lm), "ORIGINAL", "unmodified")]
    report = {"source": src, "view": lm.view, "measured": base.to_dict(), "variants": []}

    for i, sig in enumerate(signatures.ARCHETYPES, start=1):
        t = time.time()
        warped = deform.apply_signature(img, lm, sig.delta)
        dt = time.time() - t

        target = base.apply(sig.delta)
        pct = {k: (getattr(target, k) / getattr(base, k) - 1) * 100
               for k in sig.delta.to_dict()
               if getattr(base, k) not in (None, 0)}
        sub = "  ".join(f"{k.split('_')[0]} {v:+.1f}%" for k, v in pct.items())

        cv2.imwrite(f"{outdir}/{i+1:02d}_{sig.id}.jpg", warped)
        strip.append(label(crop_nose(warped, lm), sig.name.upper(), sub))
        report["variants"].append({"signature": sig.to_dict(),
                                   "target": target.to_dict(),
                                   "pct_change": pct, "render_ms": round(dt * 1000)})
        print(f"\n{sig.name}  ({dt*1000:.0f}ms)"
              f"\n  signature  {describe(sig.delta)}"
              f"\n  on you     {sub}")

    hs = min(p.shape[0] for p in strip)
    strip = [cv2.resize(p, (int(p.shape[1] * hs / p.shape[0]), hs)) for p in strip]
    sheet = np.hstack([np.pad(p, ((0, 0), (6, 6), (0, 0)), constant_values=22)
                       for p in strip])
    cv2.imwrite(f"{outdir}/contact_sheet.jpg", sheet)

    with open(f"{outdir}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {outdir}/contact_sheet.jpg and {len(strip)-1} full-frame variants")


if __name__ == "__main__":
    main()
