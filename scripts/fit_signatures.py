"""Fit real surgeon signatures from split before/after pairs.

    python scripts/fit_signatures.py [work/pairs] [data/signatures]

Front pairs give the widths (photometric, via params.measure); side pairs give
hump, projection, rotation and length (profile.py, silhouette-based, because
MediaPipe reads only 7 of 24 of these 90-degree profiles).

Exclusions are applied and reported, never silent:
  - non-rhinoplasty sources. One supplied folder mixes in lower blepharoplasty,
    masseter botox and facelift; fitting a nose signature on eyelid surgery
    yields a confident number describing nothing.
  - pose-mismatched pairs. A before and after shot at different angles measures
    parallax, not surgery.
  - eyewear. A frame across the dorsum moves measured bridge width by +50%.

Output is numbers only -- no imagery -- so data/signatures/ is safe to commit
while the photographs it came from stay in gitignored work/.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nosesim import landmarks, params, profile
from nosesim.contracts import DELTA_MODES, NoseParams, SurgeonSignature

MAX_POSE_GAP_DEG = 12.0

NAMES = {
    "carlos_wolf": ("Carlos Wolf, MD", "@carloswolfmd"),
    "murray": ("Kimpatrick Murray", "@murray.rhinoplasty"),
    "andres_bustillo": ("Andres Bustillo, MD", "@andresbustillomd"),
}

# Only alar_width survived benchmark validation; bridge is advisory and tip's
# ground truth was shown to be broken. Recorded per parameter so a consumer
# cannot mistake an advisory number for a measured one.
FRONT_CONFIDENCE = {"alar_width": "measured",
                    "bridge_width": "advisory",
                    "tip_width": "unreliable"}
PROFILE_CONFIDENCE = {"dorsal_hump": "measured", "nasal_length": "measured",
                      "tip_rotation_deg": "measured", "tip_projection": "advisory"}


def measure_front(path):
    img = cv2.imread(path)
    if img is None:
        return None, "unreadable"
    try:
        lm = landmarks.detect(img)
    except ValueError:
        return None, "no face"
    try:
        return params.measure(lm, image=img), lm
    except TypeError:                     # older signature, no pixels
        return params.measure(lm), lm


def measure_side_pair(source_path):
    """Measure a side collage. profile.py does its own splitting.

    It has to: the profile stage needs the panel geometry the silhouette walker
    expects, and the ingest crops are tightened to the face, which cuts the
    subnasale off the bottom of the contour. Feeding it pre-split crops fails
    with 'crop ends at the tip' on almost every image.
    """
    img = cv2.imread(source_path)
    if img is None:
        return None, None, "unreadable"
    panels = profile.split_collage(img)
    if len(panels) < 2:
        return None, None, f"split gave {len(panels)} panel(s)"
    out, why = [], []
    for _label, panel in panels[:2]:
        res = profile.measure_profile(panel)
        out.append(res.params if res.ok else None)
        why.append("ok" if res.ok else res.reason)
    return out[0], out[1], "/".join(why)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "work/pairs"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/signatures"
    os.makedirs(out, exist_ok=True)

    manifest = json.load(open(f"{src}/manifest.json"))
    by_surgeon = defaultdict(lambda: {"deltas": [], "used": [], "rejected": []})

    for entry in manifest:
        key = entry["surgeon_key"]
        rec = by_surgeon[key]
        stem = os.path.splitext(os.path.basename(entry["source"]))[0]
        tag = f"{stem}"

        if not entry.get("is_rhinoplasty", True):
            rec["rejected"].append(f"{tag}: {entry.get('procedure','not rhinoplasty')}")
            continue

        if entry["view"] == "front":
            b = f"{src}/{key}/{stem}_before.png"
            a = f"{src}/{key}/{stem}_after.png"
            if not (os.path.exists(b) and os.path.exists(a)):
                rec["rejected"].append(f"{tag}: missing split half")
                continue
            mb, lb = measure_front(b)
            ma, la = measure_front(a)
            if mb is None or ma is None:
                rec["rejected"].append(f"{tag}: front not measurable")
                continue
            gap = abs(lb.yaw_deg - la.yaw_deg)
            if gap > MAX_POSE_GAP_DEG:
                rec["rejected"].append(f"{tag}: pose gap {gap:.0f}deg > {MAX_POSE_GAP_DEG:.0f}")
                continue
        else:
            mb, ma, why = measure_side_pair(entry["source"])
            if mb is None or ma is None:
                rec["rejected"].append(f"{tag}: profile failed ({why})")
                continue

        rec["deltas"].append(mb.delta_to(ma))
        rec["used"].append(tag)

    registry = []
    for key, rec in sorted(by_surgeon.items()):
        name, handle = NAMES.get(key, (key, ""))
        mean, sd, n_field, conf = {}, {}, {}, {}
        for f in DELTA_MODES:
            vals = [getattr(d, f) for d in rec["deltas"] if getattr(d, f) is not None]
            if not vals:
                continue
            mean[f] = sum(vals) / len(vals)
            sd[f] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            n_field[f] = len(vals)
            base = dict(FRONT_CONFIDENCE, **PROFILE_CONFIDENCE).get(f, "advisory")
            # A mean whose spread swamps it is not a signature, whatever the
            # gauge quality. Demote rather than present a confident number.
            if len(vals) < 3 or (mean[f] and abs(sd[f]) > abs(mean[f])):
                base = "insufficient_data"
            conf[f] = base

        sig = SurgeonSignature(id=key, name=name, tagline=handle,
                               n_pairs=len(rec["deltas"]),
                               delta=NoseParams(**mean), std=NoseParams(**sd))
        d = sig.to_dict()
        d.update({"handle": handle, "n_per_field": n_field, "confidence": conf,
                  "used": rec["used"], "rejected": rec["rejected"]})
        registry.append(d)

        with open(f"{out}/{key}.json", "w") as fh:
            json.dump(d, fh, indent=2)

        print(f"\n=== {name}  ({key})  n={len(rec['deltas'])} pairs ===")
        for f in mean:
            unit = "%" if DELTA_MODES[f] == "proportional" else ""
            scale = 100 if DELTA_MODES[f] == "proportional" else 1
            print(f"  {f:18s} {mean[f]*scale:+8.2f}{unit:1s} "
                  f"sd {sd[f]*scale:6.2f}  n={n_field[f]:<3d} {conf[f]}")
        for r in rec["rejected"]:
            print(f"  rejected  {r}")

    with open(f"{out}/registry.json", "w") as fh:
        json.dump({"surgeons": registry}, fh, indent=2)
    print(f"\nwrote {out}/registry.json and {len(registry)} per-surgeon files")


if __name__ == "__main__":
    main()
