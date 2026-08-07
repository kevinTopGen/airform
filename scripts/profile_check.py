"""Run the silhouette profile measurement over the scraped side collages.

    python scripts/profile_check.py [--src DIR] [--qa DIR] [--only A|B|C]
                                    [--mediapipe]

Four things it reports, all falsifiable, none of them fitted:

  COVERAGE   how many side halves yield a measurement, and how many *pairs*
             yield two. Pairs are the unit that matters -- a signature is fitted
             from before/after differences, so a half without its partner is
             worth nothing. `--mediapipe` re-runs the landmark detector on the
             identical panels so the comparison is measured here, not quoted.

  DIRECTION  dorsal_hump before vs after, per pair. Rhinoplasty takes humps down.
             There is no ground truth for a scraped Instagram photo, so this is
             the check that decides whether the numbers mean anything, and the
             counts are printed whichever way they come out.

  DELTAS     the other three parameters, same test. Rotation should go up,
             length and projection down, on the same pairs.

  SYNTHETIC  (--synthetic) k and r against silhouettes of known geometry. The
             real pairs can only show direction -- nobody publishes the
             millimetres -- so gain is measured on shapes built to order.

  SCALE      |brow -> labrale| / |nasion -> labrale| before vs after. Brow and
             labrale are both outside the operative field, so this ratio moves
             only if the *measured* nasion slid -- the one way the choice of
             scale reference could be wrong. Reported only for pairs where the
             brow is a real supraorbital bulge in both photographs; where the
             module could not find one it prints nothing rather than a 1.00.

WHAT THIS CANNOT DO. A silhouette carries no yaw information: a head turned 20
degrees off profile traces a perfectly plausible curve, just foreshortened. The
pairs listed in FLAGGED below were judged off-profile or immediately post-op BY
EYE from the QA overlays -- not by the algorithm -- and the direction counts are
printed with and without them so the reader can see how much that judgement is
worth.

Source images are real patients' medical photographs. Nothing derived from them
is written anywhere but the (gitignored) QA directory.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nosesim.profile import measure_profile, overlay, split_collage

# Folder names are MISLABELLED upstream; the watermark inside each image is
# definitive. Keyed by the surgeon who actually operated.
SOURCES = {
    "carlos_wolf": "surgeon_A_andresbustillo",
    "murray": "surgeon_B_kimpatrick_murray",
    "andres_bustillo": "surgeon_C_johnnysalomon_md",
}
TAG = {"carlos_wolf": "A", "murray": "B", "andres_bustillo": "C"}

# Every collage in this dataset is laid out BEFORE | AFTER: surgeons A and C
# label the panels in the image, and surgeon B's left panels carry pre-op
# marking pen. Stated here because a silent flip would invert every sign below.
ORDER = ("before", "after")

# Judged by eye from work/qa/, not detected. Kept explicit and small so it is
# obvious what was excluded and why.
FLAGGED = {
    ("carlos_wolf", "4side"): "three-quarter view, and the two panels differ in yaw",
    ("murray", "1side"): "three-quarter view",
    ("murray", "3side"): "three-quarter view",
    ("murray", "2side"): "pre-op marking pen; after panel is immediate post-op",
    ("murray", "5side"): "pre-op marking pen; fresh columellar incision after",
}


def run_one(path, qa_dir, key, stem):
    img = cv2.imread(path)
    if img is None:
        return None
    out, vis = {}, []
    for (label, panel) in split_collage(img):
        res = measure_profile(panel)
        out[label] = res
        vis.append(overlay(panel, res, f"{TAG[key]} {stem} {label}"))
    h = max(v.shape[0] for v in vis)
    canvas = np.zeros((h, sum(v.shape[1] for v in vis), 3), np.uint8)
    x = 0
    for v in vis:
        canvas[:v.shape[0], x:x + v.shape[1]] = v
        x += v.shape[1]
    canvas = cv2.resize(canvas, (1100, int(canvas.shape[0] * 1100 / canvas.shape[1])))
    os.makedirs(qa_dir, exist_ok=True)
    cv2.imwrite(os.path.join(qa_dir, f"profile_{key}_{stem}.jpg"), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def mediapipe_coverage(paths):
    """Same panels, through the landmark detector, for an apples-to-apples count."""
    from nosesim import landmarks as LM

    halves = pairs = total = 0
    for path in paths:
        got = 0
        for _, panel in split_collage(cv2.imread(path)):
            total += 1
            try:
                LM.detect(panel)
                got += 1
            except Exception:
                pass
        halves += got
        pairs += got == 2
    return halves, total, pairs


SWEEPS = (("hump", (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08)),
          ("proj", (-0.09, -0.045, 0.0, 0.045, 0.09)),
          ("rot", (-10, -5, 0, 5, 10, 15)))


def synthetic():
    """Gain and correlation against silhouettes whose geometry is known exactly."""
    from nosesim import profile_synth as S

    print("\nSYNTHETIC GROUND TRUTH  (nosesim/profile_synth.py)")
    print(f"  {'parameter':<18}{'backdrop':<10}{'n':>3}{'k':>8}{'r':>9}{'rms':>9}"
          "   verdict")
    for param, levels in SWEEPS:
        for tone in ("light", "dark"):
            d = S.sweep(param, levels=levels, tone=tone)
            t, m = np.array(d["true"], float), np.array(d["meas"], float)
            if t.size < 3:
                print(f"  {d['field']:<18}{tone:<10}{t.size:>3}   too few levels measured")
                continue
            k = float(np.polyfit(t, m, 1)[0])
            r = float(np.corrcoef(t, m)[0, 1])
            rms = float(np.sqrt(np.mean((m - t) ** 2)))
            ok = "USABLE" if (r > 0.95 and k > 0.25) else "unusable"
            print(f"  {d['field']:<18}{tone:<10}{t.size:>3}{k:>8.3f}{r:>9.4f}"
                  f"{rms:>9.4f}   {ok}")
    print("  (bench.py's rule: usable when r > 0.95 and k > 0.25. A filled polygon "
          "has\n   no skin, hair or head pose, so this is a floor, not a ceiling.)")


def _direction(name, pairs, field, expect):
    vals = [(k, s, getattr(a.params, field) - getattr(b.params, field))
            for k, s, b, a in pairs]
    v = np.array([d for _, _, d in vals])
    down = int((v < 0).sum())
    hit = down if expect == "down" else v.size - down
    print(f"  {name:<16} mean {v.mean():+.4f}  sd {v.std():.4f}   "
          f"{expect} in {hit}/{v.size}")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Users/kevin/Desktop")
    ap.add_argument("--qa", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "qa"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--mediapipe", action="store_true",
                    help="also count FaceMesh detections on the same panels")
    ap.add_argument("--synthetic", action="store_true",
                    help="also score k and r against synthetic known geometry")
    args = ap.parse_args()

    rows, pairs, paths = [], [], []
    for key, folder in SOURCES.items():
        if args.only and TAG[key] != args.only.upper():
            continue
        for path in sorted(glob.glob(os.path.join(args.src, folder, "*side.png"))):
            stem = os.path.basename(path)[:-4]
            res = run_one(path, args.qa, key, stem)
            if res is None:
                continue
            paths.append(path)
            for label in ORDER:
                rows.append((key, stem, label, res[label]))
            pairs.append((key, stem, res["before"], res["after"]))

    print(f"{'surgeon':<16}{'img':<8}{'half':<8}{'hump':>9}{'proj':>8}{'len':>8}"
          f"{'nla':>8}{'scale':>8}   note")
    ok = 0
    for key, stem, label, r in rows:
        flag = "  [FLAGGED: " + FLAGGED[(key, stem)] + "]" if (key, stem) in FLAGGED else ""
        if r.ok:
            ok += 1
            p = r.params
            print(f"{key:<16}{stem:<8}{label:<8}{p.dorsal_hump:>+9.4f}"
                  f"{p.tip_projection:>8.4f}{p.nasal_length:>8.4f}"
                  f"{p.tip_rotation_deg:>8.1f}{r.scale_px:>8.0f}   ok{flag}")
        else:
            print(f"{key:<16}{stem:<8}{label:<8}{'-':>9}{'-':>8}{'-':>8}"
                  f"{'-':>8}{'-':>8}   REJECTED: {r.reason}{flag}")

    both = [(k, s, b, a) for k, s, b, a in pairs if b.ok and a.ok]
    n = len(rows)
    print(f"\nCOVERAGE  {ok}/{n} side halves ({ok / max(1, n):.0%}), "
          f"{len(both)}/{len(pairs)} complete before/after pairs")
    if args.mediapipe:
        h, t, p = mediapipe_coverage(paths)
        print(f"          MediaPipe FaceMesh on the same panels: {h}/{t} halves "
              f"({h / max(1, t):.0%}), {p}/{len(paths)} pairs")

    clean = [(k, s, b, a) for k, s, b, a in both if (k, s) not in FLAGGED]

    print("\nDIRECTION  dorsal_hump, before -> after   (expect DOWN: surgery "
          "takes humps off)")
    dec = 0
    for key, stem, b, a in both:
        d = a.params.dorsal_hump - b.params.dorsal_hump
        dec += d < 0
        mark = " FLAGGED" if (key, stem) in FLAGGED else ""
        print(f"  {key:<16}{stem:<8}{b.params.dorsal_hump:>+8.4f} -> "
              f"{a.params.dorsal_hump:>+8.4f}   {d:>+8.4f}   "
              f"{'DOWN' if d < 0 else 'UP  '}{mark}")
    cdec = sum(a.params.dorsal_hump < b.params.dorsal_hump for _, _, b, a in clean)
    print(f"  hump decreased in {dec}/{len(both)} pairs"
          f"   ({cdec}/{len(clean)} excluding flagged)")

    print("\nOTHER DELTAS  (after - before), all pairs")
    for name, fld, exp in (("tip_projection", "tip_projection", "down"),
                           ("nasal_length", "nasal_length", "down"),
                           ("nasolabial_deg", "tip_rotation_deg", "up")):
        _direction(name, both, fld, exp)
    print("  ... excluding flagged")
    for name, fld, exp in (("tip_projection", "tip_projection", "down"),
                           ("nasal_length", "nasal_length", "down"),
                           ("nasolabial_deg", "tip_rotation_deg", "up")):
        _direction(name, clean, fld, exp)

    print("\nSCALE INVARIANCE  |brow->labrale| / |nasion->labrale|, before -> after")
    drift = []
    for key, stem, b, a in both:
        rb, ra = b.extra["brow_labrale_over_scale"], a.extra["brow_labrale_over_scale"]
        if not (np.isfinite(rb) and np.isfinite(ra)):
            continue
        drift.append(abs(ra / rb - 1))
        print(f"  {key:<16}{stem:<8}{rb:>8.3f} -> {ra:>8.3f}   "
              f"{100 * (ra / rb - 1):>+6.1f}%")
    if drift:
        d = np.array(drift)
        print(f"  median |drift| {100 * np.median(d):.1f}%   max {100 * d.max():.1f}%"
              f"   n={d.size} of {len(both)} pairs (brow out of frame in the rest)")

    if args.synthetic:
        synthetic()

    print(f"\nQA overlays: {args.qa}/profile_<surgeon>_<n>.jpg")


if __name__ == "__main__":
    main()
