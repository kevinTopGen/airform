"""Turn the scraped surgeon collages into individual before/after crops.

    .venv/bin/python scripts/ingest_surgeons.py [--src DIR] [--out DIR] [--debug]

Writes to work/pairs/<surgeon_key>/<n><view>_{before,after}.png, a manifest, and
a QA contact sheet per surgeon. work/ is gitignored: these are real patients'
medical photographs and this repository is public.

TWO THINGS HERE ARE HUMAN INPUT, NOT INFERENCE, AND BOTH CHANGE THE ANSWER.

**The folder names are wrong.** Two of the three scrape folders are named after
the wrong surgeon. The watermark burnt into each collage is the ground truth and
it disagrees with the directory it sits in:

    surgeon_A_andresbustillo      -> "Carlos Wolf MD"      -> carlos_wolf
    surgeon_B_kimpatrick_murray   -> "MURRAY"              -> murray
    surgeon_C_johnnysalomon_md    -> "@ANDRESBUSTILLOMD"   -> andres_bustillo

Johnny Salomon has no data in this scrape. Fitting a signature under the folder
name would publish one surgeon's operation under another's name, which is the
single worst failure this project can produce.

**Not every collage is a rhinoplasty.** The Carlos Wolf folder is mixed: four of
its ten collages are lower blepharoplasty, masseter botox and facelift, labelled
as such in the collage's own bottom banner. Averaging a facelift into a nose
signature quietly poisons it. `PROCEDURE` below records the banner text verbatim
where it exists; `is_rhinoplasty` gates everything downstream. Nothing is
deleted -- a case that should not be fitted is still ingested and marked, so the
decision is auditable instead of invisible.

`NOTES` flags cases that are technically fine but clinically not a settled
result: surgical marking pen still on the skin, or oedema and fresh suture lines
that mean the photograph was taken in the operating room. Swelling inflates
apparent width, so an immediate post-op "after" reads as a *wider* nose and would
pull a surgeon's fitted narrowing toward zero.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nosesim.ingest import contact_sheet, debug_overlay, ingest_image, write_manifest

DEFAULT_SRC = "/Users/kevin/Desktop"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "work")

# scrape folder -> (real surgeon key, display name, watermark that proves it)
SURGEONS = {
    "surgeon_A_andresbustillo": ("carlos_wolf", "Carlos Wolf MD", "Carlos Wolf MD"),
    "surgeon_B_kimpatrick_murray": ("murray", "Kimberly Patrick Murray", "MURRAY"),
    "surgeon_C_johnnysalomon_md": ("andres_bustillo", "Andres Bustillo MD",
                                   "@ANDRESBUSTILLOMD"),
}

# (surgeon_key, file stem) -> (procedure, source)
#   "banner"   read verbatim off the collage's own printed strip
#   "inferred" no printed strip; stated from the visible operation / same patient
PROCEDURE = {
    ("carlos_wolf", "1front"): ("lower blepharoplasty", "banner"),
    ("carlos_wolf", "1side"): ("rhinoplasty", "banner"),
    ("carlos_wolf", "2front"): ("facelift", "inferred"),  # same patient as 4front
    ("carlos_wolf", "2side"): ("rhinoplasty", "inferred"),
    ("carlos_wolf", "3front"): ("masseter botox", "banner"),
    ("carlos_wolf", "3side"): ("rhinoplasty", "inferred"),
    ("carlos_wolf", "4front"): ("facelift", "banner"),
    ("carlos_wolf", "4side"): ("rhinoplasty", "inferred"),
    ("carlos_wolf", "5front"): ("rhinoplasty", "banner"),
    ("carlos_wolf", "5side"): ("rhinoplasty", "banner"),
}
# Murray brands every post "MURRAY / RHINOPLASTY"; Bustillo prints only a handle.
DEFAULT_PROCEDURE = {
    "murray": ("rhinoplasty", "banner"),
    "andres_bustillo": ("rhinoplasty", "inferred"),
}

# Cases to exclude from a signature for a reason other than the procedure.
NOTES = {
    ("murray", "2front"): ["immediate post-op: fresh suture line and blood at the "
                           "alar base in the after panel"],
    ("murray", "3front"): ["immediate post-op: nose erythematous and oedematous in "
                           "the after panel"],
    ("murray", "5front"): ["surgical marking pen on forehead and nose; immediate "
                           "post-op"],
    ("murray", "5side"): ["surgical marking pen on the cheek; immediate post-op"],
    ("murray", "2side"): ["marking pen visible under the eye"],
    ("murray", "4side"): ["marking pen visible under the eye"],
    ("carlos_wolf", "3front"): ["annotation arrows drawn over both panels"],
    ("carlos_wolf", "1front"): ["different patient/procedure from the nose cases"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--debug", action="store_true",
                    help="also write work/qa/debug/ overlays of every split")
    args = ap.parse_args()

    pairs_dir = os.path.join(args.out, "pairs")
    qa_dir = os.path.join(args.out, "qa")
    records, by_surgeon = [], {}

    for folder, (key, name, mark) in SURGEONS.items():
        src = os.path.join(args.src, folder)
        if not os.path.isdir(src):
            print(f"missing: {src}", file=sys.stderr)
            continue
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(".png"))
        files.sort(key=lambda f: (re.sub(r"\d+", "", f), int(re.findall(r"\d+", f)[0])))
        out_dir = os.path.join(pairs_dir, key)
        recs = []
        for fn in files:
            stem = os.path.splitext(fn)[0]
            view = "side" if "side" in stem else "front"
            proc, psrc = PROCEDURE.get((key, stem), DEFAULT_PROCEDURE.get(key, (None, "unknown")))
            rec = ingest_image(
                os.path.join(src, fn), surgeon_key=key, view=view, procedure=proc,
                procedure_source=psrc, is_rhinoplasty=(proc == "rhinoplasty"),
                out_dir=out_dir, stem=stem, notes=NOTES.get((key, stem)))
            recs.append(rec)
            det = "".join("Y" if h.face_detected else "." for h in rec.halves)
            print(f"{key:16s} {fn:12s} {rec.seam_method:14s} x={rec.split_x:5d} "
                  f"banner={rec.banner_box[1]}..{rec.banner_box[3]} det={det} "
                  f"{proc}{'' if rec.is_rhinoplasty else '  [EXCLUDED]'}")
            if args.debug:
                debug_overlay(rec.source, rec,
                              os.path.join(qa_dir, "debug", key, f"{stem}.jpg"))
        contact_sheet(recs, os.path.join(qa_dir, f"split_check_{key}.jpg"))
        by_surgeon[key] = recs
        records += recs

    write_manifest(records, os.path.join(pairs_dir, "manifest.json"))

    halves = [h for r in records for h in r.halves]
    print(f"\n{len(records)} collages -> {len(halves)} halves")
    for key, recs in by_surgeon.items():
        hs = [h for r in recs for h in r.halves]
        rh = [h for r in recs if r.is_rhinoplasty for h in r.halves]
        print(f"  {key:16s} {len(hs):3d} halves  "
              f"{sum(h.face_detected for h in hs):3d} mediapipe  "
              f"{sum(h.head_box_used for h in hs):3d} head-boxed  "
              f"{len(rh):3d} rhinoplasty")
    for view in ("front", "side"):
        hs = [h for r in records for h in r.halves if r.view == view]
        print(f"  {view:16s} {len(hs):3d} halves  "
              f"{sum(h.face_detected for h in hs):3d} mediapipe  "
              f"{sum(h.head_box_used for h in hs):3d} head-boxed")
    print(json.dumps({"manifest": os.path.join(pairs_dir, "manifest.json"),
                      "contact_sheets": sorted(os.path.join(qa_dir, f)
                                               for f in os.listdir(qa_dir)
                                               if f.startswith("split_check"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
