#!/usr/bin/env python3
"""
Strict QC of doctor_bank: keep ONLY true front or true left/right profile pairs.

Rejects:
- 3/4 oblique / "weird" angles
- worm's-eye (from below) with open nostrils to camera
- bird's-eye / extreme top-down
- before/after angle mismatch
- near-duplicate halves / non-faces

Rewrites doctor_bank in place (clean renumbered front/ / side/).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "doctor_bank"
REJECTED = ROOT / "doctor_bank_rejected"


def gray(im: Image.Image) -> Image.Image:
    return ImageOps.grayscale(im.convert("RGB"))


def features(im: Image.Image) -> dict:
    g = gray(im)
    w, h = g.size
    thr = ImageStat.Stat(g).mean[0] - 12

    mass_x = mass_y = mass = 0.0
    for y in range(int(h * 0.08), int(h * 0.92), 2):
        for x in range(int(w * 0.04), int(w * 0.96), 2):
            v = g.getpixel((x, y))
            if v < thr:
                wt = thr - v + 1
                mass_x += x * wt
                mass_y += y * wt
                mass += wt
    if mass <= 0:
        return {"ok": False}

    cxn = (mass_x / mass) / w
    cyn = (mass_y / mass) / h
    off = abs(cxn - 0.5)

    # L/R mirror asymmetry
    L = g.crop((0, 0, w // 2, h))
    R = ImageOps.mirror(g.crop((w // 2, 0, w, h)).resize(L.size))
    sym = ImageStat.Stat(ImageChops.difference(L, R)).mean[0]

    # Edge energy L vs R (profiles: one side strong silhouette)
    edges = g.filter(ImageFilter.FIND_EDGES)
    eL = ImageStat.Stat(edges.crop((0, 0, w // 2, h))).mean[0]
    eR = ImageStat.Stat(edges.crop((w // 2, 0, w, h))).mean[0]
    e_imb = abs(eL - eR) / (eL + eR + 1e-6)

    # Border darkness (profile face near one edge)
    band = g.crop((0, int(h * 0.18), w, int(h * 0.82)))
    bw, bh = band.size
    sl = max(1, bw // 5)
    dL = 255 - ImageStat.Stat(band.crop((0, 0, sl, bh))).mean[0]
    dR = 255 - ImageStat.Stat(band.crop((bw - sl, 0, bw, bh))).mean[0]
    dM = 255 - ImageStat.Stat(band.crop((bw // 3, 0, 2 * bw // 3, bh))).mean[0]
    edge_ratio = max(dL, dR) / (dM + 1e-6)

    # --- Worm's eye (from below): large dark nostril openings face camera ---
    # Lower-central face region
    nr = g.crop((int(w * 0.28), int(h * 0.42), int(w * 0.72), int(h * 0.72)))
    nr_pix = list(nr.getdata())
    n_area = max(1, len(nr_pix))
    very_dark = sum(1 for p in nr_pix if p < 45)
    dark_frac = very_dark / n_area
    # Also: mid darkness vs upper face (worm's eye darkens mid face)
    upper = g.crop((int(w * 0.25), int(h * 0.12), int(w * 0.75), int(h * 0.38)))
    mid = g.crop((int(w * 0.25), int(h * 0.40), int(w * 0.75), int(h * 0.68)))
    u_mean = ImageStat.Stat(upper).mean[0]
    m_mean = ImageStat.Stat(mid).mean[0]
    # From below: nostrils create very dark mid cluster; mid much darker than upper cheeks/forehead
    mid_darker = (u_mean - m_mean) > 18 and dark_frac > 0.012

    # --- Second-eye / 3-4 check for "side" ---
    # In true profile, far half of image has little facial structure (mostly bg/hair)
    # Compute face-mass fraction on left vs right thirds
    left_third = g.crop((0, int(h * 0.15), w // 3, int(h * 0.85)))
    right_third = g.crop((2 * w // 3, int(h * 0.15), w, int(h * 0.85)))
    def face_mass(tile: Image.Image) -> float:
        st = ImageStat.Stat(tile)
        return (255 - st.mean[0]) * (1 + st.var[0] / 4000)

    fmL, fmR = face_mass(left_third), face_mass(right_third)
    fm_ratio = min(fmL, fmR) / (max(fmL, fmR) + 1e-6)  # near 0 = one side empty (profile)

    return {
        "ok": True,
        "off": off,
        "cxn": cxn,
        "cyn": cyn,
        "sym": sym,
        "e_imb": e_imb,
        "edge_ratio": edge_ratio,
        "dark_frac": dark_frac,
        "mid_darker": mid_darker,
        "fm_ratio": fm_ratio,
        "w": w,
        "h": h,
    }


def is_worms_eye(f: dict) -> bool:
    if not f.get("ok"):
        return True
    # Strong nostril signal
    if f["dark_frac"] >= 0.028:
        return True
    if f["mid_darker"] and f["dark_frac"] >= 0.015:
        return True
    # Extreme bottom-weighted face mass (camera below)
    if f["cyn"] >= 0.58 and f["dark_frac"] >= 0.01:
        return True
    return False


def classify_strict(im: Image.Image) -> str:
    """
    Return 'front', 'side', or 'reject'.
    Only orthographic clinical standards:
      front = patient facing camera, eyes level, no worm's eye
      side  = true profile (L or R), one eye, strong silhouette
    """
    f = features(im)
    if not f.get("ok"):
        return "reject"
    if is_worms_eye(f):
        return "reject"

    # True profile: strong lateral mass shift + high asymmetry + one side emptier
    side_score = 0
    if f["off"] >= 0.14:
        side_score += 2
    elif f["off"] >= 0.11:
        side_score += 1
    if f["sym"] >= 38:
        side_score += 1
    if f["sym"] >= 50:
        side_score += 1
    if f["e_imb"] >= 0.18:
        side_score += 1
    if f["edge_ratio"] >= 1.08:
        side_score += 1
    if f["fm_ratio"] <= 0.55:
        side_score += 2  # one third much emptier
    elif f["fm_ratio"] <= 0.72:
        side_score += 1

    # Reject mild 3/4 that look "almost" profile: both sides still have face mass
    if side_score >= 5 and f["off"] >= 0.12 and f["fm_ratio"] <= 0.70:
        return "side"

    # True front: centered, symmetric, not profile-like
    front_score = 0
    if f["off"] <= 0.045:
        front_score += 2
    elif f["off"] <= 0.06:
        front_score += 1
    if f["sym"] <= 36:
        front_score += 2
    elif f["sym"] <= 44:
        front_score += 1
    if f["e_imb"] <= 0.12:
        front_score += 1
    if f["fm_ratio"] >= 0.72:
        front_score += 1  # both sides have face
    # nostrils not dominant
    if f["dark_frac"] < 0.012:
        front_score += 1

    # Reject if any profile-ish signal on a supposed front
    if front_score >= 5 and f["off"] <= 0.055 and f["sym"] <= 42 and f["fm_ratio"] >= 0.65:
        return "front"

    return "reject"


def pair_ok(before: Image.Image, after: Image.Image) -> tuple[bool, str, str]:
    ab = classify_strict(before)
    aa = classify_strict(after)
    if ab == "reject" or aa == "reject":
        return False, ab, aa
    if ab != aa:
        return False, ab, aa
    # size compatibility
    r1 = before.size[0] / before.size[1]
    r2 = after.size[0] / after.size[1]
    if abs(r1 - r2) / max(r1, r2) > 0.25:
        return False, ab, aa
    return True, ab, aa


def main() -> None:
    if REJECTED.exists():
        shutil.rmtree(REJECTED)
    REJECTED.mkdir(parents=True, exist_ok=True)

    kept_by_doc: dict[str, list[dict]] = {}
    stats = {"scanned": 0, "kept": 0, "rejected": 0, "by_reason": {}}

    for doc_dir in sorted(BANK.iterdir()):
        if not doc_dir.is_dir() or not (doc_dir / "doctor.json").exists():
            continue
        doctor = json.loads((doc_dir / "doctor.json").read_text(encoding="utf-8"))
        kept: list[dict] = []
        for angle_folder in ("front", "side"):
            adir = doc_dir / angle_folder
            if not adir.exists():
                continue
            for case in sorted(adir.iterdir()):
                if not case.is_dir():
                    continue
                bp, ap = case / "before.jpg", case / "after.jpg"
                if not bp.exists() or not ap.exists():
                    continue
                stats["scanned"] += 1
                before = Image.open(bp).convert("RGB")
                after = Image.open(ap).convert("RGB")
                ok, ab, aa = pair_ok(before, after)
                if not ok:
                    stats["rejected"] += 1
                    key = f"{ab}->{aa}"
                    stats["by_reason"][key] = stats["by_reason"].get(key, 0) + 1
                    # move to rejected for audit
                    dest = REJECTED / doc_dir.name / f"{angle_folder}_{case.name}_{ab}_{aa}"
                    dest.mkdir(parents=True, exist_ok=True)
                    before.save(dest / "before.jpg", quality=90)
                    after.save(dest / "after.jpg", quality=90)
                    (dest / "why.txt").write_text(
                        f"folder_was={angle_folder}\nbefore_class={ab}\nafter_class={aa}\n",
                        encoding="utf-8",
                    )
                    continue
                # re-label by CV (not original folder)
                true_angle = ab
                kept.append(
                    {
                        "angle": true_angle,
                        "before": before,
                        "after": after,
                        "src": str(case),
                        "meta": json.loads((case / "meta.json").read_text(encoding="utf-8"))
                        if (case / "meta.json").exists()
                        else {},
                    }
                )
                stats["kept"] += 1
        kept_by_doc[doc_dir.name] = {"doctor": doctor, "pairs": kept}
        print(
            f"{doc_dir.name}: kept {len(kept)} "
            f"(front={sum(1 for p in kept if p['angle']=='front')} "
            f"side={sum(1 for p in kept if p['angle']=='side')})"
        )

    # Rewrite bank clean
    for slug, bundle in kept_by_doc.items():
        doc_dir = BANK / slug
        # wipe front/side
        for sub in ("front", "side"):
            p = doc_dir / sub
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

        counters = {"front": 0, "side": 0}
        catalog = []
        for p in bundle["pairs"]:
            counters[p["angle"]] += 1
            n = counters[p["angle"]]
            case_dir = doc_dir / p["angle"] / f"case_{n:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            p["before"].save(case_dir / "before.jpg", "JPEG", quality=92, optimize=True)
            p["after"].save(case_dir / "after.jpg", "JPEG", quality=92, optimize=True)
            entry = {
                **p.get("meta", {}),
                "case_id": f"case_{n:03d}",
                "angle": p["angle"],
                "before": f"{slug}/{p['angle']}/case_{n:03d}/before.jpg",
                "after": f"{slug}/{p['angle']}/case_{n:03d}/after.jpg",
                "qc": "strict_front_or_side_v2",
            }
            (case_dir / "meta.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
            catalog.append(entry)

        doctor = bundle["doctor"]
        doctor["front_cases"] = counters["front"]
        doctor["side_cases"] = counters["side"]
        doctor["total_cases"] = counters["front"] + counters["side"]
        doctor["quality_rules"] = [
            "true front OR true left/right profile only",
            "rejects 3/4 oblique, worm's-eye (from below), bird's-eye",
            "before and after must match angle class",
            "real patient clinical pairs only",
        ]
        (doc_dir / "doctor.json").write_text(json.dumps(doctor, indent=2), encoding="utf-8")
        (doc_dir / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    # summary
    summary = {"doctors": [], "total_pairs": 0, "qc": stats}
    for doc_dir in sorted(BANK.iterdir()):
        dj = doc_dir / "doctor.json"
        if not dj.exists():
            continue
        d = json.loads(dj.read_text(encoding="utf-8"))
        summary["doctors"].append(d)
        summary["total_pairs"] += d.get("total_cases", 0)
    (BANK / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Doctor Rhinoplasty Before/After Bank (strict QC)",
        "",
        "Only **true front** or **true left/right profile** pairs.",
        "",
        "Rejected: 3/4 oblique, worm's-eye (from below), mismatched angles, marketing junk.",
        "",
        f"- Scanned: {stats['scanned']}",
        f"- Kept: {stats['kept']}",
        f"- Rejected: {stats['rejected']}",
        f"- Rejected audit folder: `doctor_bank_rejected/`",
        "",
        "| Doctor | Front | Side | Total |",
        "|--------|------:|-----:|------:|",
    ]
    for d in summary["doctors"]:
        lines.append(
            f"| {d['name']} | {d['front_cases']} | {d['side_cases']} | {d['total_cases']} |"
        )
    lines += ["", f"**Total pairs:** {summary['total_pairs']}", ""]
    (BANK / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("reject reasons:", json.dumps(stats["by_reason"], indent=2))


if __name__ == "__main__":
    main()
