#!/usr/bin/env python3
"""
Build a CLEAN per-doctor rhinoplasty before/after bank.

Quality rules (strict):
- Named Miami-area doctor only
- Real clinical patient photos only (no marketing/headers/logos)
- Same patient, same angle, before vs after
- Angle allowed: front OR side (profile) only — never 3/4 oblique
- Organized as:
    doctor_bank/
      <doctor_slug>/
        doctor.json
        front/
          case_001/before.jpg  after.jpg  meta.json
        side/
          case_001/before.jpg  after.jpg  meta.json
        catalog.json

Sources with reliable structure:
1) Dr. Zhuravsky — numbered ofN sequences (pair 1-2 side, 5-6 front for of6;
   for of8/of10 use CV angle classifier + alternating before/after)
2) Dr. Bustillo — side-by-side composite → split left=before right=after
3) Dr. Bared — side-by-side composite → split
4) Dr. Ghersi — split_vertical_1 / split_vertical_2 same stem
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "doctor_bank"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
TIMEOUT = 45
SLEEP = 0.25
MIN_EDGE = 180  # min px on shortest side after crop/split
MIN_FILE = 8_000


# ---------------------------------------------------------------------------
# HTTP / IO
# ---------------------------------------------------------------------------

def fetch(url: str, retries: int = 2) -> bytes:
    url = strip_size(url)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    err: Exception | None = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            err = e
            # don't thrash on hard 404s
            if "404" in str(e):
                break
            time.sleep(0.5 + i)
    raise RuntimeError(f"fetch failed {url}: {err}")


def fetch_text(url: str) -> str:
    return fetch(url).decode("utf-8", "replace")


def abs_url(base: str, href: str) -> str:
    return urllib.parse.urljoin(base, href)


def strip_size(url: str) -> str:
    """Prefer full-size WP upload; unwrap NitroPack CDN wrappers."""
    u = url
    # NitroPack: .../nitropack_static/.../www.example.com/wp-content/uploads/...
    m = re.search(r"(https?://[^/]+)/nitropack_static/.+?/\1(/wp-content/.+)$", u, re.I)
    if m:
        u = m.group(1) + m.group(2)
    else:
        m2 = re.search(r"nitropack_static/.+?/(https?://.+)$", u, re.I)
        if m2:
            u = m2.group(1)
        else:
            # path-only unwrap: keep host, take /wp-content/... tail
            m3 = re.search(r"(https?://[^/]+).+?(/wp-content/uploads/.+)$", u, re.I)
            if m3 and "nitropack" in u.lower():
                u = m3.group(1) + m3.group(2)
    u = re.sub(r"-\d+x\d+(?=\.(?:jpe?g|png|webp)$)", "", u, flags=re.I)
    u = re.sub(r"-scaled(?=\.(?:jpe?g|png|webp)$)", "", u, flags=re.I)
    return u


def content_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Angle classification (front / side / oblique) — pure PIL
# ---------------------------------------------------------------------------

def _gray(im: Image.Image) -> Image.Image:
    return ImageOps.grayscale(im.convert("RGB"))


def classify_angle(im: Image.Image) -> str:
    """
    Return 'front', 'side', or 'oblique'.

    Uses face mass center-of-mass + L/R mirror asymmetry on clinical headshots:
    - side/profile: mass shifted off-center, high L/R difference
    - front: mass near center, more bilateral symmetry
    - oblique (3/4): in between → rejected for this bank
    """
    g = _gray(im)
    w, h = g.size
    thr = ImageStat.Stat(g).mean[0] - 15
    # CoM of dark pixels (face against clinical backdrop)
    mass_x = 0.0
    mass = 0.0
    # sample every 2px for speed
    for y in range(int(h * 0.10), int(h * 0.90), 2):
        for x in range(int(w * 0.05), int(w * 0.95), 2):
            v = g.getpixel((x, y))
            if isinstance(v, tuple):
                v = v[0]
            if v < thr:
                wt = thr - v + 1
                mass_x += x * wt
                mass += wt
    if mass <= 0:
        return "oblique"
    cxn = (mass_x / mass) / w
    off = abs(cxn - 0.5)

    # border darkness vs mid (profiles hug one edge)
    band = g.crop((0, int(h * 0.20), w, int(h * 0.80)))
    bw, bh = band.size
    sl = max(1, bw // 5)
    dL = 255 - ImageStat.Stat(band.crop((0, 0, sl, bh))).mean[0]
    dR = 255 - ImageStat.Stat(band.crop((bw - sl, 0, bw, bh))).mean[0]
    dM = 255 - ImageStat.Stat(band.crop((bw // 3, 0, 2 * bw // 3, bh))).mean[0]
    edge_ratio = max(dL, dR) / (dM + 1e-6)

    # mirror symmetry (lower = more front-like)
    left = g.crop((0, 0, w // 2, h))
    right = ImageOps.mirror(g.crop((w // 2, 0, w, h)).resize(left.size))
    sym = ImageStat.Stat(ImageChops.difference(left, right)).mean[0]

    # side / profile
    if off >= 0.10 and sym >= 30:
        return "side"
    if off >= 0.08 and sym >= 45:
        return "side"
    if edge_ratio >= 1.05 and sym >= 35 and off >= 0.06:
        return "side"

    # front (strict — avoid letting 3/4 through)
    if off <= 0.05 and sym <= 40:
        return "front"
    if off <= 0.035 and sym <= 48:
        return "front"

    return "oblique"


def same_angle(a: str, b: str) -> bool:
    return a == b and a in ("front", "side")


def dimensions_compatible(im1: Image.Image, im2: Image.Image, tol: float = 0.20) -> bool:
    w1, h1 = im1.size
    w2, h2 = im2.size
    if min(w1, h1, w2, h2) < MIN_EDGE:
        return False
    r1, r2 = w1 / h1, w2 / h2
    return abs(r1 - r2) / max(r1, r2) <= tol


def looks_like_marketing(url: str, im: Image.Image | None = None) -> bool:
    low = url.lower()
    bad = (
        "logo", "banner", "header", "menu", "icon", "sprite", "favicon",
        "placeholder", "blank", "avatar", "team", "office", "staff",
        "travel", "package", "landing", "hero", "bg-", "background",
        "dermcare", "watermark-only", "button", "badge", "award",
        "instagram", "facebook", "youtube", "map", "wavy",
    )
    if any(b in low for b in bad):
        return True
    if im is not None:
        w, h = im.size
        if w < 120 or h < 120:
            return True
        # ultra-wide site chrome
        if w / h > 3.5 or h / w > 4.5:
            return True
    return False


# ---------------------------------------------------------------------------
# Pair record
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    doctor_slug: str
    doctor_name: str
    location: str
    website: str
    angle: str  # front | side
    before: Image.Image
    after: Image.Image
    before_url: str
    after_url: str
    source_page: str
    case_key: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------

def extract_urls(html: str, base: str) -> list[str]:
    found = re.findall(r"https?://[^\"'\s>]+", html)
    rel = re.findall(r"[\"'](/[^\"']+\.(?:jpe?g|png|webp))[\"']", html, re.I)
    out = []
    for u in found + [abs_url(base, r) for r in rel]:
        if re.search(r"\.(jpe?g|png|webp)(?:\?|$)", u, re.I):
            out.append(strip_size(u.split("?")[0]))
    # unique preserve order
    return list(dict.fromkeys(out))


def load_image(url: str) -> Image.Image | None:
    try:
        data = fetch(url)
        if len(data) < MIN_FILE:
            return None
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if looks_like_marketing(url, im):
            return None
        return im
    except Exception as e:  # noqa: BLE001
        print(f"  ! img fail {url}: {e}")
        return None


def scrape_zhuravsky() -> list[Pair]:
    """
    Listing page has all case thumbs. Pattern NNofM:
    Verified of6: 1-2 side B/A, 3-4 oblique, 5-6 front B/A.
    of8/of10: pair consecutive (1-2),(3-4),... then CV-filter angle.
    """
    name = "Dr. Ruslan Zhuravsky, MD"
    slug = "zhuravsky_ruslan"
    loc = "Miami, FL"
    site = "https://zfaceplasticsurgery.com/"
    gallery = "https://zfaceplasticsurgery.com/before-after-gallery/rhinoplasty/"
    print(f"\n== {name} ==")
    html = fetch_text(gallery)
    urls = extract_urls(html, gallery)
    # group by case id and frame index
    # e.g. 00001-01of6.jpg or 0502-1of6.jpg
    cases: dict[str, dict[int, str]] = defaultdict(dict)
    total_of: dict[str, int] = {}
    for u in urls:
        m = re.search(
            r"/uploads/\d{4}/\d{2}/([0-9A-Za-z]+)-0?(\d+)of(\d+)\.(?:jpe?g|png|webp)$",
            u,
            re.I,
        )
        if not m:
            continue
        case_id, idx, ofn = m.group(1), int(m.group(2)), int(m.group(3))
        cases[case_id][idx] = u
        total_of[case_id] = ofn

    pairs: list[Pair] = []
    for case_id, frames in sorted(cases.items()):
        ofn = total_of.get(case_id, max(frames) if frames else 0)
        # candidate index pairs
        candidates: list[tuple[int, int, str | None]] = []
        if ofn == 6 and set(frames) >= {1, 2, 5, 6}:
            candidates = [(1, 2, "side"), (5, 6, "front")]
        else:
            # generic alternating before/after
            for i in range(1, ofn, 2):
                if i in frames and (i + 1) in frames:
                    candidates.append((i, i + 1, None))

        for i, j, forced in candidates:
            bu, au = frames[i], frames[j]
            bim, aim = load_image(bu), load_image(au)
            time.sleep(SLEEP)
            if bim is None or aim is None:
                continue
            if not dimensions_compatible(bim, aim):
                continue
            ang_b = classify_angle(bim)
            ang_a = classify_angle(aim)
            if forced:
                # trust sequence for of6 but verify both not opposite
                if forced == "side" and ang_b == "front" and ang_a == "front":
                    angle = "front"  # mislabeled sequence
                elif forced == "front" and ang_b == "side" and ang_a == "side":
                    angle = "side"
                else:
                    # require neither is clearly the wrong family
                    if forced == "side" and (ang_b == "front" or ang_a == "front"):
                        continue
                    if forced == "front" and (ang_b == "side" or ang_a == "side"):
                        # allow if CV says front
                        if ang_b != "front" and ang_a != "front":
                            continue
                    angle = forced
            else:
                if not same_angle(ang_b, ang_a):
                    # if one is oblique, skip
                    continue
                angle = ang_b
            if angle not in ("front", "side"):
                continue
            pairs.append(
                Pair(
                    doctor_slug=slug,
                    doctor_name=name,
                    location=loc,
                    website=site,
                    angle=angle,
                    before=bim,
                    after=aim,
                    before_url=bu,
                    after_url=au,
                    source_page=gallery,
                    case_key=f"{case_id}_{i:02d}_{j:02d}",
                    notes=f"zhuravsky of{ofn} frames {i}/{j}; cv={ang_b}/{ang_a}",
                )
            )
        if len(pairs) and len(pairs) % 20 == 0:
            print(f"  ... {len(pairs)} pairs so far")
    print(f"  kept {len(pairs)} quality pairs from {len(cases)} cases")
    return pairs


def split_composite(im: Image.Image) -> tuple[Image.Image, Image.Image] | None:
    """Left = before, right = after for standard clinical composites."""
    w, h = im.size
    if w < h * 1.25:
        # not wide enough to be side-by-side
        return None
    mid = w // 2
    # small gutter trim
    gap = max(2, w // 200)
    left = im.crop((0, 0, mid - gap, h))
    right = im.crop((mid + gap, 0, w, h))
    if min(left.size) < MIN_EDGE or min(right.size) < MIN_EDGE:
        return None
    return left, right


def scrape_composite_gallery(
    *,
    name: str,
    slug: str,
    location: str,
    website: str,
    listing_urls: list[str],
    url_must_match: re.Pattern[str],
    max_pages: int = 12,
    max_pairs: int = 80,
) -> list[Pair]:
    print(f"\n== {name} ==")
    pages: list[str] = []
    for base in listing_urls:
        pages.append(base)
        for n in range(2, max_pages + 1):
            pages.append(base.rstrip("/") + f"/page/{n}")
            pages.append(base.rstrip("/") + f"/page/{n}/")

    # collect unique full-size composite urls + patient detail pages
    composites: dict[str, str] = {}  # url -> source page
    detail_pages: set[str] = set()

    for page in pages:
        try:
            html = fetch_text(page)
        except Exception as e:  # noqa: BLE001
            # pagination end
            continue
        time.sleep(SLEEP)
        for u in extract_urls(html, page):
            if url_must_match.search(u) and not looks_like_marketing(u):
                # prefer non-thumbnail
                if re.search(r"-\d{2,4}x\d{2,4}\.", u):
                    continue
                composites[strip_size(u)] = page
        for m in re.finditer(r"href=[\"']([^\"']+)[\"']", html, re.I):
            href = abs_url(page, m.group(1))
            if re.search(r"rhinoplasty-photos-\d+|before-and-after/.*\d+", href, re.I):
                detail_pages.add(href)

    # visit detail pages for higher-res composites
    for d in sorted(detail_pages)[:120]:
        try:
            html = fetch_text(d)
        except Exception:
            continue
        time.sleep(SLEEP)
        for u in extract_urls(html, d):
            if url_must_match.search(u) and not looks_like_marketing(u):
                if re.search(r"-\d{2,4}x\d{2,4}\.", u):
                    u = strip_size(u)
                composites[strip_size(u)] = d

    pairs: list[Pair] = []
    seen_hash: set[str] = set()
    for url, src in composites.items():
        im = load_image(url)
        time.sleep(SLEEP)
        if im is None:
            continue
        split = split_composite(im)
        if not split:
            continue
        before, after = split
        # reject if halves are nearly identical (not a true B/A) OR totally different people is ok
        # but reject if half is blank/marketing
        ang_b = classify_angle(before)
        ang_a = classify_angle(after)
        if not same_angle(ang_b, ang_a):
            continue
        angle = ang_b
        # perceptual hash-ish dedupe
        h = content_hash(before.tobytes()[:5000] + after.tobytes()[:5000])
        if h in seen_hash:
            continue
        seen_hash.add(h)
        # reject near-identical before/after (failed split / same frame)
        g1, g2 = _gray(before.resize((64, 64))), _gray(after.resize((64, 64)))
        delta = ImageStat.Stat(ImageChops.difference(g1, g2)).mean[0]
        if delta < 3.5:
            continue  # essentially the same image twice
        case_key = re.sub(r"[^a-zA-Z0-9]+", "_", Path(urllib.parse.urlparse(url).path).stem)[:60]
        pairs.append(
            Pair(
                doctor_slug=slug,
                doctor_name=name,
                location=location,
                website=website,
                angle=angle,
                before=before,
                after=after,
                before_url=url + "#left",
                after_url=url + "#right",
                source_page=src,
                case_key=case_key,
                notes=f"composite split; cv={ang_b}/{ang_a}; delta={delta:.1f}",
            )
        )
        if len(pairs) >= max_pairs:
            break
        if len(pairs) % 10 == 0:
            print(f"  ... {len(pairs)} pairs")
    print(f"  kept {len(pairs)} quality pairs from {len(composites)} composites")
    return pairs


def scrape_ghersi() -> list[Pair]:
    name = "Dr. Marcelo Ghersi, MD"
    slug = "ghersi_marcelo"
    loc = "Miami, FL"
    site = "https://aeris.co/"
    listing = "https://aeris.co/before-after-photos/primary-rhinoplasty/"
    print(f"\n== {name} ==")
    html = fetch_text(listing)
    case_links = sorted(
        {
            abs_url(listing, m.group(0) if m.group(0).startswith("http") else m.group(0))
            for m in re.finditer(
                r"(?:https?://aeris\.co)?/before-after-photos/primary-rhinoplasty/\d+/?",
                html,
            )
        }
    )
    # normalize
    norm = []
    for c in case_links:
        if not c.startswith("http"):
            c = abs_url(listing, c)
        norm.append(c.rstrip("/") + "/")
    case_links = list(dict.fromkeys(norm))

    pairs: list[Pair] = []
    for clink in case_links[:60]:
        try:
            ch = fetch_text(clink)
        except Exception as e:  # noqa: BLE001
            print(f"  case fail {clink}: {e}")
            continue
        time.sleep(SLEEP)
        urls = extract_urls(ch, clink)
        # pair split_vertical_1_X with split_vertical_2_X
        groups: dict[str, dict[str, str]] = defaultdict(dict)
        for u in urls:
            m = re.search(r"split_vertical_([12])_(.+?)\.(?:jpe?g|png|webp)$", u, re.I)
            if not m:
                continue
            which, stem = m.group(1), m.group(2)
            # strip size already done
            stem = re.sub(r"-\d+x\d+$", "", stem)
            groups[stem][which] = strip_size(u)

        for stem, roles in groups.items():
            if "1" not in roles or "2" not in roles:
                continue
            # convention: 1=before, 2=after (observed)
            bim = load_image(roles["1"])
            aim = load_image(roles["2"])
            time.sleep(SLEEP)
            if bim is None or aim is None:
                continue
            if not dimensions_compatible(bim, aim):
                continue
            ang_b, ang_a = classify_angle(bim), classify_angle(aim)
            if not same_angle(ang_b, ang_a):
                continue
            pairs.append(
                Pair(
                    doctor_slug=slug,
                    doctor_name=name,
                    location=loc,
                    website=site,
                    angle=ang_b,
                    before=bim,
                    after=aim,
                    before_url=roles["1"],
                    after_url=roles["2"],
                    source_page=clink,
                    case_key=re.sub(r"[^a-zA-Z0-9]+", "_", stem)[:60],
                    notes=f"ghersi split_vertical; cv={ang_b}/{ang_a}",
                )
            )
        if len(pairs) >= 80:
            break
    print(f"  kept {len(pairs)} quality pairs")
    return pairs


# ---------------------------------------------------------------------------
# Persist bank
# ---------------------------------------------------------------------------

def save_doctor(pairs: list[Pair]) -> dict | None:
    """Write one doctor's bank immediately to disk."""
    if not pairs:
        return None
    ROOT.mkdir(parents=True, exist_ok=True)
    slug = pairs[0].doctor_slug
    seen: set[tuple[str, str]] = set()
    uniq: list[Pair] = []
    for p in pairs:
        k = (p.angle, p.case_key)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    if not uniq:
        return None

    doc_dir = ROOT / slug
    front_dir = doc_dir / "front"
    side_dir = doc_dir / "side"
    # clean re-write for this doctor
    import shutil

    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    front_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)

    counters = {"front": 0, "side": 0}
    catalog = []
    meta0 = uniq[0]
    for p in uniq:
        counters[p.angle] += 1
        n = counters[p.angle]
        case_dir = (front_dir if p.angle == "front" else side_dir) / f"case_{n:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        bp = case_dir / "before.jpg"
        ap = case_dir / "after.jpg"
        p.before.save(bp, "JPEG", quality=92, optimize=True)
        p.after.save(ap, "JPEG", quality=92, optimize=True)
        entry = {
            "case_id": f"case_{n:03d}",
            "angle": p.angle,
            "before": str(bp.relative_to(ROOT)).replace("\\", "/"),
            "after": str(ap.relative_to(ROOT)).replace("\\", "/"),
            "before_url": p.before_url,
            "after_url": p.after_url,
            "source_page": p.source_page,
            "case_key": p.case_key,
            "notes": p.notes,
            "before_size": list(p.before.size),
            "after_size": list(p.after.size),
        }
        (case_dir / "meta.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        catalog.append(entry)

    doctor = {
        "name": meta0.doctor_name,
        "slug": slug,
        "location": meta0.location,
        "website": meta0.website,
        "front_cases": counters["front"],
        "side_cases": counters["side"],
        "total_cases": counters["front"] + counters["side"],
        "quality_rules": [
            "real patient before/after only",
            "same angle front or side only (no 3/4 oblique)",
            "no marketing/header/logo assets",
        ],
    }
    (doc_dir / "doctor.json").write_text(json.dumps(doctor, indent=2), encoding="utf-8")
    (doc_dir / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(
        f"SAVED to disk {slug}: front={counters['front']} side={counters['side']} "
        f"total={doctor['total_cases']} -> {doc_dir}"
    )
    write_summary()
    return doctor


def write_summary() -> dict:
    """Rebuild SUMMARY from whatever doctor folders exist on disk."""
    ROOT.mkdir(parents=True, exist_ok=True)
    summary = {"doctors": [], "total_pairs": 0}
    for ddir in sorted(ROOT.iterdir()):
        if not ddir.is_dir():
            continue
        dj = ddir / "doctor.json"
        if not dj.exists():
            continue
        doctor = json.loads(dj.read_text(encoding="utf-8"))
        summary["doctors"].append(doctor)
        summary["total_pairs"] += doctor.get("total_cases", 0)

    (ROOT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Doctor Rhinoplasty Before/After Bank",
        "",
        "Clean clinical pairs only — organized **per doctor**, **front** or **side**.",
        "",
        f"**Location:** `{ROOT}`",
        "",
        "## Rules",
        "",
        "- Named Miami-area surgeon",
        "- Real patient surgical before/after (not marketing)",
        "- Same angle: **front** or **side/profile** only (3/4 oblique rejected)",
        "- Each case folder: `before.jpg` + `after.jpg` + `meta.json`",
        "",
        "## Layout",
        "",
        "```",
        "doctor_bank/",
        "  <doctor_slug>/",
        "    doctor.json",
        "    catalog.json",
        "    front/case_XXX/{before,after,meta}",
        "    side/case_XXX/{before,after,meta}",
        "```",
        "",
        "## Inventory",
        "",
        "| Doctor | Location | Front | Side | Total |",
        "|--------|----------|------:|-----:|------:|",
    ]
    for d in summary["doctors"]:
        lines.append(
            f"| {d['name']} | {d['location']} | {d['front_cases']} | "
            f"{d['side_cases']} | {d['total_cases']} |"
        )
    lines += [
        "",
        f"**Total pairs:** {summary['total_pairs']}",
        "",
        "Images remain copyright of the source practices; internal analysis/training use only.",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return summary


def main() -> None:
    import shutil

    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Writing bank to: {ROOT}")

    jobs = [
        scrape_zhuravsky,
        lambda: scrape_composite_gallery(
            name="Dr. Andres Bustillo, MD, FACS",
            slug="bustillo_andres",
            location="Miami / Coral Gables, FL",
            website="https://www.drbustillo.com/",
            listing_urls=["https://www.drbustillo.com/before-and-after/rhinoplasty-photos"],
            url_must_match=re.compile(r"rhinoplasty-before-after|patient-\d+.*rhinoplasty", re.I),
        ),
        lambda: scrape_composite_gallery(
            name="Dr. Anthony Bared, MD, FACS",
            slug="bared_anthony",
            location="Miami, FL",
            website="https://www.facialplasticsurgerymiami.com/",
            listing_urls=[
                "https://www.facialplasticsurgerymiami.com/before-and-after/rhinoplasty-photos",
                "https://www.facialplasticsurgerymiami.com/before-and-after/ethnic-rhinoplasty-photos",
            ],
            url_must_match=re.compile(r"patient-\d+.*rhinoplasty.*before-after|rhinoplasty-before-after", re.I),
        ),
        scrape_ghersi,
    ]

    total = 0
    for job in jobs:
        try:
            pairs = job()
        except Exception as e:  # noqa: BLE001
            print(f"JOB FAILED: {e}")
            continue
        doc = save_doctor(pairs)
        if doc:
            total += doc["total_cases"]
        # free PIL images from memory
        del pairs

    print(f"\nDONE. Total pairs on disk: {total}")
    print(f"Open: {ROOT}")
    write_summary()


if __name__ == "__main__":
    main()
