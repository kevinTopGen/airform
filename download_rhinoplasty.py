#!/usr/bin/env python3
"""
Download publicly posted rhinoplasty before/after image sets from named Miami-area
surgeons' practice websites. Organizes into:
  rhinoplasty_dataset/<doctor_slug>/set_NNN/{before,after}.*
  rhinoplasty_dataset/<doctor_slug>/manifest.json
  rhinoplasty_dataset/doctors.json
  rhinoplasty_dataset/SUMMARY.md
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent / "rhinoplasty_dataset"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
MIN_SETS = 20
TIMEOUT = 45
SLEEP = 0.35


@dataclass
class Doctor:
    name: str
    slug: str
    location: str
    gallery_urls: list[str]
    website: str
    notes: str = ""


@dataclass
class ImagePair:
    set_id: str
    before_url: str
    after_url: str
    source_page: str
    extra: dict = field(default_factory=dict)


class LinkImgParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.imgs: list[dict] = []
        self.links: list[str] = []
        self._in_a = False
        self._a_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k: (v or "") for k, v in attrs}
        if tag == "a":
            href = ad.get("href", "")
            if href:
                self.links.append(href)
                self._in_a = True
                self._a_href = href
        if tag == "img":
            src = (
                ad.get("src")
                or ad.get("data-src")
                or ad.get("data-lazy-src")
                or ad.get("data-original")
                or ""
            )
            srcset = ad.get("srcset") or ad.get("data-srcset") or ""
            if not src and srcset:
                src = srcset.split(",")[0].strip().split(" ")[0]
            if src and not src.startswith("data:"):
                self.imgs.append(
                    {
                        "src": src,
                        "alt": ad.get("alt", ""),
                        "parent_href": self._a_href,
                        "srcset": srcset,
                    }
                )

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_a = False
            self._a_href = None


def fetch(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    last_err: Exception | None = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.0 + i)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_text(url: str) -> str:
    data = fetch(url)
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def abs_url(base: str, href: str) -> str:
    return urllib.parse.urljoin(base, href)


def strip_wp_size(url: str) -> str:
    """Prefer full-size WordPress uploads when a -NNNxNNN suffix is present."""
    return re.sub(r"-\d+x\d+(?=\.(?:jpe?g|png|webp)$)", "", url, flags=re.I)


def largest_from_srcset(srcset: str, base: str) -> str | None:
    best = None
    best_w = -1
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split()
        u = abs_url(base, bits[0])
        w = 0
        if len(bits) > 1 and bits[1].endswith("w"):
            try:
                w = int(bits[1][:-1])
            except ValueError:
                w = 0
        if w >= best_w:
            best_w = w
            best = u
    return best


def parse_page(url: str) -> tuple[list[dict], list[str], str]:
    html = fetch_text(url)
    p = LinkImgParser()
    p.feed(html)
    imgs = []
    for im in p.imgs:
        src = abs_url(url, im["src"])
        if im.get("srcset"):
            big = largest_from_srcset(im["srcset"], url)
            if big:
                src = big
        src = strip_wp_size(src)
        imgs.append({**im, "src": src, "parent_href": abs_url(url, im["parent_href"]) if im.get("parent_href") else None})
    links = [abs_url(url, l) for l in p.links]
    return imgs, links, html


def save_binary(url: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return True
    try:
        data = fetch(url)
        if len(data) < 500:
            return False
        # skip tiny placeholders / gifs
        if data[:6] in (b"GIF87a", b"GIF89a") and len(data) < 5000:
            return False
        path.write_bytes(data)
        time.sleep(SLEEP)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! download fail {url}: {e}")
        return False


def ext_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    for e in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(e):
            return e
    return ".jpg"


# ---------------------------------------------------------------------------
# Per-doctor scrapers
# ---------------------------------------------------------------------------

def scrape_zhuravsky(doc: Doctor) -> list[ImagePair]:
    """Gallery lists before (01ofN) and after (02ofN) thumbnails per case."""
    pairs: list[ImagePair] = []
    seen = set()
    for gallery in doc.gallery_urls:
        imgs, links, _ = parse_page(gallery)
        # group by case number prefix like 00001, 00002, 0502, etc.
        by_case: dict[str, dict[str, str]] = defaultdict(dict)
        for im in imgs:
            src = im["src"]
            m = re.search(
                r"/uploads/\d{4}/\d{2}/([0-9A-Za-z]+)-0?([12])of\d+",
                src,
                re.I,
            )
            if not m:
                # also try 0502-1of6 style
                m = re.search(
                    r"/uploads/\d{4}/\d{2}/([0-9A-Za-z]+)-([12])of\d+",
                    src,
                    re.I,
                )
            if not m:
                continue
            case_id, which = m.group(1), m.group(2)
            role = "before" if which in ("1", "01") else "after"
            by_case[case_id][role] = strip_wp_size(src)

        for case_id, roles in sorted(by_case.items()):
            if "before" in roles and "after" in roles:
                key = (roles["before"], roles["after"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    ImagePair(
                        set_id=f"zhuravsky_{case_id}",
                        before_url=roles["before"],
                        after_url=roles["after"],
                        source_page=gallery,
                        extra={"case": case_id},
                    )
                )
    return pairs


def scrape_bustillo(doc: Doctor) -> list[ImagePair]:
    """Paginated gallery; each patient has paired before/after thumbs + detail page."""
    pairs: list[ImagePair] = []
    seen_patients: set[str] = set()
    pages = list(doc.gallery_urls)
    # discover more pages
    for base in list(doc.gallery_urls):
        for n in range(2, 12):
            pages.append(base.rstrip("/") + f"/page/{n}")

    for page in pages:
        try:
            imgs, links, html = parse_page(page)
        except Exception as e:  # noqa: BLE001
            print(f"  page miss {page}: {e}")
            continue
        # patient detail links
        patient_links = sorted(
            {
                l
                for l in links
                if re.search(r"rhinoplasty-photos-\d+", l)
                or re.search(r"/before-and-after/.*rhinoplasty.*\d+", l)
            }
        )
        if not patient_links:
            # fall back: pair consecutive rhinoplasty images that share patient id in alt/url
            by_patient: dict[str, list[str]] = defaultdict(list)
            for im in imgs:
                src = im["src"]
                if "rhinoplasty" not in src.lower() and "rhinoplasty" not in (im.get("alt") or "").lower():
                    continue
                m = re.search(r"patient[-_]?(\d+)", src + " " + (im.get("alt") or ""), re.I)
                if not m:
                    m = re.search(r"photos-(\d+)", im.get("parent_href") or "", re.I)
                if m:
                    by_patient[m.group(1)].append(src)
            for pid, urls in by_patient.items():
                if pid in seen_patients:
                    continue
                urls = list(dict.fromkeys(urls))
                if len(urls) >= 2:
                    seen_patients.add(pid)
                    pairs.append(
                        ImagePair(
                            set_id=f"bustillo_{pid}",
                            before_url=urls[0],
                            after_url=urls[1],
                            source_page=page,
                            extra={"patient": pid},
                        )
                    )
            continue

        for plink in patient_links:
            m = re.search(r"(\d+)", plink.rsplit("-", 1)[-1])
            pid = m.group(1) if m else plink.rstrip("/").split("/")[-1]
            if pid in seen_patients:
                continue
            try:
                pimgs, _, _ = parse_page(plink)
            except Exception as e:  # noqa: BLE001
                print(f"  patient page fail {plink}: {e}")
                continue
            photo_urls = []
            for im in pimgs:
                src = im["src"]
                if any(x in src.lower() for x in ("logo", "banner", "icon", "sprite", "placeholder", "blank.gif")):
                    continue
                if not re.search(r"\.(jpe?g|png|webp)$", src, re.I):
                    continue
                # keep clinical photos
                if any(
                    k in src.lower()
                    for k in ("rhinoplasty", "patient", "before", "after", "uploads")
                ):
                    photo_urls.append(src)
            photo_urls = list(dict.fromkeys(photo_urls))
            if len(photo_urls) < 2:
                continue
            # first half often before angles, second half after — use first two as pair
            # Prefer alternating before/after naming
            befores = [u for u in photo_urls if re.search(r"before", u, re.I)]
            afters = [u for u in photo_urls if re.search(r"after", u, re.I)]
            if befores and afters:
                b, a = befores[0], afters[0]
            else:
                b, a = photo_urls[0], photo_urls[1]
            seen_patients.add(pid)
            pairs.append(
                ImagePair(
                    set_id=f"bustillo_{pid}",
                    before_url=b,
                    after_url=a,
                    source_page=plink,
                    extra={"patient": pid, "n_photos": len(photo_urls)},
                )
            )
            if len(pairs) >= 40:
                return pairs
    return pairs


def scrape_ghersi(doc: Doctor) -> list[ImagePair]:
    """aeris.co primary rhinoplasty cases."""
    pairs: list[ImagePair] = []
    seen = set()
    listing = doc.gallery_urls[0]
    imgs, links, html = parse_page(listing)
    case_links = sorted(
        {
            l
            for l in links
            if re.search(r"/primary-rhinoplasty/\d+", l)
            or re.search(r"/revision-rhinoplasty/\d+", l)
        }
    )
    # also from page2 etc if present
    more_pages = [l for l in links if "primary-rhinoplasty" in l and "page" in l]
    for mp in more_pages[:5]:
        try:
            _, links2, _ = parse_page(mp)
            case_links.update(
                l
                for l in links2
                if re.search(r"/primary-rhinoplasty/\d+", l)
                or re.search(r"/revision-rhinoplasty/\d+", l)
            )
        except Exception:
            pass

    if not case_links:
        # parse patient IDs from listing page text
        for m in re.finditer(r"/primary-rhinoplasty/(\d+)", html):
            case_links.add(abs_url(listing, f"/before-after-photos/primary-rhinoplasty/{m.group(1)}/"))

    for clink in sorted(case_links):
        if clink in seen:
            continue
        seen.add(clink)
        try:
            pimgs, _, phtml = parse_page(clink)
        except Exception as e:  # noqa: BLE001
            print(f"  case fail {clink}: {e}")
            continue
        urls = []
        for im in pimgs:
            src = im["src"]
            if any(x in src.lower() for x in ("blank.gif", "logo", "icon", "sprite", "placeholder")):
                continue
            if re.search(r"\.(jpe?g|png|webp)$", src, re.I):
                urls.append(src)
        # try data attributes / og images in html
        for m in re.finditer(r'(https?://[^"\']+\.(?:jpe?g|png|webp))', phtml, re.I):
            u = m.group(1)
            if "wp-content" in u and "blank" not in u:
                urls.append(strip_wp_size(u))
        urls = list(dict.fromkeys(urls))
        clinical = [
            u
            for u in urls
            if any(k in u.lower() for k in ("before", "after", "patient", "rhino", "gallery", "uploads"))
        ]
        if len(clinical) < 2:
            clinical = urls
        if len(clinical) < 2:
            continue
        befores = [u for u in clinical if "before" in u.lower()]
        afters = [u for u in clinical if "after" in u.lower()]
        if befores and afters:
            b, a = befores[0], afters[0]
        else:
            b, a = clinical[0], clinical[1]
        cid = clink.rstrip("/").split("/")[-1]
        pairs.append(
            ImagePair(
                set_id=f"ghersi_{cid}",
                before_url=b,
                after_url=a,
                source_page=clink,
            )
        )
        if len(pairs) >= 40:
            break
    return pairs


def scrape_bared(doc: Doctor) -> list[ImagePair]:
    pairs: list[ImagePair] = []
    seen = set()
    pages = list(doc.gallery_urls)
    for base in list(doc.gallery_urls):
        for n in range(2, 15):
            pages.append(base.rstrip("/") + f"/page/{n}")
            pages.append(base.rstrip("/") + f"?page={n}")

    patient_pages: set[str] = set()
    for page in pages:
        try:
            imgs, links, html = parse_page(page)
        except Exception as e:  # noqa: BLE001
            print(f"  page miss {page}: {e}")
            continue
        for l in links:
            if re.search(r"rhinoplasty-photos-\d+|before-and-after/.*rhinoplasty", l, re.I):
                if re.search(r"\d{3,}", l):
                    patient_pages.add(l)
        # extract pairs from listing thumbs
        by_pid: dict[str, list[str]] = defaultdict(list)
        for im in imgs:
            src = im["src"]
            alt = im.get("alt") or ""
            m = re.search(r"Patient\s+(\d+)", alt, re.I) or re.search(r"patient[-_]?(\d+)", src, re.I)
            if m and ("rhino" in src.lower() or "rhino" in alt.lower() or "before" in alt.lower()):
                by_pid[m.group(1)].append(src)
        for pid, urls in by_pid.items():
            urls = list(dict.fromkeys(urls))
            if len(urls) >= 2 and pid not in seen:
                seen.add(pid)
                pairs.append(
                    ImagePair(
                        set_id=f"bared_{pid}",
                        before_url=urls[0],
                        after_url=urls[1],
                        source_page=page,
                        extra={"patient": pid},
                    )
                )

    for plink in sorted(patient_pages):
        m = re.search(r"(\d{3,})", plink)
        pid = m.group(1) if m else plink.rstrip("/").split("/")[-1]
        if pid in seen:
            continue
        try:
            pimgs, _, _ = parse_page(plink)
        except Exception:
            continue
        urls = [
            im["src"]
            for im in pimgs
            if re.search(r"\.(jpe?g|png|webp)$", im["src"], re.I)
            and not any(x in im["src"].lower() for x in ("logo", "banner", "icon", "blank"))
        ]
        urls = list(dict.fromkeys(urls))
        if len(urls) < 2:
            continue
        seen.add(pid)
        pairs.append(
            ImagePair(
                set_id=f"bared_{pid}",
                before_url=urls[0],
                after_url=urls[1],
                source_page=plink,
            )
        )
        if len(pairs) >= 40:
            break
    return pairs


def scrape_afrooz(doc: Doctor) -> list[ImagePair]:
    pairs: list[ImagePair] = []
    seen = set()
    pages = list(doc.gallery_urls)
    for base in list(doc.gallery_urls):
        for n in range(2, 12):
            pages.append(base.rstrip("/") + f"/page/{n}/")

    case_pages: set[str] = set()
    for page in pages:
        try:
            imgs, links, html = parse_page(page)
        except Exception as e:  # noqa: BLE001
            print(f"  page miss {page}: {e}")
            continue
        for l in links:
            if "/gallery/" in l and re.search(r"/\d+/?$", l.rstrip("/")):
                case_pages.add(l)
            if "rhinoplasty" in l.lower() and re.search(r"\d{2,}", l):
                case_pages.add(l)
        # pair consecutive before/after named images
        rhino_imgs = [
            im["src"]
            for im in imgs
            if re.search(r"rhino|nose", im["src"] + " " + (im.get("alt") or ""), re.I)
            and re.search(r"\.(jpe?g|png|webp)$", im["src"], re.I)
        ]
        # group by directory/prefix
        groups: dict[str, list[str]] = defaultdict(list)
        for u in rhino_imgs:
            key = re.sub(r"[-_]?(before|after|ba|front|side|oblique|\d{2,4}x\d{2,4}).*$", "", u, flags=re.I)
            groups[key].append(u)
        for key, urls in groups.items():
            urls = list(dict.fromkeys(urls))
            if len(urls) >= 2:
                sid = re.sub(r"[^a-zA-Z0-9]+", "_", key)[-40:]
                if sid in seen:
                    continue
                seen.add(sid)
                pairs.append(
                    ImagePair(
                        set_id=f"afrooz_{sid}",
                        before_url=urls[0],
                        after_url=urls[1],
                        source_page=page,
                    )
                )

    for clink in sorted(case_pages):
        try:
            pimgs, _, _ = parse_page(clink)
        except Exception:
            continue
        urls = [
            im["src"]
            for im in pimgs
            if re.search(r"\.(jpe?g|png|webp)$", im["src"], re.I)
            and not any(x in im["src"].lower() for x in ("logo", "icon", "blank", "banner"))
        ]
        urls = list(dict.fromkeys(urls))
        if len(urls) < 2:
            continue
        sid = clink.rstrip("/").split("/")[-1]
        if sid in seen:
            continue
        seen.add(sid)
        pairs.append(
            ImagePair(
                set_id=f"afrooz_{sid}",
                before_url=urls[0],
                after_url=urls[1],
                source_page=clink,
            )
        )
        if len(pairs) >= 40:
            break
    return pairs


def scrape_generic_gallery(doc: Doctor, prefix: str) -> list[ImagePair]:
    """Heuristic: find images on gallery pages and pair sequential clinical photos."""
    pairs: list[ImagePair] = []
    seen = set()
    pages = list(doc.gallery_urls)
    # follow pagination-ish links
    for base in list(doc.gallery_urls):
        try:
            _, links, _ = parse_page(base)
            for l in links:
                if any(k in l.lower() for k in ("page/", "rhinoplasty", "before", "gallery")):
                    if l not in pages and urllib.parse.urlparse(l).netloc == urllib.parse.urlparse(base).netloc:
                        pages.append(l)
        except Exception:
            pass
    pages = pages[:25]

    for page in pages:
        try:
            imgs, links, html = parse_page(page)
        except Exception as e:  # noqa: BLE001
            print(f"  page miss {page}: {e}")
            continue

        # Case detail pages
        detail = [
            l
            for l in links
            if re.search(r"(patient|case|before|gallery).*\d+", l, re.I)
            and "rhino" in l.lower()
        ]
        for d in detail[:30]:
            if d in seen:
                continue
            try:
                pimgs, _, _ = parse_page(d)
            except Exception:
                continue
            urls = [
                im["src"]
                for im in pimgs
                if re.search(r"\.(jpe?g|png|webp)$", im["src"], re.I)
                and not any(x in im["src"].lower() for x in ("logo", "icon", "blank", "banner", "sprite"))
            ]
            urls = list(dict.fromkeys(urls))
            if len(urls) < 2:
                continue
            sid = re.sub(r"[^a-zA-Z0-9]+", "_", d.rstrip("/").split("/")[-1])[:40]
            if sid in seen:
                continue
            seen.add(sid)
            pairs.append(
                ImagePair(
                    set_id=f"{prefix}_{sid}",
                    before_url=urls[0],
                    after_url=urls[1],
                    source_page=d,
                )
            )

        # Pair consecutive images that look like clinical pairs
        clinical = []
        for im in imgs:
            src = im["src"]
            if not re.search(r"\.(jpe?g|png|webp)$", src, re.I):
                continue
            if any(x in src.lower() for x in ("logo", "icon", "blank", "banner", "sprite", "avatar")):
                continue
            if any(
                k in (src + " " + (im.get("alt") or "")).lower()
                for k in ("rhino", "nose", "before", "after", "patient", "ba-")
            ) or "/uploads/" in src or "/gallery/" in src:
                clinical.append(src)
        clinical = list(dict.fromkeys(clinical))
        # pair 0-1, 2-3, ...
        for i in range(0, len(clinical) - 1, 2):
            b, a = clinical[i], clinical[i + 1]
            key = (b, a)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                ImagePair(
                    set_id=f"{prefix}_{len(pairs)+1:03d}",
                    before_url=b,
                    after_url=a,
                    source_page=page,
                )
            )
        if len(pairs) >= 40:
            break
    return pairs


def download_pairs(doc: Doctor, pairs: list[ImagePair]) -> dict:
    out_dir = ROOT / doc.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "doctor_name": doc.name,
        "location": doc.location,
        "website": doc.website,
        "gallery_urls": doc.gallery_urls,
        "notes": doc.notes,
        "sets": [],
        "downloaded_sets": 0,
        "failed_sets": 0,
    }
    ok = 0
    for idx, pair in enumerate(pairs, start=1):
        set_dir = out_dir / f"set_{idx:03d}"
        bext = ext_from_url(pair.before_url)
        aext = ext_from_url(pair.after_url)
        bpath = set_dir / f"before{bext}"
        apath = set_dir / f"after{aext}"
        b_ok = save_binary(pair.before_url, bpath)
        a_ok = save_binary(pair.after_url, apath)
        entry = {
            "set_id": pair.set_id,
            "index": idx,
            "before_url": pair.before_url,
            "after_url": pair.after_url,
            "source_page": pair.source_page,
            "before_path": str(bpath.relative_to(ROOT)) if b_ok else None,
            "after_path": str(apath.relative_to(ROOT)) if a_ok else None,
            "ok": b_ok and a_ok,
            "extra": pair.extra,
        }
        if b_ok and a_ok:
            ok += 1
            # write set meta
            (set_dir / "meta.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        else:
            manifest["failed_sets"] += 1
            # clean partial
            if bpath.exists() and not a_ok:
                pass
        manifest["sets"].append(entry)
        if idx % 5 == 0:
            print(f"  [{doc.slug}] {ok}/{idx} sets downloaded...")
    manifest["downloaded_sets"] = ok
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "doctor.json").write_text(
        json.dumps(
            {
                "name": doc.name,
                "slug": doc.slug,
                "location": doc.location,
                "website": doc.website,
                "gallery_urls": doc.gallery_urls,
                "notes": doc.notes,
                "downloaded_sets": ok,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


DOCTORS: list[Doctor] = [
    Doctor(
        name="Dr. Ruslan Zhuravsky, MD",
        slug="zhuravsky_ruslan",
        location="Miami, FL (Aventura / NE 199th St)",
        website="https://zfaceplasticsurgery.com/",
        gallery_urls=["https://zfaceplasticsurgery.com/before-after-gallery/rhinoplasty/"],
        notes="Board-certified facial plastic surgeon; large public rhinoplasty gallery.",
    ),
    Doctor(
        name="Dr. Andres Bustillo, MD, FACS",
        slug="bustillo_andres",
        location="Miami / Coral Gables, FL",
        website="https://www.drbustillo.com/",
        gallery_urls=["https://www.drbustillo.com/before-and-after/rhinoplasty-photos"],
        notes="Board-certified facial plastic surgeon specializing in rhinoplasty.",
    ),
    Doctor(
        name="Dr. Marcelo Ghersi, MD",
        slug="ghersi_marcelo",
        location="Miami, FL",
        website="https://aeris.co/",
        gallery_urls=["https://aeris.co/before-after-photos/primary-rhinoplasty/"],
        notes="Plastic surgeon; 50+ primary rhinoplasty cases listed publicly.",
    ),
    Doctor(
        name="Dr. Anthony Bared, MD, FACS",
        slug="bared_anthony",
        location="Miami, FL",
        website="https://www.facialplasticsurgerymiami.com/",
        gallery_urls=["https://www.facialplasticsurgerymiami.com/before-and-after/rhinoplasty-photos"],
        notes="Double board-certified facial plastic surgeon; rhinoplasty specialist.",
    ),
    Doctor(
        name="Dr. Paul N. Afrooz, MD",
        slug="afrooz_paul",
        location="Coral Gables / Miami, FL",
        website="https://www.drpaulafrooz.com/",
        gallery_urls=["https://www.drpaulafrooz.com/gallery/plastic-surgery/nose/rhinoplasty/"],
        notes="Facial plastic surgeon recognized for rhinoplasty; Newsweek-listed.",
    ),
    Doctor(
        name="Dr. Ary Krau, MD",
        slug="krau_ary",
        location="Miami Beach / Miami, FL",
        website="https://www.arykraumd.com/",
        gallery_urls=["https://www.arykraumd.com/before-after-photos/rhinoplasty-before-after/"],
        notes="Board-certified plastic surgeon with rhinoplasty gallery.",
    ),
    Doctor(
        name="Dr. Joshua A. Lampert, MD",
        slug="lampert_joshua",
        location="Miami, FL",
        website="https://www.lampertmd.com/",
        gallery_urls=[
            "https://www.lampertmd.com/gallery/",
            "https://www.lampertmd.com/photo-gallery/rhinoplasty/",
            "https://www.lampertmd.com/before-and-after/rhinoplasty/",
        ],
        notes="Board-certified plastic surgeon; Miami rhinoplasty practice.",
    ),
    Doctor(
        name="Dr. Adam J. Rubinstein, MD",
        slug="rubinstein_adam",
        location="Aventura / Miami, FL",
        website="https://www.dr-rubinstein.com/",
        gallery_urls=[
            "https://www.dr-rubinstein.com/before-after-rhinoplasty-miami/",
            "https://www.dr-rubinstein.com/photo-gallery/rhinoplasty/",
        ],
        notes="Board-certified plastic surgeon in Aventura.",
    ),
    Doctor(
        name="Dr. Leonard M. Hochstein, MD",
        slug="hochstein_leonard",
        location="Miami, FL",
        website="https://www.lhochsteinmd.com/",
        gallery_urls=["https://www.lhochsteinmd.com/photo-gallery/rhinoplasty/"],
        notes="Miami plastic surgeon with public rhinoplasty photo gallery.",
    ),
    Doctor(
        name="Dr. Michael Careaga / Careaga Plastic Surgery",
        slug="careaga_miami",
        location="Miami, FL",
        website="https://www.careagaplasticsurgery.com/",
        gallery_urls=["https://www.careagaplasticsurgery.com/photo-gallery/rhinoplasty-face/"],
        notes="Careaga Plastic Surgery Miami rhinoplasty gallery.",
    ),
    Doctor(
        name="Miami Plastic Surgery (multi-surgeon practice)",
        slug="miami_plastic_surgery",
        location="Miami, FL",
        website="https://miamiplasticsurgery.com/",
        gallery_urls=["https://miamiplasticsurgery.com/gallery-procedure/face-procedures/rhinoplasty/"],
        notes="Established Miami practice rhinoplasty before/after gallery.",
    ),
    Doctor(
        name="Dr. Michael Salzhauer (Dr. Miami), MD, FACS",
        slug="salzhauer_michael",
        location="Bay Harbor Islands / Miami, FL",
        website="https://therealdrmiami.com/",
        gallery_urls=[
            "https://therealdrmiami.com/gallery",
            "https://therealdrmiami.com/gallery/rhinoplasty",
        ],
        notes="Well-known Miami plastic surgeon; public gallery by procedure.",
    ),
]

SCRAPERS = {
    "zhuravsky_ruslan": scrape_zhuravsky,
    "bustillo_andres": scrape_bustillo,
    "ghersi_marcelo": scrape_ghersi,
    "bared_anthony": scrape_bared,
    "afrooz_paul": scrape_afrooz,
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for doc in DOCTORS:
        print(f"\n=== {doc.name} ({doc.slug}) ===")
        scraper = SCRAPERS.get(doc.slug)
        try:
            if scraper:
                pairs = scraper(doc)
            else:
                pairs = scrape_generic_gallery(doc, doc.slug.split("_")[0])
        except Exception as e:  # noqa: BLE001
            print(f"  SCRAPE ERROR: {e}")
            pairs = []
        print(f"  Found {len(pairs)} candidate pairs")
        if not pairs:
            results.append(
                {
                    "doctor": doc.name,
                    "slug": doc.slug,
                    "location": doc.location,
                    "downloaded_sets": 0,
                    "status": "no_pairs_found",
                }
            )
            continue
        # Cap per doctor to keep download reasonable but >=20 when available
        pairs = pairs[: max(MIN_SETS, min(40, len(pairs)))]
        if len(pairs) > 40:
            pairs = pairs[:40]
        # ensure we try for at least min if available
        man = download_pairs(doc, pairs)
        status = "ok" if man["downloaded_sets"] >= MIN_SETS else "below_min"
        print(f"  Downloaded {man['downloaded_sets']} sets (status={status})")
        results.append(
            {
                "doctor": doc.name,
                "slug": doc.slug,
                "location": doc.location,
                "website": doc.website,
                "downloaded_sets": man["downloaded_sets"],
                "candidate_pairs": len(pairs),
                "status": status,
            }
        )

    # summary
    ok_docs = [r for r in results if r["downloaded_sets"] >= MIN_SETS]
    (ROOT / "doctors.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Rhinoplasty Before/After Dataset",
        "",
        "Public before/after image sets from **named Miami-area** rhinoplasty surgeons' practice websites.",
        "",
        f"- Doctors with ≥{MIN_SETS} sets: **{len(ok_docs)}**",
        f"- Total sets downloaded: **{sum(r['downloaded_sets'] for r in results)}**",
        "",
        "| Doctor | Location | Sets | Status |",
        "|--------|----------|------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r['doctor']} | {r.get('location','')} | {r['downloaded_sets']} | {r['status']} |"
        )
    lines += [
        "",
        "## Structure",
        "",
        "```",
        "rhinoplasty_dataset/",
        "  <doctor_slug>/",
        "    doctor.json",
        "    manifest.json",
        "    set_001/before.jpg + after.jpg + meta.json",
        "```",
        "",
        "## Provenance",
        "",
        "Each set's `meta.json` records the source page and image URLs.",
        "Images remain copyright of the respective practices/patients; for training/research use only.",
        "",
    ]
    (ROOT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone. Qualifying doctors: {len(ok_docs)}")


if __name__ == "__main__":
    main()
