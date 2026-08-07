"""Stage 0: a surgeon's before/after collage -> two clean single-face crops.

Scraped rhinoplasty results arrive as one image per case: `[BEFORE | AFTER]`
side by side, wrapped in the practice's branding. Nothing downstream can measure
that. `split_collage` turns it into two independent photographs.

Splitting at `w // 2` is wrong often enough to matter. On this scrape the
midpoint lands *inside* a panel on 13 of 31 collages, and on 5 of those it is out
by 86-162px: it would shave a strip off the right of the before photograph and
paste it onto the left of the after, which is a before-nose measured as an after.
And a surviving banner strip is worse than a bad split: it drags a face
detector's box off the face, and a brand bar in a flat colour is exactly the sort
of long clean edge a photometric width search is built to find. Four passes, in
order:

1. **Banner bands.** Grow a band in from the top and the bottom while each row
   is still >=55% of the band's own colour AND contains no unbroken non-band run
   wider than 12% of the frame. The second test is what makes it work: colour
   alone cannot separate a black brand bar from a portrait shot on black, but a
   letter stroke is thin and a face is not. So the band swallows a whole line of
   `BEFORE / AFTER / RHINOPLASTY` and stops dead on contact with a head.

2. **The seam.** Columns that are flat top to bottom are gutter -- between the
   panels and outside them -- so what is left between them are the panels, and
   unequal panels and partial carousel slivers both fall out for free. Where the
   two photographs abut with no gutter at all, the seam is the column of maximum
   left-strip/right-strip discontinuity, scored by the MEDIAN over rows so that a
   surgical marking-pen stroke down one forehead cannot outvote a real boundary.

3. **The head.** `insightface` SCRFD, which survives 90-degree profiles where
   MediaPipe does not (see README), swept over four input sizes because these are
   close-ups and a head that fills the frame overflows the detector's largest
   anchor exactly like a head that is too small. The crop is that box grown
   generously above and tightly below.

4. **Overlay type.** A watermark burnt into the middle of the photograph is
   invisible to passes 1-3: it is centred on the collage so it straddles the
   seam, and it sits on neck and hair and clothing so no row of it is flat.
   Below the chin it is found by row statistics (thin high-contrast marks on
   smooth ground); above the face, where hair has that signature too, by
   connected components that have to look typeset. Both are clamped so they can
   spend hair and margin but never eyebrow.

Detectability is recorded, never required. Side halves mostly fail MediaPipe and
that is a property of the photographs, not a bug here -- the manifest says so per
half so the failure is visible instead of silently dropping the case.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --- tuning ---------------------------------------------------------------
FLAT_TOL = 14  # max per-channel deviation still counted as "the same colour"
BANNER_FRAC = 0.55  # of a row must be the band's own colour
BANNER_RUN = 0.12  # of width: longest unbroken non-band run allowed in a row
BANNER_MAX = 0.22  # never eat more than this fraction of the height per side
BANNER_GAP = 0.05  # of height: non-banner rows the band may jump (a text block)

COL_FRAC = 0.92  # of a column must be flat for it to be gutter, not panel
MIN_PANEL = 0.12  # of width; anything narrower is a carousel sliver, not a panel
MERGE_GAP = 0.05  # of width; segments closer than this are one panel

SEAM_SEARCH = 0.30  # search the central +/-30% of width for a gutterless seam
SEAM_STRIP = 14  # px compared either side of a candidate seam

# Head box = face box grown by these fractions of its own size. Tight at the
# bottom on purpose: watermarks live under the chin and noses do not.
GROW_L = GROW_R = 0.40
GROW_TOP = 0.65
GROW_BOT = 0.12


# --- flatness -------------------------------------------------------------
def _deviation(img: np.ndarray, axis: int) -> np.ndarray:
    """Per-pixel distance from its own line's median colour.

    axis=1 -> each row against its own median (banner bands).
    axis=0 -> each column against its own median (gutters).
    """
    a = img.astype(np.int16)
    med = np.median(a, axis=axis, keepdims=True)
    return np.abs(a - med).max(axis=2)


def _flat_fraction(img: np.ndarray, axis: int, tol: int = FLAT_TOL) -> np.ndarray:
    return (_deviation(img, axis) <= tol).mean(axis=axis)


def _max_run(mask: np.ndarray) -> int:
    """Longest unbroken stretch of True."""
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return int((edges[1::2] - edges[0::2]).max())


def _is_banner_row(row: np.ndarray, colour: np.ndarray, tol: int,
                   frac: float, run: int) -> bool:
    """Is this row a strip of `colour` with, at most, lettering drawn on it?

    Two tests, and the second is the one that earns its keep. Colour alone cannot
    tell a black brand bar from a photograph shot on a black background -- both
    rows are mostly black. But what deviates from the background differs in
    *width*: a letter stroke is thin, a face is not. Capping the longest
    unbroken non-background run stops the band the moment it touches a head,
    while letting it swallow an entire line of type.
    """
    near = np.abs(row.astype(np.int16) - colour).max(axis=1) <= tol
    return bool(near.mean() >= frac) and _max_run(~near) <= run


def _band(img: np.ndarray, limit: int, gap: int) -> int:
    """Height of the banner band at the top of `img` (0 if there is none)."""
    colour = np.median(img[:3].reshape(-1, 3), axis=0).astype(np.int16)
    run = max(8, int(img.shape[1] * BANNER_RUN))
    last, miss = -1, 0
    for y in range(min(limit, img.shape[0])):
        if _is_banner_row(img[y], colour, FLAT_TOL, BANNER_FRAC, run):
            last, miss = y, 0
        else:
            miss += 1
            if miss > gap:
                break
    return last + 1


def banner_box(img: np.ndarray) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of `img` with its top and bottom banner bands removed.

    Horizontal bands only. A vertical brand bar would be indistinguishable from
    the gutter at this stage, and the column pass removes it anyway.
    """
    h, w = img.shape[:2]
    limit, gap = int(h * BANNER_MAX), int(h * BANNER_GAP)
    top = _band(img, limit, gap)
    bot = _band(img[::-1], limit, gap)
    y0, y1 = top, h - bot
    if y1 - y0 < h * 0.3:  # pathological; trust nothing and keep the whole frame
        return 0, 0, w, h
    return 0, y0, w, y1


def _segments(img: np.ndarray) -> List[Tuple[int, int]]:
    """Column spans of `img` that are not flat gutter, merged and size-filtered."""
    w = img.shape[1]
    solid = _flat_fraction(img, axis=0) < COL_FRAC
    segs, start = [], None
    for x in range(w):
        if solid[x] and start is None:
            start = x
        elif not solid[x] and start is not None:
            segs.append((start, x))
            start = None
    if start is not None:
        segs.append((start, w))

    merged: List[List[int]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] < w * MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= w * MIN_PANEL]


def _seam_by_discontinuity(img: np.ndarray, x0: int, x1: int) -> int:
    """The column inside [x0, x1) where the picture changes, with no gutter to help.

    Compares the colour of a `SEAM_STRIP`-wide strip either side. A strip mean
    averages away the vertical edges *inside* a photograph -- hair, the nasal
    shadow, a slatted wall -- while a boundary between two different photographs
    survives it.

    Scored by the MEDIAN over rows, not the mean, and that is not a detail. A
    panel boundary is a step at *every* row. A vertical surgical marking-pen
    stroke down a patient's forehead is a step at 10% of them, and on
    murray/5front the mean picked the pen and split the collage at 27% of its
    width. The median cannot be moved by a minority of rows.
    """
    a = img.astype(np.float32)
    cum = np.zeros((a.shape[0], a.shape[1] + 1, 3), np.float32)
    np.cumsum(a, axis=1, out=cum[:, 1:])
    s = SEAM_STRIP
    lo = max(x0 + s + 1, int(x0 + (x1 - x0) * (0.5 - SEAM_SEARCH)))
    hi = min(x1 - s - 1, int(x0 + (x1 - x0) * (0.5 + SEAM_SEARCH)))
    if hi <= lo:
        return (x0 + x1) // 2
    xs = np.arange(lo, hi)
    left = (cum[:, xs] - cum[:, xs - s]) / s
    right = (cum[:, xs + 1 + s] - cum[:, xs + 1]) / s
    score = np.median(np.abs(left - right).mean(axis=2), axis=0)
    return int(xs[int(np.argmax(score))])


# --- face detection -------------------------------------------------------
# SCRFD is scale-limited in BOTH directions: it resizes the input to `det_size`
# and its anchors only span a band of pixel sizes, so a head that fills the frame
# overflows the largest anchor and is missed exactly like a head that is too
# small. These collages are close-up portraits, so the default 640 misses most of
# them -- 1front_before scores 0.85 at 320 and 0.42 at 640, and nothing at 1024.
# Sweeping the input size is the whole fix.
DET_SIZES = ((256, 256), (384, 384), (512, 512), (768, 768))
DET_THRESH = 0.30
DET_KEEP = 0.45  # prefer the biggest box above this; below it, the best-scoring
DET_MIN_AREA = 0.02  # of the panel; smaller is a false positive, not a portrait

_DET = None


def _detector():
    """SCRFD from insightface buffalo_l. Detection only -- no recognition
    embedding is ever computed, so no face template of a patient is produced."""
    global _DET
    if _DET is None:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(512, 512), det_thresh=DET_THRESH)
        _DET = app.det_model
    return _DET


def face_box(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Largest detected face, or None. Works on 90-degree profiles."""
    det = _detector()
    found = []
    for size in DET_SIZES:
        try:
            boxes, _ = det.detect(img, input_size=size)
        except Exception:
            continue
        for b in boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in b[:4])
            area = max(0, x1 - x0) * max(0, y1 - y0)
            if area >= DET_MIN_AREA * img.shape[0] * img.shape[1]:
                found.append((float(b[4]), area, (x0, y0, x1, y1)))
    if not found:
        return None
    strong = [f for f in found if f[0] >= DET_KEEP]
    return max(strong or found, key=lambda f: f[1] if strong else f[0])[2]


TEXT_STROKE = 41  # px: wider than any letter stroke, narrower than a face part
TEXT_CONTRAST = 45  # how far a stroke stands off its own local background
TEXT_COVER = 0.004  # of the row's width, minimum, to call it lettering
TEXT_RUN = 0.08  # of the row's width, maximum, for a single stroke
TEXT_SAFE = 0.12  # of the crop's height: the deepest the scan may ever start
TEXT_LINE = 6  # marked rows needed inside TEXT_WIN to call it a line of type
TEXT_WIN = 25
EDGE_BAND = 0.025  # of the crop height: outer band where one marked row suffices
TOP_TEXT = 0.15  # of the crop height: how far down the top label may be looked for
TOP_SLACK = 0.05  # of the crop height: gap a label may sit below what is cleared


def _lettering_rows(panel: np.ndarray) -> np.ndarray:
    """Per row: does this row contain thin high-contrast strokes on smooth ground?

    A horizontal top-hat/black-hat with a kernel wider than any letter stroke
    keeps type and drops everything broad -- a jaw, a shadow, a hairline all sit
    inside the structuring element and cancel. What is left is scored per row on
    coverage and on the width of its widest mark, so a caption survives the test
    and a chin does not.
    """
    if panel.size == 0:
        return np.zeros(0, bool)
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (TEXT_STROKE, 1))
    mark = np.maximum(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                      cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)) > TEXT_CONTRAST
    limit = max(6, int(panel.shape[1] * TEXT_RUN))
    return np.array([m.mean() >= TEXT_COVER and 0 < _max_run(m) <= limit
                     for m in mark], dtype=bool)


def _first_caption(rows: np.ndarray, need: int = TEXT_LINE, win: int = TEXT_WIN) -> int:
    """First row of a *line* of type: `need` marked rows inside the next `win`.

    One marked row is not a caption. A specular highlight on a lip, an eyelash,
    the edge of a necklace all mark a row or two, and cutting the crop at the
    first one costs real face -- it cost four MediaPipe detections when tried.
    A set caption is tens of rows deep, so requiring density over a window keeps
    every caption and discards the isolated hits.
    """
    if rows.size < need:
        return -1
    dense = np.convolve(rows.astype(np.int32), np.ones(win, np.int32), "valid") >= need
    hit = np.flatnonzero(dense)
    if not hit.size:
        return -1
    return int(np.flatnonzero(rows[hit[0]:hit[0] + win])[0] + hit[0])


GLYPH_H = (6, 0.05)  # a letter: at least 6px tall, at most this much of the crop
GLYPH_W = 0.12  # of the crop's width, maximum, for one letter
GLYPH_FILL = 0.18  # of its own bounding box a letter must actually ink
GLYPH_RUN = 3  # letters that must line up before it counts as a line of type
WORD_ASPECT = 2.5  # w/h at which one blob is a whole word rather than a letter
WORD_W = 0.5  # of the crop's width, maximum, for one word
WORD_FILL = 0.35  # a word closes up denser than a strand of hair ever does


def _text_lines(region: np.ndarray, crop_w: int, crop_h: int) -> List[Tuple[int, int]]:
    """Row spans of `region` occupied by a line of set type.

    Needed above the face, where the row statistics used below the chin are
    useless: hair is thin, high-contrast and covers a few percent of a row, which
    is the signature of lettering exactly. What hair is not is *typeset*. So this
    goes to connected components and asks for letters -- bounded height, bounded
    width, solid enough to have been drawn rather than grown -- and then for at
    least `GLYPH_RUN` of them sharing a baseline. A strand of hair fails on
    height; a field of strands fails on alignment.

    Small type at this resolution often closes into one blob per word, so a
    single wide, short, *dense* component counts on its own. Every bound is a
    fraction of the CROP, never of the scanned strip: measured against an 80-row
    strip a 14-row capital is taller than any plausible letter, and `BEFORE`
    survived three passes of this function for exactly that reason.
    """
    if region.size == 0:
        return []
    g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (TEXT_STROKE, TEXT_STROKE))
    mark = np.maximum(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                      cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)) > TEXT_CONTRAST
    n, _, stats, _ = cv2.connectedComponentsWithStats(mark.astype(np.uint8), 8)

    h_max = GLYPH_H[1] * crop_h
    glyphs, words = [], []
    for i in range(1, n):
        x, y, gw, gh, area = stats[i]
        if not (GLYPH_H[0] <= gh <= h_max) or area < GLYPH_FILL * gw * gh:
            continue
        if 2 <= gw <= GLYPH_W * crop_w:
            glyphs.append((x, y, gw, gh))
        if gw >= WORD_ASPECT * gh and gw <= WORD_W * crop_w and area >= WORD_FILL * gw * gh:
            words.append((int(y), int(y + gh)))

    lines, used = list(words), [False] * len(glyphs)
    order = sorted(range(len(glyphs)), key=lambda i: glyphs[i][0])
    for i in order:
        if used[i]:
            continue
        x, y, gw, gh = glyphs[i]
        group, cy = [i], y + gh / 2
        for j in order:
            if used[j] or j == i:
                continue
            x2, y2, gw2, gh2 = glyphs[j]
            if abs((y2 + gh2 / 2) - cy) <= 0.6 * max(gh, gh2) and gh2 <= 2.5 * gh:
                group.append(j)
        if len(group) >= GLYPH_RUN:
            for j in group:
                used[j] = True
            ys = [int(glyphs[j][1]) for j in group]
            y1s = [int(glyphs[j][1] + glyphs[j][3]) for j in group]
            lines.append((min(ys), max(y1s)))
    return sorted(lines)


def head_box(panel: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Face box grown to a head crop, clipped to the panel.

    The bottom edge grows only as far as the first line of lettering under the
    chin. Practices burn their brand across the *photograph*, below the face and
    centred on the collage, where it straddles the seam and lands inside both
    halves -- no band or gutter detector can see it, because the rows it sits on
    are neck and hair and clothing. The scan starts at the chin, so it can shrink
    the margin to nothing but can never cut into the face.

    Every edge is clipped into the panel. On the tightest crops in this scrape --
    Murray posts a brow-to-lip fragment -- SCRFD reports the box of the whole
    head it *infers*, overhanging the panel by hundreds of pixels on all four
    sides. An unclipped edge there does not enlarge the head crop, it walks out
    of the panel and back into the banner the first pass just removed.
    """
    h, w = panel.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    top = max(0, int(y0 - GROW_TOP * bh))
    bottom = min(h, int(y1 + GROW_BOT * bh))
    # Scan below the chin, or from the last TEXT_SAFE of the crop if the chin is
    # lower than that -- which it is whenever SCRFD has inferred a chin that the
    # photograph does not show. The nose sits well above that line either way.
    chin = min(max(y1, top), bottom)
    start = max(top, min(chin, bottom - int(TEXT_SAFE * max(1, bottom - top))))
    hit = _first_caption(_lettering_rows(panel[start:bottom]))
    if hit >= 0:
        bottom = start + max(0, hit - 4)
    bottom = min(h, max(start, bottom))

    # Edge sweep. A banner the band pass has already cut down to a few rows is
    # still a few rows of somebody's brand, and by then it is far too thin to
    # read as a line of type. In the outermost EDGE_BAND of the crop a single
    # marked row is enough, because being wrong there costs 2.5% of a margin.
    band = max(4, int(EDGE_BAND * max(1, bottom - top)))
    low = np.flatnonzero(_lettering_rows(panel[max(top, bottom - band):bottom]))
    if low.size:
        bottom = max(top + 1, max(top, bottom - band) + int(low[0]))
    # Above the head there is only ever a corner label, and only above the face
    # box -- so a hit can cost hair and never an eyebrow.
    ceiling = max(top, min(y0, top + int(TOP_TEXT * max(1, bottom - top))))
    lines = _text_lines(panel[top:ceiling], w, max(1, bottom - top))
    # Walk down only while each line still hangs off the edge already cleared.
    # Anchoring to the top is what stops the scan chasing a hairline halfway
    # down the crop on the strength of one blob that happens to read as type.
    slack = max(20, int(TOP_SLACK * max(1, bottom - top)))
    cleared = 0
    for a, b in lines:
        if a > cleared + slack:
            break
        cleared = max(cleared, b + 2)
    top = max(0, min(y0, bottom - 1, top + cleared))
    cx0 = max(0, min(w - 1, int(x0 - GROW_L * bw)))
    cx1 = max(cx0, min(w, int(x1 + GROW_R * bw)))
    bottom = max(0, min(h, bottom))
    if cx1 - cx0 < 2 or bottom - top < 2:  # nothing survived; keep the panel
        return 0, 0, w, h
    return cx0, int(top), cx1, int(bottom)


# --- the split ------------------------------------------------------------
@dataclass
class Half:
    role: str  # "before" | "after"
    box: Tuple[int, int, int, int]  # final crop, in SOURCE image coordinates
    panel_box: Tuple[int, int, int, int]  # before the head crop
    face_detected: bool = False  # mediapipe FaceMesh, on the final crop
    face_view: Optional[str] = None
    face_yaw_deg: Optional[float] = None
    head_box_used: bool = False  # False = insightface found nothing, kept panel
    path: Optional[str] = None


@dataclass
class Split:
    source: str
    surgeon_key: str
    view: str
    procedure: Optional[str]
    procedure_source: str  # "banner" = printed on the collage | "inferred" | "unknown"
    is_rhinoplasty: bool
    size: Tuple[int, int]
    split_x: int
    banner_box: Tuple[int, int, int, int]
    n_segments: int
    seam_method: str  # "gutter" | "discontinuity"
    halves: List[Half] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def split_collage(img: np.ndarray) -> Tuple[Tuple[int, int, int, int], int,
                                            List[Tuple[int, int, int, int]], str, int]:
    """-> (banner_box, split_x, [left_panel, right_panel], seam_method, n_segments).

    All boxes in source coordinates.
    """
    bx0, by0, bx1, by1 = banner_box(img)
    body = img[by0:by1, bx0:bx1]

    segs = _segments(body)
    if len(segs) >= 2:
        # Two widest, kept in left-to-right order. A partial third panel is
        # narrower than either real one, so it drops out here.
        a, b = sorted(sorted(segs, key=lambda s: s[1] - s[0])[-2:])
        split_x = (a[1] + b[0]) // 2
        panels = [(a[0], a[1]), (b[0], b[1])]
        method = "gutter"
    else:
        s0, s1 = segs[0] if segs else (0, body.shape[1])
        split_x = _seam_by_discontinuity(body, s0, s1)
        panels = [(s0, split_x), (split_x, s1)]
        method = "discontinuity"

    out = []
    for px0, px1 in panels:
        # Second banner pass, per panel: overlaid corner text that survived the
        # full-width pass because the other panel's photograph covered it.
        p = body[:, px0:px1]
        _, py0, _, py1 = banner_box(p)
        out.append((bx0 + px0, by0 + py0, bx0 + px1, by0 + py1))
    return (bx0, by0, bx1, by1), bx0 + split_x, out, method, len(segs)


def ingest_image(path: str, surgeon_key: str, view: str, procedure: Optional[str],
                 procedure_source: str, is_rhinoplasty: bool, out_dir: str, stem: str,
                 notes: Optional[List[str]] = None) -> Split:
    """Split one collage, write `<stem>_before.png` / `<stem>_after.png`, describe it."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    banner, split_x, panels, method, n_seg = split_collage(img)

    rec = Split(source=path, surgeon_key=surgeon_key, view=view, procedure=procedure,
                procedure_source=procedure_source, is_rhinoplasty=is_rhinoplasty,
                size=(w, h), split_x=split_x,
                banner_box=banner, n_segments=n_seg, seam_method=method,
                notes=list(notes or []))

    os.makedirs(out_dir, exist_ok=True)
    for role, (px0, py0, px1, py1) in zip(("before", "after"), panels):
        panel = img[py0:py1, px0:px1]
        fb = face_box(panel)
        if fb is None:
            box, used = (px0, py0, px1, py1), False
        else:
            hx0, hy0, hx1, hy1 = head_box(panel, fb)
            box, used = (px0 + hx0, py0 + hy0, px0 + hx1, py0 + hy1), True

        crop = img[box[1]:box[3], box[0]:box[2]]
        dst = os.path.join(out_dir, f"{stem}_{role}.png")
        cv2.imwrite(dst, crop)

        half = Half(role=role, box=box, panel_box=(px0, py0, px1, py1),
                    head_box_used=used, path=dst)
        try:
            from .landmarks import detect

            lm = detect(crop)
            half.face_detected = True
            half.face_view = lm.view
            half.face_yaw_deg = round(lm.yaw_deg, 1)
        except Exception:
            half.face_detected = False
        rec.halves.append(half)
    return rec


def write_manifest(records: List[Split], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


# --- QA -------------------------------------------------------------------
def contact_sheet(records: List[Split], path: str, tile_h: int = 300,
                  per_row: int = 4) -> None:
    """Every before/after crop, labelled, so a human can reject a bad split.

    Automated splitting cannot verify itself: a crop that is off by 40px looks
    identical to a correct one in any metric this module could compute.
    """
    tiles = []
    for r in records:
        pair = []
        for hf in r.halves:
            im = cv2.imread(hf.path)
            if im is None:
                im = np.zeros((tile_h, tile_h // 2, 3), np.uint8)
            s = tile_h / im.shape[0]
            pair.append(cv2.resize(im, (max(1, int(im.shape[1] * s)), tile_h)))
        gap = np.full((tile_h, 6, 3), (0, 140, 255), np.uint8)
        t = np.hstack([pair[0], gap, pair[1]])
        t = cv2.copyMakeBorder(t, 44, 6, 6, 6, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        tag = os.path.basename(r.source)
        det = "".join("Y" if hf.face_detected else "n" for hf in r.halves)
        colour = (255, 255, 255) if r.is_rhinoplasty else (60, 60, 255)
        cv2.putText(t, f"{tag} {r.procedure or '?'} det:{det}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        cv2.putText(t, f"x={r.split_x} {r.seam_method} seg={r.n_segments}", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)
        tiles.append(t)

    rows = []
    for i in range(0, len(tiles), per_row):
        chunk = tiles[i:i + per_row]
        hmax = max(t.shape[0] for t in chunk)
        chunk = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(30, 30, 30))
                 for t in chunk]
        rows.append(np.hstack(chunk))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def debug_overlay(path: str, rec: Split, dst: str) -> None:
    """The source with the banner box, seam and every crop drawn on it."""
    img = cv2.imread(path)
    x0, y0, x1, y1 = rec.banner_box
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 255), 4)
    cv2.line(img, (rec.split_x, 0), (rec.split_x, img.shape[0]), (0, 0, 255), 4)
    for hf in rec.halves:
        cv2.rectangle(img, hf.panel_box[:2], hf.panel_box[2:], (255, 120, 0), 3)
        cv2.rectangle(img, hf.box[:2], hf.box[2:], (0, 255, 0), 5)
    s = 900 / max(img.shape[:2])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, cv2.resize(img, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])
