"""Stage 2b: profile NoseParams from the head SILHOUETTE, without landmarks.

Why not landmarks
-----------------
MediaPipe FaceMesh detects 10 of the 30 scraped side halves, and -- the number
that actually matters -- both halves of only 3 of the 15 before/after pairs. A
signature is fitted from differences, so a half without its partner is worth
nothing. (`scripts/profile_check.py --mediapipe` recomputes both counts on the
same panels rather than quoting them.) A 90-degree view puts half the face behind
the other half and a regressor trained on mostly-frontal faces has nothing to
anchor on. That is not a tuning problem.

What a 90-degree view *does* give, and a frontal view does not, is the whole nose
rendered as one unambiguous high-contrast curve: skin against studio backdrop.
Every quantity a rhinoplasty changes in profile -- hump, projection, rotation,
length -- is a property of that curve. So this module never detects a face. It
segments the head, walks the facing-direction boundary, and locates four
anatomical points on it, none by a threshold tuned to face size:

    pronasale    the most anterior point of the curve. That is the definition.
    nasion       apex of the deepest concavity above the tip.
    subnasale    apex of the nearest concavity below the tip.
    labrale      the most convex point of the upper lip below that.

Concavities come from the curve's own convex hull -- every hollow in a profile is
a gap under some hull edge, and a hull is invariant to scale, resolution and head
tilt -- and are then re-located as corners by `chord_deviation`, whose chord
length is set by the nose the hull just measured. Deliberately NOT used: local
extrema of the anterior projection. On a sloping forehead the profile runs
monotonically forward from hairline to tip and the nasion is not a turning point
at all; 7 of these 30 halves are like that, and an extremum search either rejects
them or walks past the brow and plants "nasion" up in the hairline.

Scale reference
---------------
NOT interpupillary distance. These crops are tight -- several cut the eyes
entirely, most have the eyes closed, and surgeon A's side collages frame
brow-to-lip only. IPD is unmeasurable on the majority of this data.

This module normalises by |nasion -> labrale|, which has to earn its place twice:

  invariant to surgery -- nasion is the bony nasofrontal suture and labrale is
    the vermilion of the upper lip. Rhinoplasty operates strictly between them
    and moves neither. (Compare nasal_length or subnasale, which surgery moves by
    design and which therefore cannot be a scale.)
  survives the crop -- both lie inside the nose-and-lips box every one of these
    collages is framed around. The brow, the eyes, the chin and the hairline are
    each cut off in at least one image; these two in none.

It also does real work: the two panels of one collage are often at different
zoom -- 854 px against 765 px of the same patient in surgeon A's 1side -- so an
un-normalised comparison would report surgery that did not happen.

The residual worry is that a hump reduction slides the *measured* nasion, since
it is the deepest point of a curve that changed shape. `profile_check.py` tests
that directly: |brow -> labrale| / |nasion -> labrale| between a patient's two
photographs moves by a median 3.4% (max 10.1%, n=7 pairs where the brow is in
frame twice), and brow and nasion cannot both be the one that moved.

    UNITS WARNING. Profile lengths here are in units of |nasion -> labrale|.
    Frontal lengths in params.py are in units of IPD. `nasal_length` is the one
    field both views produce and the two numbers are NOT interchangeable --
    subtracting one from the other is a units bug. Fit and apply profile deltas
    against profile measurements only. `ProfileResult.scale_ref` names the unit
    so a caller can assert on it.

What this cannot do
-------------------
A silhouette carries no yaw. A head turned 20 degrees off profile traces a
perfectly plausible curve, just foreshortened by cos(yaw) in every anterior
quantity, and nothing in the outline says so. Three-quarter views are therefore
measured, not rejected, and the caller has to look. That is the honest statement
of the limitation: MediaPipe would supply a yaw but only on the minority of
frames it detects at all, and its estimate saturates at +/-90 on this data.

Everything returns None rather than a guess. A rejected image is a good outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .contracts import NoseParams

SCALE_REF = "nasion_labrale"

# --- segmentation ---------------------------------------------------------
BG_TOL = 14.0          # Lab distance counted as "same colour as the backdrop"
BG_UNIFORM_FRAC = 0.55  # of edge-strip pixels agreeing on one colour: flat backdrop
BG_GRADIENT_FRAC = 0.30  # ... relaxed to this when the strip is flat row by row
BG_ROW_FRAC = 0.95     # ... which is what "flat row by row" means
MIN_HEAD_FRAC = 0.15   # head component must fill this share of the panel

# --- contour --------------------------------------------------------------
N_RESAMPLE = 900       # points along the facing edge, uniform in arc length
SMOOTH_FRAC = 0.010    # Gaussian sigma, as a fraction of contour arc length
EDGE_MARGIN = 3        # px; an edge this close to the panel frame is not skin

# --- landmark search ------------------------------------------------------
MIN_NOSE_PX = 20.0      # a "nose" smaller than this is a segmentation failure
CHORD_FRAC = 0.45       # chord half-length for the refinement pass, x nose arc
LIP_WINDOW = 0.35       # search for labrale within this x nasal length of subnasale


# ==========================================================================
# collage handling
# ==========================================================================
def _mask_banner(lab: np.ndarray, bg: np.ndarray,
                 max_frac: float = 0.30) -> np.ndarray:
    """Rows of a solid surgeon-watermark bar at the top or bottom of a panel.

    A banner row is near-uniform across x and is *not* the backdrop colour --
    that second clause is what stops the run from eating its way down through
    plain backdrop into the head, which would leave the subject touching the
    panel edge and destroy the very border strip the backdrop was measured from.
    """
    h = lab.shape[0]
    med = np.median(lab, axis=1)                       # (h, 3) per-row median
    flat = (np.linalg.norm(lab - med[:, None, :], axis=2) < BG_TOL).mean(axis=1) > 0.85
    off_bg = np.linalg.norm(med - bg, axis=1) > BG_TOL
    banner = np.zeros(h, bool)
    lim = int(h * max_frac)
    for rows in (range(min(lim, h)), range(h - 1, max(-1, h - 1 - lim), -1)):
        for r in rows:
            if not (flat[r] and off_bg[r]):
                break
            banner[r] = True
    return banner


def split_collage(img: np.ndarray, seam_search: float = 0.08,
                  gutter: float = 0.012) -> List[Tuple[str, np.ndarray]]:
    """A BEFORE|AFTER collage into its two halves, left first.

    The seam is the column of maximum horizontal gradient energy near the middle
    -- a divider line, or the hard cut between two panels. A gutter is shaved off
    each inner edge so the divider itself never enters a panel, and off each
    outer edge so a carousel arrow or the sliver of a neighbouring slide does not.
    """
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    energy = np.abs(np.diff(f, axis=1)).sum(axis=(0, 2)) / h
    lo, hi = int(w * (0.5 - seam_search)), int(w * (0.5 + seam_search))
    seam = lo + int(energy[lo:hi].argmax())
    g = max(4, int(w * gutter))
    left, right = img[:, g:seam - g], img[:, seam + g:w - g]
    return [("before", left), ("after", right)]


# ==========================================================================
# segmentation
# ==========================================================================
def backdrop(lab: np.ndarray):
    """A per-row backdrop colour model, from whichever vertical edge strip is cleaner.

    Vertical strips, not the whole border ring: a head touches the top and bottom
    of a tight crop and the watermark bar owns the bottom rows, but the strip of
    columns the subject is facing into is backdrop down almost its whole height.

    Two agreement scores gate a strip. `glob` -- does the strip agree with one
    colour -- is the ordinary case. `row` -- does each row of the strip agree with
    its own median -- additionally admits a *graded* backdrop, which is why
    surgeon A's lit blue sweep passes at glob=0.42 / row=1.00 instead of being
    thrown out as non-uniform. Wood slats run vertically and fail both (glob 0.17,
    row 0.22-0.50), so surgeon B's wall is still refused rather than modelled out
    of grain.

    Returns a list of candidate models, best first, each a tuple
    (per-row Lab (h,3), the single global Lab, explanation) -- plus a reason
    string when the list is empty.
    """
    h, w = lab.shape[:2]
    k = max(4, int(w * 0.04))
    out, why = [], []
    for name, strip in (("left", lab[:, :k]), ("right", lab[:, -k:])):
        px = strip.reshape(-1, 3)
        med = np.median(px, axis=0)
        glob = float((np.linalg.norm(px - med, axis=1) < BG_TOL).mean())
        rmed = np.median(strip, axis=1)                       # (h, 3)
        row = float((np.linalg.norm(strip - rmed[:, None, :], axis=2) < BG_TOL).mean())
        if glob >= BG_UNIFORM_FRAC:
            # Flat backdrop: one colour for the whole panel. Preferred wherever it
            # applies, because a per-row model has a degree of freedom per row and
            # every row the subject crosses is a chance to fit the subject instead.
            out.append((glob, np.repeat(med[None, :], h, axis=0), med,
                        f"flat backdrop L*a*b*={med.round(0).tolist()} "
                        f"from {name} strip ({glob:.2f})"))
        elif glob >= BG_GRADIENT_FRAC and row >= BG_ROW_FRAC:
            # Graded backdrop -- surgeon A's lit blue sweep reads flat 0.42 but
            # per-row 1.00. Smoothing the row model keeps the grade (a gradient is
            # smooth by construction) while stopping one crossed row from punching
            # a hole in the mask.
            sm = cv2.GaussianBlur(rmed.reshape(-1, 1, 3), (1, 31), 0,
                                  borderType=cv2.BORDER_REPLICATE).reshape(-1, 3)
            out.append((glob, sm, med,
                        f"graded backdrop ~{med.round(0).tolist()} from {name} strip "
                        f"(flat {glob:.2f}, per-row {row:.2f})"))
        else:
            why.append(f"{name} flat {glob:.2f} per-row {row:.2f}")
    if not out:
        return [], "backdrop not uniform (" + "; ".join(why) + ")"
    # Both sides can qualify -- a head that fills one strip with flat skin scores
    # as well as the backdrop does. Ordering by agreement puts the likelier model
    # first; `measure_profile` falls through to the other if the first yields no
    # usable facing edge, which is the only test that actually distinguishes them.
    out.sort(key=lambda c: -c[0])
    return [(m, g, h_) for _, m, g, h_ in out], ""


def head_mask(panel: np.ndarray, model=None) -> Tuple[Optional[np.ndarray], str]:
    """Largest connected foreground blob, cleaned. None if nothing head-sized.

    The connected-component step is what makes carousel arrows, watermark text
    and lens flare harmless: they are separate blobs and the head is the biggest
    thing in a portrait.
    """
    lab = cv2.cvtColor(panel, cv2.COLOR_BGR2LAB).astype(np.float32)
    if model is None:
        cands, why = backdrop(lab)
        if not cands:
            return None, why
        model = cands[0]
    bg_row, bg, how = model
    fg = (np.linalg.norm(lab - bg_row[:, None, :], axis=2) >= BG_TOL).astype(np.uint8)
    fg[_mask_banner(lab, bg)] = 0
    h, w = fg.shape
    r = max(2, int(min(h, w) * 0.006))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n < 2:
        return None, how + "; no foreground"
    i = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    if stats[i, cv2.CC_STAT_AREA] < MIN_HEAD_FRAC * h * w:
        return None, how + f"; largest blob only {stats[i, cv2.CC_STAT_AREA] / (h * w):.2f} of panel"
    mask = (lbl == i).astype(np.uint8)
    # Fill interior holes -- a dark nostril, an eye, a highlight the colour of
    # the backdrop. Flood from a *padded* corner: seeding at (0,0) of the raw
    # mask silently inverts the whole thing whenever the head touches the corner,
    # which in a tight crop is most of the time.
    ff = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    mask |= (1 - ff[1:-1, 1:-1])
    return mask, how


def facing_sign(mask: np.ndarray) -> int:
    """+1 if the subject faces image-right, -1 if left.

    Two cues, in order of decisiveness. First: these crops are framed on the
    face, so the occipital side runs off the edge of the picture while the facing
    side has backdrop in front of it -- the fraction of rows whose extreme is
    strictly inside the frame separates the two sides 0.95+ against 0.00-0.32
    across all 30 halves here. Second, as a tie-break for a centred head: the
    facing side carries a sharp local protrusion (the nose) and the back of the
    head is a smooth arc, so the peak departure from a heavily smoothed copy of
    the edge is far larger on the face side.
    """
    h, w = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size < 20:
        return 1
    score = {}
    for s in (1, -1):
        e = np.array([np.flatnonzero(mask[y])[-1 if s > 0 else 0] for y in rows],
                     dtype=np.float64)
        inside = float(((e > EDGE_MARGIN) & (e < w - 1 - EDGE_MARGIN)).mean())
        es = e * s
        k = max(3, (rows.size // 6) | 1)
        base = cv2.GaussianBlur(es.reshape(-1, 1), (1, k), 0).ravel()
        score[s] = (inside, float((es - base).max()))
    if abs(score[1][0] - score[-1][0]) > 0.10:
        return 1 if score[1][0] > score[-1][0] else -1
    return 1 if score[1][1] >= score[-1][1] else -1


# ==========================================================================
# contour
# ==========================================================================
def facing_contour(mask: np.ndarray, sign: int) -> Tuple[Optional[np.ndarray], str]:
    """The facing-direction boundary of the head, top to bottom, arc-resampled.

    Rows where the extreme touches the panel frame are dropped -- there the head
    runs out of the crop and the "silhouette" is the picture edge, not a face --
    and only the longest surviving run is kept, so a crop that clips the forehead
    costs the forehead and not the whole trace.
    """
    h, w = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size < 40:
        return None, "head spans too few rows"
    xs = np.array([np.flatnonzero(mask[y])[-1 if sign > 0 else 0] for y in rows],
                  dtype=np.float64)
    good = (xs > EDGE_MARGIN) & (xs < w - 1 - EDGE_MARGIN)
    # Bridge short breaks first. A stray highlight the colour of the backdrop can
    # knock two or three rows out of the mask, and without this the "longest
    # contiguous run" rule then throws away everything below the break -- which is
    # how a perfectly good trace loses its subnasale.
    gap_max = max(3, int(0.02 * good.size))
    i = 0
    while i < good.size:
        if good[i]:
            i += 1
            continue
        j = i
        while j < good.size and not good[j]:
            j += 1
        if 0 < i and j < good.size and j - i <= gap_max:
            xs[i:j] = np.interp(np.arange(i, j), [i - 1, j], [xs[i - 1], xs[j]])
            good[i:j] = True
        i = j
    # longest contiguous run of usable rows
    best = (0, 0)
    i = 0
    while i < good.size:
        if good[i]:
            j = i
            while j < good.size and good[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    a, b = best
    if b - a < 40:
        return None, "facing edge never leaves the frame"
    pts = np.stack([xs[a:b], rows[a:b].astype(np.float64)], axis=1)

    # resample uniformly in arc length, then smooth
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 40:
        return None, "facing edge too short"
    t = np.linspace(0, s[-1], N_RESAMPLE)
    out = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)
    sig = max(1.0, SMOOTH_FRAC * N_RESAMPLE)
    ksz = int(sig * 6) | 1
    out = cv2.GaussianBlur(out.reshape(-1, 1, 2), (1, ksz), sig,
                           borderType=cv2.BORDER_REPLICATE).reshape(-1, 2)
    return out, f"arc {s[-1]:.0f}px over rows {rows[a]}..{rows[b - 1]}"


def _anterior_axis(contour: np.ndarray, sign: int) -> np.ndarray:
    """The anterior unit vector for a possibly head-tilted profile.

    Total-least-squares line through the facing edge. The nose is a bump on an
    otherwise near-straight, near-vertical curve, so the principal direction is
    the head's own vertical and every measurement below is tilt-free.
    """
    c = contour - contour.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    down = vt[0]
    if down[1] < 0:
        down = -down
    ant = np.array([down[1], -down[0]])          # rotate -90 deg
    if ant[0] * sign < 0:
        ant = -ant
    return ant


def anterior_hull(contour: np.ndarray) -> np.ndarray:
    """Contour indices on the convex hull, in contour order.

    The facing edge is an open curve, so its convex hull is that curve's own
    anterior envelope closed off by one chord from the last point back to the
    first. Sorting the hull vertices by contour index therefore hands back
    exactly the anterior chain, and every anatomical concavity -- nasofrontal
    angle, subnasale, mentolabial sulcus -- is a gap underneath one of its edges.

    This is the reason the landmark search needs no thresholds tuned to face
    size, image resolution or head tilt: a hull is invariant to all three.
    """
    h = cv2.convexHull(contour.astype(np.float32).reshape(-1, 1, 2),
                       returnPoints=False).ravel()
    return np.unique(h)


def _deepest_gap(contour: np.ndarray, i0: int, i1: int) -> Tuple[Optional[int], float]:
    """The point between two hull vertices that lies furthest behind their chord.

    Coarse only. It says *that* there is a concavity here and roughly how big,
    which is all the bootstrap below needs; where exactly its apex lands depends
    on where the hull happened to anchor, and a crop that cuts the glabella can
    slide it a tenth of a nose down the dorsum. `chord_deviation` fixes that.
    """
    if i1 - i0 < 3:
        return None, 0.0
    p0, p1 = contour[i0], contour[i1]
    d = p1 - p0
    n = np.array([-d[1], d[0]])
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None, 0.0
    dist = np.abs((contour[i0 + 1:i1] - p0) @ (n / nn))
    j = int(dist.argmax())
    return i0 + 1 + j, float(dist[j])


def chord_deviation(contour: np.ndarray, L: int, ant: np.ndarray) -> np.ndarray:
    """Signed departure of each point from the chord spanning +/- L points.

    Curvature at a chosen scale, in pixels: positive where the profile bulges
    anteriorly (tip, brow, lip), negative in a hollow (nasofrontal angle,
    subnasale, mentolabial sulcus). Unlike the hull, both chord ends sit a fixed
    arc length away, so the answer is a local property of the corner and does not
    move when the crop moves. Unlike a derivative-based curvature it needs no
    extra smoothing -- L *is* the smoothing scale, and L is set from the size of
    the nose the hull already found.
    """
    n = contour.shape[0]
    out = np.zeros(n)
    L = max(3, min(L, (n - 1) // 2))
    i = np.arange(L, n - L)
    d = contour[i + L] - contour[i - L]
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    nrm *= np.sign(nrm @ ant)[:, None]        # +ve = anterior
    out[i] = ((contour[i] - contour[i - L]) * nrm).sum(axis=1)
    return out


def _concavity_near(dev: np.ndarray, lo: int, hi: int, side: str,
                    rel: float = 0.5) -> Tuple[Optional[int], float]:
    """The concavity nearest one end of [lo, hi) that is `rel` as deep as the best.

    Nearest-to-the-tip rather than deepest, in both directions. Below the tip the
    columella hollow is what we want and the mentolabial sulcus further down is
    often deeper; above the tip the nasofrontal angle is what we want and a
    strong brow ridge puts a supraorbital notch beyond it. Taking the nearest
    survivor rejects both. Requiring half the depth of the best candidate rejects
    the opposite failure -- the shallow supratip break of a humped nose, which
    sits between the tip and the nasion.
    """
    seg = dev[lo:hi]
    if seg.size < 5:
        return None, 0.0
    inner = np.arange(1, seg.size - 1)
    loc = inner[(seg[inner] < seg[inner - 1]) & (seg[inner] <= seg[inner + 1])
                & (seg[inner] < 0)]
    if loc.size == 0:
        return None, 0.0
    deepest = -seg[loc].min()
    keep = loc[-seg[loc] >= rel * deepest]
    j = int(keep[-1] if side == "hi" else keep[0])
    return lo + j, float(-seg[j])


# ==========================================================================
# measurement
# ==========================================================================
@dataclass
class ProfileResult:
    """Everything the profile stage knows, including why it gave up."""

    ok: bool
    reason: str
    params: Optional[NoseParams] = None
    scale_ref: str = SCALE_REF
    scale_px: Optional[float] = None
    facing: int = 1
    contour: Optional[np.ndarray] = None
    points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    extra: Dict[str, float] = field(default_factory=dict)


def measure_profile(panel: np.ndarray) -> ProfileResult:
    """A profile photograph -> NoseParams, or a stated reason for None.

    Normalised by |nasion -> labrale| (see module docstring). Never returns a
    number it could not locate the geometry for.
    """
    lab = cv2.cvtColor(panel, cv2.COLOR_BGR2LAB).astype(np.float32)
    models, why = backdrop(lab)
    if not models:
        return ProfileResult(False, f"segmentation: {why}")
    # Both edge strips can pass the uniformity gate -- a head filling one strip
    # with flat skin agrees with itself as well as a backdrop does. Rather than
    # guess, run the whole extraction against each hypothesis and keep the first
    # that produces a face. Every gate below is a test of the hypothesis too.
    out = None
    for model in models:
        out = _measure_with(panel, lab, model)
        if out.ok:
            return out
    return out


def _measure_with(panel, lab, model) -> ProfileResult:
    """One backdrop hypothesis, all the way through."""
    mask, how = head_mask(panel, model)
    if mask is None:
        return ProfileResult(False, f"segmentation: {how}")
    sign = facing_sign(mask)
    contour, cinfo = facing_contour(mask, sign)
    if contour is None:
        return ProfileResult(False, f"contour: {cinfo}", facing=sign)
    note = f"{model[2]}; {cinfo}"

    ant = _anterior_axis(contour, sign)
    a = (contour - contour[0]) @ ant           # anterior projection, px
    rng = float(a.max() - a.min())
    if rng < MIN_NOSE_PX:
        return ProfileResult(False, "contour is flat -- not a face profile",
                             facing=sign, contour=contour)

    # Pronasale is the most anterior point of the whole profile -- that is its
    # definition, and it needs no threshold.
    i_tip = int(a.argmax())

    # Nasion and subnasale are the apexes of the hollows either side of the tip.
    # Note what is NOT used to find them: a local minimum of the anterior
    # projection. On a sloping forehead the profile runs monotonically forward
    # from hairline to tip and the nasion is not a turning point at all -- 7 of
    # these 30 halves are like that, and an extremum search either rejects them
    # or, worse, walks past the brow and plants "nasion" up in the hairline.
    hull = anterior_hull(contour)
    above = hull[hull < i_tip]
    below = hull[hull > i_tip]
    if above.size == 0 or above[-1] < 3:
        return ProfileResult(False, "no nasion: crop starts below the nasofrontal angle",
                             facing=sign, contour=contour)
    if below.size == 0:
        return ProfileResult(False, "no subnasale: crop ends at the tip",
                             facing=sign, contour=contour)
    # Hull vertices run contiguously along the convex tip and the convex brow, so
    # what matters is not a vertex but an *edge that bridges something*. Score
    # every hull edge by the depth of the hollow underneath it.
    edges = [(int(hull[k]), int(hull[k + 1]),
              *_deepest_gap(contour, int(hull[k]), int(hull[k + 1])))
             for k in range(hull.size - 1)]
    up = [e for e in edges if e[1] <= i_tip and e[2] is not None]
    dn = [e for e in edges if e[0] >= i_tip and e[2] is not None]
    if not up or max(e[3] for e in up) < 2.0:
        return ProfileResult(False, "no nasion: no concavity between brow and dorsum",
                             facing=sign, contour=contour)
    if not dn or max(e[3] for e in dn) < 2.0:
        return ProfileResult(False, "no subnasale: no concavity below the tip",
                             facing=sign, contour=contour)

    # Above the tip take the DEEPEST hollow: a humped nose has a shallow supratip
    # break in the way and a "nearest" rule would stop there. Below the tip take
    # the NEAREST: nothing at all lies between the tip and the columella base,
    # while the mentolabial sulcus further down is often the deeper of the two.
    b0, _, c_nasion, _ = max(up, key=lambda e: e[3])
    dn_max = max(e[3] for e in dn)
    _, c1, c_sub, _ = next(e for e in dn if e[3] >= 0.25 * dn_max)
    if b0 < 3:
        return ProfileResult(False, "no nasion: crop starts below the nasofrontal angle",
                             facing=sign, contour=contour)

    # Refinement. The hull located each hollow but anchored its chord wherever the
    # crop happened to end, which slides the apex; re-find both apexes as corners
    # at a chord scale set by the nose the hull just measured.
    L = max(4, int(round(CHORD_FRAC * (i_tip - c_nasion))))
    dev = chord_deviation(contour, L, ant)
    i_nasion, _ = _concavity_near(dev, max(L, b0), i_tip, "hi")
    i_sub, _ = _concavity_near(dev, i_tip, min(c1, contour.shape[0] - L), "lo")
    if i_nasion is None:
        i_nasion = c_nasion
    if i_sub is None:
        i_sub = c_sub

    nose_px = float(np.linalg.norm(contour[i_tip] - contour[i_nasion]))
    if nose_px < MIN_NOSE_PX:
        return ProfileResult(False, f"nose height {nose_px:.0f}px -- segmentation failed",
                             facing=sign, contour=contour)

    # Labial point (labrale superius): the most convex point of the upper lip.
    # Two failure modes had to be closed here, because this point sets the scale
    # and a swap between the two photographs of one patient corrupts every number:
    #   * a local maximum of the anterior projection can be a sub-pixel philtral
    #     shoulder just under the subnasale -- convexity at a lip-sized chord sees
    #     the vermilion instead;
    #   * the hull vertex closing the subnasal hollow is the CHIN whenever the lip
    #     sits behind the subnasale-to-chin chord, which happened in 10 of 25
    #     halves here and halved the reported nasal_length -- hence the window.
    # The window is a straight-line 35% of nasal length below the subnasale --
    # a distance, not an arc-index count, because arc length per index varies
    # several-fold along one contour and an index window that fits the upper lip
    # on one face reaches the chin on the next. Anthropometric sn-ls is 24-30% of
    # n-sn, so 35% clears the lip and stops well short of anything else.
    lim = LIP_WINDOW * float(np.linalg.norm(contour[i_sub] - contour[i_nasion]))
    far = np.flatnonzero(np.linalg.norm(contour[i_sub + 2:] - contour[i_sub], axis=1) > lim)
    hi = int(i_sub + 2 + far[0]) if far.size else contour.shape[0] - 1
    if hi - i_sub < 5:
        return ProfileResult(False, "no labial point: crop ends before the upper lip",
                             facing=sign, contour=contour)
    dev_lip = chord_deviation(contour, max(3, (hi - i_sub) // 2), ant)
    i_lab = i_sub + 2 + int(np.argmax(dev_lip[i_sub + 2:hi]))

    # The dorsum must be longer than the columella. If it is not, the "tip" is
    # something else -- a chin on a very small nose, or a segmentation artefact.
    ratio = (i_tip - i_nasion) / max(1, i_sub - i_tip)
    if not 0.7 <= ratio <= 6.0:
        return ProfileResult(False, f"dorsum/columella arc ratio {ratio:.2f} is not a nose",
                             facing=sign, contour=contour)

    # Brow, for QA only: the anterior-most point within one dorsum length above
    # the nasion, which is the supraorbital ridge. Anatomical, not the hull vertex
    # b0 -- b0 lands wherever the crop ends and is useless as a control point.
    # Nothing measured below depends on it; profile_check.py uses it to test
    # whether the nasion drifts against fixed anatomy between before and after.
    up_win = i_nasion - (i_tip - i_nasion)
    i_brow = None
    if up_win > 2 and i_nasion - up_win > 4:
        cand = up_win + int(np.argmax(a[up_win:i_nasion]))
        # A brow that is no more anterior than the nasion is not a brow, it is
        # the search window's own edge. Reporting it would make the invariance
        # check below trivially pass, so it is withheld.
        if a[cand] - a[i_nasion] >= 0.05 * (a[i_tip] - a[i_nasion]):
            i_brow = cand

    P = {"nasion": contour[i_nasion], "pronasale": contour[i_tip],
         "subnasale": contour[i_sub], "labrale": contour[i_lab]}
    if i_brow is not None:
        P["brow"] = contour[i_brow]

    scale = float(np.linalg.norm(P["labrale"] - P["nasion"]))
    if scale < 4 * MIN_NOSE_PX:
        return ProfileResult(False, f"scale reference only {scale:.0f}px",
                             facing=sign, contour=contour,
                             points={k: tuple(map(float, v)) for k, v in P.items()})

    # dorsal hump: signed peak deviation of the dorsum from the nasion->tip chord
    chord = P["pronasale"] - P["nasion"]
    nrm = np.array([chord[1], -chord[0]])
    nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
    if nrm @ ant < 0:
        nrm = -nrm
    dev = (contour[i_nasion:i_tip + 1] - P["nasion"]) @ nrm
    j = int(np.abs(dev).argmax())
    hump = float(dev[j]) / scale

    # tip projection: perpendicular stand-off of the tip from the nasion->
    # subnasale line, i.e. from the plane the nose is mounted on. Head-tilt free.
    base = P["subnasale"] - P["nasion"]
    bn = np.array([base[1], -base[0]])
    bn = bn / (np.linalg.norm(bn) + 1e-9)
    proj = float(abs((P["pronasale"] - P["nasion"]) @ bn)) / scale

    length = float(np.linalg.norm(P["subnasale"] - P["nasion"])) / scale

    # nasolabial angle at subnasale, between the columella and the upper lip.
    # Both arms are chords from subnasale, so the angle is invariant to head
    # tilt and to image scale without any facial-axis assumption at all.
    v1 = P["pronasale"] - P["subnasale"]
    v2 = P["labrale"] - P["subnasale"]
    cosang = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    rot = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    params = NoseParams(nasal_length=length, dorsal_hump=hump,
                        tip_projection=proj, tip_rotation_deg=rot)
    return ProfileResult(
        True, note, params=params, scale_px=scale, facing=sign, contour=contour,
        points={k: (float(v[0]), float(v[1])) for k, v in P.items()},
        extra={"hump_apex_frac": j / max(1, i_tip - i_nasion),
               "nose_px": nose_px,
               "dorsum_columella_ratio": ratio,
               "brow_labrale_over_scale":
                   (float(np.linalg.norm(P["brow"] - P["labrale"])) / scale)
                   if "brow" in P else float("nan")})


# ==========================================================================
# QA
# ==========================================================================
_COLOURS = {"nasion": (0, 255, 255), "pronasale": (0, 0, 255),
            "subnasale": (255, 0, 255), "labrale": (0, 255, 0),
            "brow": (255, 200, 0)}


def overlay(panel: np.ndarray, res: ProfileResult, title: str = "") -> np.ndarray:
    """Debug render: the traced contour, every located point, and the numbers."""
    vis = panel.copy()
    if res.contour is not None:
        cv2.polylines(vis, [res.contour.astype(np.int32)], False, (255, 255, 0), 2)
    for name, (x, y) in res.points.items():
        c = _COLOURS.get(name, (255, 255, 255))
        cv2.circle(vis, (int(x), int(y)), 9, c, -1)
        cv2.circle(vis, (int(x), int(y)), 9, (0, 0, 0), 2)
        cv2.putText(vis, name, (int(x) + 14, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    if res.ok and "nasion" in res.points and "pronasale" in res.points:
        n = tuple(int(v) for v in res.points["nasion"])
        t = tuple(int(v) for v in res.points["pronasale"])
        s = tuple(int(v) for v in res.points["subnasale"])
        cv2.line(vis, n, t, (200, 200, 200), 1)
        cv2.line(vis, n, s, (200, 200, 200), 1)

    lines = [title]
    if res.ok and res.params is not None:
        p = res.params
        lines += [f"hump  {p.dorsal_hump:+.4f}", f"proj  {p.tip_projection:.4f}",
                  f"len   {p.nasal_length:.4f}", f"nla   {p.tip_rotation_deg:.1f} deg",
                  f"scale {res.scale_px:.0f}px"]
    else:
        lines += ["REJECTED"] + [res.reason[i:i + 40] for i in range(0, len(res.reason), 40)]
    y = 30
    for ln in lines:
        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        y += 30
    return vis
