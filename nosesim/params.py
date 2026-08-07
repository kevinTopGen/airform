"""Stage 2: Landmarks -> NoseParams.

This is the step that replaces "train a model". Seven scalars, each one a
measurement a surgeon would actually recognise, normalised by interpupillary
distance so they compare across photos, cameras and people.

Averaging seven robust scalars over ~15 before/after pairs is stable. Averaging
60 raw landmark deltas over 15 pairs is mostly averaging landmark noise.

WHERE THE THREE FRONTAL WIDTHS NOW COME FROM
Not from the landmarks. `scripts/bench.py` scored every technique against warps
of known size and FaceMesh's lateral nose vertices report almost none of a real
change -- alar k=0.063, bridge r=-0.178, tip k=0.039 -- because those vertices
come from the model's shape prior and the sidewall gives the regressor nothing
to override it with. `photometric_v2` measures shading instead of landmarks and
scores alar k=0.878 / bridge k=0.466 / tip k=0.596, every one at r>0.99.

So the widths come from `photometric_v2` when it succeeds and from the landmark
bands only when it returns nothing, and `sources(p)` says which happened for
each number -- a fallback value is a *different kind of thing* from a measured
one and a consumer fitting a signature has to be able to tell them apart.

The nose frame, the IPD normalisation, `nasal_length` and every profile
parameter are unchanged: vertical landmarks sit on real edges (tip, subnasale,
radix) and behave. Only the widths were ever the problem.

THE GAIN CORRECTION, AND WHY IT IS NOT A MULTIPLY
bench fits `observed_fraction = k * requested_fraction`. A gauge with k=0.466
reports a 4.7% narrowing when the nose really narrowed 10%. Rendering, by
contrast, is verified to deliver 100% of what it is asked. Feed an uncorrected
signature back in and it renders k*T -- a fraction of the real surgery -- with
nothing in the pipeline looking broken. Hence a gain of 1/k.

The obvious implementation is wrong. Multiplying the measurement by 1/k does
nothing at all, because every width delta is *proportional* (contracts.py) and
a constant factor cancels out of `after/before`:

    (c*m_after) / (c*m_before) - 1  ==  m_after/m_before - 1

The correction has to be non-linear in the measurement. bench's fit says the
gauge compresses *fractional* change by k, i.e. dlog(m)/dlog(x) = k, so the
true width goes as x ~ m^(1/k) and the corrected reading is

    c = C * (m / C) ** (1/k)

`C` is a per-gauge anchor. It cancels out of every delta -- c_a/c_b depends only
on m_a/m_b -- so it cannot affect a signature, a fit or a render. All it does is
choose units, and it is set to the gauge's reading on the calibration subject so
that a face near that operating point reads the same number before and after
this change. To first order this is identical to inverting bench's own linear
fit; it differs only in being scale-invariant, so it does not quietly assume
every patient is the size of the one the benchmark was built from.

Two gates, both taken from bench.py's own USABLE rule, decide whether a gain is
applied at all:

    r >= 0.95   the response must track reality. This is non-negotiable: 1/k on
                a gauge that is reporting noise amplifies the noise and nothing
                else. scripts/calibrate.py refuses to emit a gain in exactly
                this case and this module refuses for the same reason.
    k >= 0.25   a gain above 4x multiplies noise faster than signal. This is
                what keeps a 16x correction off the landmark fallback, whose
                alar r=0.965 would otherwise clear the correlation gate.

Fail either and the number is passed through raw, with the reason recorded in
`sources(p)`. k and r are read from bench/results.json, never hardcoded, so
re-running the benchmark re-derives the gains.
"""

from __future__ import annotations

import functools
import hashlib
from collections import OrderedDict
from typing import Dict, NamedTuple, Optional

import numpy as np

from . import landmarks as _lm_module
from . import measure as _adapters
from .contracts import Landmarks, NoseParams
from .landmarks import NOSE_TIP, RADIX, nose_frame, scale_ref
from .nose_region import band_width, nose_indices

# Slices of the nose, as a fraction of radix->subnasale distance.
#
# The subnasale sits at the base of the columella, *below* where the alae flare
# widest, so the widest point of the nose lands near t=0.75 rather than t=1.
# Alar width is therefore the max lateral extent anywhere over the lower nose,
# not the span at t~1 -- that would measure the nostril sills instead.
BRIDGE_BAND = (0.25, 0.55)
ALAR_BAND = (0.55, 1.30)
TIP_BAND = (0.60, 0.88)
TIP_LOBULE_FRAC = 0.60  # of alar half-width; excludes the wings from the tip

WIDTH_FIELDS = NoseParams.WIDTH_FIELDS

# The adapter that measures the frontal widths, and the name recorded when it
# could not and the landmark bands above had to stand in.
FRONT_ADAPTER = "photometric_v2"
FALLBACK_SOURCE = "landmark_band"
# Nearest calibrated proxy for the fallback: mp_mesh reads the same landmark set
# through base.BANDS rather than the slightly wider bands here. It fails the
# gain gate by a factor of four either way, so the difference never becomes
# load-bearing -- but the refusal is data-driven rather than asserted.
FALLBACK_BENCH_PROXY = "mp_mesh"

# bench.py's USABLE thresholds. Kept in sync deliberately: a parameter this
# module is willing to gain-correct is exactly a parameter bench calls usable.
GAIN_R_MIN, GAIN_K_MIN = 0.95, 0.25

# Gauge anchors, in IPD units: each adapter's reading on the calibration subject
# (bench/<tone>/baseline.png), which is where its k was fitted. Sets units only
# -- it cancels out of every delta. Regenerate after re-running the benchmark on
# a different subject with:
#
#   .venv/bin/python -c "from nosesim.measure import get; \
#       print(get('photometric_v2').measure('bench/normal/baseline.png'))"
GAUGE_ANCHOR: Dict[str, Dict[str, float]] = {
    "photometric_v2": {"alar_width": 0.611822,
                       "bridge_width": 0.197370,
                       "tip_width": 0.233300},
}


class Gauge(NamedTuple):
    """How much of a real change one adapter reports for one parameter."""

    k: Optional[float]        # slope of observed vs requested fractional change
    r: Optional[float]        # correlation, worst case over the scored tones
    gain: Optional[float]     # 1/k, or None when the gates refuse it
    anchor: Optional[float]   # units anchor; irrelevant to every delta
    reason: str


def measure(lm: Landmarks, image=None) -> NoseParams:
    """A nose, in seven numbers, with provenance attached.

    `image` is the BGR array `lm` was detected from. It is optional only for
    compatibility: without pixels the frontal widths cannot be measured
    photometrically and fall back to the landmark bands. Callers that already
    went through `landmarks.detect` need not pass it -- see `remember_image`.

    The returned NoseParams is the plain contract type. Provenance rides along
    as an attribute, not a field, so `to_dict`, `delta_to`, `apply` and every
    other contract operation are untouched; read it with `sources(p)`.
    """
    pts = lm.as_array()
    idx = nose_indices(pts)
    R, u, v, L = nose_frame(pts)
    ipd = scale_ref(pts)

    p = NoseParams(nasal_length=L / ipd)
    src = {"nasal_length": {"source": "landmark_frame", "raw": L / ipd,
                            "k": None, "r": None, "gain": None,
                            "calibrated": False, "view": lm.view,
                            "reason": "radix->subnasale, uncorrected: bench "
                                      "scores only the three widths, so there is "
                                      "no measured k to invert"}}

    if lm.view in ("frontal", "three_quarter"):
        photo = _adapter_widths(lm, image)
        bands = _band_widths(pts, idx, R, u, v, L)
        gauges = _gauges(FRONT_ADAPTER)
        fallback_gauges = _gauges(FALLBACK_BENCH_PROXY)

        for name in WIDTH_FIELDS:
            raw = photo.get(name)
            if raw is not None:
                g = gauges[name]
                value = _corrected(raw, g)
                source = FRONT_ADAPTER
            else:
                b = bands.get(name)
                if b is None:
                    continue
                g = fallback_gauges[name]
                raw = b / ipd
                value = _corrected(raw, g)
                source = FALLBACK_SOURCE
            setattr(p, name, value)
            # `view` is recorded because the benchmark corpus is frontal only:
            # a gain measured head-on has never been validated at three_quarter,
            # and a consumer weighting a scraped pair deserves to know that.
            src[name] = {"source": source, "raw": raw, "k": g.k, "r": g.r,
                         "gain": g.gain, "calibrated": g.gain is not None,
                         "view": lm.view, "reason": g.reason}

    if lm.view == "profile":
        p.dorsal_hump = _dorsal_hump(pts, idx, ipd)
        p.tip_projection = _tip_projection(pts, ipd)
        p.tip_rotation_deg = _tip_rotation(pts)
        for name in ("dorsal_hump", "tip_projection", "tip_rotation_deg"):
            src[name] = {"source": "landmark_frame", "raw": getattr(p, name),
                         "k": None, "r": None, "gain": None, "calibrated": False,
                         "view": lm.view,
                         "reason": "profile parameters are unbenchmarked"}

    object.__setattr__(p, "sources", src)
    return p


def sources(p: NoseParams) -> Dict[str, dict]:
    """Per-field provenance for a measurement produced by `measure`.

    {field: {source, raw, k, r, gain, calibrated, view, reason}}. Empty for anything
    that is not a measurement -- a delta or a target has no provenance, because
    it was computed rather than observed.
    """
    return dict(getattr(p, "sources", {}) or {})


def trusted(p: NoseParams, field: str) -> bool:
    """Did `field` come from the benchmarked adapter with a calibrated gain?

    The one question a signature fitter has to ask before averaging a delta.
    False means the number is a fallback, an uncalibrated gauge, or absent --
    all of which are reasons to drop the field rather than fit on it.
    """
    s = sources(p).get(field)
    return bool(s and s["source"] == FRONT_ADAPTER and s["calibrated"])


# --- gauge calibration ----------------------------------------------------

@functools.lru_cache(maxsize=None)
def _gauges(adapter: str) -> Dict[str, Gauge]:
    """Resolve {parameter: Gauge} for one adapter from bench/results.json.

    Scored per (parameter, tone). The gate uses the *worst* tone -- a gauge that
    tracks under studio light and wanders under flat light is not calibrated,
    it is calibrated under one condition -- while k is taken from the untouched
    'normal' tone, which is what an actual photograph is.
    """
    scored = _adapters.bench_scores().get(adapter, {})
    by_param: Dict[str, Dict[str, dict]] = {}
    for key, v in scored.items():
        name, _, tone = key.partition("|")
        by_param.setdefault(name, {})[tone or "normal"] = v

    anchors = GAUGE_ANCHOR.get(adapter, {})
    out = {}
    for name in WIDTH_FIELDS:
        tones = by_param.get(name)
        anchor = anchors.get(name)
        if not tones:
            out[name] = Gauge(None, None, None, anchor,
                              f"no bench entry for {adapter}/{name}; uncorrected")
            continue

        r = min(float(t["r"]) for t in tones.values())
        k = float((tones.get("normal") or next(iter(tones.values())))["k"])

        if r < GAIN_R_MIN:
            reason = (f"r={r:.3f} < {GAIN_R_MIN}: response does not track the "
                      f"request, so 1/k would amplify noise. Uncorrected.")
        elif k < GAIN_K_MIN:
            reason = (f"k={k:.3f} < {GAIN_K_MIN}: a {1/k:.0f}x gain multiplies "
                      f"noise faster than signal. Uncorrected.")
        elif anchor is None or anchor <= 0:
            reason = (f"k={k:.3f}, r={r:.3f} usable but no gauge anchor for "
                      f"{adapter}/{name}; uncorrected.")
        else:
            out[name] = Gauge(k, r, 1.0 / k, anchor,
                              f"k={k:.3f}, r={r:.3f}: gain {1/k:.3f} applied")
            continue
        out[name] = Gauge(k, r, None, anchor, reason)
    return out


def _corrected(value: Optional[float], g: Gauge) -> Optional[float]:
    """Undo the gauge's compression of fractional change.

    c = C*(m/C)^(1/k). Identity when no gain survived the gates, which is the
    only safe default: an uncorrected number is merely attenuated, an
    incorrectly corrected one is wrong in an unknown direction.
    """
    if value is None or g.gain is None or not g.anchor:
        return value
    if not (np.isfinite(value) and value > 0):
        return value
    return float(g.anchor * (value / g.anchor) ** g.gain)


# --- the two width measurements -------------------------------------------

def _band_widths(pts, idx, R, u, v, L) -> Dict[str, Optional[float]]:
    """The incumbent landmark-band widths, in pixels. Fallback only."""
    alar = band_width(pts, idx, R, u, v, L, *ALAR_BAND)
    bridge = band_width(pts, idx, R, u, v, L, *BRIDGE_BAND)
    tip = None
    if alar:
        tip = band_width(pts, idx, R, u, v, L, *TIP_BAND,
                         b_limit=TIP_LOBULE_FRAC * alar / 2)
    return {"alar_width": alar, "bridge_width": bridge, "tip_width": tip}


def _adapter_widths(lm: Landmarks, image=None) -> Dict[str, float]:
    """photometric_v2's widths, already IPD-normalised. {} if unavailable.

    Memoised on the landmark fingerprint, and the memo is consulted *before* the
    pixels are. That ordering is load-bearing, not an optimisation.

    `deform.plan` re-measures the face it was just handed and divides the caller's
    target by it. If the first call resolved photometrically and the second fell
    back to the landmark bands, the ratio would be a ratio between two different
    gauges -- a -8% bridge request rendering as -38% -- and nothing downstream
    would look wrong. So once a face has been measured photometrically it stays
    measured: the memo is small (a dict of three floats) and outlives the frame
    cache by three orders of magnitude, which is what makes a mid-pipeline gauge
    switch impossible rather than merely unlikely.

    An empty result is cached only when the adapter actually ran and produced
    nothing. "No pixels available" is not an answer and must not be memoised, or
    a later caller that does supply the image would be served the fallback.
    """
    key = _fingerprint(lm)
    if key in _WIDTHS:
        _WIDTHS.move_to_end(key)
        return _WIDTHS[key]

    img = image if image is not None else _IMAGES.get(key)
    if img is None:
        return {}

    try:
        got = _adapters.measure_image(FRONT_ADAPTER, img)
    except Exception:
        got = None

    out = {k: float(v) for k, v in (got or {}).items()
           if k in WIDTH_FIELDS and v and np.isfinite(v) and v > 0}
    _remember(_WIDTHS, key, out, _WIDTH_MEMO)
    return out


# --- image registry -------------------------------------------------------
#
# A photometric measurement needs pixels; `measure(lm)` is handed only
# landmarks, and `deform.plan` and `signatures.fit` both call it that way. So
# `landmarks.detect` is wrapped once, at import, to remember which image each
# Landmarks came from. It is a shim, not a design: the clean fix is for
# Landmarks to carry its image, which is a change to contracts.py.
#
# The wrapper adds a side effect and changes no return value, holds at most a
# handful of recent frames, and everything degrades to the landmark bands if it
# is ever bypassed.

_IMAGE_CACHE = 8       # frames, so bounded by megabytes
_WIDTH_MEMO = 4096     # three floats each, so bounded by nothing that matters

_IMAGES: "OrderedDict[str, object]" = OrderedDict()
_WIDTHS: "OrderedDict[str, dict]" = OrderedDict()


def _fingerprint(lm: Landmarks) -> str:
    a = np.ascontiguousarray(lm.as_array(), dtype=np.float64)
    return hashlib.sha1(a.tobytes()).hexdigest()


def _remember(store: OrderedDict, key, value, limit: int):
    store[key] = value
    store.move_to_end(key)
    while len(store) > limit:
        store.popitem(last=False)


def remember_image(lm: Landmarks, image) -> None:
    """Associate a detected Landmarks with the pixels it was detected from."""
    _remember(_IMAGES, _fingerprint(lm), image, _IMAGE_CACHE)


def _install_image_registry():
    detect = getattr(_lm_module, "detect")
    if getattr(detect, "_airform_registers_image", False):
        return

    @functools.wraps(detect)
    def detect_and_remember(image_bgr):
        lm = detect(image_bgr)
        try:
            remember_image(lm, image_bgr)
        except Exception:
            pass  # measurement must never fail because bookkeeping did
        return lm

    detect_and_remember._airform_registers_image = True
    _lm_module.detect = detect_and_remember


_install_image_registry()


# --- profile parameters (unchanged) ---------------------------------------

def _dorsal_hump(pts, idx, ipd):
    """Peak deviation of the dorsum from the straight radix->tip line.

    Positive = convex (a hump), negative = scooped. Meaningless head-on, which
    is why measure() gates it on view.
    """
    R, T = pts[RADIX], pts[NOSE_TIP]
    axis = T - R
    n = np.array([-axis[1], axis[0]]) / (np.linalg.norm(axis) + 1e-9)
    dev = (pts[idx] - R) @ n
    return float(np.abs(dev).max() / ipd) * float(np.sign(dev[np.abs(dev).argmax()]))


def _tip_projection(pts, ipd):
    from .landmarks import SUBNASALE

    return float(np.linalg.norm(pts[NOSE_TIP] - pts[SUBNASALE]) / ipd)


def _tip_rotation(pts):
    """Nasolabial angle proxy: columella direction vs the facial vertical."""
    from .landmarks import SUBNASALE

    c = pts[NOSE_TIP] - pts[SUBNASALE]
    return float(np.degrees(np.arctan2(-c[1], abs(c[0]) + 1e-9)))


def pose_gap_deg(a: Landmarks, b: Landmarks) -> float:
    """Head-pose mismatch between a before/after pair.

    The single highest-value data filter in the pipeline. If the 'after' photo
    was shot at a different angle, the measured delta is mostly parallax and it
    will poison the surgeon's mean. Reject pairs above ~10 degrees.
    """
    return abs(a.yaw_deg - b.yaw_deg)
