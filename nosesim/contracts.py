"""Frozen data contracts. Every agent codes against these and nothing else.

The whole architecture rests on one idea: a rhinoplasty is a small vector of
interpretable, scale-free numbers. Measure them on a face, subtract to get a
surgeon's signature, add to apply it. No model is trained anywhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Literal, Optional

View = Literal["frontal", "three_quarter", "profile"]

DeltaMode = Literal["absolute", "proportional"]

# How a delta in each field transfers from the patient it was fitted on to the
# patient it is applied to.
#
#   absolute      new = old + delta
#   proportional  new = old * (1 + delta)      delta is a fraction, -0.107 = -10.7%
#
# Widths are proportional. A surgeon who takes a 40mm alar base to 36mm performs
# "-10%", not "-4mm": apply -4mm to a 30mm base and you have done a different,
# much more aggressive operation. Storing widths absolutely means a signature
# fitted on one practice's patient demographic mistransfers to any nose whose
# baseline differs -- which is both a correctness bug and a fairness bug, since
# nasal width varies systematically with ancestry.
#
# Lengths, projections and angles stay absolute. Shortening a nose by 2mm or
# rotating a tip 5 degrees is the same manoeuvre whatever the starting nose, and
# "-10% of a 95 degree nasolabial angle" is not a thing a surgeon can do -- an
# angle has no meaningful zero to take a percentage of.
DELTA_MODES: Dict[str, DeltaMode] = {
    "alar_width": "proportional",
    "bridge_width": "proportional",
    "tip_width": "proportional",
    "nasal_length": "absolute",
    "dorsal_hump": "absolute",
    "tip_rotation_deg": "absolute",
    "tip_projection": "absolute",
}

# The seven numbers, in declaration order. Every numeric field has a mode.
FIELD_NAMES = tuple(DELTA_MODES)


@dataclass
class Landmarks:
    """Output of the landmark stage. Pixel coordinates in the source image."""

    points: List[List[float]]  # (N, 2) — MediaPipe FaceMesh, 478 pts w/ irises
    view: View
    yaw_deg: float  # + = subject's head turned to their left
    conf: float
    width: int
    height: int

    def as_array(self):
        import numpy as np

        return np.asarray(self.points, dtype=np.float64)


@dataclass
class NoseParams:
    """A nose, in seven numbers.

    Every value is normalised by interpupillary distance, so it is invariant to
    image resolution and camera distance. Widths/lengths are ratios; rotation is
    degrees. A *measurement* and a *delta* share this type — a delta is what you
    get by comparing two measurements.

    A delta is not always a subtraction. Each field has a MODE (see DELTA_MODES):
    width deltas are *proportional* (a fraction of the patient's own baseline),
    everything else is *absolute*. `delta_to` produces the right kind and `apply`
    consumes it; both honour `modes` for per-field overrides. On a measurement
    `modes` is meaningless and stays empty.

    `frontal` and `profile` mark which fields a given view can actually observe.
    Reading a profile-only field off a frontal photo yields None, not a guess.
    """

    alar_width: Optional[float] = None  # nostril wing span            [frontal]
    bridge_width: Optional[float] = None  # dorsum width, mid-height   [frontal]
    tip_width: Optional[float] = None  # tip lobule width              [frontal]
    nasal_length: Optional[float] = None  # radix -> subnasale         [both]
    dorsal_hump: Optional[float] = None  # bulge off radix-tip line    [profile]
    tip_rotation_deg: Optional[float] = None  # nasolabial angle       [profile]
    tip_projection: Optional[float] = None  # tip stand-off from face  [profile]

    # Per-field overrides of DELTA_MODES. Only meaningful on a delta.
    modes: Dict[str, DeltaMode] = field(default_factory=dict)

    FRONTAL_FIELDS = ("alar_width", "bridge_width", "tip_width", "nasal_length")
    PROFILE_FIELDS = ("nasal_length", "dorsal_hump", "tip_rotation_deg", "tip_projection")
    WIDTH_FIELDS = ("alar_width", "bridge_width", "tip_width")

    def observable(self, view: View) -> tuple:
        return self.PROFILE_FIELDS if view == "profile" else self.FRONTAL_FIELDS

    def mode(self, name: str) -> DeltaMode:
        """'absolute' or 'proportional' for one field of this delta."""
        return self.modes.get(name, DELTA_MODES[name])

    def with_modes(self, **modes: DeltaMode) -> "NoseParams":
        """Copy with some fields forced to a mode, e.g. .with_modes(alar_width='absolute')."""
        bad = set(modes) - set(FIELD_NAMES)
        if bad:
            raise KeyError(f"not NoseParams fields: {sorted(bad)}")
        return NoseParams(modes={**self.modes, **modes},
                          **{n: getattr(self, n) for n in FIELD_NAMES})

    def delta_to(self, other: "NoseParams") -> "NoseParams":
        """The operation that turns self into other, field-wise.

        Proportional fields give other/self - 1; absolute fields give other-self.
        None wherever either side is unobserved (or self is 0 and no fraction of
        it exists).
        """
        out, modes = {}, {}
        for name in FIELD_NAMES:
            a, b = getattr(self, name), getattr(other, name)
            if a is None or b is None:
                out[name] = None
                continue
            m = self.mode(name)
            if m == "proportional":
                if abs(a) < 1e-9:
                    out[name] = None  # no fraction of zero is meaningful
                    continue
                out[name] = b / a - 1.0
            else:
                out[name] = b - a
            modes[name] = m
        return NoseParams(modes=modes, **out)

    def apply(self, delta: "NoseParams") -> "NoseParams":
        """self, operated on. Fields the delta doesn't specify pass through.

        Proportional fields scale (a * (1+d)), absolute fields add (a + d), per
        `delta.mode(field)`. The result is a plain measurement, so it carries no
        modes of its own.
        """
        out = {}
        for name in FIELD_NAMES:
            a, d = getattr(self, name), getattr(delta, name)
            if a is None or d is None:
                out[name] = a
            elif delta.mode(name) == "proportional":
                out[name] = a * (1.0 + d)
            else:
                out[name] = a + d
        return NoseParams(**out)

    def scaled(self, k: float) -> "NoseParams":
        """This delta at k of full strength. Modes preserved.

        Linear in the stored quantity, so a proportional -10% at k=0.5 is -5%.
        That is what a before/after slider should do.
        """
        return NoseParams(modes=dict(self.modes),
                          **{n: (None if getattr(self, n) is None else getattr(self, n) * k)
                             for n in FIELD_NAMES})

    def as_absolute(self, baseline: "NoseParams") -> "NoseParams":
        """This delta rewritten in absolute units against one specific baseline.

        `baseline.apply(d)` == `baseline.apply(d.as_absolute(baseline))`. For
        callers that want to reason in IPD units on a known face; the
        proportional form is the one that transfers between faces.
        """
        out = {}
        for name in FIELD_NAMES:
            d, b = getattr(self, name), getattr(baseline, name)
            if d is None:
                out[name] = None
            elif self.mode(name) == "proportional":
                out[name] = None if b is None else b * d
            else:
                out[name] = d
        return NoseParams(modes={n: "absolute" for n in FIELD_NAMES}, **out)

    def to_dict(self):
        """The numbers only — modes are metadata and stay out, so that
        `NoseParams(**p.to_dict())` and arithmetic over the values keep working."""
        return {n: getattr(self, n) for n in FIELD_NAMES if getattr(self, n) is not None}


assert FIELD_NAMES == tuple(f.name for f in fields(NoseParams) if f.name != "modes"), (
    "DELTA_MODES must list every numeric NoseParams field, in declaration order")


@dataclass
class SurgeonSignature:
    """What a surgeon does to a nose, averaged over their public before/afters.

    `delta` is mixed-mode: read `delta.mode(field)` before interpreting a number.
    Widths are fractions of the patient's own baseline (-0.107 = -10.7% narrower),
    everything else is in IPD units. Apply it with `patient.apply(sig.delta)`.
    """

    id: str
    name: str
    tagline: str
    n_pairs: int
    delta: NoseParams
    std: NoseParams = field(default_factory=NoseParams)

    def target_for(self, baseline: NoseParams, strength: float = 1.0) -> NoseParams:
        """The measurement this surgeon would produce from `baseline`."""
        return baseline.apply(self.delta.scaled(strength))

    def to_dict(self):
        d = asdict(self)
        d["delta"] = self.delta.to_dict()
        d["std"] = self.std.to_dict()
        d["delta_modes"] = {k: self.delta.mode(k) for k in d["delta"]}
        return d


def dumps(obj) -> str:
    return json.dumps(obj.to_dict() if hasattr(obj, "to_dict") else asdict(obj), indent=2)
