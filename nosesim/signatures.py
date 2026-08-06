"""Stage 3: before/after pairs -> SurgeonSignature.

The entire "training" step, in one mean and one standard deviation.

`fit` is the real thing and is ready for scraped data today. `ARCHETYPES` is
placeholder data so every downstream agent can build against a populated
registry before a single Instagram photo has been collected — swapping in real
signatures later changes no code, only this dict.

Surgeons are deliberately unnamed here. Attaching a real practice's name to a
predicted surgical outcome is a defamation surface, and the archetypes read
better in a demo anyway.
"""

from __future__ import annotations

import statistics
from typing import List, Tuple

from .contracts import FIELD_NAMES, Landmarks, NoseParams, SurgeonSignature
from .params import measure, pose_gap_deg

MAX_POSE_GAP_DEG = 10.0


def fit(surgeon_id, name, tagline, pairs: List[Tuple[Landmarks, Landmarks]],
        max_pose_gap=MAX_POSE_GAP_DEG) -> SurgeonSignature:
    """Mean and spread of the per-pair operation over a surgeon's public results.

    Per-pair the operation is `delta_to`, which is mode-aware: widths come out as
    after/before - 1 (a fraction), everything else as after - before. So the mean
    width delta answers "by what percentage does this surgeon narrow a base",
    which is the question that transfers to a patient whose nose is not the size
    of this surgeon's average patient's.
    """
    deltas, rejected = [], 0
    for before, after in pairs:
        if pose_gap_deg(before, after) > max_pose_gap:
            rejected += 1  # parallax, not surgery
            continue
        deltas.append(measure(before).delta_to(measure(after)))

    if not deltas:
        raise ValueError(f"{surgeon_id}: no usable pairs ({rejected} rejected on pose)")

    mean, std, modes = {}, {}, {}
    for name_ in FIELD_NAMES:
        vals = [getattr(d, name_) for d in deltas if getattr(d, name_) is not None]
        if vals:
            mean[name_] = sum(vals) / len(vals)
            std[name_] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            # every delta measured the same way; carry that forward
            modes[name_] = next(d.mode(name_) for d in deltas
                                if getattr(d, name_) is not None)

    return SurgeonSignature(id=surgeon_id, name=name, tagline=tagline,
                            n_pairs=len(deltas),
                            delta=NoseParams(modes=modes, **mean),
                            std=NoseParams(modes=modes, **std))


def holdout_error(sig: SurgeonSignature, before: Landmarks, after: Landmarks):
    """Predict the held-out 'after' from the 'before' and report the residual.

    Cheap, and it is the difference between claiming the signatures mean
    something and showing it.
    """
    pred = measure(before).apply(sig.delta)
    truth = measure(after)
    return {name: abs(getattr(pred, name) - getattr(truth, name))
            for name in FIELD_NAMES
            if getattr(pred, name) is not None and getattr(truth, name) is not None}


# --- Placeholder registry -------------------------------------------------
# Width deltas are PROPORTIONAL: fractions of the patient's own baseline, so
# -0.107 narrows any alar base by 10.7% whether that base started at 30mm or
# 45mm. nasal_length stays ABSOLUTE, in fractions of interpupillary distance
# (a typical nasal length is ~0.75 IPD, so -0.045 is a ~2mm shortening).
#
# These are the same five operations as before, re-expressed: each width value
# is the percentage change the old absolute value happened to produce on the
# reference face, so the renders are unchanged on that face and now transfer
# correctly to any other.

ARCHETYPES = [
    SurgeonSignature(
        id="conservative", name="The Conservative", n_pairs=0,
        tagline="Preserves ethnic character. You, slightly edited.",
        delta=NoseParams(bridge_width=-0.041, tip_width=-0.049,
                         alar_width=-0.023, nasal_length=-0.005)),
    SurgeonSignature(
        id="alar", name="The Base Reducer", n_pairs=0,
        tagline="Weir excisions. Narrows the base, leaves the bridge alone.",
        delta=NoseParams(bridge_width=-0.014, tip_width=-0.055,
                         alar_width=-0.095, nasal_length=0.0)),
    SurgeonSignature(
        id="dorsal", name="The Dorsal Preservationist", n_pairs=0,
        tagline="Narrows the bridge, keeps the tip you were born with.",
        delta=NoseParams(bridge_width=-0.157, tip_width=-0.033,
                         alar_width=-0.015, nasal_length=-0.008)),
    SurgeonSignature(
        id="tip", name="The Tip Sculptor", n_pairs=0,
        tagline="Everything happens below the supratip. Shorter, rotated.",
        delta=NoseParams(bridge_width=-0.027, tip_width=-0.151,
                         alar_width=-0.034, nasal_length=-0.045)),
    SurgeonSignature(
        id="signature", name="The Miami Signature", n_pairs=0,
        tagline="You will not be mistaken for un-operated.",
        delta=NoseParams(bridge_width=-0.163, tip_width=-0.192,
                         alar_width=-0.107, nasal_length=-0.038)),
]

BY_ID = {s.id: s for s in ARCHETYPES}
