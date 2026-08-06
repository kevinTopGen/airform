"""Stage 1: image -> Landmarks.

MediaPipe FaceMesh, not dlib. dlib's 68-point model gives the nose nine points
and collapses off-frontal; FaceMesh gives 478 with dense nasal coverage and
survives real head pose.
"""

from __future__ import annotations

import cv2
import numpy as np

from .contracts import Landmarks

_FM = None


def _mesh():
    global _FM
    if _FM is None:
        import mediapipe as mp

        _FM = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # -> 478 pts, adds iris centres
            min_detection_confidence=0.5,
        )
    return _FM


# Landmarks a rhinoplasty cannot move. Everything is normalised against these.
R_IRIS, L_IRIS = 468, 473
R_CANTHUS_IN, L_CANTHUS_IN = 133, 362
R_FACE_EDGE, L_FACE_EDGE = 234, 454
NOSE_TIP, RADIX, SUBNASALE = 1, 168, 2


def detect(image_bgr) -> Landmarks:
    h, w = image_bgr.shape[:2]
    res = _mesh().process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError("no face detected")

    lms = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64)

    yaw = estimate_yaw(pts)
    view = "frontal" if abs(yaw) < 20 else ("three_quarter" if abs(yaw) < 55 else "profile")
    return Landmarks(points=pts.tolist(), view=view, yaw_deg=float(yaw),
                     conf=1.0, width=w, height=h)


def estimate_yaw(pts) -> float:
    """Rough yaw from facial asymmetry about the nose tip.

    Cheap and good enough for gating: all we need to know is whether the
    profile-only parameters are trustworthy on this photo.
    """
    tip = pts[NOSE_TIP]
    left = np.linalg.norm(pts[L_FACE_EDGE] - tip)
    right = np.linalg.norm(pts[R_FACE_EDGE] - tip)
    ratio = (left - right) / (left + right)
    return float(np.clip(ratio * 190.0, -90, 90))


def scale_ref(pts) -> float:
    """Interpupillary distance — the normalising constant for every parameter.

    Iris centres when available (sub-pixel, from refine_landmarks), inner canthi
    as fallback.
    """
    if len(pts) > L_IRIS:
        ipd = np.linalg.norm(pts[L_IRIS] - pts[R_IRIS])
        if ipd > 1:
            return float(ipd)
    return float(np.linalg.norm(pts[L_CANTHUS_IN] - pts[R_CANTHUS_IN]) * 1.45)


def nose_frame(pts):
    """Local coordinate system for the nose.

    Origin at the radix, +u running down the dorsum to the subnasale, +v lateral.
    Working in this frame means the parameters stay meaningful under head tilt —
    'narrow the bridge' is perpendicular to the dorsum, not to the image.
    """
    R = pts[RADIX]
    S = pts[SUBNASALE]
    axis = S - R
    L = float(np.linalg.norm(axis))
    u = axis / L
    v = np.array([-u[1], u[0]])
    return R, u, v, L


def to_local(pts, R, u, v):
    """-> (a, b): a = distance down the dorsum, b = signed lateral offset."""
    rel = pts - R
    return rel @ u, rel @ v


def to_global(a, b, R, u, v):
    return R + np.outer(a, u) + np.outer(b, v)
