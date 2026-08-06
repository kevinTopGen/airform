"""MediaPipe Tasks API FaceLandmarker -- the modern interface to the same mesh.

mp_mesh.py drives the legacy `mp.solutions.face_mesh` graph. This adapter drives
`mediapipe.tasks.python.vision.FaceLandmarker` against the bundled
`face_landmarker.task`, and measures the result with *exactly* the same code
path (nose_indices -> nose_frame -> band_width / scale_ref) so any difference in
the benchmark numbers is attributable to the landmarks, not to the measurement.

The task bundle also emits a 4x4 facial transformation matrix, which is a real
head-pose estimate rather than the landmark-asymmetry heuristic in
landmarks.estimate_yaw. `pose(path)` exposes it as degrees; the width
measurement does not use it.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..landmarks import nose_frame, scale_ref
from ..nose_region import band_width, nose_indices
from .base import BANDS

NAME = "mp_tasks"

_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "face_landmarker.task",
)

_LANDMARKER = None


def available() -> bool:
    if not os.path.exists(_MODEL):
        return False
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
    except Exception:
        return False
    return True


def _landmarker():
    """Lazily build the FaceLandmarker; it is expensive and stateless in IMAGE mode."""
    global _LANDMARKER
    if _LANDMARKER is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        _LANDMARKER = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_MODEL),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
        )
    return _LANDMARKER


def _detect(img):
    """-> (pts Nx2 in pixels, 4x4 facial transformation matrix or None)."""
    import mediapipe as mp

    h, w = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = _landmarker().detect(mp_img)
    if not res.face_landmarks:
        return None, None

    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]],
                   dtype=np.float64)
    mat = None
    mats = getattr(res, "facial_transformation_matrixes", None)
    if mats:
        mat = np.array(mats[0], dtype=np.float64)
    return pts, mat


def measure(path):
    img = cv2.imread(path)
    if img is None:
        return None
    pts, _ = _detect(img)
    if pts is None:
        return None

    # Identical measurement to mp_mesh.py, deliberately.
    idx = nose_indices(pts)
    R, u, v, L = nose_frame(pts)
    ipd = scale_ref(pts)

    out = {}
    for name, (lo, hi) in BANDS.items():
        w = band_width(pts, idx, R, u, v, L, lo, hi)
        if w is not None:
            out[name] = w / ipd
    return out


# --------------------------------------------------------------------------
# Head pose, from the facial transformation matrix. Not used by measure();
# exposed for the parallax filter, which currently uses estimate_yaw().
# --------------------------------------------------------------------------

def pose(path):
    """-> {'yaw','pitch','roll'} in degrees, or None.

    The task bundle solves a rigid fit of the canonical face metric mesh to the
    detected landmarks and hands back a 4x4 model->camera matrix. The upper-left
    3x3 is a rotation; decomposed here as intrinsic Y-X-Z (yaw, pitch, roll) in
    the OpenGL-style frame MediaPipe uses (+X right, +Y up, -Z into the scene).
    Sign convention below: yaw > 0 = subject's face turned to image-right,
    pitch > 0 = chin up, roll > 0 = head tilted clockwise in the image.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    _, mat = _detect(img)
    if mat is None:
        return None
    return pose_from_matrix(mat)


def pose_from_matrix(mat):
    Rm = np.asarray(mat, dtype=np.float64)[:3, :3]
    # Y-X-Z Euler decomposition of a rotation matrix.
    sy = -Rm[1, 2]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.9999:
        yaw = np.arctan2(Rm[0, 2], Rm[2, 2])
        roll = np.arctan2(Rm[1, 0], Rm[1, 1])
    else:  # gimbal lock
        yaw = np.arctan2(-Rm[2, 0], Rm[0, 0])
        roll = 0.0
    return {"yaw": float(np.degrees(yaw)),
            "pitch": float(np.degrees(pitch)),
            "roll": float(np.degrees(roll))}


def translation(path):
    """-> (tx, ty, tz) of the fitted face in canonical-mesh units, or None.

    tz is a real metric depth proxy (the canonical mesh has fixed size), so it
    is the piece a parallax filter needs beyond rotation.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    _, mat = _detect(img)
    if mat is None:
        return None
    return tuple(float(x) for x in np.asarray(mat)[:3, 3])
