"""
landmarks.py
------------
Wraps MediaPipe's Face Mesh solution and exposes just the landmarks
we care about for gaze estimation: the iris center points and the
eye-corner points.

MediaPipe Face Mesh (with refine_landmarks=True) gives us 478 facial
landmarks per face, including 10 dedicated iris points that regular
face-mesh models don't provide. That refinement is what makes gaze
estimation possible without training a custom model ourselves.

Landmark index reference (MediaPipe Face Mesh, refine_landmarks=True):
  Right eye iris: 469, 470, 471, 472   (person's right eye)
  Left eye iris:  474, 475, 476, 477   (person's left eye)
  Right eye corners: 33 (outer), 133 (inner)
  Left eye corners:  362 (inner), 263 (outer)
"""

from __future__ import annotations  # lets us use `EyeLandmarks | None` on Python 3.9+
from dataclasses import dataclass
import mediapipe as mp
import numpy as np


RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_EYE_CORNERS = (33, 133)   # outer, inner
LEFT_EYE_CORNERS = (362, 263)   # inner, outer


@dataclass
class EyeLandmarks:
    right_iris_center: np.ndarray
    left_iris_center: np.ndarray
    right_corners: tuple
    left_corners: tuple


class FaceMeshDetector:
    def __init__(self, max_faces: int = 1, min_detection_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,  # required to get iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr) -> EyeLandmarks | None:
        """
        Runs face mesh detection on a single frame.
        Returns EyeLandmarks in pixel coordinates, or None if no face found.
        """
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB for MediaPipe
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        def to_px(idx):
            lm = face.landmark[idx]
            return np.array([lm.x * w, lm.y * h])

        right_iris_pts = np.array([to_px(i) for i in RIGHT_IRIS])
        left_iris_pts = np.array([to_px(i) for i in LEFT_IRIS])

        return EyeLandmarks(
            right_iris_center=right_iris_pts.mean(axis=0),
            left_iris_center=left_iris_pts.mean(axis=0),
            right_corners=(to_px(RIGHT_EYE_CORNERS[0]), to_px(RIGHT_EYE_CORNERS[1])),
            left_corners=(to_px(LEFT_EYE_CORNERS[0]), to_px(LEFT_EYE_CORNERS[1])),
        )

    def close(self):
        self.face_mesh.close()
