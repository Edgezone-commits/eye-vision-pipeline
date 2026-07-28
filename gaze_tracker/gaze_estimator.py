"""
gaze_estimator.py
------------------
Turns raw iris/eye-corner landmarks into a human-readable gaze
direction: LEFT, RIGHT, or CENTER.

Method: for each eye, we compute where the iris center sits between
the two eye corners as a ratio from 0.0 to 1.0:

    ratio = distance(outer_corner, iris_center) / distance(outer_corner, inner_corner)

  ratio close to 0.0  -> iris near outer corner -> looking toward that side
  ratio close to 1.0  -> iris near inner corner -> looking the other way
  ratio near 0.5      -> looking roughly center

This is a simple geometric heuristic, not a trained model. It's a
reasonable first pass and is genuinely how many introductory gaze
systems start, but it has known limitations documented in the README
(head-pose sensitivity, no per-user calibration). A production system
would calibrate per-user and/or use a regression model trained on
labeled gaze data.
"""

from dataclasses import dataclass
import numpy as np
from .landmarks import EyeLandmarks


@dataclass
class GazeResult:
    direction: str      # "LEFT", "RIGHT", or "CENTER"
    ratio: float         # averaged horizontal ratio, for debugging/logging


def _horizontal_ratio(iris_center: np.ndarray, outer, inner) -> float:
    eye_width = np.linalg.norm(inner - outer)
    if eye_width == 0:
        return 0.5
    dist_from_outer = np.linalg.norm(iris_center - outer)
    return float(np.clip(dist_from_outer / eye_width, 0.0, 1.0))


def estimate_gaze(eyes: EyeLandmarks, left_thresh: float = 0.42, right_thresh: float = 0.58) -> GazeResult:
    right_ratio = _horizontal_ratio(
        eyes.right_iris_center, eyes.right_corners[0], eyes.right_corners[1]
    )
    left_ratio = _horizontal_ratio(
        eyes.left_iris_center, eyes.left_corners[0], eyes.left_corners[1]
    )
    avg_ratio = (right_ratio + left_ratio) / 2.0

    if avg_ratio < left_thresh:
        direction = "RIGHT"   # from the viewer's perspective on a mirrored webcam feed
    elif avg_ratio > right_thresh:
        direction = "LEFT"
    else:
        direction = "CENTER"

    return GazeResult(direction=direction, ratio=avg_ratio)
