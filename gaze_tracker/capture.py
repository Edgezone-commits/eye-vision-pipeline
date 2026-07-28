"""
capture.py
----------
Thin wrapper around OpenCV's VideoCapture.

Keeping this in its own module (instead of calling cv2.VideoCapture
directly inside main.py) means:
  - main.py doesn't need to know *how* frames are captured
  - we can later swap the webcam for a video file or IP camera
    without touching the rest of the pipeline
This separation of concerns is a small thing, but it's exactly the
kind of structure that makes a CV pipeline easy to extend later
(e.g. swapping in a recorded clinical video instead of a live webcam).
"""

import cv2


class VideoStream:
    def __init__(self, source: int = 0, width: int = 640, height: int = 480):
        """
        source: camera index (0 = default webcam)
        width/height: requested capture resolution
        """
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check that it's connected and "
                "not being used by another application."
            )

    def read_frame(self):
        """Returns a single BGR frame, or None if capture failed."""
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def release(self):
        self.cap.release()
