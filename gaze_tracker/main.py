"""
main.py
-------
Entry point for the real-time gaze tracker.

Pipeline: capture frame -> detect face/iris landmarks -> estimate
gaze direction -> overlay result on frame -> display.

Run with:  python -m gaze_tracker.main
Press 'q' to quit. Press 'l' to log the current gaze reading to
gaze_log.csv (a tiny stand-in for the "data collection" stage of a
CV pipeline).
"""

import csv
import time
import cv2

from .capture import VideoStream
from .landmarks import FaceMeshDetector
from .gaze_estimator import estimate_gaze


def main():
    stream = VideoStream(source=0)
    detector = FaceMeshDetector(max_faces=1)

    print("Gaze tracker running. Press 'q' to quit, 'l' to log a reading.")

    try:
        while True:
            frame = stream.read_frame()
            if frame is None:
                print("Failed to read frame from webcam.")
                break

            eyes = detector.process(frame)

            if eyes is not None:
                result = estimate_gaze(eyes)
                label = f"{result.direction}  (ratio={result.ratio:.2f})"
                color = (0, 255, 0)
            else:
                label = "No face detected"
                color = (0, 0, 255)

            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
            cv2.imshow("Gaze Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l") and eyes is not None:
                _log_reading(result.direction, result.ratio)
                print(f"Logged: {label}")

    finally:
        stream.release()
        detector.close()
        cv2.destroyAllWindows()


def _log_reading(direction: str, ratio: float, path: str = "gaze_log.csv"):
    """Appends a timestamped gaze reading to a CSV log file."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), direction, round(ratio, 3)])


if __name__ == "__main__":
    main()
