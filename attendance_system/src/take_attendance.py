"""
Kiosk attendance pipeline (deployed end-to-end path)
====================================================

The complete deployed flow in one command:

    image/webcam frame -> SCRFD detect+landmarks -> ArcFace embed
                       -> open-set identify (tau from config)
                       -> append to today's attendance CSV

Attendance log: ``data/attendance/attendance_YYYY-MM-DD.csv`` with columns
``time,student_id,name,score`` — one row per student per day (re-recognising
an already-marked student is reported but not duplicated). ``student_id`` is
the stable roster id, so logs survive name spelling fixes.

Modes
-----
* Photo mode:  ``python src/take_attendance.py IMG1 [IMG2 ...]``
  Marks every face in each image that clears the threshold (supports group
  photos). Unknown faces are reported, never logged.
* Webcam mode: ``python src/take_attendance.py --webcam [N]``
  Live preview with box + name + score; SPACE marks the largest face,
  ESC quits. One student at a time — kiosk interaction model.

Viva concepts: open-set rejection at deployment; why the threshold is a
calibrated constant, not a magic number; audit trail via stable ids;
failure handling (no face / unknown face / already marked).
"""

from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import preprocessing as pp
from detect_embed import FaceBackend
from gallery import Recognizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("take_attendance")


# ---- attendance log --------------------------------------------------------
class AttendanceLog:
    """Append-only per-day CSV; one row per student per day."""

    FIELDS = ["time", "student_id", "name", "score"]

    def __init__(self, day: str | None = None):
        self.day = day or datetime.now().strftime("%Y-%m-%d")
        config.ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = config.ATTENDANCE_DIR / f"attendance_{self.day}.csv"
        self.present: set[str] = set()
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                self.present = {r["student_id"] for r in csv.DictReader(f)}
        else:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDS)

    def mark(self, student_id: str, name: str, score: float) -> bool:
        """Record one student; False if already marked today."""
        if student_id in self.present:
            return False
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [datetime.now().strftime("%H:%M:%S"), student_id, name,
                 f"{score:.4f}"])
        self.present.add(student_id)
        return True


# ---- photo mode ------------------------------------------------------------
def process_photo(path: Path, fb: FaceBackend, rec: Recognizer,
                  logbook: AttendanceLog) -> None:
    img = pp.load_image_upright(path)
    faces = fb.detect(img)
    if not faces:
        print(f"{path.name}: NO FACE DETECTED")
        return
    for face in faces:
        if getattr(face, "is_spoof", False):
            print(f"{path.name}: SPOOF DETECTED (prob: {face.spoof_probability:.3f}) - rejecting")
            continue
            
        m = rec.identify(face.embedding)
        if m is None:
            best = rec.scores(face.embedding).max()
            print(f"{path.name}: UNKNOWN face (best score {best:.3f} "
                  f"< tau {rec.threshold})")
            continue
        if logbook.mark(m.student_id, m.name, m.score):
            print(f"{path.name}: PRESENT  {m.student_id} {m.name}"
                  f"  (score {m.score:.3f}, margin {m.margin:.3f})")
        else:
            print(f"{path.name}: already marked  {m.student_id} {m.name}")


# ---- webcam mode -----------------------------------------------------------
def run_webcam(cam_index: int, fb: FaceBackend, rec: Recognizer,
               logbook: AttendanceLog) -> None:
    """Live kiosk with multi-frame confirmation.

    A student is marked only after CONFIRM_FRAMES consecutive frames agree on
    the same identity — a single lucky/blurry frame can neither mark the
    right person at a bad moment nor the wrong person at all. SPACE still
    forces an immediate mark (operator override); ESC quits.
    """
    import cv2

    CONFIRM_FRAMES = 5
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"cannot open webcam #{cam_index}")
    print(f"auto-marks after {CONFIRM_FRAMES} consistent frames; "
          "SPACE = force-mark, ESC = quit")
    streak_id, streak = None, 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        faces = fb.detect(frame_bgr[:, :, ::-1])
        if faces and getattr(faces[0], "is_spoof", False):
            m = None
        else:
            m = rec.identify(faces[0].embedding) if faces else None

        # consecutive-agreement counter drives auto-marking
        if m is not None and m.student_id == streak_id:
            streak += 1
        else:
            streak_id, streak = (m.student_id if m else None), 1 if m else 0
        if m is not None and streak >= CONFIRM_FRAMES:
            if logbook.mark(m.student_id, m.name, m.score):
                print(f"PRESENT  {m.student_id} {m.name}  ({m.score:.3f})")
            streak = 0                    # re-confirm before any next mark

        if faces:
            x1, y1, x2, y2 = faces[0].bbox.astype(int)
            is_spoof = getattr(faces[0], "is_spoof", False)
            if is_spoof:
                color = (0, 0, 255)
                label = f"SPOOF {faces[0].spoof_probability:.2f}"
            else:
                known = m is not None
                marked = known and m.student_id in logbook.present
                color = ((180, 130, 0) if marked else (0, 200, 0)) if known \
                    else (0, 0, 255)
                label = (f"{m.name} {m.score:.2f}"
                         + (" MARKED" if marked else f" {streak}/{CONFIRM_FRAMES}")
                         if known else "unknown")
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("attendance kiosk", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                                    # ESC
            break
        if key == 32 and m is not None:                  # SPACE override
            if logbook.mark(m.student_id, m.name, m.score):
                print(f"PRESENT  {m.student_id} {m.name}  ({m.score:.3f})")
            else:
                print(f"already marked  {m.student_id} {m.name}")
    cap.release()
    cv2.destroyAllWindows()


# ---- CLI -------------------------------------------------------------------
def main(argv: list[str]) -> None:
    rec = Recognizer(threshold=config.RECOG_THRESHOLD)
    fb = FaceBackend()
    logbook = AttendanceLog()
    print(f"attendance log: {logbook.path} "
          f"({len(logbook.present)} already marked today)")

    if argv and argv[0] == "--webcam":
        run_webcam(int(argv[1]) if len(argv) > 1 else 0, fb, rec, logbook)
    elif argv:
        for a in argv:
            process_photo(Path(a), fb, rec, logbook)
    else:
        raise SystemExit(__doc__)

    print(f"\n{len(logbook.present)}/{len(rec.ids)} students marked present "
          f"on {logbook.day}")


if __name__ == "__main__":
    main(sys.argv[1:])
