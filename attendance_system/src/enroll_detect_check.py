"""
Enrollment detection/suitability check (SCRFD over the whole enroll set)
=======================================================================

Runs the deployed SCRFD detector on every enrollment image and reports, per
image: number of faces found, best detection score, the subject face's share
of the frame, and pose proxy (eye-line tilt / asymmetry from 5 landmarks).
Quantifies how well the pre-trained stack handles this data BEFORE we commit
an enrollment strategy — in particular how profiles and hallway bystanders
behave.

Output: reports/enroll_inspection/detection_report.csv + console summary.
"""

from __future__ import annotations

import csv
import logging

import numpy as np

import config
import preprocessing as pp
from detect_embed import FaceBackend

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("enroll_detect")

ENROLL_DIR = config.RAW_SESSIONS_DIR / "phone_enroll"


def pose_proxy(landmarks: np.ndarray) -> float:
    """Rough frontalness proxy in [0,1]: 1 = symmetric/frontal, ->0 = profile.

    Uses the horizontal balance of eyes/mouth around the nose x-coordinate.
    """
    le, re, nose, lm, rm = landmarks
    eye_span = abs(re[0] - le[0]) + 1e-6
    # nose offset from the eye midpoint, normalized by eye span
    mid = (le[0] + re[0]) / 2
    offset = abs(nose[0] - mid) / eye_span
    return float(max(0.0, 1.0 - offset))


def main() -> None:
    fb = FaceBackend()
    students = sorted(d for d in ENROLL_DIR.iterdir() if d.is_dir())
    rows = []
    for sdir in students:
        for p in sorted(sdir.iterdir()):
            if p.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            img = pp.load_image_upright(p)
            H, W = img.shape[:2]
            faces = fb.detect(img)
            if not faces:
                rows.append(dict(student=sdir.name, file=p.name, n_faces=0,
                                 det_score="", frame_frac="", frontalness="",
                                 status="NO_FACE"))
                continue
            f = faces[0]  # largest = subject
            frac = f.area / (W * H)
            rows.append(dict(student=sdir.name, file=p.name,
                             n_faces=len(faces),
                             det_score=round(f.det_score, 3),
                             frame_frac=round(frac, 3),
                             frontalness=round(pose_proxy(f.landmarks), 2),
                             status="OK" if len(faces) == 1 else "MULTI_FACE"))

    out = config.REPORTS_DIR / "enroll_inspection" / "detection_report.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] != "NO_FACE"]
    noface = [r for r in rows if r["status"] == "NO_FACE"]
    multi = [r for r in rows if r["status"] == "MULTI_FACE"]
    scores = sorted(r["det_score"] for r in ok)
    fronts = sorted(r["frontalness"] for r in ok)
    print("\n============ ENROLL DETECTION REPORT ============")
    print(f"images processed     : {len(rows)}")
    print(f"faces detected in     : {len(ok)}/{len(rows)} "
          f"({100*len(ok)/len(rows):.1f}%)")
    print(f"NO face detected      : {len(noface)}  "
          f"{[r['student']+'/'+r['file'] for r in noface][:6]}")
    print(f"MULTI-face (bystander): {len(multi)}  "
          f"{[r['student']+'/'+r['file'] for r in multi][:6]}")
    print(f"det_score            : min {scores[0]}, median {scores[len(scores)//2]}")
    print(f"frontalness proxy    : min {fronts[0]}, median {fronts[len(fronts)//2]}"
          f"  (low = profile)")
    students_all_ok = sum(
        1 for s in students
        if all(r["status"] != "NO_FACE" for r in rows if r["student"] == s.name))
    print(f"students with >=1 usable view : {students_all_ok if False else ''}")
    per_student_ok = {}
    for r in ok:
        per_student_ok[r["student"]] = per_student_ok.get(r["student"], 0) + 1
    worst = sorted(per_student_ok.items(), key=lambda kv: kv[1])[:5]
    print(f"fewest detected views/student : {worst}")
    print(f"report -> {out}")
    print("=================================================")


if __name__ == "__main__":
    main()
