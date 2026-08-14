"""
End-to-end smoke test
=====================

One command that exercises the complete deployed chain and fails loudly if
any link is broken. Run it after ANY change to code, data, or environment:

    python src/smoke_test.py

Checks (in pipeline order):
  1. config paths resolve; enrollment data, roster, gallery, probes exist
  2. image loading (EXIF-upright, RGB)
  3. face detection on a known portrait (exactly one face, sane confidence)
  4. embedding (512-d, unit length)
  5. identification (correct student, score clears the threshold)
  6. open-set rejection (a known non-enrolled face scores below threshold)
  7. attendance logging (mark + same-day dedup) into a THROWAWAY file

Uses a temporary attendance log; never touches real logs.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import preprocessing as pp

PASS, FAIL = "  ok  ", "FAILED"
failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    print(f"[{PASS if cond else FAIL}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures += 1


def main() -> None:
    print("=== smoke test:", datetime.now().isoformat(timespec="seconds"), "===")

    # 1. files and artifacts
    check("enrollment dir", config.PHONE_ENROLL_DIR.is_dir())
    check("labelled DSLR dir", config.SESSIONS["dslr_labelled"]["images_dir"].is_dir())
    check("roster.csv", config.ROSTER_CSV.exists())
    check("gallery.npz", config.GALLERY_NPZ.exists())

    from gallery import Recognizer
    rec = Recognizer(threshold=config.RECOG_THRESHOLD)
    check("gallery loads", len(rec.ids) >= 40, f"{len(rec.ids)} students")

    # 2. load — first labelled DSLR portrait that is an enrolled student;
    #    the expected name is read from the roster at runtime (not hard-coded)
    enrolled = set(rec.roster.values())
    genuine_path = next((p for p in config.session_images("dslr_labelled")
                         if p.stem in enrolled), None)
    check("genuine probe available", genuine_path is not None)
    expected_name = genuine_path.stem if genuine_path else None
    img = pp.load_image_upright(genuine_path)
    check("image loads upright", img.ndim == 3 and img.shape[0] > img.shape[1],
          f"shape {img.shape}")

    # 3. detect
    from detect_embed import FaceBackend
    fb = FaceBackend()
    faces = fb.detect(img)
    check("detects exactly one face", len(faces) == 1)
    check("detection confidence sane", faces and faces[0].det_score > 0.5,
          f"{faces[0].det_score:.3f}" if faces else "no face")

    # 4. embed
    emb = faces[0].embedding
    check("embedding is 512-d unit vector",
          emb.shape == (512,) and abs(float(np.linalg.norm(emb)) - 1) < 1e-3)

    # 5. identify (genuine)
    m = rec.identify(emb)
    check("identifies correct student", m is not None and m.name == expected_name,
          f"{m.name} @ {m.score:.3f}" if m else "rejected")

    # 6. open-set rejection (DSC files are known non-enrolled people)
    impostors = sorted(config.SESSIONS["dslr_labelled"]["images_dir"].glob("DSC*.jpeg"))
    check("impostor probe available", bool(impostors))
    if impostors:
        e2 = fb.embed_path(impostors[0])
        check("rejects non-enrolled face", e2 is not None and rec.identify(e2) is None,
              f"best {float(rec.scores(e2).max()):.3f} < tau {config.RECOG_THRESHOLD}"
              if e2 is not None else "no face")

    # 7. attendance logging in a sandbox
    import take_attendance as ta
    real_dir = config.ATTENDANCE_DIR
    with tempfile.TemporaryDirectory() as td:
        config.ATTENDANCE_DIR = Path(td)
        logbook = ta.AttendanceLog()
        first = logbook.mark(m.student_id, m.name, m.score)
        second = logbook.mark(m.student_id, m.name, m.score)
        check("marks attendance", first is True)
        check("deduplicates same-day mark", second is False)
    config.ATTENDANCE_DIR = real_dir

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
