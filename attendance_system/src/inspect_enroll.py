"""
Enrollment dataset inspection (phone, multi-view)
=================================================

Audits the second dataset (per-student phone captures used for enrollment)
BEFORE any pipeline integration. Walks ``data/raw_sessions/phone_enroll/
<name>/*`` and records per image: format, colour mode, EXIF orientation,
upright resolution (megapixels), a blur score and mean brightness; flags any
file that fails to decode.

Outputs (reports/enroll_inspection/):
    image_metadata_enroll.csv    one row per image
    contact_sheet_enroll.jpg     first image of every student (visual check)
    roster_template.csv          proposed real-name -> pseudonymous id map

No identity IDs are assigned to data yet; this is inspection only.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import config
import preprocessing as pp  # registers HEIC opener on import

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("inspect_enroll")

ENROLL_DIR = config.RAW_SESSIONS_DIR / "phone_enroll"
REPORT_DIR = config.REPORTS_DIR / "enroll_inspection"
ANALYSIS_W = 512
THUMB_W, COLS = 200, 8
TAG_ORIENTATION = 274


def blur_score(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    students = sorted(d for d in ENROLL_DIR.iterdir() if d.is_dir())
    rows, corrupt, first_imgs = [], [], []

    for sdir in students:
        imgs = sorted(p for p in sdir.iterdir()
                      if p.suffix.lower() in config.IMAGE_EXTENSIONS
                      or p.suffix.lower() == ".dng")
        first_done = False
        for p in imgs:
            if p.suffix.lower() == ".dng":
                rows.append(dict(student=sdir.name, file=p.name, fmt="DNG",
                                 mode="", orientation="", width="", height="",
                                 megapixels="", blur="", brightness="",
                                 note="raw — has sibling JPG; skipped for use"))
                continue
            try:
                with Image.open(p) as im:
                    fmt = im.format
                    orient = int(im.getexif().get(TAG_ORIENTATION, 1))
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    w, h = im.size
                    small = im.resize((ANALYSIS_W,
                                       max(1, round(h * ANALYSIS_W / w))))
                    g = np.asarray(small.convert("L"))
            except Exception as e:  # noqa: BLE001
                corrupt.append((str(p), repr(e)))
                continue
            rows.append(dict(student=sdir.name, file=p.name, fmt=fmt,
                             mode="RGB", orientation=orient, width=w, height=h,
                             megapixels=round(w * h / 1e6, 1),
                             blur=round(blur_score(g), 1),
                             brightness=round(float(g.mean()), 1), note=""))
            if not first_done:
                first_imgs.append((sdir.name, small.copy()))
                first_done = True

    # ---- CSV ---------------------------------------------------------------
    csv_path = REPORT_DIR / "image_metadata_enroll.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("metadata -> %s", csv_path)

    # ---- roster template (pseudonymization proposal) -----------------------
    roster_path = REPORT_DIR / "roster_template.csv"
    with open(roster_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["real_name", "student_id"])
        for i, sdir in enumerate(students, 1):
            w.writerow([sdir.name, f"P{i:02d}"])
    log.info("roster template -> %s", roster_path)

    # ---- contact sheet -----------------------------------------------------
    thumbs = []
    for name, im in first_imgs:
        s = THUMB_W / im.width
        thumbs.append((im.resize((THUMB_W, round(im.height * s))), name))
    cell_h = max(t.height for t, _ in thumbs) + 20
    rowsn = -(-len(thumbs) // COLS)
    sheet = Image.new("RGB", (COLS * (THUMB_W + 6) + 6, rowsn * (cell_h + 6) + 6),
                      "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, name) in enumerate(thumbs):
        x = 6 + (i % COLS) * (THUMB_W + 6)
        y = 6 + (i // COLS) * (cell_h + 6)
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + im.height + 3), name, fill="black")
    sheet_path = REPORT_DIR / "contact_sheet_enroll.jpg"
    sheet.save(sheet_path, quality=85)
    log.info("contact sheet -> %s", sheet_path)

    # ---- summary -----------------------------------------------------------
    usable = [r for r in rows if r["fmt"] != "DNG"]
    per_student = {}
    for r in usable:
        per_student.setdefault(r["student"], 0)
        per_student[r["student"]] += 1
    mps = [r["megapixels"] for r in usable if r["megapixels"] != ""]
    blurs = sorted(r["blur"] for r in usable if r["blur"] != "")
    orients = {}
    for r in usable:
        orients[r["orientation"]] = orients.get(r["orientation"], 0) + 1
    fmts = {}
    for r in usable:
        fmts[r["fmt"]] = fmts.get(r["fmt"], 0) + 1

    print("\n================ ENROLLMENT DATASET SUMMARY ================")
    print(f"students (folders)   : {len(students)}")
    print(f"usable images        : {len(usable)}  (+{len(rows)-len(usable)} DNG skipped)")
    print(f"images/student       : min {min(per_student.values())}, "
          f"max {max(per_student.values())}, "
          f"mean {sum(per_student.values())/len(per_student):.1f}")
    print(f"formats              : {fmts}")
    print(f"EXIF orientations    : {orients}  (non-1 => rotation stored in EXIF)")
    print(f"resolution (MP)      : {min(mps)} - {max(mps)} MP")
    print(f"blur score (var-lap) : min {blurs[0]}, median {blurs[len(blurs)//2]}"
          f" (lower = softer)")
    print(f"corrupt/undecodable  : {len(corrupt)} {corrupt[:3] if corrupt else ''}")
    print(f"contact sheet        : {sheet_path}")
    print("===========================================================")


if __name__ == "__main__":
    main()
