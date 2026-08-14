"""
Phase 1 — Dataset Inspection
============================

Analyzes the raw face-portrait dataset BEFORE any preprocessing decisions.

For every image this script records:
  * file name, file size, SHA-256 hash (exact-duplicate detection)
  * stored pixel dimensions and EXIF orientation flag
  * capture timestamp, camera make/model, ISO, exposure (from EXIF)
  * brightness mean / std  (exposure consistency check)
  * variance-of-Laplacian blur score (focus quality check)

Outputs (written to reports/dataset_inspection/):
  * image_metadata.csv   - one row per image, all metrics
  * contact_sheet.jpg    - thumbnail grid used for manual identity labeling
  * labels_template.csv  - empty filename -> student_id mapping to fill in

Design notes
------------
* Images are opened with Pillow and passed through ImageOps.exif_transpose()
  so orientation matches what a human sees.  OpenCV ignores EXIF, which is
  exactly why orientation is measured and corrected here, once, explicitly.
* Blur and brightness are computed on a fixed-width (1024 px) grayscale copy
  so scores are comparable across images regardless of native resolution.
* No face detection is performed at this stage — the "no pre-trained model"
  constraint means the face-localization strategy is a separate, explicit
  design decision (Phase 3), made only after this inspection.

The same audit applies to every capture session (``--session``); output
files are suffixed with the session tag (legacy unsuffixed names are kept
for the original DSLR session).

Usage:
    python src/inspect_dataset.py [--session dslr]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import config

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
REPORT_DIR = config.REPORTS_DIR / "dataset_inspection"
SHEET_MAX = 120                # contact-sheet thumbnail cap per session

ANALYSIS_WIDTH = 1024        # width used for blur / brightness metrics
THUMB_WIDTH = 260            # contact-sheet thumbnail width
SHEET_COLUMNS = 7            # contact-sheet grid columns

# EXIF tag ids (see JEITA CP-3451 specification)
TAG_MAKE, TAG_MODEL, TAG_ORIENTATION = 271, 272, 274
EXIF_IFD_POINTER = 0x8769
TAG_DATETIME_ORIGINAL, TAG_ISO, TAG_EXPOSURE = 36867, 34855, 33434

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("inspect")


@dataclass
class ImageRecord:
    """All inspection metrics for a single raw image."""
    filename: str
    file_mb: float
    sha256_16: str            # first 16 hex chars — enough to spot duplicates
    stored_width: int
    stored_height: int
    exif_orientation: int
    upright_width: int        # dimensions AFTER orientation correction
    upright_height: int
    capture_time: str
    camera: str
    iso: str
    exposure_s: str
    brightness_mean: float    # 0-255 grayscale mean
    brightness_std: float
    blur_laplacian_var: float # higher = sharper (scale-dependent, see notes)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian — a standard focus measure."""
    g = gray.astype(np.float64)
    lap = (-4.0 * g[1:-1, 1:-1]
           + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def inspect_image(path: Path) -> ImageRecord:
    with Image.open(path) as img:
        stored_w, stored_h = img.size
        exif = img.getexif()
        orientation = int(exif.get(TAG_ORIENTATION, 1))
        camera = f"{exif.get(TAG_MAKE, '?')} {exif.get(TAG_MODEL, '?')}".strip()
        ifd = exif.get_ifd(EXIF_IFD_POINTER)
        capture_time = str(ifd.get(TAG_DATETIME_ORIGINAL, "?"))
        iso = str(ifd.get(TAG_ISO, "?"))
        exposure = str(ifd.get(TAG_EXPOSURE, "?"))

        upright = ImageOps.exif_transpose(img)
        upright_w, upright_h = upright.size

        # fixed-width grayscale copy for comparable quality metrics
        scale = ANALYSIS_WIDTH / upright_w
        small = upright.convert("L").resize(
            (ANALYSIS_WIDTH, max(1, round(upright_h * scale))), Image.BILINEAR)
        gray = np.asarray(small)

    return ImageRecord(
        filename=path.name,
        file_mb=round(path.stat().st_size / 2**20, 2),
        sha256_16=sha256_of(path),
        stored_width=stored_w, stored_height=stored_h,
        exif_orientation=orientation,
        upright_width=upright_w, upright_height=upright_h,
        capture_time=capture_time, camera=camera, iso=iso, exposure_s=exposure,
        brightness_mean=round(float(gray.mean()), 1),
        brightness_std=round(float(gray.std()), 1),
        blur_laplacian_var=round(laplacian_variance(gray), 1),
    )


def build_contact_sheet(paths: list[Path], out_path: Path) -> None:
    """Numbered thumbnail grid so identities can be labeled quickly."""
    thumbs = []
    for p in paths:
        with Image.open(p) as img:
            img = ImageOps.exif_transpose(img)
            ratio = THUMB_WIDTH / img.width
            thumbs.append(img.resize(
                (THUMB_WIDTH, round(img.height * ratio)), Image.BILINEAR))

    cell_h = max(t.height for t in thumbs) + 26          # caption space
    rows = -(-len(thumbs) // SHEET_COLUMNS)
    sheet = Image.new("RGB", (SHEET_COLUMNS * (THUMB_WIDTH + 8) + 8,
                              rows * (cell_h + 8) + 8), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (thumb, p) in enumerate(zip(thumbs, paths)):
        x = 8 + (i % SHEET_COLUMNS) * (THUMB_WIDTH + 8)
        y = 8 + (i // SHEET_COLUMNS) * (cell_h + 8)
        sheet.paste(thumb, (x, y))
        draw.text((x + 2, y + thumb.height + 4),
                  f"{i + 1:02d}  {p.stem}", fill="black")
    sheet.save(out_path, quality=88)
    log.info("contact sheet -> %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="dslr",
                    choices=sorted(config.KNOWN_SOURCES))
    args = ap.parse_args()
    suffix = "" if args.session == "dslr" else f"_{args.session}"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = config.session_images(args.session)
    if not paths:
        raise SystemExit(f"No images found for session '{args.session}'")
    log.info("inspecting %d images (session '%s') ...",
             len(paths), args.session)

    records = [inspect_image(p) for p in paths]

    # ---- per-image CSV -----------------------------------------------------
    csv_path = REPORT_DIR / f"image_metadata{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)
    log.info("metadata table -> %s", csv_path)

    # ---- contact sheet (evenly-spaced sample for large sessions) -----------
    sheet_paths = (paths if len(paths) <= SHEET_MAX
                   else paths[::-(-len(paths) // SHEET_MAX)])
    build_contact_sheet(sheet_paths, REPORT_DIR / f"contact_sheet{suffix}.jpg")

    # ---- console summary ---------------------------------------------------
    dims = {(r.upright_width, r.upright_height) for r in records}
    orients = sorted({r.exif_orientation for r in records})
    dupes = len(records) - len({r.sha256_16 for r in records})
    bright = [r.brightness_mean for r in records]
    blur = [r.blur_laplacian_var for r in records]

    times = []
    for r in records:
        try:
            times.append((datetime.strptime(r.capture_time,
                          "%Y:%m:%d %H:%M:%S"), r.filename))
        except ValueError:
            pass
    times.sort()

    print("\n================ DATASET SUMMARY ================")
    print(f"images                : {len(records)}")
    print(f"cameras               : {sorted({r.camera for r in records})}")
    print(f"upright resolutions   : {sorted(dims)}")
    print(f"EXIF orientations     : {orients}")
    print(f"exact duplicates      : {dupes}")
    print(f"brightness mean range : {min(bright):.0f} - {max(bright):.0f} (0-255)")
    print(f"blur score  range     : {min(blur):.0f} - {max(blur):.0f} "
          f"(median {sorted(blur)[len(blur)//2]:.0f}; low = suspect)")
    if times:
        span = times[-1][0] - times[0][0]
        print(f"capture window        : {times[0][0]} -> {times[-1][0]} ({span})")
        print("\nshot-to-shot gaps (person changes show up as larger gaps):")
        for (t1, f1), (t2, f2) in zip(times, times[1:]):
            gap = (t2 - t1).total_seconds()
            marker = "  <-- likely NEW PERSON" if gap >= 20 else ""
            print(f"  {f1} -> {f2}: {gap:5.0f}s{marker}")
    print("=================================================")


if __name__ == "__main__":
    main()
