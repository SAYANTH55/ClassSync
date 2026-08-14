"""
Phase 3a — Classical face-region proposals (background segmentation)
====================================================================

Generates PROPOSED face bounding boxes for the studio portraits using pure
image processing — no learned components of any kind. The proposals are:

  1. a labour-saving prefill for the manual annotation tool
     (the researcher verifies/adjusts each box -> ground truth), and
  2. the documented "classical baseline" for the dissertation's face-
     localization comparison (Chapter: rejected as primary method due to
     brittleness outside controlled conditions).

Method (constraint-compliant: zero trained parameters)
------------------------------------------------------
1. Load image EXIF-upright, downscale to a working width, grayscale.
2. Segment the person by CONNECTIVITY, not intensity: Canny edge map ->
   dilated into a "barrier"; every smooth region connected to the top/side
   image borders is background; regions the barrier fences off from the
   borders form the person's silhouette (largest component kept, holes
   closed).  NOTE: a simpler corner-sampled intensity threshold was tried
   first and FAILED — the studio backdrop is vignetted (bright centre,
   dark corners), so no single background value exists. Documented in the
   methodology as evidence of classical-method brittleness.
3. Row-width profile of the silhouette: the head spans from the first
   non-empty row down to the "shoulder row" where width jumps towards its
   maximum. Face box geometry is derived from head height/width with fixed
   anthropometric ratios.
4. Coordinates are mapped back to FULL-RESOLUTION upright pixel space and
   written to data/labels/proposals.csv. An overlay contact sheet is written
   for visual QA.

Scope: the head-geometry heuristics assume the STUDIO framing (uniform
backdrop, head-and-shoulders). The script therefore targets one session at a
time (default: dslr). Running it on cluttered phone data is possible for
comparison figures but proposals there are expected to be poor — phone
sessions are annotated manually (sample) and, later, by the trained
detector. Results are MERGED into proposals.csv per filename, so multiple
sessions coexist.

Usage:
    python src/propose_boxes.py [--session dslr]
"""

from __future__ import annotations

import argparse
import csv
import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("propose")

# ---- tunable constants (documented in methodology) -------------------------
WORK_WIDTH = 512          # analysis resolution (px)
CANNY_LO, CANNY_HI = 40, 120   # Canny hysteresis thresholds
SIDE_SEED_FRAC = 0.70     # top fraction of side borders used as background seeds
MIN_ROW_FRAC = 0.02       # row counts below this fraction of width = empty
SHOULDER_FRAC = 0.70      # row width >= this * max width  ->  shoulder row
HEAD_W_ROWS = (0.30, 0.80)  # rows (as frac of head height) used for head width
FACE_W_FRAC = 0.90        # face box width as fraction of head width
FACE_TOP_FRAC = 0.12      # face box starts this far below the hair top
FACE_BOT_FRAC = 1.00      # face box ends at this fraction of head height
OVERLAY_THUMB_W = 260


def segment_silhouette(gray: np.ndarray) -> np.ndarray:
    """Boolean foreground mask of the person against the studio backdrop.

    Robust to the backdrop's vignette because it never models background
    intensity: background is defined as "smooth AND connected to the top or
    upper-side image borders"; the person's edge outline acts as a barrier
    that fences the body interior off from those borders.
    """
    h, w = gray.shape
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    barrier = cv2.dilate(edges, kernel5, iterations=1) > 0

    # label smooth (non-barrier) regions; 4-connectivity so the thin barrier
    # actually separates regions
    free = (~barrier).astype(np.uint8)
    _, labels = cv2.connectedComponents(free, connectivity=4)

    side_rows = int(SIDE_SEED_FRAC * h)
    seed_labels = (set(labels[0, :].tolist())
                   | set(labels[:side_rows, 0].tolist())
                   | set(labels[:side_rows, w - 1].tolist()))
    seed_labels.discard(0)  # 0 = barrier pixels, never a seed

    background = np.isin(labels, list(seed_labels))
    fg = (~background).astype(np.uint8)  # person interior + its edge outline

    n, comp, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sil = (comp == largest).astype(np.uint8)

    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, kernel7, iterations=2)
    return sil.astype(bool)


def propose_face_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Derive (x, y, w, h) face box from the silhouette's row-width profile."""
    h, w = mask.shape
    widths = mask.sum(axis=1)
    nonempty = np.flatnonzero(widths > MIN_ROW_FRAC * w)
    if nonempty.size < 10:
        return None
    top = int(nonempty[0])

    max_w = int(widths[top:].max())
    shoulder_rows = np.flatnonzero(widths[top:] >= SHOULDER_FRAC * max_w)
    if shoulder_rows.size == 0:
        return None
    shoulder = top + int(shoulder_rows[0])
    head_h = shoulder - top
    if head_h < 10:
        return None

    # head width & centre measured on the middle of the head (skips hair spike)
    r0 = top + int(HEAD_W_ROWS[0] * head_h)
    r1 = top + int(HEAD_W_ROWS[1] * head_h)
    spans = []
    for r in range(r0, max(r1, r0 + 1)):
        cols = np.flatnonzero(mask[r])
        if cols.size:
            spans.append((cols[0], cols[-1]))
    if not spans:
        return None
    lefts, rights = zip(*spans)
    head_w = float(np.median(np.array(rights) - np.array(lefts)))
    cx = float(np.median((np.array(rights) + np.array(lefts)) / 2))

    fw = FACE_W_FRAC * head_w
    y0 = top + FACE_TOP_FRAC * head_h
    y1 = top + FACE_BOT_FRAC * head_h
    fh = y1 - y0
    x0 = cx - fw / 2
    return (int(round(x0)), int(round(y0)), int(round(fw)), int(round(fh)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="dslr",
                    choices=sorted(config.KNOWN_SOURCES))
    args = ap.parse_args()
    if config.SESSIONS[args.session]["kind"] != "stills":
        log.info("note: heuristics are tuned for the studio session; expect "
                 "weak proposals on '%s' data", args.session)
    paths = config.session_images(args.session)
    if not paths:
        raise SystemExit(f"no images found for session '{args.session}'")
    out_rows, thumbs = [], []

    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            full_w, full_h = im.size
            scale = WORK_WIDTH / full_w
            small = im.convert("L").resize(
                (WORK_WIDTH, round(full_h * scale)), Image.BILINEAR)
        gray = np.asarray(small)

        mask = segment_silhouette(gray)
        box = propose_face_box(mask)
        if box is None:
            log.warning("%s: no proposal (segmentation failed)", p.name)
            out_rows.append([p.name, "", "", "", "", "bgseg-v1-FAILED"])
            thumbs.append((p, None))
            continue

        x, y, w, h = (int(round(v / scale)) for v in box)  # back to full res
        out_rows.append([p.name, x, y, w, h, "bgseg-v1"])
        thumbs.append((p, (x, y, w, h)))

    # merge with existing proposals (other sessions' rows are preserved)
    csv_path = config.LABELS_CSV.parent / "proposals.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["filename", "face_x", "face_y", "face_w", "face_h", "method"]
    merged: dict[str, list] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                merged[r["filename"]] = [r.get(k, "") for k in header]
    for row in out_rows:
        merged[row[0]] = row
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(merged[k] for k in sorted(merged))
    ok = sum(1 for r in out_rows if r[1] != "")
    log.info("proposals -> %s  (%d/%d succeeded for session '%s'; "
             "%d rows total)", csv_path, ok, len(out_rows), args.session,
             len(merged))

    # ---- QA overlay sheet (evenly-spaced sample for large sessions) --------
    if len(thumbs) > 112:
        thumbs = thumbs[::-(-len(thumbs) // 112)]
    sheet_dir = config.REPORTS_DIR / "preprocessing"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cols, pad = 7, 8
    cells = []
    for p, box in thumbs:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            s = OVERLAY_THUMB_W / im.width
            thumb = im.resize((OVERLAY_THUMB_W, round(im.height * s)),
                              Image.BILINEAR).convert("RGB")
        if box:
            x, y, w, h = (v * s for v in box)
            ImageDraw.Draw(thumb).rectangle([x, y, x + w, y + h],
                                            outline=(0, 255, 0), width=3)
        cells.append((thumb, p.stem))
    cell_h = max(t.height for t, _ in cells) + 22
    rows_n = -(-len(cells) // cols)
    sheet = Image.new("RGB", (cols * (OVERLAY_THUMB_W + pad) + pad,
                              rows_n * (cell_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (thumb, name) in enumerate(cells):
        x = pad + (i % cols) * (OVERLAY_THUMB_W + pad)
        y = pad + (i // cols) * (cell_h + pad)
        sheet.paste(thumb, (x, y))
        draw.text((x + 2, y + thumb.height + 3), name, fill="black")
    sheet_path = sheet_dir / f"proposals_overlay_{args.session}.jpg"
    sheet.save(sheet_path, quality=88)
    log.info("QA overlay -> %s", sheet_path)


if __name__ == "__main__":
    main()
