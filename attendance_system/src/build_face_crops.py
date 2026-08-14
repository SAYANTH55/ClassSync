"""
Phase 3c — Build the processed face-crop dataset
================================================

Combines the identity labels (``labels.csv``, all capture sessions) with
face geometry and produces the model-ready dataset::

    data/processed/faces_<size>/<student_id>/<student_id>_<source>_<seq>.png

plus ``manifest.csv`` recording, for every crop, its source image, geometry,
alignment status and generator parameters — full provenance, so any crop can
be traced back to the raw pixel region that produced it.

Geometry sources are merged per image with a fixed precedence (best wins):

    annotations.csv   human ground truth                (highest)
    detections.csv    own trained HOG+SVM detector      (future phases)
    proposals.csv     classical baseline — PROVISIONAL  (lowest)

The manifest records which tier produced every crop, so detector-assisted
labels are always distinguishable from human ground truth downstream.

Alignment: crops are eye-aligned (similarity transform) when eye coordinates
exist in the annotation row; otherwise a plain expanded-square crop is used
and the manifest records ``aligned=no``.

The processed tree is derived data: it is deleted and rebuilt on every run.
A QA contact sheet of all crops is written to reports/preprocessing/.

Usage:
    python src/build_face_crops.py [--size 112] [--margin 0.25]
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

import config
import preprocessing as pp

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("crops")

THUMB_W, SHEET_COLS = 112, 10


GEOMETRY_TIERS = [  # merged in this order; later (better) tiers overwrite
    ("proposals-PROVISIONAL", "proposals.csv"),
    ("detector", "detections.csv"),
    ("annotations", "annotations.csv"),
]


def read_geometry() -> dict[str, tuple[dict, str]]:
    """filename -> (geometry row, tier tag), best available tier per image."""
    geo: dict[str, tuple[dict, str]] = {}
    for tag, fname in GEOMETRY_TIERS:
        path = config.LABELS_CSV.parent / fname
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("face_x"):      # failed/blank rows are skipped
                    geo[r["filename"]] = (r, tag)
    if not geo:
        raise SystemExit("no geometry files found — run annotate_faces.py "
                         "or propose_boxes.py first.")
    return geo


def read_identities() -> list[dict[str, str]]:
    with open(config.LABELS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=112, help="output side (px)")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="box expansion fraction before squaring")
    ap.add_argument("--sessions",
                    help="comma list of capture sessions (default: all)")
    args = ap.parse_args()

    sessions = (set(args.sessions.split(",")) if args.sessions
                else set(config.KNOWN_SOURCES))
    geometry = read_geometry()
    id_rows = [r for r in read_identities() if r["source"] in sessions]

    # processed tree is derived. Full build: wipe. Partial (--sessions):
    # refresh only the selected sessions' crops, keep the rest intact.
    out_root = config.PROCESSED_DIR / f"faces_{args.size}"
    full_build = sessions == set(config.KNOWN_SOURCES)
    if full_build:
        if out_root.exists():
            shutil.rmtree(out_root)
    else:
        for src in sessions:
            for f in out_root.glob(f"*/*_{src}_*"):
                f.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    # deterministic per-(student, source) sequence numbers, matching
    # organize_dataset.py so processed names mirror organized names
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in id_rows:
        groups[(row["student_id"], row["source"])].append(row)

    manifest, thumbs, skipped = [], [], []
    for (sid, src), members in sorted(groups.items()):
        for seq, row in enumerate(sorted(members, key=lambda r: r["filename"]),
                                  start=1):
            fn = row["filename"]
            g, geo_tier = geometry.get(fn, (None, None))
            if g is None:
                skipped.append(fn)
                continue
            img = pp.load_image_upright(config.find_image(src, fn))
            box = tuple(int(g[k]) for k in ("face_x", "face_y",
                                            "face_w", "face_h"))
            eyes_known = bool(g.get("eye_lx"))
            sq = pp.expand_to_square(box, args.margin)
            if eyes_known:
                crop_img = pp.align_face(
                    img,
                    (float(g["eye_lx"]), float(g["eye_ly"])),
                    (float(g["eye_rx"]), float(g["eye_ry"])),
                    out_size=args.size)
            else:
                crop_img = pp.resize_square(pp.crop(img, sq), args.size)

            out_dir = out_root / sid
            out_dir.mkdir(exist_ok=True)
            out_name = f"{sid}_{src}_{seq:03d}.png"
            Image.fromarray(crop_img).save(out_dir / out_name)

            manifest.append({
                "crop": f"{sid}/{out_name}", "source_file": fn,
                "student_id": sid, "capture_source": src,
                "face_x": box[0], "face_y": box[1],
                "face_w": box[2], "face_h": box[3],
                "aligned": "yes" if eyes_known else "no",
                "geometry_from": geo_tier,
                "out_size": args.size, "margin": args.margin,
            })
            thumbs.append((crop_img, sid))

    manifest_path = out_root / "manifest.csv"
    if not full_build and manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f)
                    if r["capture_source"] not in sessions]
        manifest = kept + manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    # ---- QA sheet (evenly-spaced sample, capped for large sessions) --------
    if len(thumbs) > 200:
        stride = -(-len(thumbs) // 200)
        thumbs = thumbs[::stride]
    sheet_dir = config.REPORTS_DIR / "preprocessing"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    pad, cell_h = 6, THUMB_W + 20
    rows_n = -(-len(thumbs) // SHEET_COLS)
    sheet = Image.new("RGB", (SHEET_COLS * (THUMB_W + pad) + pad,
                              rows_n * (cell_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (arr, sid) in enumerate(thumbs):
        x = pad + (i % SHEET_COLS) * (THUMB_W + pad)
        y = pad + (i // SHEET_COLS) * (cell_h + pad)
        sheet.paste(Image.fromarray(arr).resize((THUMB_W, THUMB_W)), (x, y))
        draw.text((x + 2, y + THUMB_W + 3), sid, fill="black")
    sheet_path = sheet_dir / "crops_sheet.jpg"
    sheet.save(sheet_path, quality=90)

    aligned_n = sum(1 for m in manifest if m["aligned"] == "yes")
    by_tier = defaultdict(int)
    for m in manifest:
        by_tier[m["geometry_from"]] += 1
    print("\n============ CROP BUILD SUMMARY ============")
    print(f"sessions built       : {sorted(sessions)}")
    print(f"crops in manifest    : {len(manifest)}  ->  {out_root}")
    print(f"geometry tiers       : {dict(sorted(by_tier.items()))}")
    print(f"eye-aligned          : {aligned_n}/{len(manifest)}")
    print(f"skipped (no geometry): {len(skipped)}"
          + (f"  e.g. {skipped[:5]}" if skipped else ""))
    print(f"QA sheet             : {sheet_path}")
    print("============================================")


if __name__ == "__main__":
    main()
