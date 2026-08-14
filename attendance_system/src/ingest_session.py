"""
Session ingestion — normalize a new capture session into pipeline-ready images
==============================================================================

Input layout (created by the researcher when collecting data; immutable):

    data/raw_sessions/<session>/<student_id>/
        VID_0231.mp4          # videos: frames sampled at config.FRAME_FPS
        IMG_1223.jpg          # stills: copied byte-identically

Output (derived, write-once per original unless --force):

    data/frames/<session>/<session>_<sid>_<stem>_f0012.jpg   (video frames)
    data/frames/<session>/<session>_<sid>_<stem>.jpg         (stills)
    data/frames/<session>/ingestion_manifest.csv
    reports/ingestion/<session>_qa_sheet.jpg

Why this step exists
--------------------
* Guarantees the project invariant that filenames are GLOBALLY UNIQUE across
  sessions (tag + student id + sanitized stem are baked into every name).
* Pre-labels data: the per-student folder assigns identity at capture time,
  so organize_dataset.py can fill labels.csv rows automatically.
* Full traceability: the manifest records, per output image, its origin file,
  frame index and timestamp — any training crop can be traced back through
  the crop manifest -> labels -> ingestion manifest -> raw video.
* Determinism: same originals + same config -> same outputs (fixed sampling
  grid, fixed encoder settings).

Video rotation: some phones store rotation as metadata that OpenCV ignores.
Inspect the QA sheet after ingesting; if a student's frames are sideways,
re-run with  --student SXX --rotate 90|180|270 --force.

Usage:
    python src/ingest_session.py phone1
    python src/ingest_session.py phone1 --student S07 --rotate 90 --force
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil

import cv2
from PIL import Image, ImageDraw

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("ingest")

ROTATE_CODE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}
QA_THUMB, QA_COLS, QA_MAX = 180, 8, 96


def sanitize(stem: str) -> str:
    """File-stem cleanup so output names stay portable and parseable."""
    return re.sub(r"[^A-Za-z0-9-]+", "-", stem).strip("-")


def ingest_video(path, sid, session, out_dir, rotate, writer) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log.error("cannot open video %s", path.name)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / config.FRAME_FPS))
    stem = sanitize(path.stem)
    n_out = frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step == 0:
            if rotate:
                frame = cv2.rotate(frame, ROTATE_CODE[rotate])
            name = f"{session}_{sid}_{stem}_f{n_out:04d}.jpg"
            cv2.imwrite(str(out_dir / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_JPEG_QUALITY])
            writer.writerow([name, sid, f"{sid}/{path.name}", "video",
                             frame_i, round(frame_i / fps, 3), rotate])
            n_out += 1
        frame_i += 1
    cap.release()
    return n_out


def ingest_still(path, sid, session, out_dir, writer) -> int:
    name = f"{session}_{sid}_{sanitize(path.stem)}{path.suffix.lower()}"
    shutil.copy2(path, out_dir / name)   # byte-identical; EXIF preserved,
    writer.writerow([name, sid, f"{sid}/{path.name}", "still", "", "", 0])
    return 1                             # handled later by load_image_upright


def qa_sheet(session: str) -> None:
    imgs = config.session_images(session)
    if not imgs:
        return
    stride = max(1, len(imgs) // QA_MAX)
    sample = imgs[::stride][:QA_MAX]
    cells = []
    for p in sample:
        with Image.open(p) as im:
            im = im.convert("RGB")
            s = QA_THUMB / max(im.size)
            cells.append((im.resize((round(im.width * s), round(im.height * s))),
                          p.stem[:28]))
    cell = QA_THUMB + 22
    rows = -(-len(cells) // QA_COLS)
    sheet = Image.new("RGB", (QA_COLS * (QA_THUMB + 6) + 6,
                              rows * (cell + 6) + 6), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, name) in enumerate(cells):
        x = 6 + (i % QA_COLS) * (QA_THUMB + 6)
        y = 6 + (i // QA_COLS) * (cell + 6)
        sheet.paste(im, (x + (QA_THUMB - im.width) // 2,
                         y + (QA_THUMB - im.height) // 2))
        draw.text((x + 2, y + QA_THUMB + 4), name, fill="black")
    out = config.REPORTS_DIR / "ingestion" / f"{session}_qa_sheet.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    log.info("QA sheet (%d/%d sampled) -> %s", len(sample), len(imgs), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", choices=sorted(
        s for s, c in config.SESSIONS.items() if c["kind"] == "ingested"))
    ap.add_argument("--student", help="ingest only this student id (e.g. S07)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--force", action="store_true",
                    help="re-ingest originals whose outputs already exist")
    args = ap.parse_args()

    cfg = config.SESSIONS[args.session]
    src_root, out_dir = cfg["originals_dir"], cfg["images_dir"]
    if not src_root.exists():
        raise SystemExit(
            f"{src_root} not found.\nCreate it with one folder per student "
            f"(S01, S02, ...) containing that student's videos/stills.")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "ingestion_manifest.csv"
    new_manifest = not manifest_path.exists()
    n_frames = n_stills = n_skipped = 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if new_manifest:
            writer.writerow(["filename", "student_id", "origin", "origin_kind",
                             "frame_index", "time_s", "rotate_deg"])
        for sdir in sorted(d for d in src_root.iterdir() if d.is_dir()):
            sid = sdir.name
            if not re.match(config.STUDENT_ID_PATTERN, sid):
                log.warning("skipping folder '%s' (not a student id)", sid)
                continue
            if args.student and sid != args.student:
                continue
            for f in sorted(sdir.iterdir()):
                token = f"{args.session}_{sid}_{sanitize(f.stem)}"
                existing = list(out_dir.glob(f"{token}*"))
                if existing and not args.force:
                    n_skipped += 1
                    continue
                for old in existing:
                    old.unlink()
                if f.suffix.lower() in config.VIDEO_EXTENSIONS:
                    n_frames += ingest_video(f, sid, args.session, out_dir,
                                             args.rotate, writer)
                elif f.suffix.lower() in config.IMAGE_EXTENSIONS:
                    n_stills += ingest_still(f, sid, args.session,
                                             out_dir, writer)

    qa_sheet(args.session)
    print("\n============ INGESTION SUMMARY ============")
    print(f"session          : {args.session}")
    print(f"video frames     : {n_frames}")
    print(f"stills copied    : {n_stills}")
    print(f"originals skipped: {n_skipped} (already ingested; --force redoes)")
    print(f"images_dir total : {len(config.session_images(args.session))}")
    print("NEXT: python src/organize_dataset.py init --session "
          f"{args.session}")
    print("===========================================")


if __name__ == "__main__":
    main()
