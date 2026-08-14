"""
Per-student face thumbnails for the web UI.

Generates a small square face crop for each enrolled student from their first
enrollment photo, cached to data/cache/thumbs/<student_id>.jpg. The web app
serves these so lists show a real face instead of initials. Detection-only
(no recognition / anti-spoofing) — this is cosmetic, not part of the pipeline.

Usage:
    python src/build_thumbnails.py            # build for everyone
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

import config
import preprocessing as pp
from gallery import load_roster

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("build_thumbnails")

THUMB_DIR = config.CACHE_DIR / "thumbs"
SIZE = 256          # output thumbnail side (px)
MARGIN = 1.6        # face-box expansion for a nice head-and-shoulders crop


def _det_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=config.EMBED_MODEL_NAME,
                       root=str(config.EMBED_MODEL_ROOT),
                       providers=["CPUExecutionProvider"],
                       allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _frontalness(face) -> float:
    """0..1 — how front-facing the head is, from the 5 landmarks. A frontal
    face has the nose centred between the eyes and the eyes roughly level."""
    lm = face.kps
    leye, reye, nose = lm[0], lm[1], lm[2]
    eye_mid_x = (leye[0] + reye[0]) / 2
    eye_dist = abs(reye[0] - leye[0]) + 1e-6
    horiz = 1 - min(1.0, abs(nose[0] - eye_mid_x) / (eye_dist / 2 + 1e-6))
    level = 1 - min(1.0, abs(reye[1] - leye[1]) / (eye_dist + 1e-6))
    return 0.7 * horiz + 0.3 * level


def build_one(student_id: str, name: str, app) -> bool:
    """Write data/cache/thumbs/<student_id>.jpg; True on success.

    Chooses the MOST FRONTAL of the student's enrollment photos so the
    thumbnail is a front-facing portrait, not a side profile.
    """
    # candidate source images: the front-facing DSLR studio portrait (if the
    # student has one) PLUS the phone selfies. We then pick the most frontal
    # of ALL of them — the studio portrait usually wins, which fixes students
    # whose selfies are all side-facing.
    imgs = []
    dslr = config.SESSIONS["dslr_labelled"]["images_dir"] / f"{name}.jpeg"
    if dslr.exists():
        imgs.append(dslr)
    folder = config.PHONE_ENROLL_DIR / name
    if folder.is_dir():
        imgs += sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in config.IMAGE_EXTENSIONS)
    if not imgs:
        return False

    best = None                                    # (score, rgb, face)
    for p in imgs:
        rgb = pp.load_image_upright(p)
        faces = app.get(rgb[:, :, ::-1])
        if not faces:
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                * (f.bbox[3] - f.bbox[1]))
        score = _frontalness(f)
        if best is None or score > best[0]:
            best = (score, rgb, f)

    if best is not None:
        _, rgb, f = best
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = f.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * MARGIN
    else:                                          # no face in any -> centre
        rgb = pp.load_image_upright(imgs[0])
        h, w = rgb.shape[:2]
        cx, cy, side = w / 2, h / 2, min(h, w)

    half = side / 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    thumb = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(THUMB_DIR / f"{student_id}.jpg"),
                thumb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def build_all() -> None:
    roster = load_roster()
    app = _det_app()
    n = 0
    for sid, name in sorted(roster.items()):
        if build_one(sid, name, app):
            n += 1
        else:
            log.warning("no thumbnail for %s (%s)", sid, name)
    log.info("built %d/%d thumbnails -> %s", n, len(roster), THUMB_DIR)


if __name__ == "__main__":
    build_all()
