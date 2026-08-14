"""
RGB aligned-crop cache for the deep-learning baselines
======================================================

Same SCRFD detection + 5-point norm_crop alignment used everywhere else in
the project, but keeping RGB (the CNN baselines consume colour; the classical
cache is grayscale). Writes ``data/cache/crops_rgb.npz``:

    gallery/<student>/<file>  ->  (112, 112, 3) uint8   (phone enrollment)
    probe/<stem>              ->  (112, 112, 3) uint8   (labelled DSLR)

Run under the face311 env (insightface). The training/evaluation script
(evaluate_deep.py) runs under the TensorFlow env and reads only this npz —
the two environments never need each other's packages.

Usage:
    python src/build_crops_rgb.py
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

import config
import preprocessing as pp

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("build_crops_rgb")

OUT = config.CACHE_DIR / "crops_rgb.npz"
CROP = 112


def main() -> None:
    from detect_embed import FaceBackend
    from insightface.utils.face_align import norm_crop
    fb = FaceBackend()
    out: dict[str, np.ndarray] = {}

    def crop_of(path) -> np.ndarray | None:
        img = pp.load_image_upright(path)
        faces = fb.detect(img)
        return norm_crop(img, faces[0].landmarks, image_size=CROP) if faces else None

    for d in sorted(config.PHONE_ENROLL_DIR.iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in config.IMAGE_EXTENSIONS:
                c = crop_of(p)
                if c is None:
                    log.warning("no face: %s/%s", d.name, p.name)
                    continue
                out[f"gallery/{d.name}/{p.name}"] = c
    for p in config.session_images("dslr_labelled"):
        c = crop_of(p)
        if c is None:
            log.warning("no face: %s", p.name)
            continue
        out[f"probe/{p.stem}"] = c

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    log.info("aligned %d RGB crops -> %s", len(out), OUT)


if __name__ == "__main__":
    main()
