"""
Student enrollment CLI
======================

Adds (or extends) one student in the enrollment data and rebuilds the
gallery, with validation BEFORE anything is written:

  * every supplied image must contain exactly one detectable face;
  * the new images must all be the same person (pairwise similarity check);
  * if the student already exists, new images must match the existing
    templates (protects against enrolling someone under the wrong name —
    the class of bug the project audit caught in the raw dataset).

Usage:
    python src/enroll.py "Full Name" img1.jpg img2.heic ...
    python src/enroll.py --remove "Full Name"     # unenroll (keeps images
                                                  # on disk, removes folder
                                                  # from phone_enroll)
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np

import config
import gallery

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("enroll")

MIN_CENTROID_SIM = 0.35    # each image must be this close to the group centre
MIN_MATCH_EXISTING = 0.35  # ...and clear this vs existing templates


class EnrollmentError(Exception):
    """Validation failure — callers (CLI, web API) decide how to present it."""


def enroll(name: str, image_paths: list[Path]) -> None:
    from detect_embed import FaceBackend
    fb = FaceBackend()

    embs = []
    for p in image_paths:
        if not p.exists():
            raise EnrollmentError(f"missing file: {p}")
        e = fb.embed_path(p)
        if e is None:
            raise EnrollmentError(f"REJECTED: no detectable face in {p.name}")
        embs.append(e)
    E = np.stack(embs)

    # all-same-person check via the group CENTROID: every image must point
    # near the average face direction. This tolerates natural pose/expression
    # variation (which stays near the centre) but flags a stray photo of a
    # different person (which sits far from it). More robust than a
    # min-pairwise test, which a single hard pair of extreme-angle genuine
    # shots can trip — and it names the offending file.
    if len(E) > 1:
        centroid = E.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        to_centroid = E @ centroid
        worst = int(np.argmin(to_centroid))
        if float(to_centroid[worst]) < MIN_CENTROID_SIM:
            raise EnrollmentError(
                f"REJECTED: '{image_paths[worst].name}' does not match the "
                f"others (similarity {to_centroid[worst]:.3f} to the group "
                f"average < {MIN_CENTROID_SIM}) — is it a different person?")

    # consistency with existing enrollment, if any
    dest = config.PHONE_ENROLL_DIR / name
    if dest.exists() and config.GALLERY_NPZ.exists():
        rec = gallery.Recognizer()
        sid = {v: k for k, v in rec.roster.items()}.get(name)
        if sid is not None:
            T = rec.templates[rec.ids.index(sid)]
            best = float(np.max(E @ T.T))
            if best < MIN_MATCH_EXISTING:
                raise EnrollmentError(
                    f"REJECTED: new images do not match existing '{name}' "
                    f"templates (best {best:.3f} < {MIN_MATCH_EXISTING})")

    dest.mkdir(parents=True, exist_ok=True)
    for p in image_paths:
        target = dest / p.name
        if target.exists():
            raise EnrollmentError(f"REJECTED: {target} already exists")
        shutil.copy2(p, target)
    log.info("copied %d image(s) -> %s", len(image_paths), dest)

    # embedding cache is stale for this student — drop their entry so the
    # gallery rebuild re-embeds them (others still come from cache)
    cache = config.CACHE_DIR / "embeddings_phone_enroll.npz"
    if cache.exists():
        z = dict(np.load(cache).items())
        z.pop(name, None)
        np.savez(cache, **z)

    gallery.build_gallery()
    log.info("enrollment complete: %s", name)


def remove(name: str) -> None:
    src = config.PHONE_ENROLL_DIR / name
    if not src.exists():
        raise EnrollmentError(f"no enrollment folder for '{name}'")
    # move aside rather than delete — raw data stays recoverable
    aside = config.RAW_SESSIONS_DIR / "unenrolled" / name
    aside.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(aside))
    cache = config.CACHE_DIR / "embeddings_phone_enroll.npz"
    if cache.exists():
        z = dict(np.load(cache).items())
        z.pop(name, None)
        np.savez(cache, **z)
    gallery.build_gallery()
    log.info("unenrolled %s (images moved to %s)", name, aside)


if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        if len(args) >= 2 and args[0] == "--remove":
            remove(args[1])
        elif len(args) >= 2:
            enroll(args[0], [Path(a) for a in args[1:]])
        else:
            raise SystemExit(__doc__)
    except EnrollmentError as e:
        raise SystemExit(str(e))
