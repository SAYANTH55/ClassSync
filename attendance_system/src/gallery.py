"""
Roster + multi-template ArcFace gallery + open-set recogniser (deployed path)
=============================================================================

Three responsibilities, one file (they share every data structure):

1. ROSTER — the canonical list of enrolled students. Derived from the
   phone-enrollment folder names and persisted to ``data/labels/roster.csv``
   as ``student_id,name`` (S01, S02, ... alphabetical on first build).
   ID assignments are STABLE: rebuilding keeps existing ids and only appends
   new students. Every later artifact (gallery, attendance log, evaluation)
   keys on ``student_id``, so a spelling fix in a name never breaks joins.

2. GALLERY — for each enrolled student, ALL phone-image ArcFace embeddings
   are kept as templates (multi-template enrollment). Matching scores a probe
   by the MAX cosine over a student's templates: pose/expression variation in
   enrollment then works in our favour rather than being averaged away.
   Saved to ``data/processed/gallery.npz`` (arrays keyed by student_id, plus
   an ``__order__`` index) — derived data, rebuildable at any time.

3. RECOGNIZER — open-set identification:
       score(probe, student) = max cosine over that student's templates
       identify(probe)       = argmax student if its score >= threshold,
                               else UNKNOWN (open-set rejection)
   The threshold is chosen empirically by the DSLR cross-device evaluation
   (evaluate_dslr.py), not guessed.

Viva concepts: open-set vs closed-set identification; multi-template
enrollment vs single centroid; cosine similarity in angular space; why the
threshold must be calibrated on cross-device data (the deployment gap).

Usage:
    python src/gallery.py build      # roster.csv + gallery.npz from phone data
    python src/gallery.py show       # print roster + gallery summary
    from gallery import Recognizer
    rec = Recognizer(); rec.identify(embedding)  # -> Match | None
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("gallery")


# ---- roster ----------------------------------------------------------------
def load_roster() -> dict[str, str]:
    """roster.csv -> {student_id: name} (empty dict if not built yet)."""
    if not config.ROSTER_CSV.exists():
        return {}
    with open(config.ROSTER_CSV, newline="", encoding="utf-8") as f:
        return {r["student_id"]: r["name"] for r in csv.DictReader(f)}


def build_roster() -> dict[str, str]:
    """Create/refresh roster.csv from phone_enroll folder names.

    Existing id->name assignments are preserved; new students are appended
    with the next free id. Returns {student_id: name}.
    """
    names = sorted(d.name for d in config.PHONE_ENROLL_DIR.iterdir()
                   if d.is_dir() and any(d.iterdir()))
    roster = load_roster()
    known = set(roster.values())
    next_num = 1 + max((int(sid[1:]) for sid in roster), default=0)
    for name in names:
        if name not in known:
            roster[f"S{next_num:02d}"] = name
            next_num += 1
    dropped = known - set(names)
    if dropped:  # never silently unenroll — surface it, keep the ids reserved
        log.warning("roster names no longer in phone_enroll: %s", sorted(dropped))

    config.ROSTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ROSTER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "name"])
        for sid in sorted(roster):
            w.writerow([sid, roster[sid]])
    log.info("roster: %d students -> %s", len(roster), config.ROSTER_CSV)
    return roster


# ---- gallery build ---------------------------------------------------------
def build_gallery() -> None:
    """Embed every phone enrollment image; save multi-template gallery."""
    roster = build_roster()
    name_to_id = {v: k for k, v in roster.items()}

    # reuse the audit-phase embedding cache when present (identical pipeline)
    cache = config.CACHE_DIR / "embeddings_phone_enroll.npz"
    cached = dict(np.load(cache).items()) if cache.exists() else {}

    fb = None
    arrays: dict[str, np.ndarray] = {}
    for name, sid in sorted(name_to_id.items()):
        if not (config.PHONE_ENROLL_DIR / name).is_dir():
            # roster keeps the id reserved (old logs stay valid) but an
            # unenrolled student contributes no gallery templates
            log.info("skipping %s (%s): no enrollment folder", sid, name)
            continue
        if name in cached:
            arrays[sid] = cached[name].astype(np.float32)
            continue
        if fb is None:  # lazy — only load models if the cache missed
            from detect_embed import FaceBackend
            fb = FaceBackend()
        embs = []
        for p in sorted((config.PHONE_ENROLL_DIR / name).iterdir()):
            if p.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            e = fb.embed_path(p)
            if e is None:
                log.warning("no face in %s/%s — skipped", name, p.name)
                continue
            embs.append(e)
        if not embs:
            raise SystemExit(f"student '{name}' has zero usable images")
        arrays[sid] = np.stack(embs).astype(np.float32)

    order = np.array(sorted(arrays), dtype="U8")
    config.GALLERY_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.GALLERY_NPZ, __order__=order, **arrays)
    n_tpl = sum(len(a) for a in arrays.values())
    log.info("gallery: %d students, %d templates (%d-d) -> %s",
             len(arrays), n_tpl, config.EMBED_DIM, config.GALLERY_NPZ)


# ---- recognizer ------------------------------------------------------------
@dataclass
class Match:
    """Result of identifying one probe embedding."""
    student_id: str
    name: str
    score: float              # max cosine over the student's templates
    second_id: str            # runner-up (margin diagnostics)
    second_score: float

    @property
    def margin(self) -> float:
        return self.score - self.second_score


class Recognizer:
    """Open-set identifier over the multi-template gallery."""

    def __init__(self, threshold: float = 0.0):
        if not config.GALLERY_NPZ.exists():
            raise SystemExit("gallery not built — run: python src/gallery.py build")
        z = np.load(config.GALLERY_NPZ)
        self.ids: list[str] = [str(s) for s in z["__order__"]]
        self.templates = [z[sid] for sid in self.ids]      # ragged: (n_i, 512)
        self.roster = load_roster()
        self.threshold = threshold
        log.info("recognizer: %d students, threshold=%.3f",
                 len(self.ids), threshold)

    def scores(self, emb: np.ndarray) -> np.ndarray:
        """Per-student max-cosine scores for one probe embedding (512,)."""
        return np.array([float(np.max(t @ emb)) for t in self.templates])

    def identify(self, emb: np.ndarray) -> Match | None:
        """Best match if it clears the open-set threshold, else None."""
        s = self.scores(emb)
        i, j = np.argsort(s)[-1], np.argsort(s)[-2]
        m = Match(self.ids[i], self.roster[self.ids[i]], float(s[i]),
                  self.ids[j], float(s[j]))
        return m if m.score >= self.threshold else None


# ---- CLI -------------------------------------------------------------------
def _show() -> None:
    roster = load_roster()
    z = np.load(config.GALLERY_NPZ)
    print(f"\n{'id':5s} {'name':16s} templates")
    for sid in [str(s) for s in z["__order__"]]:
        print(f"{sid:5s} {roster[sid]:16s} {len(z[sid])}")
    print(f"\n{len(roster)} students, "
          f"{sum(len(z[str(s)]) for s in z['__order__'])} templates")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build_gallery, "show": _show}[cmd]()
