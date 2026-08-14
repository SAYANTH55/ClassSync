"""
DSLR dataset engineering review (inspection only — no architecture changes)
===========================================================================

Reviews the manually renamed DSLR portrait set against (a) the pristine
originals kept in ``data/raw`` and (b) the phone enrollment dataset, then
dry-runs the deployed SCRFD+ArcFace stack on it.

Checks performed
----------------
1. Integrity: readability, SHA-256 hash-match of every renamed file to a
   pristine original (renames don't alter bytes), duplicates, files whose
   names are still camera names, and which originals are missing entirely.
2. Identity reconciliation vs phone enrollment folders: exact matches,
   near-miss spellings (difflib), phone students with no DSLR portrait,
   DSLR names with no phone folder.
3. Detection/embedding dry run: SCRFD on every DSLR image (score, faces
   found, largest-face choice), ArcFace embedding, and — for name-matched
   students — cross-device similarity: DSLR embedding vs that student's
   phone templates (genuine) and vs the best other student (impostor).
   This previews exactly how useful DSLR data is as probe or gallery.
4. Aligned 112x112 crops (insightface norm_crop — same alignment the
   embeddings use) saved as a montage for visual QA.

Outputs: reports/dslr_check/ (report.csv, aligned_montage.jpg) and cached
embeddings in data/cache/ so later phases don't recompute.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image

import config
import preprocessing as pp
from detect_embed import FaceBackend, cosine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("dslr_check")

RENAMED_DIR = Path(r"E:\Attendance monitoring system"
                   r"\Attendance_monitoring_system-20260718T093929Z-1-001"
                   r"\Attendance_monitoring_system")
PRISTINE_DIR = config.SESSIONS["dslr"]["images_dir"]
ENROLL_DIR = config.RAW_SESSIONS_DIR / "phone_enroll"
OUT_DIR = config.REPORTS_DIR / "dslr_check"
CACHE_DIR = config.DATA_DIR / "cache"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phone_embeddings(fb: FaceBackend) -> dict[str, np.ndarray]:
    """Largest-face ArcFace embedding for every phone image, cached to disk.

    Returns {student_name: (n_imgs, 512) array}.
    """
    cache = CACHE_DIR / "embeddings_phone_enroll.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        out = {}
        for key in z.files:
            out[key] = z[key]
        log.info("phone embeddings loaded from cache (%d students)", len(out))
        return out
    log.info("embedding all phone enrollment images (one-off, cached) ...")
    out: dict[str, list[np.ndarray]] = {}
    for sdir in sorted(d for d in ENROLL_DIR.iterdir() if d.is_dir()):
        embs = []
        for p in sorted(sdir.iterdir()):
            if p.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            e = fb.embed_best(pp.load_image_upright(p))
            if e is not None:
                embs.append(e)
        if embs:
            out[sdir.name] = np.stack(embs)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    log.info("cached -> %s", cache)
    return {k: np.asarray(v) for k, v in out.items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. integrity & hash reconciliation --------------------------------
    pristine = {sha256(p): p.name for p in sorted(PRISTINE_DIR.glob("*.jpeg"))}
    renamed = sorted(p for p in RENAMED_DIR.iterdir()
                     if p.suffix.lower() in config.IMAGE_EXTENSIONS)

    rows, seen_hashes, dup_pairs = [], {}, []
    for p in renamed:
        h = sha256(p)
        if h in seen_hashes:
            dup_pairs.append((seen_hashes[h], p.name))
        seen_hashes[h] = p.name
        rows.append({"file": p.name, "stem": p.stem, "hash": h,
                     "original": pristine.get(h, "NOT-IN-ORIGINALS"),
                     "is_camera_name": p.stem.upper().startswith("DSC")})
    matched_hashes = {r["hash"] for r in rows}
    missing_originals = sorted(v for k, v in pristine.items()
                               if k not in matched_hashes)

    # ---- 2. identity reconciliation ---------------------------------------
    phone_names = sorted(d.name for d in ENROLL_DIR.iterdir() if d.is_dir())
    phone_with_imgs = sorted(
        d.name for d in ENROLL_DIR.iterdir() if d.is_dir()
        and any(f.suffix.lower() in config.IMAGE_EXTENSIONS
                for f in d.iterdir()))
    named = [r for r in rows if not r["is_camera_name"]]
    dslr_names = [r["stem"] for r in named]

    exact = sorted(set(dslr_names) & set(phone_names))
    dslr_only, fuzzy = [], []
    for n in dslr_names:
        if n in phone_names:
            continue
        close = difflib.get_close_matches(n, phone_names, n=1, cutoff=0.75)
        (fuzzy if close else dslr_only).append((n, close[0] if close else ""))
    phone_missing_dslr = sorted(
        set(phone_with_imgs) - set(dslr_names)
        - {m for _, m in fuzzy if m})

    # ---- 3. SCRFD + ArcFace dry run ---------------------------------------
    fb = FaceBackend()
    gallery = phone_embeddings(fb)
    from insightface.utils import face_align

    crops, det_rows = [], []
    for r in rows:
        img = pp.load_image_upright(RENAMED_DIR / r["file"])
        faces = fb.detect(img)
        if not faces:
            det_rows.append({**r, "n_faces": 0, "det_score": "",
                             "genuine_cos": "", "impostor_cos": "",
                             "impostor_name": ""})
            continue
        f = faces[0]
        crop = face_align.norm_crop(img[:, :, ::-1], landmark=f.landmarks,
                                    image_size=112)[:, :, ::-1]
        crops.append((crop, r["stem"][:14]))

        # cross-device similarity (named files with a phone match only)
        gname = r["stem"] if r["stem"] in gallery else None
        if gname is None:
            close = difflib.get_close_matches(r["stem"], list(gallery), 1, 0.75)
            gname = close[0] if close else None
        genuine, imp_best, imp_name = "", "", ""
        if gname:
            genuine = round(max(cosine(f.embedding, e)
                                for e in gallery[gname]), 3)
            imp_best, imp_name = max(
                ((round(max(cosine(f.embedding, e) for e in embs), 3), n)
                 for n, embs in gallery.items() if n != gname),
                key=lambda t: t[0])
        det_rows.append({**r, "n_faces": len(faces),
                         "det_score": round(f.det_score, 3),
                         "genuine_cos": genuine,
                         "impostor_cos": imp_best, "impostor_name": imp_name})

    # ---- montage -----------------------------------------------------------
    cols, cell, pad = 9, 112, 5
    rowsn = -(-len(crops) // cols)
    sheet = Image.new("RGB", (cols * (cell + pad) + pad,
                              rowsn * (cell + 22 + pad) + pad), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for i, (arr, name) in enumerate(crops):
        x = pad + (i % cols) * (cell + pad)
        y = pad + (i // cols) * (cell + 22 + pad)
        sheet.paste(Image.fromarray(arr), (x, y))
        draw.text((x + 1, y + cell + 3), name, fill="black")
    sheet.save(OUT_DIR / "aligned_montage.jpg", quality=90)

    with open(OUT_DIR / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
        w.writeheader()
        w.writerows(det_rows)

    # ---- console report ----------------------------------------------------
    genuines = [r["genuine_cos"] for r in det_rows if r["genuine_cos"] != ""]
    impostors = [r["impostor_cos"] for r in det_rows if r["impostor_cos"] != ""]
    print("\n================= DSLR REVIEW =================")
    print(f"files in renamed dir      : {len(rows)} (originals: {len(pristine)})")
    print(f"hash-matched to originals : {sum(1 for r in rows if r['original'] != 'NOT-IN-ORIGINALS')}")
    print(f"not from original set     : {[r['file'] for r in rows if r['original'] == 'NOT-IN-ORIGINALS']}")
    print(f"originals missing entirely: {missing_originals}")
    print(f"duplicate contents        : {dup_pairs if dup_pairs else 'none'}")
    print(f"still camera-named        : {sorted(r['file'] for r in rows if r['is_camera_name'])}")
    print("---- identity reconciliation vs phone enrollment ----")
    print(f"exact name matches        : {len(exact)}")
    print(f"near-miss spellings       : {fuzzy}")
    print(f"DSLR-only names           : {[n for n, _ in dslr_only]}")
    print(f"phone students w/o DSLR   : {phone_missing_dslr}")
    print(f"phone folders w/o images  : {sorted(set(phone_names) - set(phone_with_imgs))}")
    print("---- SCRFD + ArcFace dry run ----")
    print(f"faces detected            : {sum(1 for r in det_rows if r['n_faces'])}/{len(det_rows)}")
    print(f"multi-face images         : {[r['file'] for r in det_rows if isinstance(r['n_faces'], int) and r['n_faces'] > 1]}")
    if genuines:
        print(f"genuine cos (same person, DSLR vs phone): "
              f"min {min(genuines)}, median {sorted(genuines)[len(genuines)//2]}, max {max(genuines)}")
        print(f"impostor cos (best wrong person)        : "
              f"min {min(impostors)}, median {sorted(impostors)[len(impostors)//2]}, max {max(impostors)}")
        overlap = [r["stem"] for r in det_rows
                   if r["genuine_cos"] != "" and r["impostor_cos"] != ""
                   and r["genuine_cos"] <= r["impostor_cos"]]
        print(f"identities where genuine <= impostor    : {overlap if overlap else 'NONE — perfect separation'}")
    print(f"report  -> {OUT_DIR / 'report.csv'}")
    print(f"montage -> {OUT_DIR / 'aligned_montage.jpg'}")
    print("===============================================")


if __name__ == "__main__":
    main()
