"""
Classical baselines on the identical protocol: Eigenfaces + LBPH
================================================================

The comparative-study companion to evaluate_dslr.py. Both classical methods
are evaluated on EXACTLY the same split as the deployed ArcFace recogniser:

    gallery = all phone enrollment images (multi-template per student)
    probes  = labelled DSLR portraits (46 genuine + 7 impostor)

Fairness controls (so the comparison isolates the REPRESENTATION):
  * identical face detection + landmark alignment (SCRFD + 5-pt norm_crop
    to 112x112) for every method — classical methods see the same aligned
    crops ArcFace sees, just grayscaled;
  * identical multi-template matching rule (best match over a student's
    gallery images);
  * identical metrics: rank-1 accuracy, genuine/impostor separation,
    zero-FAR threshold sweep.

Methods
-------
* EIGENFACES (Turk & Pentland 1991): PCA on the flattened gallery crops;
  identity scored by cosine similarity in the eigenface subspace.
  Holistic, linear, illumination-sensitive — the historical baseline.
* LBPH (Ahonen et al. 2006): Local Binary Pattern histograms per spatial
  grid cell, chi-square distance. Local texture, more illumination-robust
  than PCA, still handcrafted. Score = -distance (higher = better) so the
  same sweep machinery applies.

Aligned grayscale crops are cached to data/cache/crops_gray.npz (one-off;
detection reruns only if the cache is missing).

Usage:
    python src/evaluate_classical.py
"""

from __future__ import annotations

import csv
import logging

import cv2
import numpy as np

import config
import preprocessing as pp
from gallery import load_roster

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("evaluate_classical")

OUT_DIR = config.REPORTS_DIR / "eval_classical"
CROP_CACHE = config.CACHE_DIR / "crops_gray.npz"
CROP = 112
PCA_COMPONENTS = 150          # < n_gallery-1; retains ~all useful variance


# ---- aligned grayscale crops (shared preprocessing) ------------------------
def build_crop_cache() -> dict[str, np.ndarray]:
    """SCRFD-align every gallery+probe image once; cache as uint8 gray."""
    if CROP_CACHE.exists():
        z = np.load(CROP_CACHE)
        log.info("aligned crops loaded from cache (%d)", len(z.files))
        return dict(z.items())

    from detect_embed import FaceBackend
    from insightface.utils.face_align import norm_crop
    fb = FaceBackend()
    out: dict[str, np.ndarray] = {}

    def crop_of(path) -> np.ndarray | None:
        img = pp.load_image_upright(path)
        faces = fb.detect(img)
        if not faces:
            return None
        aligned = norm_crop(img, faces[0].landmarks, image_size=CROP)
        return cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)

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

    CROP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CROP_CACHE, **out)
    log.info("aligned %d crops -> %s", len(out), CROP_CACHE)
    return out


# ---- shared evaluation harness --------------------------------------------
def evaluate(method: str, score_probe, gallery_names: list[str],
             probes: dict[str, np.ndarray], roster_names: set[str]) -> dict:
    """Run one method through the common rank-1 + sweep protocol.

    ``score_probe(crop) -> np.ndarray`` of per-student scores aligned with
    ``gallery_names`` (higher = more similar).
    """
    rows, g_scores, i_scores = [], [], []
    for stem, crop in sorted(probes.items()):
        s = score_probe(crop)
        i = int(np.argmax(s))
        pred, score = gallery_names[i], float(s[i])
        genuine = stem in roster_names
        (g_scores if genuine else i_scores).append(score)
        rows.append({"file": stem, "kind": "genuine" if genuine else "impostor",
                     "pred": pred, "score": round(score, 4),
                     "correct": (pred == stem) if genuine else ""})
    genuine_rows = [r for r in rows if r["kind"] == "genuine"]
    n_correct = sum(r["correct"] is True for r in genuine_rows)
    g, i = np.array(g_scores), np.array(i_scores)

    # zero-FAR sweep on this method's own score scale
    taus = np.linspace(min(i.min(), g.min()), g.max(), 400)
    best_tar = 0.0
    for t in taus:
        if float(np.mean(i >= t)) == 0.0:
            # TAR = accepted AND correct
            tar = float(np.mean([(r["score"] >= t) and (r["correct"] is True)
                                 for r in genuine_rows]))
            best_tar = max(best_tar, tar)
    res = {"method": method, "rank1": n_correct, "n_genuine": len(genuine_rows),
           "rank1_pct": 100 * n_correct / len(genuine_rows),
           "tar_at_far0": best_tar,
           "g_min": float(g.min()), "g_med": float(np.median(g)),
           "i_med": float(np.median(i)), "i_max": float(i.max()),
           "separated": bool(i.max() < g.min()), "rows": rows}
    log.info("%-11s rank-1 %d/%d (%.1f%%)  TAR@FAR=0: %.3f  separated: %s",
             method, n_correct, len(genuine_rows), res["rank1_pct"],
             best_tar, res["separated"])
    return res


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crops = build_crop_cache()
    roster_names = set(load_roster().values())

    # gallery arrays grouped per student, probe dict by stem
    gal: dict[str, list[np.ndarray]] = {}
    for key, c in crops.items():
        parts = key.split("/")
        if parts[0] == "gallery":
            gal.setdefault(parts[1], []).append(c)
    probes = {k.split("/", 1)[1]: c for k, c in crops.items()
              if k.startswith("probe/")}
    gallery_names = sorted(gal)
    log.info("gallery: %d students / %d crops; probes: %d",
             len(gal), sum(map(len, gal.values())), len(probes))

    results = []

    # ---- Eigenfaces --------------------------------------------------------
    from sklearn.decomposition import PCA
    X, owner = [], []
    for n in gallery_names:
        for c in gal[n]:
            X.append(c.astype(np.float32).ravel() / 255.0)
            owner.append(n)
    X = np.stack(X)
    pca = PCA(n_components=min(PCA_COMPONENTS, len(X) - 1),
              random_state=config.RANDOM_SEED).fit(X)
    G = pca.transform(X)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    owner = np.array(owner)

    def eigen_score(crop: np.ndarray) -> np.ndarray:
        v = pca.transform(crop.astype(np.float32).ravel()[None] / 255.0)[0]
        v /= np.linalg.norm(v)
        sims = G @ v
        return np.array([float(sims[owner == n].max()) for n in gallery_names])

    results.append(evaluate("Eigenfaces", eigen_score, gallery_names,
                            probes, roster_names))

    # ---- LBPH (own implementation; cv2.face was removed in OpenCV 5) -------
    # Standard formulation (Ahonen et al. 2006): 8-neighbour radius-1 LBP
    # code per pixel, 256-bin histogram per cell of an 8x8 spatial grid,
    # concatenated; chi-square distance between histograms. Score = -distance.
    def lbp_hist(img: np.ndarray) -> np.ndarray:
        f = img.astype(np.int16)
        c = f[1:-1, 1:-1]
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                  (1, 1), (1, 0), (1, -1), (0, -1)]
        code = np.zeros_like(c, dtype=np.uint8)
        for bit, (dy, dx) in enumerate(shifts):
            nb = f[1 + dy:f.shape[0] - 1 + dy, 1 + dx:f.shape[1] - 1 + dx]
            code |= ((nb >= c) << bit).astype(np.uint8)
        cells = []
        step = code.shape[0] // 8
        for gy in range(8):
            for gx in range(8):
                cell = code[gy * step:(gy + 1) * step,
                            gx * step:(gx + 1) * step]
                h = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
                cells.append(h / max(h.sum(), 1.0))
        return np.concatenate(cells)                      # (8*8*256,)

    lbph_gal = {n: np.stack([lbp_hist(c) for c in gal[n]])
                for n in gallery_names}

    def lbph_score(crop: np.ndarray) -> np.ndarray:
        h = lbp_hist(crop)
        out = []
        for n in gallery_names:
            H = lbph_gal[n]
            chi2 = ((H - h) ** 2 / (H + h + 1e-10)).sum(axis=1)
            out.append(-float(chi2.min()))                # best template
        return np.array(out)

    results.append(evaluate("LBPH", lbph_score, gallery_names,
                            probes, roster_names))

    # ---- ArcFace row (from the deployed evaluation, same protocol) ---------
    arc_csv = config.REPORTS_DIR / "eval_dslr" / "per_probe.csv"
    with open(arc_csv, newline="", encoding="utf-8") as f:
        arc = list(csv.DictReader(f))
    g = np.array([float(r["score"]) for r in arc if r["kind"] == "genuine"])
    i = np.array([float(r["score"]) for r in arc if r["kind"] == "impostor"])
    nc = sum(r["correct"] == "True" for r in arc if r["kind"] == "genuine")
    results.append({"method": "ArcFace", "rank1": nc, "n_genuine": len(g),
                    "rank1_pct": 100 * nc / len(g), "tar_at_far0": 1.0,
                    "g_min": float(g.min()), "g_med": float(np.median(g)),
                    "i_med": float(np.median(i)), "i_max": float(i.max()),
                    "separated": bool(i.max() < g.min()), "rows": []})

    # ---- outputs -----------------------------------------------------------
    for r in results:
        if r["rows"]:
            with open(OUT_DIR / f"per_probe_{r['method'].lower()}.csv", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(r["rows"][0]))
                w.writeheader(); w.writerows(r["rows"])
    with open(OUT_DIR / "comparison.csv", "w", newline="",
              encoding="utf-8") as f:
        keys = ["method", "rank1", "n_genuine", "rank1_pct", "tar_at_far0",
                "g_min", "g_med", "i_med", "i_max", "separated"]
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r[k] for k in keys])

    print("\n=============== COMPARATIVE RESULTS (identical protocol) ===============")
    print(f"{'method':12s} {'rank-1':>10s} {'TAR@FAR=0':>10s} "
          f"{'genuine med':>12s} {'impostor max':>13s} {'separated':>10s}")
    for r in results:
        print(f"{r['method']:12s} {r['rank1']:>3d}/{r['n_genuine']}"
              f" ({r['rank1_pct']:5.1f}%) {r['tar_at_far0']:>10.3f}"
              f" {r['g_med']:>12.3f} {r['i_max']:>13.3f}"
              f" {str(r['separated']):>10s}")
    print(f"outputs -> {OUT_DIR}")
    print("========================================================================")


if __name__ == "__main__":
    main()
