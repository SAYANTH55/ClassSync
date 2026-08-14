"""
Per-identity performance analysis of the deployed recogniser (bias audit)
=========================================================================

Fairness question: does the system work EQUALLY WELL for every student, or
do aggregate numbers (100% rank-1) hide weak individuals? Aggregates cannot
answer this — per-identity analysis can.

Method: the full probe-vs-gallery score matrix S[probe, student] (max cosine
over templates, as deployed). From it, per enrolled student with a DSLR probe:

  * genuine score      — their probe vs their own templates
  * margin             — genuine minus best wrong-student score (decision
                         safety margin; the number that must stay above the
                         threshold band for reliability)
  * lamb score         — highest score any OTHER student's probe achieves
                         against THEIR templates (how imitable they are)
  * wolf score         — highest score THEIR probe achieves against another
                         student's templates (how much they imperil others)

This is the classic Doddington-zoo framing (goats = low genuine, lambs =
easily imitated, wolves = good impersonators), the standard label-free
per-identity bias audit in biometrics.

Demographics note (for the write-up): the cohort is predominantly South
Asian, so this audit cannot certify cross-demographic fairness — and NO
demographic attributes are inferred from faces or names here (that would
itself be a biased measurement). A subgroup analysis would require
self-reported attributes with consent; state this as a limitation.

Outputs: reports/bias_audit/ (per_student.csv, bias_audit.png, console
summary). Pure numpy over cached embeddings — reruns instantly.

Usage:
    python src/analyze_bias.py        # face311 env (embeds probes once)
"""

from __future__ import annotations

import csv
import logging

import numpy as np

import config
from gallery import Recognizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("analyze_bias")

OUT_DIR = config.REPORTS_DIR / "bias_audit"
PROBE_CACHE = config.CACHE_DIR / "embeddings_dslr_probes.npz"


def probe_embeddings() -> dict[str, np.ndarray]:
    """Labelled-DSLR probe embeddings, cached after the first run."""
    if PROBE_CACHE.exists():
        z = np.load(PROBE_CACHE)
        return {k: z[k] for k in z.files}
    import preprocessing as pp
    from detect_embed import FaceBackend
    fb = FaceBackend()
    out = {}
    for p in config.session_images("dslr_labelled"):
        e = fb.embed_best(pp.load_image_upright(p))
        if e is None:
            log.warning("no face: %s", p.name)
            continue
        out[p.stem] = e
    PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PROBE_CACHE, **out)
    log.info("cached %d probe embeddings -> %s", len(out), PROBE_CACHE)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = Recognizer()
    name_of = rec.roster
    id_of = {v: k for k, v in name_of.items()}
    probes = probe_embeddings()

    # score matrix over enrolled-with-probe students only
    students = [name_of[sid] for sid in rec.ids]          # gallery order
    S = {stem: rec.scores(e) for stem, e in probes.items()}

    rows = []
    enrolled_stems = [s for s in probes if s in id_of]
    for stem in enrolled_stems:
        gi = students.index(stem)
        genuine = float(S[stem][gi])
        wrong = np.delete(S[stem], gi)
        margin = genuine - float(wrong.max())
        lamb = max(float(S[o][gi]) for o in enrolled_stems if o != stem)
        wolf = float(wrong.max())
        rows.append({"student_id": id_of[stem], "name": stem,
                     "genuine": round(genuine, 4), "margin": round(margin, 4),
                     "lamb": round(lamb, 4), "wolf": round(wolf, 4),
                     "n_templates": len(rec.templates[rec.ids.index(id_of[stem])])})
    rows.sort(key=lambda r: r["margin"])

    with open(OUT_DIR / "per_student.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    g = np.array([r["genuine"] for r in rows])
    m = np.array([r["margin"] for r in rows])

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    order = np.argsort([r["genuine"] for r in rows])
    names_sorted = [rows[i]["name"] for i in order]
    ax1.barh(range(len(rows)), [rows[i]["genuine"] for i in order],
             color="tab:blue", alpha=0.8, label="genuine score")
    ax1.barh(range(len(rows)), [rows[i]["wolf"] for i in order],
             color="tab:orange", alpha=0.8, label="best wrong-student score")
    ax1.axvline(config.RECOG_THRESHOLD, ls="--", c="k", lw=1,
                label=f"tau = {config.RECOG_THRESHOLD}")
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels(names_sorted, fontsize=6)
    ax1.set_xlabel("max cosine")
    ax1.set_title("per-student genuine vs best-impostor (sorted)")
    ax1.legend(fontsize=8)
    ax2.scatter(g, [r["lamb"] for r in rows], s=25)
    for r in rows[:5]:                       # annotate the 5 weakest margins
        ax2.annotate(r["name"], (r["genuine"], r["lamb"]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    ax2.axvline(config.RECOG_THRESHOLD, ls="--", c="k", lw=1)
    ax2.set_xlabel("genuine score (goat axis: low = hard to match)")
    ax2.set_ylabel("lamb score (high = easily imitated)")
    ax2.set_title("Doddington zoo scatter")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bias_audit.png", dpi=150)

    # ---- summary -----------------------------------------------------------
    print("\n================= PER-IDENTITY BIAS AUDIT =================")
    print(f"students audited        : {len(rows)} "
          f"(enrolled with a DSLR probe)")
    print(f"genuine score           : mean {g.mean():.3f}  sd {g.std():.3f}  "
          f"min {g.min():.3f}  max {g.max():.3f}")
    print(f"decision margin         : mean {m.mean():.3f}  sd {m.std():.3f}  "
          f"min {m.min():.3f}")
    print(f"all margins positive    : {bool((m > 0).all())}")
    print(f"all genuine above tau   : {bool((g > config.RECOG_THRESHOLD).all())}")
    print("\nweakest 5 by margin:")
    for r in rows[:5]:
        print(f"  {r['student_id']} {r['name']:14s} genuine {r['genuine']:.3f}"
              f"  margin {r['margin']:.3f}  lamb {r['lamb']:.3f}")
    print(f"outputs -> {OUT_DIR}")
    print("===========================================================")


if __name__ == "__main__":
    main()
