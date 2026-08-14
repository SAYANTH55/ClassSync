"""
Cross-device evaluation: DSLR probes vs the phone-built gallery
===============================================================

The headline experiment for the deployed recogniser. The gallery is built
exclusively from phone self-captures; the probes are studio DSLR portraits
taken ~4 months earlier on different hardware. This is exactly the deployment
gap an attendance kiosk faces (enrollment device != capture device), so the
numbers here are honest estimates of field performance.

Probe set (from the labelled DSLR session):
  * GENUINE probes  — files whose stem is an enrolled student name
                      (closed-set question: is rank-1 the right person?)
  * IMPOSTOR probes — a DSLR-only student (no phone enrollment) + the DSC* unknowns
                      (open-set question: are they rejected at threshold?)

Reported:
  * rank-1 identification accuracy over genuine probes
  * genuine / impostor score distributions (max cosine vs best gallery match)
  * threshold sweep: TAR (genuine accepted AND correct) vs FAR (impostor
    accepted) at each tau; recommended tau = highest TAR with FAR = 0,
    plus the margin to the nearest impostor score
  * per-probe CSV + distribution/sweep plot in reports/eval_dslr/

Viva concepts: open-set identification; TAR/FAR trade-off; why threshold
calibration must use cross-device probes; rank-1 vs verification metrics.

Usage:
    python src/evaluate_dslr.py
"""

from __future__ import annotations

import csv
import logging

import numpy as np

import config
import preprocessing as pp
from detect_embed import FaceBackend
from gallery import Recognizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("evaluate_dslr")

OUT_DIR = config.REPORTS_DIR / "eval_dslr"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = Recognizer()                       # threshold applied later via sweep
    name_to_id = {v: k for k, v in rec.roster.items()}
    fb = FaceBackend()

    rows = []
    for path in config.session_images("dslr_labelled"):
        emb = fb.embed_best(pp.load_image_upright(path))
        if emb is None:                      # audit says this cannot happen
            log.error("no face detected in %s", path.name)
            continue
        m = rec.identify(emb)                # threshold 0 -> always a Match
        true_id = name_to_id.get(path.stem)  # None -> impostor probe
        rows.append({
            "file": path.name,
            "true_id": true_id or "",
            "kind": "genuine" if true_id else "impostor",
            "pred_id": m.student_id,
            "pred_name": m.name,
            "score": round(m.score, 4),
            "margin": round(m.margin, 4),
            "correct": (m.student_id == true_id) if true_id else "",
        })

    genuine = [r for r in rows if r["kind"] == "genuine"]
    impostor = [r for r in rows if r["kind"] == "impostor"]
    g_scores = np.array([r["score"] for r in genuine])
    i_scores = np.array([r["score"] for r in impostor])
    n_correct = sum(r["correct"] is True for r in genuine)
    errors = [r for r in genuine if r["correct"] is False]

    # ---- threshold sweep (open-set): TAR vs FAR ----------------------------
    taus = np.round(np.arange(0.20, 0.751, 0.005), 3)
    sweep = []
    for t in taus:
        tar = float(np.mean([(r["score"] >= t) and (r["correct"] is True)
                             for r in genuine]))
        far = float(np.mean(i_scores >= t)) if len(i_scores) else 0.0
        sweep.append((t, tar, far))
    zero_far = [(t, tar) for t, tar, far in sweep if far == 0.0]
    tau_star, tar_star = max(zero_far, key=lambda x: x[1]) if zero_far else (None, None)
    # operating point: midpoint of the empty band between the distributions
    band_lo, band_hi = float(i_scores.max()), float(g_scores.min())
    tau_mid = round((band_lo + band_hi) / 2, 3) if band_lo < band_hi else None

    # ---- outputs -----------------------------------------------------------
    with open(OUT_DIR / "per_probe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(OUT_DIR / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["tau", "TAR", "FAR"]); w.writerows(sweep)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0.0, 1.0, 0.025)
    ax1.hist(g_scores, bins=bins, alpha=0.7, label=f"genuine (n={len(g_scores)})")
    ax1.hist(i_scores, bins=bins, alpha=0.7, label=f"impostor (n={len(i_scores)})")
    if tau_mid:
        ax1.axvline(tau_mid, ls="--", c="k", lw=1, label=f"tau = {tau_mid}")
    ax1.set(xlabel="max cosine vs gallery", ylabel="probes",
            title="DSLR probes vs phone gallery")
    ax1.legend()
    ax2.plot(taus, [s[1] for s in sweep], label="TAR (correct accept)")
    ax2.plot(taus, [s[2] for s in sweep], label="FAR (impostor accept)")
    if tau_mid:
        ax2.axvline(tau_mid, ls="--", c="k", lw=1)
    ax2.set(xlabel="threshold tau", ylabel="rate", title="open-set sweep")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "evaluation.png", dpi=150)

    # ---- summary -----------------------------------------------------------
    print("\n================ DSLR EVALUATION ================")
    print(f"gallery students        : {len(rec.ids)}")
    print(f"genuine probes          : {len(genuine)}")
    print(f"impostor probes         : {len(impostor)}")
    print(f"rank-1 accuracy         : {n_correct}/{len(genuine)}"
          f"  ({100 * n_correct / len(genuine):.1f}%)")
    for r in errors:
        print(f"  MISID: {r['file']} -> {r['pred_name']} ({r['score']:.3f})")
    print(f"genuine scores          : min {g_scores.min():.3f}, "
          f"median {np.median(g_scores):.3f}, max {g_scores.max():.3f}")
    print(f"impostor scores         : min {i_scores.min():.3f}, "
          f"median {np.median(i_scores):.3f}, max {i_scores.max():.3f}")
    if tau_mid:
        print(f"separation band         : [{band_lo:.3f}, {band_hi:.3f}] "
              f"(width {band_hi - band_lo:.3f})")
        print(f"recommended tau         : {tau_mid}  "
              f"(TAR {tar_star:.3f}, FAR 0.000 at tau={tau_star})")
    else:
        print("distributions OVERLAP — no zero-FAR threshold exists")
    print(f"outputs -> {OUT_DIR}")
    print("=================================================")


if __name__ == "__main__":
    main()
