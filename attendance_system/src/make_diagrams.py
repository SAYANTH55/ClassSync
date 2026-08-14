"""
Phase 12 (started early) — Dissertation figures as code
=======================================================

Generates the system-architecture and flowchart figures used in the
dissertation. Figures are produced programmatically (matplotlib) so they are
version-controlled, restyleable, and always in sync with the actual design.

Outputs (300 dpi PNG) -> docs/diagrams/
    fig_system_architecture.png
    fig_preprocessing_flowchart.png
    fig_kiosk_inference_flowchart.png
    fig_detector_training_flowchart.png

Usage:
    python src/make_diagrams.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

import config

OUT = config.DOCS_DIR / "diagrams"

# ---- palette ----------------------------------------------------------------
C_PROC = "#dbe9f6"     # processing step
C_DATA = "#fdf2d0"     # data store / artifact
C_DECIDE = "#e8f6e8"   # decision
C_NOTE = "#f5e6f0"     # annotation / constraint note
C_EDGE = "#4a4a4a"
FS = 9


def box(ax, x, y, w, h, text, fc=C_PROC, fs=FS, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1,
                                edgecolor=C_EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def diamond(ax, cx, cy, w, h, text, fs=FS - 0.5):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2),
                          (cx + w / 2, cy), (cx, cy - h / 2)],
                         closed=True, linewidth=1,
                         edgecolor=C_EDGE, facecolor=C_DECIDE))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2, label="", curve=0.0, fs=FS - 1, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.1, color=C_EDGE, linestyle=ls,
                                 connectionstyle=f"arc3,rad={curve}"))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 1.2, label, ha="center", va="bottom",
                fontsize=fs, style="italic")


def canvas(w_in, h_in, xmax=100, ymax=100):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}")


# ---------------------------------------------------------------------------
# Figure 1 — system architecture
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = canvas(12.5, 7.2)
    ax.text(24, 97, "OFFLINE — enrollment & training", fontsize=11,
            ha="center", weight="bold")
    ax.text(80, 97, "ONLINE — kiosk attendance", fontsize=11,
            ha="center", weight="bold")
    ax.plot([51, 51], [2, 95], color="#bbbbbb", ls="--", lw=1)

    # offline column
    box(ax, 4, 84, 40, 8, "Data acquisition — multiple capture sessions\nsession 1: DSLR studio stills · sessions 2+: per-student\nsmartphone videos/stills → deterministic ingestion (3 fps)", C_DATA, FS - 0.5)
    box(ax, 4, 70, 40, 8, "Annotation (ground truth)\nface box + eye centres · identity labels\nfull manual (session 1) · stratified sample (phone)", C_PROC, FS - 0.5)
    box(ax, 4, 56, 40, 8, "Preprocessing module\nEXIF-upright · crop+margin · eye alignment\n· resize 112×112 (INTER_AREA)", C_PROC)
    box(ax, 4, 42, 40, 8, "Processed dataset + provenance manifest\nsession-disjoint train / val / test splits", C_DATA)
    box(ax, 4, 28, 19, 9, "Detector training\nHOG + linear SVM\n(hard-negative mining)", C_PROC, FS - 0.5)
    box(ax, 25, 28, 19, 9, "Recognition training\nfrom-scratch CNN\n(softmax over N students)", C_PROC, FS - 0.5)
    box(ax, 4, 14, 40, 7, "Trained artifacts:  detector weights · CNN weights\n· decision threshold τ (chosen on validation set)", C_DATA)

    arrow(ax, (24, 84), (24, 78))
    arrow(ax, (24, 70), (24, 64))
    arrow(ax, (24, 56), (24, 50))
    arrow(ax, (13.5, 42), (13.5, 37))
    arrow(ax, (34.5, 42), (34.5, 37))
    arrow(ax, (13.5, 28), (13.5, 21))
    arrow(ax, (34.5, 28), (34.5, 21))

    # online column
    box(ax, 58, 84, 36, 8, "Kiosk camera frame\n(one student at a time)", C_DATA)
    box(ax, 58, 70, 36, 8, "Face detection\nsliding window · own HOG+SVM · NMS", C_PROC)
    box(ax, 58, 56, 36, 8, "Preprocessing (same module as training)\ncrop → align → resize → scale to [0,1]", C_PROC)
    box(ax, 58, 42, 36, 8, "CNN inference\nclass probabilities for N students", C_PROC)
    box(ax, 58, 28, 36, 9, "Decision logic\nmax prob ≥ τ ?  ·  not yet marked today?\nelse: rejected & logged as unknown", C_PROC)
    box(ax, 58, 14, 36, 7, "Attendance ledger\nCSV / Excel export · GUI display · audit log", C_DATA)

    arrow(ax, (76, 84), (76, 78))
    arrow(ax, (76, 70), (76, 64))
    arrow(ax, (76, 56), (76, 50))
    arrow(ax, (76, 42), (76, 37))
    arrow(ax, (76, 28), (76, 21))

    # weights flow across lanes
    arrow(ax, (44, 17.5), (58, 31), label="deployed weights + τ", curve=-0.25)
    box(ax, 4, 2, 90, 8,
        "Constraint boundary: every learned parameter (SVM weights, CNN weights) is trained exclusively\n"
        "on data collected and annotated within this project — no pre-trained models anywhere in the pipeline.",
        C_NOTE, FS - 0.5)
    save(fig, "fig_system_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2 — preprocessing flowchart
# ---------------------------------------------------------------------------
def fig_preprocessing():
    fig, ax = canvas(8.5, 10.5)
    cx = 38
    box(ax, cx - 18, 92, 36, 5, "raw image (immutable, data/raw)", C_DATA)
    box(ax, cx - 18, 82, 36, 5, "EXIF orientation transpose\n(applied exactly once)")
    diamond(ax, cx, 71, 30, 10, "manual ground truth\navailable?")
    box(ax, 68, 74, 28, 6, "annotations.csv\nface box + eye centres", C_DATA, FS - 1)
    box(ax, 68, 62, 28, 7, "classical proposal (bg-seg)\nPROVISIONAL — flagged\nin manifest", C_NOTE, FS - 1)
    box(ax, cx - 18, 52, 36, 5, "expand box by margin (0.25) → square")
    diamond(ax, cx, 41, 30, 10, "eye centres\nknown?")
    box(ax, 68, 44, 28, 6, "similarity-transform\nalignment (eyes → canonical)", C_PROC, FS - 1)
    box(ax, 68, 33, 28, 6, "plain crop\n(replicate-padded)", C_PROC, FS - 1)
    box(ax, cx - 16, 22, 32, 6, "resize to 112×112\nINTER_AREA (anti-aliasing)")
    box(ax, cx - 16, 12, 32, 6, "lossless PNG +\nprovenance manifest row", C_DATA)
    box(ax, cx - 16, 2, 32, 6, "QA contact sheet\n(visual verification)", C_DATA)
    box(ax, 68, 12, 28, 10,
        "normalization note:\n[0,1] scaling at train time;\nmean/std from TRAIN split\nonly (no leakage)", C_NOTE, FS - 1)

    arrow(ax, (cx, 92), (cx, 87.5))
    arrow(ax, (cx, 82), (cx, 76.5))
    arrow(ax, (cx + 15, 71), (68, 76), label="yes", curve=-0.15)
    arrow(ax, (cx + 15, 71), (68, 66), label="no", curve=0.15)
    arrow(ax, (68, 74), (cx + 4, 57.5), curve=-0.1)
    arrow(ax, (68, 64), (cx + 4, 57), curve=0.1)
    arrow(ax, (cx, 52), (cx, 46.5))
    arrow(ax, (cx + 15, 41), (68, 47), label="yes", curve=-0.15)
    arrow(ax, (cx + 15, 41), (68, 36), label="no", curve=0.15)
    arrow(ax, (68, 45), (cx + 4, 28), curve=-0.1)
    arrow(ax, (68, 35), (cx + 4, 27), curve=0.1)
    arrow(ax, (cx, 22), (cx, 18.5))
    arrow(ax, (cx, 12), (cx, 8.5))
    save(fig, "fig_preprocessing_flowchart.png")


# ---------------------------------------------------------------------------
# Figure 3 — kiosk inference loop
# ---------------------------------------------------------------------------
def fig_kiosk():
    fig, ax = canvas(8.5, 10.5)
    cx = 40
    box(ax, cx - 15, 93, 30, 5, "capture kiosk camera frame", C_DATA)
    box(ax, cx - 15, 84, 30, 5, "own HOG+SVM detector\n(pyramid · sliding window · NMS)")
    diamond(ax, cx, 74, 28, 9, "face found?")
    box(ax, cx - 15, 62, 30, 5, "preprocess crop\n(same module as training)")
    box(ax, cx - 17, 53, 34, 5, "CNN forward pass → probabilities")
    diamond(ax, cx, 43, 30, 9, "max probability ≥ τ ?")
    diamond(ax, cx, 29, 30, 9, "already marked\nthis session?")
    box(ax, cx - 15, 15, 30, 6, "append to attendance ledger\n(id, date, time, confidence)", C_DATA)
    box(ax, cx - 15, 5, 30, 5, "GUI feedback → next student", C_PROC)
    box(ax, 74, 40.5, 22, 7, "log rejected capture\nas 'unknown' (audit)", C_NOTE, FS - 1)
    box(ax, 74, 26.5, 22, 7, "ignore duplicate\n(idempotent marking)", C_NOTE, FS - 1)

    arrow(ax, (cx, 93), (cx, 89.5))
    arrow(ax, (cx, 84), (cx, 78.5))
    arrow(ax, (cx, 69.5), (cx, 67), label="yes")
    ax.plot([cx + 14, 93, 93], [74, 74, 95.5], color=C_EDGE, lw=1.1)
    ax.text(70, 75.2, "no", fontsize=FS - 1, style="italic")
    arrow(ax, (93, 95.5), (55, 95.5))
    arrow(ax, (cx, 62), (cx, 58.5))
    arrow(ax, (cx, 53), (cx, 47.5))
    arrow(ax, (cx, 38.5), (cx, 33.5), label="yes")
    arrow(ax, (cx + 15, 43), (74, 44), label="no")
    arrow(ax, (cx, 24.5), (cx, 21), label="no")
    arrow(ax, (cx + 15, 29), (74, 30), label="yes")
    arrow(ax, (cx, 15), (cx, 10.5))
    arrow(ax, (cx - 15, 7.5), (12, 7.5))
    arrow(ax, (12, 7.5), (12, 95.5))
    arrow(ax, (12, 95.5), (cx - 15, 95.5))
    save(fig, "fig_kiosk_inference_flowchart.png")


# ---------------------------------------------------------------------------
# Figure 4 — detector training with hard-negative mining
# ---------------------------------------------------------------------------
def fig_detector_training():
    fig, ax = canvas(10.5, 7.8)
    box(ax, 4, 84, 42, 9, "human-annotated face boxes\n(ground truth, annotations.csv)", C_DATA)
    box(ax, 54, 84, 42, 9, "random non-face patches\n(backgrounds of own images)", C_DATA)
    box(ax, 4, 68, 42, 8, "positive windows → own HOG\nimplementation (NumPy)")
    box(ax, 54, 68, 42, 8, "negative windows → own HOG\nimplementation (NumPy)")
    box(ax, 29, 52, 42, 8, "train linear SVM (hinge loss)\non OWN data only")
    box(ax, 29, 36, 42, 8, "scan training images\npyramid + sliding window + NMS")
    diamond(ax, 50, 24, 34, 10, "false positives\nremaining?")
    box(ax, 4, 30, 20, 9, "harvest false alarms\nas HARD negatives", C_NOTE, FS - 1)
    box(ax, 29, 4, 42, 9, "final detector → frozen weights\nevaluated vs. ground truth (IoU, miss rate)", C_DATA)

    arrow(ax, (25, 84), (25, 76))
    arrow(ax, (75, 84), (75, 76))
    arrow(ax, (25, 68), (40, 60), curve=-0.1)
    arrow(ax, (75, 68), (60, 60), curve=0.1)
    arrow(ax, (50, 52), (50, 44))
    arrow(ax, (50, 36), (50, 29))
    arrow(ax, (33, 24), (14, 30), label="yes", curve=0.15)
    arrow(ax, (14, 39), (29, 55), label="retrain", curve=0.25)
    arrow(ax, (50, 19), (50, 13), label="no")
    save(fig, "fig_detector_training_flowchart.png")


if __name__ == "__main__":
    fig_architecture()
    fig_preprocessing()
    fig_kiosk()
    fig_detector_training()
