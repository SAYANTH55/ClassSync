"""
Deep-learning baselines on the identical protocol: scratch CNN + MobileNetV2
============================================================================

Fills the last two rows of the comparative table. Both networks are trained
as 47-class classifiers on the phone enrollment crops ONLY (245 images,
~5/student) and evaluated on the labelled DSLR probes — the same
session-disjoint, cross-device split every other method faced.

Evaluation mirrors the embedding protocol exactly: the classifier head is
discarded and the PENULTIMATE-layer activation is used as a face embedding;
gallery templates = training images' own embeddings; probes scored by max
cosine per student; rank-1 + zero-FAR sweep. (Closed-set softmax accuracy is
also reported, but the embedding route is what makes rows comparable and
lets the open-set question be asked at all.)

Methods
-------
* SCRATCH-CNN — compact VGG-style net (4 conv blocks -> 128-d embedding),
  trained from random init. Expected to struggle: 245 images cannot teach a
  network what identity-invariance looks like (limitation L2 made visible).
* MOBILENETV2-TL — ImageNet-pretrained base, frozen; only a small embedding
  head trains. Transfer learning: generic visual features reused, so the
  scarce data only has to learn the projection, not vision itself.

Runs under the TensorFlow env (miniconda); consumes only crops_rgb.npz built
by build_crops_rgb.py under face311 — no insightface needed here.

Usage:
    python src/evaluate_deep.py
"""

from __future__ import annotations

import csv
import logging
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("evaluate_deep")

OUT_DIR = config.REPORTS_DIR / "eval_deep"
CROPS = config.CACHE_DIR / "crops_rgb.npz"
EMB_DIM = 128
EPOCHS_SCRATCH = 60
EPOCHS_TL = 30
BATCH = 32


# ---- data ------------------------------------------------------------------
def load_split():
    z = np.load(CROPS)
    gal_x, gal_y, names = [], [], []
    probes: dict[str, np.ndarray] = {}
    for key in z.files:
        parts = key.split("/")
        if parts[0] == "gallery":
            if parts[1] not in names:
                names.append(parts[1])
            gal_x.append(z[key])
            gal_y.append(parts[1])
        else:
            probes[parts[1]] = z[key]
    names = sorted(names)
    idx = {n: i for i, n in enumerate(names)}
    X = np.stack(gal_x).astype(np.float32) / 255.0
    y = np.array([idx[n] for n in gal_y])
    return X, y, names, probes


def augmented_dataset(X, y):
    import tensorflow as tf
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 1)),
        tf.keras.layers.RandomContrast(0.2),
    ])
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(X), seed=config.RANDOM_SEED).batch(BATCH)
    ds = ds.map(lambda a, b: (aug(a, training=True), b),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---- models ----------------------------------------------------------------
def scratch_cnn(n_classes: int):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input((112, 112, 3))
    x = inp
    for filters in (32, 64, 128, 256):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    emb = L.Dense(EMB_DIM, name="embedding")(x)
    x = L.Dropout(0.4)(emb)
    out = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def mobilenet_tl(n_classes: int):
    import tensorflow as tf
    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(
        input_shape=(112, 112, 3), include_top=False, weights="imagenet")
    base.trainable = False                       # pure transfer learning
    inp = L.Input((112, 112, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    emb = L.Dense(EMB_DIM, name="embedding")(x)
    x = L.Dropout(0.3)(emb)
    out = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)


# ---- shared evaluation (same protocol as classical/arcface) ----------------
def evaluate(method, model, X, y, names, probes, roster_names):
    import tensorflow as tf
    embedder = tf.keras.Model(model.input,
                              model.get_layer("embedding").output)

    def embed(batch):
        e = embedder.predict(batch, verbose=0)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    G = embed(X)
    P_stems = sorted(probes)
    P = embed(np.stack([probes[s] for s in P_stems]).astype(np.float32) / 255.0)

    rows, g_scores, i_scores = [], [], []
    softmax_pred = model.predict(
        np.stack([probes[s] for s in P_stems]).astype(np.float32) / 255.0,
        verbose=0).argmax(1)
    for k, stem in enumerate(P_stems):
        sims = G @ P[k]
        per_student = np.array([float(sims[y == i].max())
                                for i in range(len(names))])
        i_best = int(per_student.argmax())
        pred, score = names[i_best], float(per_student[i_best])
        genuine = stem in roster_names
        (g_scores if genuine else i_scores).append(score)
        rows.append({"file": stem, "kind": "genuine" if genuine else "impostor",
                     "pred": pred, "score": round(score, 4),
                     "softmax_pred": names[softmax_pred[k]],
                     "correct": (pred == stem) if genuine else ""})

    genuine_rows = [r for r in rows if r["kind"] == "genuine"]
    n_correct = sum(r["correct"] is True for r in genuine_rows)
    n_softmax = sum(r["softmax_pred"] == r["file"] for r in genuine_rows)
    g, i = np.array(g_scores), np.array(i_scores)
    best_tar = 0.0
    for t in np.linspace(min(i.min(), g.min()), g.max(), 400):
        if float(np.mean(i >= t)) == 0.0:
            tar = float(np.mean([(r["score"] >= t) and (r["correct"] is True)
                                 for r in genuine_rows]))
            best_tar = max(best_tar, tar)
    res = {"method": method, "rank1": n_correct, "n_genuine": len(genuine_rows),
           "rank1_pct": 100 * n_correct / len(genuine_rows),
           "softmax_rank1": n_softmax, "tar_at_far0": best_tar,
           "g_min": float(g.min()), "g_med": float(np.median(g)),
           "i_med": float(np.median(i)), "i_max": float(i.max()),
           "separated": bool(i.max() < g.min()), "rows": rows}
    log.info("%-14s rank-1 %d/%d (%.1f%%)  softmax %d/%d  TAR@FAR=0 %.3f"
             "  separated %s", method, n_correct, len(genuine_rows),
             res["rank1_pct"], n_softmax, len(genuine_rows), best_tar,
             res["separated"])
    return res


def main() -> None:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from gallery import load_roster
    roster_names = set(load_roster().values())
    X, y, names, probes = load_split()
    log.info("train: %d crops / %d students; probes: %d",
             len(X), len(names), len(probes))
    ds = augmented_dataset(X, y)

    results = []
    for method, builder, epochs in (
            ("Scratch-CNN", scratch_cnn, EPOCHS_SCRATCH),
            ("MobileNetV2-TL", mobilenet_tl, EPOCHS_TL)):
        log.info("training %s (%d epochs, CPU) ...", method, epochs)
        model = builder(len(names))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        hist = model.fit(ds, epochs=epochs, verbose=0)
        log.info("%s final train acc: %.3f", method,
                 hist.history["accuracy"][-1])
        results.append(evaluate(method, model, X, y, names, probes,
                                roster_names))

    for r in results:
        with open(OUT_DIR / f"per_probe_{r['method'].lower()}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(r["rows"][0]))
            w.writeheader(); w.writerows(r["rows"])
    with open(OUT_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["method", "rank1", "n_genuine", "rank1_pct", "softmax_rank1",
                "tar_at_far0", "g_min", "g_med", "i_med", "i_max", "separated"]
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r[k] for k in keys])

    print("\n================== DEEP BASELINES (identical protocol) ==================")
    for r in results:
        print(f"{r['method']:15s} rank-1 {r['rank1']}/{r['n_genuine']}"
              f" ({r['rank1_pct']:.1f}%)  softmax {r['softmax_rank1']}"
              f"/{r['n_genuine']}  TAR@FAR=0 {r['tar_at_far0']:.3f}"
              f"  separated {r['separated']}")
    print(f"outputs -> {OUT_DIR}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
