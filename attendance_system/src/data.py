"""
Phase 5 — Dataset splits and input pipeline
===========================================

Turns the processed face crops into ``tf.data`` pipelines for training,
validation and test, and maintains the canonical class index (student_id ->
integer label) used by both training and the attendance system.

Crops live at ``data/processed/faces_<size>/<student_id>/<sid>_<source>_<seq>.png``
so both the identity label and the capture session are recoverable from the
path — enabling the two split strategies below.

Split strategies
----------------
* ``session`` (preferred once multiple sessions exist): each capture session
  goes entirely to one split, e.g. train on phone sessions, test on the DSLR
  gallery. Measures generalization to UNSEEN capture conditions (§3.9). No
  augmented or original view of a test session is ever seen in training.
* ``stratified`` (fallback / single-session): a per-class random partition
  with a fixed seed. Classes with too few images to populate every split are
  reported; their under-represented splits fall back to what is available
  (acceptable only for the current provisional single-image dataset).

The pipeline decodes PNGs, scales to [0, 1] (float32), batches and prefetches.
Dataset-level mean/std standardization is intentionally NOT applied here; if
adopted later it must be fit on the training split only (§3.6.2).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("data")

CLASS_INDEX_PATH = config.MODELS_DIR / "class_index.json"
_NAME_RE = re.compile(r"^(S\d{2,3})_([A-Za-z0-9]+)_\d+\.png$")


@dataclass(frozen=True)
class Crop:
    path: Path
    student_id: str
    source: str


def list_crops(size: int = config.INPUT_SIZE) -> list[Crop]:
    """All processed crops of the given size, parsed from their filenames."""
    root = config.PROCESSED_DIR / f"faces_{size}"
    crops: list[Crop] = []
    for p in sorted(root.glob("*/*.png")):
        m = _NAME_RE.match(p.name)
        if not m:
            log.warning("skipping unrecognized crop name: %s", p.name)
            continue
        crops.append(Crop(p, m.group(1), m.group(2)))
    if not crops:
        raise SystemExit(f"no crops in {root} — run build_face_crops.py first")
    return crops


def build_class_index(crops: list[Crop], persist: bool = True) -> dict[str, int]:
    """Deterministic student_id -> label map; persisted for inference reuse."""
    classes = sorted({c.student_id for c in crops})
    index = {sid: i for i, sid in enumerate(classes)}
    if persist:
        CLASS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        CLASS_INDEX_PATH.write_text(json.dumps(index, indent=2))
        log.info("class index (%d classes) -> %s", len(index), CLASS_INDEX_PATH)
    return index


def load_class_index() -> dict[str, int]:
    return json.loads(CLASS_INDEX_PATH.read_text())


def _split_stratified(crops, val_frac, test_frac, seed):
    import random
    rng = random.Random(seed)
    by_class: dict[str, list[Crop]] = defaultdict(list)
    for c in crops:
        by_class[c.student_id].append(c)

    train, val, test = [], [], []
    thin = 0
    for sid, items in sorted(by_class.items()):
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        if n >= 3:
            test += items[:n_test]
            val += items[n_test:n_test + n_val]
            train += items[n_test + n_val:]
        else:                       # too few to split — all to train
            train += items
            thin += 1
    if thin:
        log.warning("%d/%d classes had <3 crops: entirely in TRAIN (no val/"
                    "test for them). Expected for the provisional dataset.",
                    thin, len(by_class))
    return train, val, test


def _split_by_session(crops, train_sessions, test_sessions, val_sessions):
    def pick(sset):
        return [c for c in crops if c.source in sset] if sset else []
    return (pick(train_sessions), pick(val_sessions), pick(test_sessions))


def _to_dataset(items, class_index, batch, shuffle, augmenter, seed):
    if not items:
        return None
    paths = [str(c.path) for c in items]
    labels = [class_index[c.student_id] for c in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_png(tf.io.read_file(path),
                               channels=config.INPUT_CHANNELS)
        img = tf.image.resize(img, [config.INPUT_SIZE, config.INPUT_SIZE])
        return tf.cast(img, tf.float32) / 255.0, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch)
    if augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_datasets(size: int = config.INPUT_SIZE,
                  strategy: str = "stratified",
                  val_frac: float = 0.15,
                  test_frac: float = 0.15,
                  batch: int = 32,
                  augmenter=None,
                  train_sessions: set[str] | None = None,
                  val_sessions: set[str] | None = None,
                  test_sessions: set[str] | None = None,
                  seed: int = config.RANDOM_SEED):
    """Return (train_ds, val_ds, test_ds, class_index, counts).

    ``augmenter`` is applied to the training dataset only.
    """
    crops = list_crops(size)
    class_index = build_class_index(crops)

    if strategy == "session":
        tr, va, te = _split_by_session(crops, train_sessions or set(),
                                       test_sessions or set(),
                                       val_sessions or set())
    elif strategy == "stratified":
        tr, va, te = _split_stratified(crops, val_frac, test_frac, seed)
    else:
        raise ValueError(f"unknown split strategy: {strategy}")

    counts = {"train": len(tr), "val": len(va), "test": len(te),
              "classes": len(class_index)}
    log.info("split '%s': %s", strategy, counts)
    return (
        _to_dataset(tr, class_index, batch, True, augmenter, seed),
        _to_dataset(va, class_index, batch, False, None, seed),
        _to_dataset(te, class_index, batch, False, None, seed),
        class_index, counts,
    )


if __name__ == "__main__":
    crops = list_crops()
    idx = build_class_index(crops, persist=False)
    by_src = defaultdict(int)
    for c in crops:
        by_src[c.source] += 1
    print("\n============ DATASET OVERVIEW ============")
    print(f"crops              : {len(crops)}")
    print(f"classes            : {len(idx)}")
    print(f"crops per session  : {dict(by_src)}")
    print(f"crops per class     : "
          f"{min(sum(1 for c in crops if c.student_id == s) for s in idx)}"
          f" (min) .. "
          f"{max(sum(1 for c in crops if c.student_id == s) for s in idx)} (max)")
    print("=========================================")
