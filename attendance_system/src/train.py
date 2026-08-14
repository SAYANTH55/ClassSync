"""
Phase 7 — Recognition CNN training
==================================

Ties the input pipeline (data.py), augmentation (augment.py) and architecture
(model.py) into a reproducible training run with checkpointing, early
stopping, LR scheduling, a CSV history log and training-curve figures.

Outputs
-------
    models/face_cnn.keras                 best checkpoint (by monitored metric)
    models/class_index.json               student_id -> label (from data.py)
    reports/training/history.csv          per-epoch metrics
    reports/training/curves.png           loss / accuracy curves

Modes
-----
* Normal:  ``python src/train.py --epochs 60``
* Session-disjoint (once phone data exists):
      ``python src/train.py --strategy session
             --train-sessions phone1,phone2 --test-sessions dslr``
* Smoke test:  ``python src/train.py --smoke``
      A few fast epochs to PROVE the end-to-end loop runs on whatever crops
      exist. On the current 1-image-per-student dataset this only memorizes
      (no held-out data) — it verifies plumbing, NOT model quality.

Constraint: weights are initialized randomly (He init) and trained solely on
project data. No pre-trained weights are loaded at any point.
"""

from __future__ import annotations

import argparse
import logging

import keras

import config
from augment import build_augmenter
from data import make_datasets
from model import build_model, compile_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("train")


def plot_curves(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, h["loss"], label="train")
    if "val_loss" in h:
        ax1.plot(epochs, h["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    if "acc" in h:
        ax2.plot(epochs, h["acc"], label="train")
        if "val_acc" in h:
            ax2.plot(epochs, h["val_acc"], label="val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    log.info("training curves -> %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--strategy", choices=["stratified", "session"],
                    default="stratified")
    ap.add_argument("--train-sessions", default="")
    ap.add_argument("--val-sessions", default="")
    ap.add_argument("--test-sessions", default="")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="fast plumbing test (few epochs)")
    args = ap.parse_args()

    keras.utils.set_random_seed(config.RANDOM_SEED)
    epochs = 3 if args.smoke else args.epochs
    augmenter = None if (args.no_augment or args.smoke) else build_augmenter()

    def sess(s):
        return set(x for x in s.split(",") if x) or None

    train_ds, val_ds, test_ds, class_index, counts = make_datasets(
        strategy=args.strategy, batch=args.batch, augmenter=augmenter,
        train_sessions=sess(args.train_sessions),
        val_sessions=sess(args.val_sessions),
        test_sessions=sess(args.test_sessions))

    if train_ds is None:
        raise SystemExit("training split is empty — check split settings")
    if val_ds is None:
        log.warning("no validation split — monitoring TRAIN loss instead "
                    "(expected on the provisional single-image dataset)")

    model = compile_model(build_model(counts["classes"]), learning_rate=args.lr)
    monitor = "val_loss" if val_ds is not None else "loss"

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hist_csv = config.REPORTS_DIR / "training" / "history.csv"
    hist_csv.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.MODELS_DIR / "face_cnn.keras", monitor=monitor,
            save_best_only=True, mode="min"),
        keras.callbacks.CSVLogger(hist_csv),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                          patience=5, min_lr=1e-6),
    ]
    if val_ds is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=monitor, patience=12, restore_best_weights=True))

    log.info("training %d epochs on %d crops (%d classes)%s",
             epochs, counts["train"], counts["classes"],
             "  [SMOKE TEST]" if args.smoke else "")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=2)
    plot_curves(history, config.REPORTS_DIR / "training" / "curves.png")

    if test_ds is not None:
        log.info("evaluating on held-out test split ...")
        results = model.evaluate(test_ds, verbose=0, return_dict=True)
        log.info("test metrics: %s", {k: round(v, 4) for k, v in results.items()})

    print("\n============ TRAINING SUMMARY ============")
    print(f"mode                 : {'SMOKE (plumbing only)' if args.smoke else 'full'}")
    print(f"classes              : {counts['classes']}")
    print(f"crops train/val/test : {counts['train']}/{counts['val']}/{counts['test']}")
    print(f"final train loss     : {history.history['loss'][-1]:.4f}")
    print(f"final train acc      : {history.history.get('acc', ['n/a'])[-1]}")
    print(f"best model saved     : {config.MODELS_DIR / 'face_cnn.keras'}")
    if args.smoke:
        print("NOTE: smoke test verifies the loop RUNS; accuracy is not a "
              "result (no held-out data on the current dataset).")
    print("==========================================")


if __name__ == "__main__":
    main()
