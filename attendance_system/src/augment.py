"""
Phase 4 — Data augmentation (recognition training)
==================================================

Builds the augmentation pipeline applied to TRAINING crops only. Augmentation
synthesizes plausible appearance variation the enrollment data lacks, acting
as a regularizer for the from-scratch CNN. It is parameter-free (no learned
or pre-trained component) and is applied on-the-fly each epoch.

Transforms and their justification (dissertation §4)
----------------------------------------------------
* Horizontal flip   — a face and its mirror are both valid views; doubles
  effective pose coverage. (No vertical flip: faces are not up/down
  symmetric.)
* Small rotation (±8%·2pi ≈ ±14°) and translation/zoom — tolerance to the
  imperfect framing the detector will produce at deployment; the eye-
  alignment step already removes gross rotation, so the range is modest.
* Brightness and contrast jitter — robustness to lighting differences across
  rooms and, together with the multi-device data, to camera exposure/tone
  differences (the cross-device generalization goal, §3.3).

Applied AFTER the train/val/test split and to the training split only, so no
augmented view of a sample can leak into validation or test (§3.9).

Usage:
    python src/augment.py            # render a before/after grid on a crop
    from augment import build_augmenter
    aug = build_augmenter()
    x_aug = aug(x_train, training=True)
"""

from __future__ import annotations

import keras
from keras import layers

import config


def build_augmenter(rotation: float = 0.04,
                    zoom: float = 0.10,
                    translation: float = 0.08,
                    brightness: float = 0.15,
                    contrast: float = 0.15,
                    name: str = "augment") -> keras.Sequential:
    """Return the training-time augmentation pipeline.

    Ranges are intentionally conservative because inputs are already eye-
    aligned; over-strong augmentation would fight the normalization.
    Parameters are fractions: rotation/zoom/translation as fractions of a
    full turn / image size, brightness/contrast as jitter factors.
    """
    keras.utils.set_random_seed(config.RANDOM_SEED)
    return keras.Sequential([
        layers.RandomFlip("horizontal", name="aug_flip"),
        layers.RandomRotation(rotation, fill_mode="reflect", name="aug_rot"),
        layers.RandomTranslation(translation, translation,
                                 fill_mode="reflect", name="aug_shift"),
        layers.RandomZoom(zoom, zoom, fill_mode="reflect", name="aug_zoom"),
        layers.RandomBrightness(brightness, value_range=(0.0, 1.0),
                                name="aug_bright"),
        layers.RandomContrast(contrast, name="aug_contrast"),
    ], name=name)


def _preview() -> None:
    """Render an original crop and several augmentations to a QA image."""
    import numpy as np
    from PIL import Image

    crops = sorted((config.PROCESSED_DIR / f"faces_{config.INPUT_SIZE}")
                   .glob("*/*.png"))
    if not crops:
        raise SystemExit("no processed crops found — run build_face_crops.py")
    src = crops[0]
    arr = np.asarray(Image.open(src).convert("RGB")).astype("float32") / 255.0
    aug = build_augmenter()

    n = 7
    batch = np.repeat(arr[None, ...], n, axis=0)
    out = aug(batch, training=True).numpy()

    cell, pad = config.INPUT_SIZE, 6
    strip = Image.new("RGB", ((n + 1) * (cell + pad) + pad, cell + 2 * pad),
                      "white")
    strip.paste(Image.fromarray((arr * 255).astype("uint8")), (pad, pad))
    for i in range(n):
        img = Image.fromarray((out[i].clip(0, 1) * 255).astype("uint8"))
        strip.paste(img, ((i + 1) * (cell + pad) + pad, pad))
    out_path = config.REPORTS_DIR / "preprocessing" / "augmentation_preview.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path, quality=92)
    print(f"leftmost = original {src.name}; the rest are augmentations")
    print(f"augmentation preview -> {out_path}")


if __name__ == "__main__":
    _preview()
