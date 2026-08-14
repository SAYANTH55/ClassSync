"""
Phase 3b — Face preprocessing primitives
========================================

Reusable, side-effect-free functions shared by every later stage (detector
training, CNN training, kiosk inference). No function here reads config or
touches the filesystem beyond loading the image it is given — scripts compose
these primitives.

Geometric normalization policy (dissertation §Methodology):

* EXIF-safe loading: every image enters the pipeline through
  :func:`load_image_upright`, which applies the EXIF orientation transpose
  exactly once. OpenCV's ``imread`` is never used for dataset images because
  it ignores EXIF (the DSLR set stores frames rotated with orientation=8).
* Crop geometry: the annotated/detected face box is expanded by a margin
  (context helps CNNs; a too-tight crop loses chin/forehead under box jitter)
  and made square, so resizing to the CNN input never distorts aspect ratio.
* Resampling: ``cv2.INTER_AREA`` for downscaling — face crops from the 53 MP
  portraits shrink by ~15x per axis; area averaging avoids the aliasing that
  bilinear/bicubic introduce at strong minification.
* Alignment: when both eye centres are known (manual annotation now, detector
  landmarks later), a similarity transform maps them to fixed canonical
  positions, removing in-plane rotation and scale variation — a classical
  normalization step (FERET protocol) with large accuracy impact.
* Intensity: crops are stored as uint8 PNG (lossless). Scaling to [0, 1]
  happens at training/inference time via :func:`to_float01`. Dataset-level
  standardization (mean/std) is deliberately NOT done here: those statistics
  must be computed on the training split only, after Phase 5, to avoid
  information leaking from validation/test data into preprocessing.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

try:  # cv2 is required only for warp/resize; import guarded for testability
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:  # HEIC/HEIF support (iPhone captures) — registers a PIL opener
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

Box = tuple[int, int, int, int]          # x, y, w, h  (pixels, upright space)
Point = tuple[float, float]              # x, y


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_image_upright(path) -> np.ndarray:
    """Load an image as an RGB uint8 array with EXIF orientation applied."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"))


# ---------------------------------------------------------------------------
# box geometry
# ---------------------------------------------------------------------------
def expand_to_square(box: Box, margin: float) -> Box:
    """Grow *box* by ``margin`` (fraction of its larger side) on every side,
    then pad the shorter dimension so the result is square (centre kept).

    The result may extend beyond the image; :func:`crop` handles that by
    replicate-padding, so faces near a border are never distorted.
    """
    x, y, w, h = box
    m = margin * max(w, h)
    x, y, w, h = x - m, y - m, w + 2 * m, h + 2 * m
    side = max(w, h)
    x -= (side - w) / 2
    y -= (side - h) / 2
    return (int(round(x)), int(round(y)), int(round(side)), int(round(side)))


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    """Crop *box* from *img*, replicate-padding any part outside the image."""
    x, y, w, h = box
    H, W = img.shape[:2]
    pad_l, pad_t = max(0, -x), max(0, -y)
    pad_r, pad_b = max(0, x + w - W), max(0, y + h - H)
    if any((pad_l, pad_t, pad_r, pad_b)):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_REPLICATE)
        x, y = x + pad_l, y + pad_t
    return img[y:y + h, x:x + w]


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def align_face(img: np.ndarray, eye_left: Point, eye_right: Point,
               out_size: int, eye_row: float = 0.38,
               eye_dist: float = 0.42) -> np.ndarray:
    """Similarity-transform crop that places the eyes at canonical positions.

    Parameters
    ----------
    eye_left, eye_right : subject's eyes in image coordinates; "left" = the
        eye with the smaller x (viewer's left).
    out_size : side of the square output crop in pixels.
    eye_row : vertical position of the eye line in the output (fraction).
    eye_dist : distance between the eyes in the output (fraction of width).
    """
    (lx, ly), (rx, ry) = eye_left, eye_right
    dst_l = ((0.5 - eye_dist / 2) * out_size, eye_row * out_size)
    dst_r = ((0.5 + eye_dist / 2) * out_size, eye_row * out_size)

    # closed-form similarity transform from one point pair + implied rotation
    src_v = np.array([rx - lx, ry - ly])
    dst_v = np.array([dst_r[0] - dst_l[0], dst_r[1] - dst_l[1]])
    scale = np.linalg.norm(dst_v) / (np.linalg.norm(src_v) + 1e-9)
    angle = np.arctan2(src_v[1], src_v[0]) - np.arctan2(dst_v[1], dst_v[0])
    cos, sin = scale * np.cos(angle), scale * np.sin(angle)
    # rotate about the left eye, then translate it onto its canonical spot
    M = np.array([[cos, sin, dst_l[0] - (cos * lx + sin * ly)],
                  [-sin, cos, dst_l[1] - (-sin * lx + cos * ly)]])
    return cv2.warpAffine(img, M, (out_size, out_size),
                          flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# resizing / intensity
# ---------------------------------------------------------------------------
def resize_square(img: np.ndarray, out_size: int) -> np.ndarray:
    """Resize a (square) crop to the model input size.

    INTER_AREA when shrinking (anti-aliasing), INTER_LINEAR when enlarging.
    """
    interp = cv2.INTER_AREA if img.shape[0] > out_size else cv2.INTER_LINEAR
    return cv2.resize(img, (out_size, out_size), interpolation=interp)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """RGB -> single-channel uint8 (ITU-R BT.601 luma, as used by OpenCV)."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def to_float01(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]. Train-split mean/std standardization
    is applied later, with statistics from training data only (no leakage)."""
    return img.astype(np.float32) / 255.0
