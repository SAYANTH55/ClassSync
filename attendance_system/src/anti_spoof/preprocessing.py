import cv2
import numpy as np

from . import utils


def crop_face(img_bgr: np.ndarray, bbox_xyxy, scale: float) -> np.ndarray:
    """Expand the detection box by `scale` about its centre, CLAMP to the image
    bounds, and resize to the model input size. Matches yakhyo's `crop_face`
    exactly (the preprocessing the ONNX exports were validated with)."""
    src_h, src_w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    box_w, box_h = x2 - x1, y2 - y1

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x1 + box_w / 2, y1 + box_h / 2

    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(src_w - 1, int(cx + new_w / 2))
    ny2 = min(src_h - 1, int(cy + new_h / 2))

    crop = img_bgr[ny1:ny2 + 1, nx1:nx2 + 1]
    if crop.size == 0:
        crop = np.zeros((utils.INPUT_SIZE[1], utils.INPUT_SIZE[0], 3), np.uint8)
    return cv2.resize(crop, utils.INPUT_SIZE)


def preprocess_face(img_bgr: np.ndarray, bbox_xyxy, scale: float) -> np.ndarray:
    """Crop → RAW BGR float32 [0,255] → CHW → (1,3,80,80). CRITICAL: MiniFASNet
    here is fed unnormalised pixel values (no /255, no mean/std) — that is how
    the exports were trained/validated. Dividing by 255 silently breaks it."""
    crop = crop_face(img_bgr, bbox_xyxy, scale)
    t = crop.astype(np.float32)                    # raw [0,255], BGR
    t = np.transpose(t, (2, 0, 1))                 # HWC -> CHW
    return t[np.newaxis, :, :, :]                  # (1,3,80,80)
