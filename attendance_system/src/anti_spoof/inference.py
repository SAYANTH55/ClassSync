import logging

import numpy as np

from . import model_loader
from . import preprocessing
from . import utils

log = logging.getLogger("anti_spoof.inference")

# Ensemble of (session, scale) loaded lazily. If loading fails and FAIL_CLOSED
# is True, every face is reported as SPOOF (rejected) so nothing is marked
# without a working liveness model. If FAIL_CLOSED is False, faces are treated
# as live (recognition keeps working, but spoofs are not blocked).
_sessions = None
_load_failed = False
FAIL_CLOSED = True


def get_sessions():
    global _sessions, _load_failed
    if _load_failed:
        return None
    if _sessions is None:
        try:
            _sessions = model_loader.load_models()
        except Exception as e:
            log.error("anti-spoof models failed to load (%s); FAIL_CLOSED=%s",
                      e, FAIL_CLOSED)
            _load_failed = True
            return None
    return _sessions


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _verdict(live: bool, live_prob: float, loaded: bool) -> dict:
    return {"live": live, "confidence": float(live_prob),
            "spoof_probability": float(1.0 - live_prob), "model_loaded": loaded}


def analyze_face(img_bgr: np.ndarray, bbox) -> dict:
    """Liveness verdict for one face (bbox = [x1,y1,x2,y2] in img_bgr pixels).

    Ensembles the MiniFASNet variants: each model gets its own crop scale, its
    softmax is summed, and the averaged 3-class probability decides. Class index
    1 = live. Accept only if argmax == 1 AND live prob ≥ threshold.
    """
    sessions = get_sessions()
    if not sessions:                       # model unavailable
        # FAIL_CLOSED → reject; else treat as live
        return _verdict(not FAIL_CLOSED, 0.0 if FAIL_CLOSED else 1.0, False)

    try:
        summed = np.zeros((1, 3), dtype=np.float32)
        for sess, scale in sessions:
            tensor = preprocessing.preprocess_face(img_bgr, bbox, scale)
            out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
            summed += softmax(out)
        probs = (summed / len(sessions))[0]           # averaged 3-class probs
        label = int(np.argmax(probs))
        live_prob = float(probs[utils.LIVE_CLASS])
        is_live = (label == utils.LIVE_CLASS) and (live_prob >= utils.LIVENESS_THRESHOLD)
        return _verdict(is_live, live_prob, True)
    except Exception as e:
        # a runtime error must not silently pass a spoof
        log.error("anti-spoof inference error (%s); FAIL_CLOSED=%s", e, FAIL_CLOSED)
        return _verdict(not FAIL_CLOSED, 0.0 if FAIL_CLOSED else 1.0, False)
