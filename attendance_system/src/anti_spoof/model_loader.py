import logging
import urllib.request

import onnxruntime as ort

from . import utils

log = logging.getLogger("anti_spoof.model_loader")


def _download_if_needed(spec: dict) -> None:
    path = utils.MODEL_DIR / spec["file"]
    if path.exists():
        return
    utils.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading anti-spoof model %s ...", spec["file"])
    urllib.request.urlretrieve(spec["url"], path)
    log.info("downloaded -> %s", path)


def load_models() -> list[tuple[ort.InferenceSession, float]]:
    """Load every MiniFASNet variant; return [(session, crop_scale), ...].
    Raises if any model is missing and cannot be downloaded (caller decides the
    fail policy)."""
    sessions = []
    for spec in utils.MODELS:
        _download_if_needed(spec)
        sess = ort.InferenceSession(
            str(utils.MODEL_DIR / spec["file"]),
            providers=["CPUExecutionProvider"],
        )
        sessions.append((sess, spec["scale"]))
        log.info("anti-spoof model ready: %s (scale %.1f)",
                 spec["file"], spec["scale"])
    return sessions
