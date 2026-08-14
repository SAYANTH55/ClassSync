"""
Pre-trained detection + face embedding backend (deployed recogniser)
====================================================================

Thin, well-documented wrapper over insightface's ``buffalo_l`` pack, which
provides — in one model set — an SCRFD-10GF face detector with 5-point
landmarks (``det_10g.onnx``) and the ArcFace ResNet-50 face-embedding network
(``w600k_r50.onnx``, trained on WebFace-600K; 512-d). This is the project's
DEPLOYED path for both stages:

    detection + landmarks  -> replaces the classical HOG proposals and the
                              manual eye annotation (landmarks are automatic)
    ArcFace embedding      -> the face representation compared for recognition

Why pre-trained here (viva justification)
-----------------------------------------
ArcFace is trained with an additive angular-margin loss on millions of faces
across tens of thousands of identities. The resulting 512-d embedding places
same-identity faces close together and different identities far apart in
angular (cosine) space, REGARDLESS of who was in the training set. For a
fixed cohort of ~56 students with few images each, this is decisive: identity
can be decided by distance to enrolled templates without training a
classifier from scratch on scarce data (addresses limitation L2), and a new
student is enrolled by storing one embedding — no retraining. The from-scratch
CNN and classical methods remain in the project as benchmarked baselines.

Concepts to be ready to defend: transfer learning / representation reuse;
metric learning and angular-margin (ArcFace) loss; cosine similarity as an
identity score; open-set vs closed-set recognition; threshold selection; and
demographic bias in face models (to be measured on this cohort, not assumed
away).

EXIF: images are loaded upright via preprocessing.load_image_upright (the DSLR
frames carry orientation flag 8); insightface consumes BGR uint8 arrays.

Usage:
    python src/detect_embed.py            # self-test on sample portraits
    from detect_embed import FaceBackend
    fb = FaceBackend(); faces = fb.detect(img_rgb); emb = fb.embed_best(img_rgb)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import config
import preprocessing as pp

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("detect_embed")


@dataclass
class Face:
    """One detected face."""
    bbox: np.ndarray          # (4,) x1, y1, x2, y2 in upright pixel coords
    landmarks: np.ndarray     # (5, 2) eyes, nose, mouth corners
    embedding: np.ndarray | None # (512,) L2-normalized ArcFace embedding, None if spoof
    det_score: float
    is_spoof: bool = False
    live_confidence: float = 1.0
    spoof_probability: float = 0.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class FaceBackend:
    """Lazy singleton-ish wrapper around insightface FaceAnalysis."""

    def __init__(self, det_size: int = 640):
        from insightface.app import FaceAnalysis  # imported lazily (heavy)

        config.EMBED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        # buffalo_l is a 5-model pack; FaceAnalysis.get() runs EVERY loaded
        # model on every face. We use only detection + recognition — loading
        # the 2D/3D-landmark and gender/age models costs ~145 ms/frame for
        # outputs we discard (measured by profile_pipeline.py).
        self.app = FaceAnalysis(
            name=config.EMBED_MODEL_NAME,
            root=str(config.EMBED_MODEL_ROOT),
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        # ctx_id=-1 -> CPU; det_size trades recall vs speed
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        log.info("insightface '%s' ready (CPU, det_size=%d)",
                 config.EMBED_MODEL_NAME, det_size)

    def detect(self, img_rgb: np.ndarray) -> list[Face]:
        """Detect all faces in an RGB uint8 image; sorted largest-first."""
        import anti_spoof
        from insightface.app.common import Face as IFFace

        bgr = img_rgb[:, :, ::-1]
        out = []
        
        # Run detection only
        det_model = self.app.models['detection']
        bboxes, kpss = det_model.detect(bgr, max_num=0, metric='default')
        if bboxes.shape[0] == 0:
            return []

        rec_model = self.app.models.get('recognition')

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            
            iface = IFFace(bbox=bbox, kps=kps, det_score=det_score)

            # Anti-spoofing layer — runs BEFORE ArcFace; gates embedding.
            liveness_data = anti_spoof.analyze_face(bgr, bbox)

            embedding = None
            if liveness_data["live"] and rec_model is not None:
                # Only proceed to ArcFace if face is live
                rec_model.get(bgr, iface)
                if iface.normed_embedding is not None:
                    embedding = iface.normed_embedding.astype(np.float32)
            # per-face pipeline trace (task 15): see every liveness decision
            log.info("face %d | live=%.3f spoof=%.3f -> %s | ArcFace=%s",
                     i, liveness_data["confidence"],
                     liveness_data["spoof_probability"],
                     "LIVE(accept)" if liveness_data["live"] else "SPOOF(REJECT)",
                     "ran" if embedding is not None else "skipped")
            
            out.append(Face(
                bbox=iface.bbox.astype(float),
                landmarks=iface.kps.astype(float),
                embedding=embedding,
                det_score=float(iface.det_score),
                is_spoof=not liveness_data["live"],
                live_confidence=liveness_data["confidence"],
                spoof_probability=liveness_data["spoof_probability"]
            ))

        out.sort(key=lambda f: f.area, reverse=True)
        return out

    def embed_best(self, img_rgb: np.ndarray) -> np.ndarray | None:
        """Embedding of the largest detected LIVE face (kiosk: one main subject)."""
        faces = self.detect(img_rgb)
        for f in faces:
            if not f.is_spoof:
                return f.embedding
        return None

    def embed_path(self, path) -> np.ndarray | None:
        return self.embed_best(pp.load_image_upright(path))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (already L2-normalized) embeddings."""
    return float(np.dot(a, b))


def _self_test() -> None:
    """Verify detection + embedding, and sanity-check the similarity metric."""
    import config as cfg

    paths = cfg.session_images("dslr")[:3]
    if not paths:
        raise SystemExit("no DSLR portraits found")
    fb = FaceBackend()

    embs, names = [], []
    for p in paths:
        img = pp.load_image_upright(p)
        faces = fb.detect(img)
        if not faces:
            log.warning("no face detected in %s", p.name)
            continue
        f = faces[0]
        embs.append(f.embedding)
        names.append(p.stem)
        log.info("%s: %d face(s), best det_score=%.3f, bbox=%s, emb_dim=%d",
                 p.name, len(faces), f.det_score,
                 f.bbox.round().astype(int).tolist(), f.embedding.shape[0])

    # sanity: self vs horizontally-flipped self (should be HIGH), vs another
    # person (should be LOWER) — demonstrates the embedding encodes identity
    img0 = pp.load_image_upright(paths[0])
    flip0 = fb.embed_best(img0[:, ::-1, :])
    print("\n============ EMBEDDING SELF-TEST ============")
    print(f"embedding dim         : {embs[0].shape[0]}  (expected {cfg.EMBED_DIM})")
    print(f"norm ~1 (L2)          : {np.allclose(np.linalg.norm(embs[0]), 1, atol=1e-3)}")
    if flip0 is not None:
        print(f"self vs flipped-self  : cos = {cosine(embs[0], flip0):+.3f}  (expect high)")
    if len(embs) >= 2:
        print(f"person0 vs person1    : cos = {cosine(embs[0], embs[1]):+.3f}  (expect lower)")
    if len(embs) >= 3:
        print(f"person0 vs person2    : cos = {cosine(embs[0], embs[2]):+.3f}  (expect lower)")
    print("=============================================")


if __name__ == "__main__":
    _self_test()
