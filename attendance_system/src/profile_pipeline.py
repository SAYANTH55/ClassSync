"""
Latency profile of the deployed pipeline (per-stage, per-config)
================================================================

Answers the deployment question: how many milliseconds does one attendance
check cost on this machine, and WHERE do they go?

Stages timed independently on a webcam-sized frame (1280x720):

    detect   — SCRFD forward pass + NMS      (expected dominant cost;
               scales ~quadratically with det_size)
    embed    — norm_crop alignment + ArcFace forward on the 112x112 crop
               (constant cost — input size is fixed by alignment)
    match    — cosine scores vs all gallery templates (pure numpy)

Method notes (the habits that make timings trustworthy):
  * 3 warm-up runs per config are discarded — first inferences pay one-off
    allocation/optimization costs that would poison the stats;
  * median and p95 reported, never mean — means are wrecked by OS hiccups;
  * the same frame is reused so configs differ only in the knob under test.

Usage:
    python src/profile_pipeline.py          # face311 env
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import config
import preprocessing as pp
from gallery import Recognizer

WARMUP = 3
RUNS = 15
FRAME_WH = (1280, 720)          # typical webcam capture size
DET_SIZES = (640, 480, 320)


def stats(ms: list[float]) -> str:
    a = np.sort(np.array(ms))
    return f"median {a[len(a)//2]:7.1f} ms   p95 {a[int(len(a)*0.95)-1]:7.1f} ms"


def main() -> None:
    # webcam-sized test frame from a real portrait (content realistic:
    # one face, cluttered margins after resize)
    src = config.session_images("dslr_labelled")[0]
    img = pp.load_image_upright(src)
    # letterbox, never bare-resize: a 3:4 portrait squashed into a 16:9
    # frame distorts faces beyond what the detector was trained on
    # (bare cv2.resize here produced ZERO detections — real-world bug)
    w, h = FRAME_WH
    scale = min(w / img.shape[1], h / img.shape[0])
    nw, nh = int(img.shape[1] * scale), int(img.shape[0] * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    frame[y0:y0 + nh, x0:x0 + nw] = resized

    rec = Recognizer(threshold=config.RECOG_THRESHOLD)
    print(f"\nframe {FRAME_WH[0]}x{FRAME_WH[1]}, gallery {len(rec.ids)} "
          f"students / {sum(len(t) for t in rec.templates)} templates, "
          f"{RUNS} timed runs after {WARMUP} warm-ups\n")

    from detect_embed import FaceBackend
    for det_size in DET_SIZES:
        fb = FaceBackend(det_size=det_size)

        t_detect, t_embed, t_match = [], [], []
        emb = None
        for i in range(WARMUP + RUNS):
            t0 = time.perf_counter()
            faces = fb.detect(frame)
            t1 = time.perf_counter()
            # fb.detect already embeds internally (insightface pipeline);
            # isolate embed cost by re-running the recognition model alone
            from insightface.utils.face_align import norm_crop
            crop = norm_crop(frame, faces[0].landmarks, image_size=112)
            rmodel = fb.app.models["recognition"]
            e = rmodel.get_feat(crop[:, :, ::-1]).flatten()
            e = e / np.linalg.norm(e)
            t2 = time.perf_counter()
            scores = rec.scores(e.astype(np.float32))
            t3 = time.perf_counter()
            if i >= WARMUP:
                t_detect.append((t1 - t0) * 1000)
                t_embed.append((t2 - t1) * 1000)
                t_match.append((t3 - t2) * 1000)
            emb = e

        total = (np.median(t_detect) + np.median(t_embed)
                 + np.median(t_match))
        best = rec.ids[int(np.argmax(rec.scores(emb.astype(np.float32))))]
        print(f"det_size={det_size}")
        print(f"  detect (SCRFD+NMS)   {stats(t_detect)}")
        print(f"  embed  (align+Arc)   {stats(t_embed)}")
        print(f"  match  (cosine x{sum(len(t) for t in rec.templates)})"
              f"  {stats(t_match)}")
        print(f"  TOTAL median {total:7.1f} ms  (~{1000/total:.1f} fps)"
              f"   [sanity: top match {rec.roster[best]}]\n")


if __name__ == "__main__":
    main()
