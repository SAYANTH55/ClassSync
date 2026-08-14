"""
Phase 3d — Manual face annotation tool (ground truth creation)
==============================================================

Interactive OpenCV tool for annotating, per image:
  * one face bounding box  (drag with the mouse; prefilled from proposals.csv)
  * two eye centres        (left-click each eye)

Output: ``data/labels/annotations.csv`` — the project's ground truth for
(1) aligned recognition crops, (2) detector training positives, and
(3) quantitative detector evaluation (IoU vs. this human reference).

Controls
--------
  mouse drag        draw a new face box
  left click        place an eye centre (3rd click restarts the pair)
  ENTER / SPACE     accept image -> autosave -> next
  b                 go back one image
  r                 restore the classical proposal box
  c                 clear box + eyes
  q / ESC           save and quit (progress is kept; tool resumes later)

The tool displays a scaled view; all saved coordinates are in FULL-RESOLUTION
upright pixel space (EXIF-corrected), matching preprocessing.py conventions.

Works identically for every capture session (DSLR portraits, ingested phone
frames/stills): images are resolved through the session registry in
config.py. Use ``--session`` to annotate one session at a time (recommended
for phone frames: annotate the stratified sample chosen for detector
training/evaluation rather than every frame).

Usage:
    python src/annotate_faces.py                 # all sessions
    python src/annotate_faces.py --session dslr  # one session
"""

from __future__ import annotations

import argparse
import csv
import logging

import cv2
import numpy as np

import config
import preprocessing as pp

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("annotate")

MAX_DISP_H, MAX_DISP_W = 900, 1400
ANN_PATH = config.LABELS_CSV.parent / "annotations.csv"
PROP_PATH = config.LABELS_CSV.parent / "proposals.csv"
FIELDS = ["filename", "face_x", "face_y", "face_w", "face_h",
          "eye_lx", "eye_ly", "eye_rx", "eye_ry", "geometry_source"]


class State:
    """Mutable annotation state for the currently displayed image."""

    def __init__(self) -> None:
        self.box: tuple[int, int, int, int] | None = None   # full-res
        self.eyes: list[tuple[float, float]] = []           # full-res
        self.source = "manual"
        self.drag_start: tuple[int, int] | None = None
        self.drag_now: tuple[int, int] | None = None


def load_csv(path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f) if r.get("face_x")}


def save_annotations(done: dict[str, dict]) -> None:
    ANN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANN_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for fn in sorted(done):
            w.writerow(done[fn])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=sorted(config.KNOWN_SOURCES),
                    help="annotate only this capture session (default: all)")
    args = ap.parse_args()
    wanted = {args.session} if args.session else None
    paths = [p for _src, p in config.iter_images(wanted)]
    if not paths:
        raise SystemExit("no images found for the selected session(s)")
    proposals = load_csv(PROP_PATH)
    done = load_csv(ANN_PATH)
    log.info("%d images, %d already annotated", len(paths), len(done))

    # start at the first un-annotated image
    idx = next((i for i, p in enumerate(paths) if p.name not in done),
               len(paths) - 1)

    state = State()
    win = "annotate  (ENTER=accept  b=back  r=proposal  c=clear  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    cached: dict[str, np.ndarray] = {}

    def load_state_for(p) -> tuple[np.ndarray, float]:
        """Reset per-image state from annotation/proposal; return view+scale."""
        if p.name not in cached:
            cached.clear()                       # keep at most one 53 MP image
            cached[p.name] = pp.load_image_upright(p)
        img = cached[p.name]
        h, w = img.shape[:2]
        s = min(1.0, MAX_DISP_H / h, MAX_DISP_W / w)
        state.box, state.eyes, state.source = None, [], "manual"
        row = done.get(p.name) or proposals.get(p.name)
        if row:
            state.box = tuple(int(row[k]) for k in
                              ("face_x", "face_y", "face_w", "face_h"))
            state.source = ("previous" if p.name in done else "proposal")
            if row.get("eye_lx"):
                state.eyes = [(float(row["eye_lx"]), float(row["eye_ly"])),
                              (float(row["eye_rx"]), float(row["eye_ry"]))]
        view = cv2.cvtColor(cv2.resize(img, (round(w * s), round(h * s)),
                                       interpolation=cv2.INTER_AREA),
                            cv2.COLOR_RGB2BGR)
        return view, s

    def on_mouse(event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drag_start = state.drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.drag_start:
            state.drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state.drag_start:
            x0, y0 = state.drag_start
            state.drag_start = None
            if abs(x - x0) < 5 and abs(y - y0) < 5:        # click -> eye
                if len(state.eyes) >= 2:
                    state.eyes = []
                state.eyes.append((x / scale, y / scale))
            else:                                          # drag -> box
                bx, by = min(x0, x), min(y0, y)
                bw, bh = abs(x - x0), abs(y - y0)
                state.box = (round(bx / scale), round(by / scale),
                             round(bw / scale), round(bh / scale))
                state.source = "manual"

    cv2.setMouseCallback(win, on_mouse)
    view, scale = load_state_for(paths[idx])

    while True:
        frame = view.copy()
        color = {"manual": (0, 255, 0), "previous": (255, 200, 0),
                 "proposal": (0, 255, 255)}[state.source]
        if state.box:
            x, y, w, h = (round(v * scale) for v in state.box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if state.drag_start and state.drag_now:
            cv2.rectangle(frame, state.drag_start, state.drag_now,
                          (0, 255, 0), 1)
        for i, (ex, ey) in enumerate(state.eyes):
            cv2.circle(frame, (round(ex * scale), round(ey * scale)),
                       6, (0, 0, 255) if i == 0 else (255, 0, 255), 2)
        status = (f"[{idx + 1}/{len(paths)}] {paths[idx].name}   "
                  f"annotated: {len(done)}/{len(paths)}   "
                  f"box: {state.source if state.box else 'NONE'}   "
                  f"eyes: {len(state.eyes)}/2")
        cv2.rectangle(frame, (0, frame.shape[0] - 28),
                      (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, status, (8, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, frame)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):                                   # ENTER / SPACE
            if state.box is None or len(state.eyes) != 2:
                log.warning("need a face box AND two eyes before accepting")
                continue
            (lx, ly), (rx, ry) = sorted(state.eyes)           # left = min x
            done[paths[idx].name] = dict(zip(FIELDS, [
                paths[idx].name, *state.box,
                round(lx, 1), round(ly, 1), round(rx, 1), round(ry, 1),
                "proposal-accepted" if state.source == "proposal"
                else "manual"]))
            save_annotations(done)
            if idx + 1 >= len(paths):
                log.info("all images annotated — done!")
                break
            idx += 1
            view, scale = load_state_for(paths[idx])
        elif key == ord("b") and idx > 0:
            idx -= 1
            view, scale = load_state_for(paths[idx])
        elif key == ord("r") and paths[idx].name in proposals:
            row = proposals[paths[idx].name]
            state.box = tuple(int(row[k]) for k in
                              ("face_x", "face_y", "face_w", "face_h"))
            state.eyes, state.source = [], "proposal"
        elif key == ord("c"):
            state.box, state.eyes, state.source = None, [], "manual"
        elif key in (ord("q"), 27):                           # q / ESC
            break

    save_annotations(done)
    cv2.destroyAllWindows()
    log.info("saved %d annotations -> %s", len(done), ANN_PATH)
    if len(done) == len(paths):
        log.info("NEXT: python src/build_face_crops.py   (rebuilds crops "
                 "eye-aligned from ground truth)")


if __name__ == "__main__":
    main()
