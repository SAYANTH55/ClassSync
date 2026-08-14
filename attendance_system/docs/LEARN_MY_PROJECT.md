# ClassSync — Learn My Project From Zero

*A teaching guide for your evaluation. Written from your ACTUAL code, with your
real model names, files, thresholds and logic. Every technical word is explained
in **simple English first**, then technically. Where a thing is not built, it
says **NOT IMPLEMENTED**; where the code can't confirm it, **NOT VERIFIED**.*

> **The one sentence to remember:** three different AI models do three different
> jobs — **SCRFD = where is the face**, **MiniFASNet = is it a real live face or a
> photo/screen**, **ArcFace = who is the person** — and your code glues them into
> an attendance system.

---

## 1. THE BIG PICTURE

**The problem.** Taking attendance by calling names or passing a sheet is slow
and easy to cheat — a friend can mark you present when you're absent ("proxy
attendance").

**What ClassSync does.** A camera at the front of the class looks at the
students, recognises their faces, checks each face is a **real live person and
not a photo on a phone**, and marks who is present — many students at once,
automatically.

**Who uses it.** A teacher/lecturer runs it; the students are the subjects.

**Why it suits a classroom (not just one webcam user).** Your system detects and
recognises **every face in the frame at the same time**, each with its own
confirmation counter — so a whole seated class is handled, not one person walking
up.

**Your real pipeline (this is correct as written):**
```
CAMERA / IMAGE
  ↓
FACE DETECTION      (SCRFD — find every face + 5 landmarks)
  ↓
ANTI-SPOOFING       (MiniFASNet — live or photo/screen?)  ← runs BEFORE ArcFace
  ↓
FACE ALIGNMENT      (use landmarks → 112×112 straight face)
  ↓
FACE EMBEDDING      (ArcFace — face → 512 numbers)
  ↓
FACE MATCHING       (cosine similarity vs stored templates, MAX per student)
  ↓
THRESHOLD           (≥ 0.447 → a student, else "unknown")
  ↓
CONFIRMATION        (seen in 5 frames → mark)
  ↓
ATTENDANCE          (one CSV row per student per day)
```
**One correction to the order you may have seen:** in your code, **anti-spoofing
happens BEFORE ArcFace** — a spoof is stopped before any recognition, so it can
never be marked. (Alignment happens right before ArcFace.)

---

## 2. DATA COLLECTION

You have **three separate kinds of images** — keep them apart, examiners love
this distinction:

- **Phone enrolment selfies** — `data/raw_sessions/phone_enroll/<Name>/`. About
  5 selfies per student on their own phones. **48 students, 256 templates.**
  → **This is ENROLMENT DATA. It builds the gallery the system recognises
  against.**
- **DSLR studio portraits** — an external folder; 53 images (46 named students +
  a DSLR-only student + some `DSC*` strangers) taken on a Sony camera.
  → **This is EVALUATION DATA only. It is NEVER used to enrol.**
- **Attendance CSV files** — `data/attendance/attendance_YYYY-MM-DD.csv`.
  → **This is ATTENDANCE DATA — the daily record of who was present.**

**Four words, never mixed:**
| Word | Simple meaning |
|---|---|
| **Enrolment data** | the phone selfies used to *learn* each student |
| **Gallery** | the stored face vectors built from enrolment data (`gallery.npz`) |
| **Evaluation data** | the DSLR portraits used only to *test* accuracy |
| **Attendance data** | the daily CSV record of presence |

**Why phone images for enrolment?** They're easy to collect and look like what a
webcam sees. **Why DSLR for evaluation?** It's a *different camera* — so testing
"phone-built gallery vs DSLR faces" proves the system works across devices (the
realistic deployment gap), not just on the same phone. **What the DSLR test
actually checks:** does a gallery made from phone selfies still recognise the
same people when the picture comes from a completely different camera months
later? (Answer in your results: yes, 100%.)

---

## 3. ENROLMENT (what happens when you add a student)

**Simple:** a student gives ~5 photos; the system turns each photo's face into a
list of numbers and saves them under that student's name.

**Step by step (your `enroll.py`):**
```
each PHOTO
  → SCRFD detects the face (+ landmarks)
  → face CROP + ALIGNMENT to 112×112
  → ArcFace → 512-D EMBEDDING
  → stored as a TEMPLATE for that student
  → all templates together = the GALLERY (gallery.npz)
```
- **Does every photo make one embedding?** Yes — **1 photo → 1 embedding**. 5
  photos → 5 embeddings.
- **How many embeddings can one student have?** As many as their photos (some
  have 4, most ~5, one has 11).
- **What is a "template" in your project?** One stored embedding for a student.
- **Where are they stored?** In `data/processed/gallery.npz`, grouped by student
  ID. The **roster** (`data/labels/roster.csv`) maps student ID ↔ name.
- **Are original photos or embeddings stored?** The gallery stores **embeddings**
  (the number-vectors). The original photos also stay on disk in the enrolment
  folder, but recognition uses the embeddings.
- **Does it check the images are the same person?** **Yes** — the **centroid
  consistency check**: it averages the embeddings (the "centroid") and every
  photo must be within cosine **0.35** of that average; a stray photo of a
  different person is far from the centre → the whole enrolment is **rejected**,
  naming the bad file. (`enroll.py`, `MIN_CENTROID_SIM = 0.35`.)

---

## 4. SCRFD — from zero (FACE DETECTION)

**Simple:** SCRFD is the "face finder." It looks at a picture and says "there are
faces here, here and here," and marks 5 dots on each (eyes, nose, mouth corners).

- **What it stands for:** *Sample and Computation Redistribution for Face
  Detection.* Your model file: `det_10g.onnx`.
- **Why you need it:** you can't recognise a face until you know **where** it is.
- **Input:** an image.
- **Output per face:** a **bounding box** (a rectangle `[x1,y1,x2,y2]` — 4
  numbers), a **confidence** (0–1), and **5 landmarks** (5 named points).
- **Bounding box** = "a rectangle drawn around the face."
- **Landmarks** = "5 marked points: left eye, right eye, nose, left mouth corner,
  right mouth corner."
- **Multiple faces / a classroom:** SCRFD returns a **list of all faces** in one
  pass. 10 people → 10 boxes. Your code then loops over each.
- **Does SCRFD make embeddings?** **No.** **Does it identify who someone is?**
  **No.** It only finds and locates faces.

**Say this in the viva:**
> **SCRFD = WHERE is the face. ArcFace = WHO is the person. MiniFASNet = is it a
> REAL face or a spoof.**

---

## 5. FACE CROPPING AND ALIGNMENT

**Simple:** cut out just the face from the big photo, then rotate/resize it so
the eyes and nose sit in a standard place — like everyone taking a passport photo
in the same pose.

- **Who crops:** the code uses SCRFD's bounding box to cut out the face region.
- **Bounding box** = the rectangle around the face (from SCRFD).
- **The 5 landmarks** are used to **align** — the face is warped so those 5
  points land on 5 fixed reference positions.
- **Why alignment is needed:** ArcFace was trained on aligned faces; if the face
  is tilted, recognition is worse. Alignment removes pose so only *identity*
  differences remain.
- **Why 112×112:** that is the exact input size ArcFace expects.
- **Which code:** insightface's `norm_crop`, called inside
  `detect_embed.FaceBackend.detect`.
- **What is actually sent to ArcFace:** a **112×112 aligned face crop** (not the
  whole photo).

**Example:** Sayanth tilts his head 20°. Alignment rotates the crop so his eyes
are level and 112×112 — the same standard pose used for everyone.

---

## 6. ARCFACE — from zero (FACE EMBEDDING)

**Simple:** ArcFace is a machine that looks at one face and turns it into a list
of **512 numbers** that acts like that person's "face fingerprint."

- **What it is / why you use it:** a neural network that outputs a face
  **embedding** (identity vector). You use it because these vectors let you
  recognise people the model has *never seen in training* — so you can enrol any
  student without retraining.
- **Did you train it?** **No.** It is **pretrained**.
- **"Pretrained"** = "someone already trained it on a huge dataset, and you just
  download and use the finished model." ArcFace here was trained on
  **WebFace-600K** (~600,000 identities). Model file: `w600k_r50.onnx` (a
  ResNet-50 network).
- **"Learned features"** = the patterns the network learned to look for. During
  training it discovered, on its own, which visual cues separate people (face
  shape, spacing of features, texture) and packed them into its internal numbers
  ("weights"). We don't tell it "look at the nose" — it **learned** what matters.
- **What the weights do:** they transform face pixels into the 512-number vector.
- **What ArcFace outputs:** **ONE embedding = 512 numbers**, scaled to length 1.

**The 512-D confusion, cleared up:**
```
ONE FACE  →  ArcFace  →  ONE EMBEDDING  →  512 NUMBERS
```
- It is **NOT 512 embeddings.** It is **one** embedding that happens to be a list
  of **512** values. "512-dimensional" just means "a list of 512 numbers."
- **Why you can't say "number 1 = nose, number 2 = eye":** the 512 numbers are
  **not human-labelled features.** They are abstract coordinates the network
  invented during training. No single number means "nose." Identity is spread
  across all 512 together. That's what "learned features" means — the machine's
  own internal code for a face, not a checklist we designed.

---

## 7. EMBEDDING (taught separately)

- **What is an embedding?** Simple: "a face turned into a list of numbers (a
  vector) that represents identity." Technical: a point in a 512-dimensional
  space where same-person faces cluster together.
- **Why convert an image into an embedding?** Because comparing raw images is
  unreliable — lighting, angle, and background change the pixels a lot, even for
  the same person. Embeddings strip away those and keep **identity**.
- **Why not compare original images directly?** Two photos of the same person can
  have totally different pixels (different light/pose) → pixel comparison fails.
  Two different people in the same lighting can have *similar* pixels → also
  fails. Embeddings fix both.
- **Why a vector is useful:** you can measure "closeness" between two vectors with
  simple maths (cosine similarity).
- **512-dimensional** = the vector has 512 values.
- **Same person → similar embeddings; different people → dissimilar.** That is
  exactly what ArcFace's training guarantees.

**Numeric idea:**
```
Photo A (Sayanth) → [0.03, -0.11, 0.08, ... ]   (512 numbers)
Photo B (Sayanth) → [0.04, -0.09, 0.07, ... ]   (512 numbers, very similar)
Photo C (someone else) → [-0.20, 0.15, -0.05, ...] (quite different)
```
Comparing A and B mathematically (see §11) gives a **high** score (~0.8);
comparing A and C gives a **low** score (~0.25).

---

## 8. IMAGE vs EMBEDDING vs TEMPLATE vs GALLERY vs ROSTER vs ATTENDANCE

| Term | Simple meaning | In your project |
|---|---|---|
| **Image** | a photo | an enrolment selfie or a camera frame |
| **Embedding** | a face as 512 numbers | ArcFace output |
| **Template** | a *stored* embedding for a student | inside `gallery.npz` |
| **Gallery** | all students' templates together | `gallery.npz` |
| **Roster** | ID ↔ name list | `roster.csv` (S01 → Student A …) |
| **Attendance record** | who was present, when | a row in the day's CSV |

**Example — Student B has 5 enrolment photos:**
```
Photo 1 → embedding → template 1
Photo 2 → embedding → template 2
Photo 3 → embedding → template 3
Photo 4 → embedding → template 4
Photo 5 → embedding → template 5   → all 5 stored under his student ID in the gallery
```
**Why multiple templates help:** each captures a slightly different
angle/expression; at recognition the system uses whichever stored template best
matches the current pose (see §12).

---

## 9. LIVE ATTENDANCE (multiple students at once)

**10 students sitting in class:**
```
Camera frame
  → SCRFD detects all 10 faces
  → FOR EACH face (independently):
        MiniFASNet: live or spoof?  (spoof → stop)
        align → ArcFace → 512-D embedding
        compare to gallery → best student + score
        threshold → recognised or unknown
  → each recognised student's 5-frame counter goes up
  → students who reach 5/5 get marked
```
**How it recognises many people at once:** SCRFD returns all faces, and the code
**loops** over every one — each is detected, spoof-checked, embedded, and matched
on its own. There is **one counter per student**, so several can be confirmed and
marked in the same session.

---

## 10. ANTI-SPOOFING / MiniFASNet

**Why recognition alone isn't enough:** ArcFace only sees a 2-D face image. A
sharp **photo on a phone**, a **printed photo**, or a **digital image** makes
almost the same embedding as the real face — so without a check, a photo could
mark someone present. A **video replay** (playing a video of the person) is the
hardest attack.

- **What MiniFASNet is:** a small, fast neural network for **anti-spoofing
  (liveness)** — it decides *real live person* vs *fake (photo/screen)*.
- **Where it came from / pretrained / did you train it:** from the open-source
  *minivision Silent-Face-Anti-Spoofing* project (ONNX exports); **pretrained;
  you did NOT train it.** Trained on real faces vs spoof attacks (prints,
  screens). Exact training-set details are upstream → **NOT VERIFIED** beyond
  "real vs spoof."
- **Why you chose it:** it's the standard lightweight passive liveness model and
  runs on CPU.
- **Two models (ensemble):** `MiniFASNetV2` (crop scale 2.7) + `MiniFASNetV1SE`
  (crop scale 4.0). Both run; outputs averaged.
- **Input it receives:** a face crop resized to **80×80**, kept **BGR**, fed as
  **raw pixels 0–255** (no /255 — this was a real bug you fixed).
- **Output:** 3 numbers → probabilities. **Class index 1 = LIVE.**
- **LIVE** = a real person in front of the camera. **SPOOF** = a photo/screen.
  **Spoof probability / confidence** = how sure the model is it's fake / live.
- **How LIVE/SPOOF is decided:** average the two models; **LIVE** only if class 1
  is the largest **and** its probability ≥ **0.5** (`LIVENESS_THRESHOLD`).
- **Where in the pipeline / before ArcFace?** Inside
  `detect_embed.FaceBackend.detect`, MiniFASNet runs **BEFORE ArcFace**.
- **If spoof is detected:** ArcFace is **skipped** → no embedding → the face is
  never recognised and **cannot be marked**.
- **Fail-open or fail-closed?** **Fail-CLOSED** (`FAIL_CLOSED = True`): if the
  model is missing/errors, faces are **rejected** (nobody marked) — the safe
  choice for a security feature.
- **Real test you did:** holding a phone showing a face produced a **red "spoof —
  photo/screen" box, 0 recognized, 1 spoof blocked, and no mark** — confirmed
  working. Real webcam faces score ~0.9 live and pass.

---

## 11. COSINE SIMILARITY (from zero)

**Simple:** a number that says how *alike* two vectors are — high when they point
the same way, low when they point differently.

- **Why you need it:** to compare a live face's embedding with the stored
  templates.
- **What's compared:** two 512-number embeddings (live vs a template).
- **Score meaning:** **high (→1) = same person; low (→0.2) = different person.**
- **Range:** −1 to 1 (in practice ~0 to 1 here).

**Tiny example (2-D so you can picture it):**
```
A = [1, 0]      B = [0.9, 0.1]      C = [0, 1]
cosine(A,B) ≈ 0.99  (almost same direction → same person)
cosine(A,C) = 0     (perpendicular → different person)
```
For 512 numbers it's the same idea, just more values: multiply matching numbers
and add them up (because the vectors are length 1, that sum **is** the cosine).

- **Why it fits face embeddings:** ArcFace is trained so identity is encoded in
  the **direction** of the vector — cosine measures direction, so it's the
  natural match.

---

## 12. MULTI-TEMPLATE MATCHING (very important — your real logic)

Your `gallery.py` (`Recognizer.scores`) does this for a live embedding:

```
FOR each student:
    score = MAX( cosine(live, template) over all that student's templates )
THEN pick the student with the highest score.
```

**Example — Student A has 4 templates:**
```
Template 1 = 0.55
Template 2 = 0.83   ← best
Template 3 = 0.60
Template 4 = 0.52
A's score = MAX = 0.83
```
Do the same for every student, then choose the **best student** overall.

```
LIVE EMBEDDING
  → compare against ALL templates of ALL students
  → keep the BEST template score PER student
  → choose the BEST student
  → (then apply the threshold, §13)
```

**Why the max, and why many angles help:** if the live face is turned left and
one stored template was also turned left, that template matches strongly → high
max score. So **more stored angles = more chances a stored template matches the
current pose = more robust recognition.** (This is also why a slightly
side-facing enrolment photo is *good* for recognition even if it looks odd as a
thumbnail — the thumbnail is only a display picture, not a matching input.)

---

## 13. THRESHOLD (your real value)

- **Exact value:** **τ = 0.447**, in `config.RECOG_THRESHOLD`.
- **What a threshold is:** a cut-off score. If the best match is at least this
  good, accept it; otherwise reject.
- **Why it exists:** to avoid forcing every face onto *some* enrolled student —
  strangers must be allowed to be "unknown."
- **How it was chosen (calibrated):** from your DSLR cross-device test — it's the
  **midpoint of the empty gap** between genuine scores (0.607–0.896) and impostor
  scores (0.252–0.286). At 0.447 the test had **0 false accepts and 0 false
  rejects**. (It was measured, not guessed.)
- **score ≥ 0.447 →** recognised as that student.
- **score < 0.447 →** **UNKNOWN** — not marked.
- **This is "open-set recognition":** the system can say "I don't know this
  person," instead of always picking the closest enrolled student. So an
  outsider walking in is labelled unknown, never marked.

**Example:** live face best score 0.71 (Sayanth) ≥ 0.447 → Sayanth. A stranger's
best score 0.28 < 0.447 → unknown.

---

## 14. FIVE-FRAME CONFIRMATION

- **Why multiple frames:** one blurry/lucky frame shouldn't decide attendance.
  Requiring several frames makes it reliable.
- **What "streak" means:** a per-student counter of how many recent frames
  recognised them. The label shows `streak / needed`, e.g. **3/5**.
- **Why it reduces false attendance:** a momentary wrong guess won't reach 5.
- **Leaves the frame / recognition briefly fails:** the streak **decays −1** (down
  to 0) on a missed frame — it does **NOT** hard-reset. This tolerates a quick
  head-turn.
- **How it's finally marked:** when the streak reaches **5** (`CONFIRM_FRAMES =
  5`) and the student isn't already marked today, a CSV row is written.

```
recognised this frame  → streak +1
NOT recognised         → streak -1 (min 0)
streak reaches 5       → MARK (once per day)
```

---

## 15. MULTI-FACE CLASSROOM MODE

**What changed:** originally the live camera used only the **largest** face and
one counter (a single-person kiosk). It was changed so **every** face is
recognised with its **own** counter — a classroom.

**Why the engine didn't change:** `FaceBackend.detect` already returned **all**
faces (each already embedded); the old code just discarded all but the biggest.
So only the live-session wrapper (`ws/camera.py` `ClassroomSession`) and the UI
(`Camera.jsx`) changed.

- **SCRFD** detects all faces; the code **loops** over each.
- **Per-face** anti-spoof → embed → identify.
- **Per-student streaks** (a dictionary `student_id → count`).
- **Many students confirmed at once**, each reaching 5/5 on their own.
- **Unknown faces** → labelled unknown, no streak, never marked.
- **Spoofed faces** → red "spoof", no embedding, never marked.
- **Duplicate prevention:** once marked, a student is in the day's "present" set →
  never marked again.
- **Leaves the frame / turns head:** streak decays (−1), doesn't reset.
- **Frontend:** draws a coloured box + label on **each** face (green = marked,
  amber = confirming n/5, grey = unknown, red = spoof), plus live counters and a
  "Marked present" list.

---

## 16. ATTENDANCE STORAGE (your real implementation)

- **Format: CSV files — NOT a SQL database.** One file per day:
  `data/attendance/attendance_YYYY-MM-DD.csv`. (`take_attendance.AttendanceLog`.)
- **Columns:** `time, student_id, name, score`. The **date** is the filename.
- **student_id** = stable ID (S01…); **name**; **time** = HH:MM:SS of marking;
  **score** = the cosine similarity at that moment.
- **Duplicate prevention / one per day:** the log keeps the set of already-present
  IDs for the day and ignores repeats → exactly one row per student per day.
- **Clear day:** `DELETE /api/attendance/{date}` **archives** the day's file to
  `data/attendance/_archive/` (recoverable, not truly deleted).
- **Archive:** reports only read top-level daily files, so archived files don't
  reappear.
- **Reports:** `attendance_report.py` + the API read the CSVs + roster to produce
  day views, absentee lists, and a per-student summary.

---

## 17. FASTAPI BACKEND

**What is FastAPI?** Simple: a toolkit for building a web server in Python.
**Why you used it:** it supports **async + WebSockets** (needed to stream camera
frames) and runs in the **same Python environment as your AI models**, so no
separate service — and it also serves your React app, so one command runs
everything.

**Endpoints that actually exist:**
| Method | Path | Purpose | Input | Output |
|---|---|---|---|---|
| GET | `/api/health` | server/gallery status | — | status, gallery size, threshold |
| GET | `/api/settings` | model + config info | — | model names, threshold, sizes |
| GET | `/api/dashboard` | today's counts + recent + chart | — | counts, recent marks, day series |
| GET | `/api/students` | list students | — | students + template counts |
| GET | `/api/students/{id}/photo` | face thumbnail | — | a JPEG (or 404) |
| POST | `/api/students` | enrol a student | `name` + `images[]` | `{ok, message}` |
| DELETE | `/api/students/{id}` | remove a student | — | `{ok, message}` |
| GET | `/api/attendance/{date}` | present + absent | — | day lists |
| GET | `/api/attendance/summary` | per-student % | — | summary rows |
| GET | `/api/attendance/{date}/export` | download CSV | — | a CSV file |
| DELETE | `/api/attendance/{date}` | clear/archive a day | — | `{ok, cleared}` |
| WS | `/ws/camera` | live recognition | JPEG frames | JSON per frame |

**WebSocket (`/ws/camera`):** a **two-way, always-open** connection. The browser
sends JPEG frames; the server replies per frame with `{state:"frame", faces:[…],
counts:{…}, marked:[…]}`. This is how the live camera works.

---

## 18. REACT FRONTEND

**What is React?** Simple: a JavaScript library for building web page UIs from
reusable pieces ("components"). Built by Vite into plain files your server sends.

**Pages:**
- **Dashboard** — today's totals, an animated ring, a 14-day chart, recent marks.
- **Camera** — the live attendance screen (below).
- **Students** — roster table (now with **face photos**), add/remove students.
- **Reports** — Daily (present/absent, search, export, Clear day) + Summary
  (per-student %, below-75% flag).
- **Settings** — threshold explained, model names, gallery size (read-only).

**Live camera page shows, per face:**
- a **bounding box** coloured by state,
- **name** + **confirmation counter** ("Name 3/5"),
- recognition **states:** confirming (amber) → marked (green) / unknown (grey),
- **spoof state:** red "spoof — photo/screen",
- **live counters:** detected / recognized / marked / "N spoof blocked",
- a **"Marked present" roster** filling in as students are confirmed.

---

## 19. COMPLETE MODEL TABLE

| Component | Model (file) | Purpose | Pretrained? | Trained by us? | Input | Output |
|---|---|---|---|---|---|---|
| Detector | **SCRFD** (`det_10g.onnx`) | find faces + landmarks | Yes (WIDER FACE) | No | image (BGR) | boxes + 5 landmarks + score |
| Embedder | **ArcFace ResNet-50** (`w600k_r50.onnx`) | face → identity vector | Yes (WebFace-600K) | No | 112×112 aligned face | 512-D unit embedding |
| Anti-spoof #1 | **MiniFASNetV2** (scale 2.7) | live vs spoof | Yes (minivision) | No | 80×80 raw BGR | 3-class probs |
| Anti-spoof #2 | **MiniFASNetV1SE** (scale 4.0) | live vs spoof (ensemble) | Yes (minivision) | No | 80×80 raw BGR | 3-class probs |

(The scratch-CNN and MobileNetV2 in `evaluate_deep.py` are **evaluation
baselines** trained by us — **not** part of the deployed system.)

---

## 20. TRAINING vs INFERENCE

- **Training** (simple): teaching a model by showing it millions of examples and
  slowly adjusting its internal numbers ("weights") until it does the job. Slow,
  done once, needs huge data + GPUs.
- **Inference** (simple): *using* a trained model to get an answer for a new
  input. Fast.
- **Who trained your models:** insightface trained SCRFD (WIDER FACE) and ArcFace
  (WebFace-600K); minivision trained MiniFASNet. **You did none of this.**
- **What your project does at runtime:** **inference only** — it runs the fixed
  models forward.
- **Why you don't retrain ArcFace when a student joins:** because ArcFace already
  turns *any* face into a good identity vector. Adding a student just means
  **storing their new embeddings** in the gallery — **no weights change**. That's
  the big advantage of the embedding approach.

---

## 21. YOUR CONTRIBUTION (crucial for evaluation)

**Pretrained (not your work — tools you used):** SCRFD, ArcFace, MiniFASNet.

**Your engineering/research contribution (what you actually built):**
- **Dataset preparation** — collecting phone + DSLR images; auditing and fixing a
  real duplicate-identity data bug (`dslr_check.py`).
- **Enrolment pipeline** — validated add/remove with the **centroid consistency
  check** (`enroll.py`).
- **Gallery + multi-template recognition** — stable IDs + **max-cosine over
  templates** (`gallery.py`).
- **Threshold calibration** — deriving τ = 0.447 from the score distributions.
- **Open-set rejection** — the "unknown" logic.
- **Multi-face classroom mode** — per-student streaks with decay
  (`ClassroomSession`).
- **Confirmation logic** — the 5-frame rule.
- **Anti-spoofing integration** — sourcing real models, fixing the raw-BGR
  preprocessing, ensembling, fail-closed gating **before** ArcFace.
- **Backend + frontend** — FastAPI server + React web app (dashboard, camera,
  reports, students, settings; face-photo thumbnails).
- **Attendance system** — per-day CSV, dedup, clear/archive, reports.
- **Evaluation + baselines** — the cross-device test + the 5-method comparison.
- **Testing** — the 14-check `smoke_test.py`.
- **Privacy/data handling** — pseudonymous IDs, git-ignored data, private repo.

**Honest one-liner:** *"I didn't train the face models — using pretrained ArcFace
is the correct choice because I can add a student by storing a vector, no
retraining. My work is the whole system around them: the data pipeline, the
gallery and calibrated open-set recognition, the classroom multi-face + anti-spoof
logic, the evaluation, and the full web app."*

---

## 22. EVALUATION (your real results)

**Protocol:** cross-device — gallery from **phone** selfies, tested on **DSLR**
portraits (different camera). `evaluate_dslr.py`.

**Headline:** **Rank-1 = 46/46 = 100%**, **TAR@FAR=0 = 1.000**; genuine scores
0.607–0.896 vs impostor 0.252–0.286 → a clean **0.32-wide gap**; τ = 0.447 sits
in the middle.

**Baselines (identical protocol):** Eigenfaces 26.1%, LBPH 34.8%, scratch-CNN
4.3%, MobileNetV2-transfer 17.4%, **ArcFace 100%**. Plus a per-student bias audit
(`analyze_bias.py`) — all margins positive.

**Latency (`profile_pipeline.py`, CPU):** detection ~230 ms, embedding ~130 ms
per face, matching ~0.6 ms; ~2 FPS single-face.

**Metrics in simple English:**
- **Accuracy / Rank-1** — % of test faces whose top match is the correct person.
- **TAR** — % of genuine faces correctly accepted. **FAR** — % of impostors
  wrongly accepted. "**TAR@FAR=0**" = TAR while allowing **zero** false accepts.
- **Open-set separation** — the *gap* between genuine and impostor scores (bigger
  = safer).
- **Decision margin** — best score minus runner-up (confidence of each decision).
- **Latency** — time per stage/frame.
- **Precision / Recall / F1 / ROC** — **NOT the reported metrics** here → treat as
  **NOT IMPLEMENTED** as named outputs.

---

## 23. LIMITATIONS (real ones)

- **CPU speed:** ~2 FPS single-face; slower with many faces (≈230+K×130 ms) — a
  seated-classroom sweep, not real-time video.
- **Distance / small (back-row) faces:** SCRFD can miss them.
- **Profile faces:** ArcFace accuracy drops for strong side views.
- **Lighting / camera:** thresholds are tuned for this setup; a very different
  camera may need re-tuning.
- **Anti-spoofing:** passive 2-D; strong vs photos/screens, but a perfect
  high-quality **video replay** is the hardest case — not guaranteed.
- **Occlusion (mask/hand), twins/look-alikes:** not specifically tested → **NOT
  VERIFIED**.
- **No SQL database, no login/authentication** → **NOT IMPLEMENTED** (fine for one
  classroom).
- **Fail-closed cost:** if the anti-spoof model is missing, nobody is marked.

---

## 24. FILE-BY-FILE

- **`config.py`** — all paths + constants (τ=0.447, model names, folders).
  Connects: everything.
- **`preprocessing.py`** — `load_image_upright` (EXIF-upright RGB, HEIC).
- **`detect_embed.py`** — `FaceBackend.detect`: SCRFD → **anti-spoof** → (if live)
  ArcFace; returns `Face` objects. Connects: `anti_spoof`, `gallery`, camera.
- **`anti_spoof/`** — `utils` (models, scales, LIVE_CLASS=1, threshold 0.5),
  `preprocessing` (80×80 raw BGR), `model_loader` (load both ONNX), `inference`
  (`analyze_face`, ensemble, `FAIL_CLOSED=True`).
- **`gallery.py`** — roster + gallery + `Recognizer` (`scores` = max cosine per
  student; `identify` = best ≥ τ else None). Connects: camera, enrol.
- **`take_attendance.py`** — `AttendanceLog` (per-day CSV, dedup) + a single-face
  CLI kiosk.
- **`attendance_report.py`** — day/summary/absentee reporting.
- **`enroll.py`** — `enroll` (validate + centroid check + rebuild), `remove`.
- **`smoke_test.py`** — 14-check end-to-end verification.
- **`dslr_check.py`** — dataset audit (hashes, identity reconciliation).
- **`evaluate_dslr.py`** — the headline cross-device evaluation.
- **`evaluate_classical.py`** — Eigenfaces + LBPH baselines. **`evaluate_deep.py`**
  — scratch-CNN + MobileNetV2 baselines (**only place training happens**, for
  baselines).
- **`profile_pipeline.py`** — per-stage latency. **`build_thumbnails.py`** — face
  thumbnails for the UI.
- **Backend:** `main.py` (app + SPA), `deps.py` (engine singletons + lock),
  `schemas.py`, `routers/{dashboard,attendance,students,settings}.py`,
  `ws/camera.py` (`ClassroomSession`, `_process`).
- **Frontend:** `App.jsx`, `api.js`, `pages/{Dashboard,Camera,Reports,Students,
  Settings}.jsx`, `components/{Sidebar,Topbar,Logo,Ring,Avatar}.jsx`,
  `hooks/useCountUp.js`.

---

## 25. EXPLAIN IT LIKE YOU'RE PRESENTING (ordered)

1. **Problem** — attendance is slow and easy to fake with a photo.
2. **Data** — phone selfies to enrol; DSLR portraits only to test.
3. **Enrolment** — each photo → an embedding → stored as templates in the gallery;
   a centroid check ensures all photos are the same person.
4. **SCRFD** — finds every face + 5 landmarks.
5. **Alignment** — landmarks straighten the face to 112×112.
6. **ArcFace** — turns the face into 512 numbers.
7. **512-D embedding** — one vector of 512 values = the face "fingerprint."
8. **Template** — a stored embedding for a student.
9. **Gallery** — all templates of all students.
10. **Live camera** — sends frames over a WebSocket.
11. **MiniFASNet** — checks each face is live, **before** recognition; spoof →
    stop.
12. **ArcFace again** — embed the live face.
13. **Cosine similarity** — compare to templates.
14. **Threshold 0.447** — accept above it, else "unknown."
15. **Five-frame confirmation** — seen 5 frames → mark.
16. **Attendance** — one CSV row per student per day.
17. **Multi-face** — every face handled independently, own counter.
18. **Evaluation** — 100% cross-device, beats 4 baselines.
19. **Contribution** — the whole system around the pretrained models.

---

## 26. 2-MINUTE VERSION (say it naturally)

"My project is ClassSync, a face-recognition attendance system for classrooms
that also blocks cheating with photos. Normal attendance is slow and someone can
mark an absent friend present, so I automated it with a camera.

It has two parts. First, enrolment: each student gives about five phone selfies.
I use a face detector called SCRFD to find the face, then a pretrained model
called ArcFace to turn each face into a list of 512 numbers — an embedding, which
is basically a face fingerprint. I store these as templates in a gallery, and I
check all the photos are the same person before saving.

Second, live attendance: the camera streams frames to my server. For every face,
I first run an anti-spoofing model, MiniFASNet, which decides if it's a real
person or a photo on a phone. If it's a spoof, I stop there — I don't even
recognise it. If it's live, I make its embedding and compare it to the gallery
using cosine similarity, taking the best match per student. If that best score is
above my threshold of 0.447, it's that student; otherwise it's 'unknown'. To
avoid mistakes, a student must be recognised in five frames before I mark them,
and I store attendance once per day in a CSV file. It handles many students at
once, each with their own counter.

I didn't train the face models — using pretrained ArcFace is the right choice
because I can add a new student just by storing their vector, with no retraining.
My contribution is the whole system: the data pipeline, the gallery and the
calibrated open-set recognition, the multi-face classroom logic, the
anti-spoofing integration, and a full web app in FastAPI and React. I also
evaluated it — enrol on phones, test on a different DSLR camera — and it got 100%
recognition with zero false accepts, beating four other methods."

---

## 27. 30-SECOND VERSION

"ClassSync takes classroom attendance with a camera. Students enrol with a few
selfies. In class, for every face, I first check it's a real person and not a
photo — using MiniFASNet — then recognise it with ArcFace embeddings and cosine
similarity, and mark them present after five steady frames. It handles many
students at once, rejects unknowns, blocks phone-photo spoofs, and scored 100% on
a cross-device test. The face models are pretrained; my work is the whole system
around them."

---

## 28. 50 QUESTIONS (short + detailed)

1. **What is SCRFD?** — Short: the face detector. Detailed: a single-stage CNN
   that finds all faces + 5 landmarks; pretrained on WIDER FACE; file
   `det_10g.onnx`.
2. **Why SCRFD?** — Short: fast, multi-face, gives landmarks. Detailed: needed for
   alignment and classroom multi-face.
3. **What is ArcFace?** — Short: the face recogniser (embedder). Detailed:
   ResNet-50 trained with angular-margin loss on WebFace-600K; outputs a 512-D
   vector.
4. **Why ArcFace?** — Short: its embeddings generalise to new people. Detailed:
   enrol by storing a vector, no retraining.
5. **What is an embedding?** — Short: a face as 512 numbers. Detailed: a point in
   identity space; same person → close.
6. **Why 512 dimensions?** — Short: that's the model's output size. Detailed: 512
   learned values encode identity together.
7. **Is it 512 embeddings?** — Short: no, one embedding of 512 numbers.
8. **What is a template?** — Short: a stored embedding for a student. Detailed:
   several per student in `gallery.npz`.
9. **What is a gallery?** — Short: all students' templates. Detailed: the database
   of known faces.
10. **Who creates the embedding?** — Short: ArcFace. Not SCRFD.
11. **Who assigns the 512 numbers?** — Short: ArcFace, from its learned weights.
12. **What are learned features?** — Short: patterns the network discovered in
    training. Detailed: not human-labelled; no single number = "nose."
13. **Did you train ArcFace?** — Short: no, pretrained. Detailed: I only run
    inference.
14. **What does pretrained mean?** — Short: already trained by others; I download
    and use it.
15. **Why not retrain when a student joins?** — Short: I just store new
    embeddings. Detailed: weights never change; that's the point.
16. **What is cosine similarity?** — Short: how aligned two vectors are, −1..1.
    Detailed: dot product of unit vectors; high = same person.
17. **Why 0.447?** — Short: calibrated midpoint between genuine and impostor
    scores. Detailed: 0 false accept/reject on the DSLR test.
18. **What happens below threshold?** — Short: unknown, not marked.
19. **Unknown person?** — Short: labelled unknown (open-set), never marked.
20. **Phone image shown?** — Short: MiniFASNet flags spoof → ArcFace skipped →
    not marked.
21. **What is MiniFASNet?** — Short: the anti-spoofing model. Detailed: ensemble
    of two small CNNs, class 1 = live, threshold 0.5.
22. **Why MiniFASNet?** — Short: standard lightweight liveness model on CPU.
23. **What is liveness detection?** — Short: real person vs photo/screen.
24. **Why multiple templates?** — Short: more angles → more robust matching.
25. **Why 5 frames?** — Short: one frame could be wrong; 5 makes it reliable.
26. **How does multi-face work?** — Short: SCRFD returns all faces; loop + a
    counter per student.
27. **Someone leaves the frame?** — Short: their streak decays, doesn't reset.
28. **Why DSLR images?** — Short: to test on a *different camera* (cross-device).
29. **Your contribution?** — Short: the whole system around pretrained models.
30. **Limitations?** — Short: CPU speed, back-row/profile faces, passive
    anti-spoof, no DB/auth.
31. **Detection vs recognition?** — Short: where vs who.
32. **What is alignment?** — Short: straighten the face to 112×112 using
    landmarks.
33. **Why 112×112?** — Short: ArcFace's required input size.
34. **BGR vs RGB?** — Short: colour-channel order; each model needs its own; we
    match it.
35. **What is ONNX?** — Short: a model file format; runs via ONNX Runtime on CPU.
36. **What is inference?** — Short: running a trained model to get outputs.
37. **Fail-open vs fail-closed?** — Short: pass vs reject when the model is
    missing; we use fail-closed.
38. **Can a spoof be marked?** — Short: no — no embedding is even made.
39. **How is attendance stored?** — Short: one CSV per day (`time,student_id,
    name,score`).
40. **Do you use a database?** — Short: no, CSV files.
41. **Duplicate marking?** — Short: prevented — one row per student per day.
42. **How many students/images?** — Short: 48 students, 256 templates.
43. **Enrolment vs evaluation data?** — Short: phone selfies enrol; DSLR only
    tests.
44. **Main result?** — Short: 100% rank-1, zero false accepts, cross-device.
45. **What are baselines?** — Short: Eigenfaces/LBPH/scratch-CNN/MobileNetV2 — all
    far worse.
46. **TAR/FAR?** — Short: accept rate for genuine / for impostors; we report
    TAR@FAR=0 = 1.0.
47. **Why FastAPI?** — Short: async + WebSockets, same env as the AI.
48. **Why a WebSocket for the camera?** — Short: keep sending frames + getting
    results continuously.
49. **Why React?** — Short: component UI, built to static files served by the
    server.
50. **How does a new student appear instantly?** — Short: enrol → embeddings added
    → gallery reloaded; no training.

---

## 29. FINAL MEMORY SHEET

**THE 20 THINGS I MUST REMEMBER**
1. **SCRFD = where**, **MiniFASNet = real or fake**, **ArcFace = who.**
2. All three are **pretrained; I did not train them** (only inference).
3. **Enrol = phone selfies; DSLR = evaluation only.**
4. **48 students, 256 templates** in `gallery.npz`.
5. **1 face → 1 embedding = 512 numbers** (not 512 embeddings).
6. The 512 numbers are **learned features** — no single one means "nose."
7. **Templates** = stored embeddings; each student has several.
8. Recognition = **max cosine over a student's templates**, then best student.
9. **Threshold τ = 0.447**, calibrated (midpoint of genuine vs impostor).
10. **Below threshold → unknown** (open-set rejection).
11. **MiniFASNet runs BEFORE ArcFace**; spoof → no embedding → never marked.
12. Anti-spoof = **2-model ensemble, 80×80 raw BGR, class 1 = live, 0.5,
    fail-closed.**
13. **Multi-face:** every face handled independently, **per-student streaks**.
14. **Mark at 5/5**; streak **decays −1** on a miss (no reset).
15. **Attendance = CSV**, one row per student per day; **no SQL DB.**
16. Enrolment has a **centroid same-person check (≥ 0.35).**
17. **Result: 100% rank-1, TAR@FAR=0 = 1.0**, cross-device; baselines 4–35%.
18. **Backend = FastAPI (REST + /ws/camera); frontend = React; CPU only.**
19. **My contribution = the whole system** around pretrained models.
20. **Key line:** pretrained ArcFace + a stored-vector gallery = add a student
    with **no retraining**.

**The final pipeline (corrected to your code):**
```
CAMERA
  ↓
SCRFD
  ↓
FACE BOX + 5 LANDMARKS
  ↓
MINIFASNET  →  LIVE / SPOOF   (spoof → STOP: no ArcFace, no mark)
  ↓ (live)
FACE ALIGNMENT (112×112)
  ↓
ARCFACE
  ↓
1 EMBEDDING = 512 NUMBERS
  ↓
COMPARE WITH STORED TEMPLATES  (max cosine per student)
  ↓
COSINE SIMILARITY
  ↓
THRESHOLD (0.447)
  ↓
KNOWN / UNKNOWN
  ↓
5-FRAME CONFIRMATION (decay on miss)
  ↓
ATTENDANCE (one CSV row per student per day)
```
*Only correction vs your draft: **MiniFASNet comes before ArcFace and before
alignment's embedding step** — a spoof is rejected first, so it never reaches
ArcFace.*
