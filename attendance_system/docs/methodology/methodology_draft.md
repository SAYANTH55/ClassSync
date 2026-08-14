# Chapter 3 — Methodology (Working Draft)

> **Status:** draft v1, 18 July 2026. Sections 3.7–3.9 describe designed-but-
> not-yet-executed work and are written in the future/planned tense; they will
> be converted to past tense as phases complete. Citation keys in [brackets]
> are placeholders for the reference manager.

## 3.1 Research Design Overview

This project develops a face-recognition attendance monitoring system in
which **every learned component is trained from scratch** on data collected
within the project. The system comprises two independent inference stages —
face *detection* (localizing a face in the camera frame) and face
*recognition* (identifying the detected face among enrolled students) —
followed by attendance decision logic (Figure `fig_system_architecture.png`).

A constraint governs all design decisions and is stated here precisely: *no
component of the system contains parameters learned from data external to
this project.* Published **algorithms** — histogram-of-oriented-gradients
feature extraction [Dalal & Triggs 2005], support vector machine
optimization [Cortes & Vapnik 1995], convolutional neural networks
[LeCun et al. 1998] — are used as mathematical procedures; every **learned
parameter** (detector weights, recognition network weights, decision
thresholds) is derived exclusively from data collected and annotated by the
author. This distinction (algorithms as public knowledge; weights as the
model) defines the term "from scratch" as used throughout this dissertation.

The deployment target is a **kiosk-mode** capture station: students present
themselves one at a time to a fixed camera. This mirrors commercial
attendance terminals and constrains the detection problem to a single, large,
approximately frontal face — a scenario in which classically trained
detectors remain effective.

## 3.2 Ethical Considerations and Data Protection

Facial images are identifiable personal data. The following controls are
applied: (i) written participant consent is collected and archived for all
imaged students (in progress; see Limitations register, L5); (ii)
participants are pseudonymized at the earliest possible point — an opaque
identifier (S01–S56) is assigned in a single mapping file, and no filename,
figure, log, or result artifact contains a real name; (iii) image data is
excluded from version control and never uploaded to public repositories;
(iv) raw data is stored on a local encrypted-capable volume and treated as
immutable.

## 3.3 Data Collection

**Session 1 (completed).** Fifty-six studio portraits were captured on
9 March 2026 with a Sony ILCE-7RM5 (61 MP full-frame) under controlled
lighting against a uniform grey backdrop, one portrait per student
(6336 × 8448 px upright; 33-minute session; single camera).

**Sessions 2+ (planned, from 20 July 2026).** Because a single image per
identity cannot support supervised training (§3.6, Limitation L2), a
multi-session smartphone protocol is used: per student, at least two videos
of ~15 s in different rooms and lighting conditions, with slow head rotation
(±30°), recorded with and without spectacles where applicable; casual still
photographs may supplement the videos. Media are stored per student at
capture time (`raw_sessions/<session>/<student_id>/`), assigning identity
labels at the moment of collection. A deterministic ingestion step samples
video frames at 3 fps and normalizes all media into uniquely named images
(session tag, student id, and source file are encoded in every filename),
with an ingestion manifest recording each image's origin file, frame index
and timestamp. This yields 60–90 training images per student per session
while preserving an unbroken provenance chain from any training crop back to
the raw video (§3.5, §3.10).

**Why heterogeneous devices.** Capturing across different devices is a
deliberate design choice, not a convenience. Camera pipelines differ in
sensor noise characteristics, optics, tone mapping, sharpening, and JPEG
compression; a network trained on a single device can bind identity to these
device-specific low-level statistics rather than to facial structure — a
form of dataset bias [Torralba & Efros 2011] that inflates in-corpus
accuracy and collapses under deployment. The kiosk camera will differ from
both the studio DSLR and the collection phones, so cross-device robustness
is a *requirement*, not an optimization: training data spanning several
phone models regularizes the model toward device-invariant facial features,
and the session-disjoint evaluation protocol (§3.9) measures exactly this
generalization rather than single-domain memorization. The controlled DSLR
gallery additionally provides an upper-bound test condition free of motion
blur and compression artefacts.

## 3.4 Dataset Inspection

Prior to any preprocessing decision, the raw dataset was audited with a
deterministic inspection script recording, per image: pixel dimensions, EXIF
orientation, capture timestamp, camera identity, exposure parameters,
brightness statistics, a variance-of-Laplacian focus score computed at a
fixed analysis width, and a SHA-256 content hash. Findings: no duplicate or
corrupted files; consistent exposure (grayscale means 106–172/255); all
images stored with EXIF orientation flag 8, i.e. physically rotated on disk —
a property that silently corrupts pipelines whose loaders ignore EXIF
metadata, and which therefore motivated a single mandatory EXIF-correcting
load function through which all images enter the pipeline (§3.6.1).

## 3.5 Data Organization and Labeling Protocol

Identity labels are maintained in a single auditable mapping file
(`labels.csv`) rather than encoded in manually sorted directories,
eliminating a common source of silent label noise in small-scale datasets.
From this file, a build script constructs the class-organized dataset tree
deterministically; every copied file is verified by SHA-256 against its
immutable original. Organized filenames follow
`<student_id>_<source>_<seq>.<ext>`, where `source` is the capture-session
tag (e.g. `dslr`, `phone1`). Labels were assigned provisionally in filename
order and are verified by the researcher against a generated contact sheet;
verification status is tracked per row (Limitation L3 until closed).

**Multi-session organization.** Capture sessions are declared once in a
central session registry (device, date, image location); all tooling
resolves data through it, so adding a session requires no pipeline changes.
The label file is append-only across sessions: new sessions add rows
(pre-labeled from the ingestion filename prefix) without touching verified
earlier rows, and the organized tree can be rebuilt per session, leaving
other sessions' verified copies untouched. Image filenames are globally
unique across sessions by construction, so every metadata file
(labels, annotations, manifests) can key on the bare filename
unambiguously. DSLR and phone data therefore coexist in the same
directories, distinguishable at a glance and fully traceable: crop manifest
→ labels row → ingestion manifest → raw file.

## 3.6 Preprocessing Pipeline

The pipeline (Figure `fig_preprocessing_flowchart.png`) is implemented as a
side-effect-free module of composable functions used identically in training
and deployment, preventing train/serve skew.

### 3.6.1 Geometric normalization

(i) **EXIF-safe loading**: orientation metadata is applied exactly once at
load. (ii) **Face localization**: from human annotation (ground truth) or,
during deployment, the trained detector (§3.7). (iii) **Crop geometry**: the
face box is expanded by a 25 % margin (context tolerance against box jitter)
and padded to square so that resizing never distorts aspect ratio; crops
extending beyond the frame are replicate-padded. (iv) **Alignment**: where
both eye centres are known, a similarity transform maps them to canonical
positions (eye line at 38 % of crop height, inter-ocular distance 42 % of
crop width), removing in-plane rotation and scale variation — a classical
normalization with substantial recognition impact [FERET protocol,
Phillips et al. 2000]. (v) **Resampling**: area-averaging interpolation
(`INTER_AREA`) is used for the ~15× per-axis minification from the 53 MP
portraits, avoiding aliasing artefacts of bilinear/bicubic resampling; the
working input resolution is 112 × 112 px. Crops are stored losslessly (PNG)
with a provenance manifest recording, for every crop, its source image, raw
pixel geometry, alignment status, and generator parameters.

### 3.6.2 Intensity normalization and leakage prevention

Pixel intensities are scaled to [0, 1] at model input. Dataset-level
standardization statistics (mean/standard deviation), where used, are
computed **on the training split only** and applied unchanged to validation
and test data; computing such statistics over the full dataset before
splitting constitutes information leakage and is explicitly avoided.

### 3.6.3 Manual annotation (ground truth)

Each image receives a human-annotated face bounding box and two eye centres
via a purpose-built annotation tool. Human annotation serves three roles:
aligned training crops for recognition; positive samples for detector
training; and the reference against which detector accuracy is quantified
(§3.9). To accelerate annotation, boxes are prefilled by a classical
proposal generator (§3.6.4) and verified or corrected by the researcher.

**Annotation at scale.** Exhaustive manual annotation is applied to the
56-image DSLR session but is infeasible for the several thousand phone
frames. The protocol is therefore two-tier: a stratified sample of phone
frames (per student, spanning videos and lighting conditions) is annotated
manually; the remaining frames are localized by the detector of §3.7 once
trained. Every crop's geometry source (human / detector / classical
proposal) is recorded in the provenance manifest, detector-derived boxes are
spot-checked via QA contact sheets, and detector *evaluation* uses only
human-annotated geometry — detector outputs are never used to score the
detector itself.

### 3.6.4 Classical face localization baseline

A zero-parameter baseline was implemented to (a) prefill annotation and
(b) ground the argument for a trained detector. An initial design —
foreground segmentation by intensity difference from a corner-sampled
background estimate — **failed on all 56 images**: the studio backdrop is
vignetted (bright centre, dark periphery), so no single background intensity
exists, and the brightest backdrop region itself was segmented as
foreground. The revised design segments by *connectivity* instead of
intensity: a Canny edge map, dilated into a barrier, separates smooth
regions; regions connected to the top/side image borders are background, and
the largest fenced-off region is the person's silhouette. The face box is
then derived from the silhouette's row-width profile (head span above the
shoulder width-step) with fixed anthropometric ratios. This method localized
55/56 faces correctly, failing on the one portrait with a headscarf, where
the head-contour assumption does not hold. Both failure modes — lighting
sensitivity and headwear sensitivity — are retained in the dissertation as
empirical evidence of classical-method brittleness, motivating §3.7.

## 3.7 Face Detection Trained from Scratch (planned)

The deployed detector is a sliding-window classifier over a hand-implemented
HOG representation with a linear SVM trained on project data only
(Figure `fig_detector_training_flowchart.png`): positives are HOG vectors of
annotated face windows; initial negatives are random non-face patches from
the same images; the classifier is then iteratively hardened by
**hard-negative mining** — false detections on training images are added to
the negative set and the SVM retrained [Dalal & Triggs 2005; Felzenszwalb
et al. 2010]. Inference scans an image pyramid and merges detections by
non-maximum suppression. All three components (HOG, sliding-window scan,
NMS) are implemented in NumPy by the author; the SVM is optimized with a
standard hinge-loss solver operating solely on project data.

## 3.8 Face Recognition CNN (planned; architecture in Chapter 4)

Recognition is formulated as **closed-set classification** — a softmax
distribution over the N enrolled students — matching the fixed-cohort
semantics of classroom attendance. Unknown-person handling uses a confidence
threshold τ on the maximum class probability, selected on validation data;
captures below τ are rejected and logged rather than marked. Metric-learning
formulations (Siamese/triplet) were considered and deferred: their pair/
triplet data appetite and training instability present disproportionate risk
under the project's data budget, while offering benefits (open-set
enrollment) not required by the fixed-cohort use case. The CNN architecture,
regularization, and augmentation policy are specified in Chapter 4;
training does not begin until the multi-session dataset (§3.3) is complete.

## 3.9 Evaluation Methodology

**Detection** is evaluated against human annotation: mean intersection-
over-union, miss rate, and false positives per image.
**Recognition** is evaluated with accuracy, per-class precision/recall/F1
(macro-averaged), the confusion matrix, and — for the rejection mechanism —
an ROC-style curve of false-accept vs. false-reject rates as τ varies.
**Split policy:** train/validation/test splits are *session-disjoint*: no
capture session contributes to more than one split, so reported test
performance reflects generalization to unseen capture conditions rather than
memorization of session artefacts. The single-session DSLR portraits are
reserved as an additional controlled test gallery. Augmented images, where
used, never cross split boundaries (augmentation is applied after splitting,
to training data only).

## 3.10 Reproducibility

All preprocessing is scripted and deterministic; derived datasets are
rebuildable from raw data plus the label/annotation files. Environment
versions are pinned (`requirements.txt`); all stochastic procedures use a
fixed seed (42); dissertation figures are generated programmatically from
version-controlled code. Raw data is immutable; every derived artifact
carries provenance metadata sufficient to trace it to raw pixels.

## 3.11 Limitations

A living limitations register (`docs/LIMITATIONS.md`) is maintained
throughout the project; at the time of writing it records the single-session
provenance of the initial dataset (L1), its one-image-per-identity structure
blocking CNN training pending further collection (L2), pending identity
verification (L3), the supervisor's ruling excluding all pre-trained
detectors with the consequent from-scratch detector design (L4, resolved),
and pending consent archival (L5).
