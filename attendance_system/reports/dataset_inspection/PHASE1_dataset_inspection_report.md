# Phase 1 — Dataset Inspection Report

**Project:** AI-Based Face Recognition Attendance Monitoring System Using a CNN Trained from Scratch
**Phase:** 1 of 12 — Dataset Inspection
**Date of inspection:** 18 July 2026
**Constraint:** No pre-trained models or transfer learning at any stage (supervisor requirement).

---

## 1. Provenance

| Item | Value |
|---|---|
| Source archive | `Attendance_monitoring_system-20260718T093929Z-1-001.zip` (737.9 MB, Google Drive export) |
| Camera | Sony ILCE-7RM5 (A7R V, 61 MP full-frame) — single camera for all images |
| Capture session | **One session**: 9 March 2026, 12:07:57 → 12:40:56 (33 minutes) |
| Setting | Indoor studio: plain grey backdrop, controlled professional lighting |
| Images | 56 JPEG files, `DSC09046`–`DSC09190` (56 kept of ~145 shutter actuations) |

## 2. Technical properties

| Property | Finding |
|---|---|
| Native resolution | 8448 × 6336 stored; **6336 × 8448 portrait after EXIF correction** (all 56 identical) |
| EXIF orientation | Flag = 8 on every image (stored rotated 90°). **Pipelines must apply EXIF transpose before any processing** — OpenCV's `imread` ignores EXIF and would produce sideways faces. |
| File size | 9.8 – 15.3 MB per image (mean ≈ 13.2 MB) |
| Exact duplicates | 0 (SHA-256 verified) |
| Brightness (grayscale mean) | 106 – 172 / 255 — consistent studio exposure, no under/over-exposed frames |
| Focus (variance of Laplacian @ 1024 px width) | 75 – 204, median 134. Softest frames: `DSC09099`, `DSC09069`, `DSC09101`, `DSC09137`, `DSC09167`. None rejected — at 53 MP even the softest frame yields a sharp face crop at CNN input scale. |

## 3. Content characteristics (visual inspection + contact sheet)

- **One person per image** — formal head-and-shoulders studio portraits.
- Uniform appearance: dark blazer, white shirt, university lanyard; frontal pose; neutral-to-smiling expression.
- Identical background and lighting across all images.
- Contact-sheet review indicates the 56 images depict **≈56 distinct individuals (one image per person)** — pending confirmation by manual labeling (`labels_template.csv`).

## 4. Critical assessment — data sufficiency

**Finding F1 (blocking): the dataset cannot train a from-scratch CNN classifier in its current form.**
With one image per identity there is no possibility of a train/validation/test split within a class. Augmenting a single image and distributing the augmented copies across splits constitutes **data leakage** (validation samples would be near-duplicates of training samples), which invalidates all reported metrics.

**Finding F2: zero intra-class variation.** All variation dimensions relevant to deployment (lighting, pose, expression, background, camera, scale) are constant. A model trained only on this data cannot be expected to generalise to classroom/webcam conditions (*domain shift*).

**Finding F3 (positive): the dataset is an excellent enrollment gallery.** Uniform, high-resolution, well-exposed portraits are ideal as reference/identity images and as a clean **held-out test set** captured under controlled conditions.

## 5. Recommendations

1. **R1 — Confirm identities.** Fill `labels_template.csv` using `contact_sheet.jpg` (assign `student_id` S01…S56; names optional).
2. **R2 — Collect a training set (required).** Per student: two short smartphone videos (~15 s each) in different rooms/lighting, slowly varying head pose ±30°, with/without glasses where applicable. Frame extraction at ~3 fps yields 60–90 usable images per person. This satisfies the from-scratch constraint (data collection, not model reuse) and is standard practice in attendance-system literature.
3. **R3 — Session-disjoint evaluation.** Reserve the DSLR portraits (this dataset) and/or one video session as test data captured in conditions disjoint from training — a methodological strength examiners look for.
4. **R4 — Ethics documentation.** Obtain and archive written consent from every participant (images are identifiable personal data). Record consent status in the dissertation's ethics section.
5. **R5 — Raw data immutability.** `data/raw/` is never modified; all processing writes to derived directories.

## 6. Reproducibility

- Inspection script: `src/inspect_dataset.py` (deterministic, no random components).
- Environment: Python 3.13.11, Pillow 12.1.0, NumPy 2.2.6 (OpenCV 4.13.0, TensorFlow 2.21.0 available for later phases), Windows 11.
- Outputs: `image_metadata.csv` (per-image metrics), `contact_sheet.jpg` (labeling aid), `labels_template.csv` (identity mapping, to be completed).

---

*Prepared as part of Phase 1. Phase 2 (Dataset Organization) begins after identity confirmation (R1) and a decision on training-data collection (R2).*
