# MaskGen pipeline: `create_project` → `detect` → `segment`

This guide explains how the three core tasks work together, what they read/write, the config keys they rely on, and how to run them.

---

## High‑level flow

1. **`create_project`**

   * Pulls a sample of images (by species) from the *AgIR field* SQLite DB.
   * Copies those images from long‑term storage (LTS) to a local project folder under `project_maskgen_dir/developed-images`.
   * Writes a **project temp CSV** containing the sampled rows plus a `local_developed_image_path` column.

2. **`detect`**

   * Loads the temp CSV.
   * For each row, resolves the on‑disk image path (preferring `local_developed_image_path`).
   * Runs YOLO weed detection.
   * Writes the detection back into the CSV (`bbox_xywh`, `det_pred_conf`, `detection_note`).

3. **`segment`**

   * Loads the temp CSV.
   * For each row with a valid `bbox_xywh`, crops the image ROI and runs a UNet to get a binary mask.
   * Saves **two** mask artifacts (crop‑space mask and full‑frame mask) and a crop image; updates CSV with paths and notes.

---

## Shared assumptions & artifacts

* **Project temp CSV**: single source of truth passed between tasks (created by `create_project`, appended by `detect`, also appended by `segment`).
* **Image location resolution**: all tasks try to use `local_developed_image_path` first; otherwise they derive `<stem>.<extension>` under `project_maskgen_dir/developed-images`.
* **Repo‑relative paths** are used where possible in CSV for portability.

---

## `create_project`

### What it does

* Connects to the AgIR field SQLite DB and loads `field_data`.
* Filters to `extension == 'jpg'` and `is_preprocessed == true`, then excludes `image_index in (0,1)`.
* Samples **N images per species** according to `cfg.create_project.species_images`.
* Copies those images from **LTS** (`cfg.paths.longterm_storage`) into the **project** under `project_maskgen_dir/developed-images`.
* Writes the **project temp CSV** to `cfg.paths.project_temp_db` with a new `local_developed_image_path` column.

### Key inputs (config)

* `paths.base_dir` — repo root (used for relative pathing)
* `paths.project_maskgen_dir` — project working dir (receives images/masks)
* `paths.longterm_storage` — LTS root (source images)
* `paths.project_temp_db` — CSV file written/updated across tasks
* `paths.agir_field_db` — SQLite DB file
* `create_project.species_images` — mapping of `{species: n_images}` to sample
* `create_project.seed` — random seed for sampling

### Columns added/updated in CSV

* `local_developed_image_path` — repo‑relative path to the copied image

### Output on disk

* `project_maskgen_dir/developed-images/` — copied JPGs
* `paths.project_temp_db` — CSV for downstream tasks

---

## `detect`

### What it does

* Loads a YOLO model from `cfg.paths.yolo_weed_detection_model`.
* For each row in the temp CSV, resolves the image path and runs detection.
* Stores **one** bbox per image (the highest‑confidence detection if multiple are found).
* Persists results back into the same CSV.

### Key inputs (config)

* `paths.base_dir`
* `paths.project_maskgen_dir`
* `paths.project_temp_db`
* `paths.yolo_weed_detection_model` — YOLO weights file

### Columns added/updated in CSV

* `bbox_xywh` — JSON string `[x, y, w, h]` (ints)
* `det_pred_conf` — float confidence score
* `detection_note` — notes such as “No detection” or file missing

### Output on disk

* (No new files; CSV is updated in place.)

### Detection behavior

* If **no detections**: `bbox_xywh = NaN`, `det_pred_conf = NaN`, `detection_note = "No detection"`.
* If **multiple detections**: keep the **max‑confidence** bbox; a note is recorded internally and `detection_note` remains empty unless there’s an issue.

---

## `segment`

### What it does

* Loads the temp CSV and a UNet from `cfg.paths.unet_segmentation_model`.
* For each row with a valid `bbox_xywh`:

  1. Resolve the image path; crop the ROI.
  2. Predict a **binary mask** over the crop (with `sigmoid > 0.5`).
  3. If the crop is huge, segment in **tiles** and merge.
  4. **Embed** the crop‑mask back into full‑image coordinates.
  5. Save artifacts and update CSV.

### Key inputs (config)

* `paths.base_dir`
* `paths.project_maskgen_dir`
* `paths.project_temp_db`
* `paths.unet_segmentation_model` — UNet weights file

### Columns added/updated in CSV

* `initial_mask_path` — repo‑relative path to **full‑frame mask** PNG
* `initial_cutout_mask_path` — repo‑relative (or absolute fallback) path to **crop‑space mask** PNG
* `cutout_name` — PNG cutout filename
* `seg_note` — error/skip reasons (e.g., no bbox, image missing)

### Output on disk

* `project_maskgen_dir/initial_masks/<stem>_mask.png` — **full‑frame mask** (binary 0/255)
* `project_maskgen_dir/cutouts/<stem>_0_mask.png` — **crop‑space mask** (binary 0/255)
* `project_maskgen_dir/cutouts/<stem>_0.jpg` — **crop image** (for review)
* `project_maskgen_dir/cutouts/<stem>_0.png` — **masked crop** (background zeroed)

### Model & inference details

* UNet (`encoder=resnet34`, `encoder_weights=imagenet`, `classes=1`) loaded to CPU/GPU depending on availability.
* Inference wrapped in `torch.no_grad()`; masks thresholded at 0.5.
* Tile inference used if H or W of the crop exceeds \~4k (configurable in code).

---

## Orchestration patterns

You can run tasks individually or through an orchestrator. Two common options in this repo:

* **Via `mask_gen.py` task registry**: iterates the boolean map at `cfg.tasks.mask_gen` and runs enabled tasks in order.
* **Via `main.py`**: a broader pipeline entry point with a `mode` switch (e.g., `mode=mask_gen`) and a run‑scoped logger/report.

---

## How to run (examples)

> These are examples; adjust to your repo layout and Hydra config filenames.

### A) Run via `main.py` (preferred when logging/reporting across modes)

```bash
python -m src.main mode=mask_gen \
  tasks.mask_gen.create_project=true \
  tasks.mask_gen.detect=true \
  tasks.mask_gen.segment=true \
  hydra.run.dir=./runs/${now:%Y-%m-%d_%H-%M-%S}
```

### B) Run individual tasks (debugging)

```bash
# create_project only
python -m src.mask_gen_utils.create_project

# detect only
python -m src.mask_gen_utils.detect

# segment only
python -m src.mask_gen_utils.segment
```

### C) Run via `mask_gen.py` directly (iterates enabled tasks)

```bash
python -m src.mask_gen \
  tasks.mask_gen.create_project=true \
  tasks.mask_gen.detect=true \
  tasks.mask_gen.segment=true
```

---

## Minimal config checklist

```yaml
paths:
  base_dir: /abs/path/to/repo
  project_maskgen_dir: ${paths.base_dir}/projects/my_project
  longterm_storage: /mnt/research-projects/.../longterm_images3
  project_temp_db: ${paths.project_maskgen_dir}/project_temp.csv
  agir_field_db: /abs/path/to/AgIR_field.db
  yolo_weed_detection_model: /abs/path/to/yolo.pt
  unet_segmentation_model: /abs/path/to/unet.pth

create_project:
  species_images: { "ragweed parthenium": 50, "amaranthus palmeri": 50 }
  seed: 42

tasks:
  mask_gen:
    create_project: true
    detect: true
    segment: true
```

---

## Operational notes & pitfalls

* **DB schema assumptions**: `create_project` expects columns including `extension`, `is_preprocessed`, `image_index`, and `app_species` in `field_data`. If your DB doesn’t have these, add a view or adjust the loader filter.
* **Missing files**: `detect` will mark `detection_note = "Image not found"` if the derived path doesn’t exist. Verify `local_developed_image_path` or your `<stem>.<extension>` naming.
* **Multiple detections**: only the most confident detection is retained per image (simplifies the segmentation step).
* **GPU/CPU**: segmentation selects CUDA if available; otherwise CPU. Large crops are tiled to avoid OOM.
* **Path relativity**: outputs try to be **repo‑relative**; if directories are outside `base_dir`, absolute paths may end up in the CSV.

---

## What you get at the end

* A curated set of images in the project folder.
* A temp CSV with detection and segmentation metadata/paths.
* Initial masks and cutouts on disk, ready for inspection, refinement, or export to downstream tools.

---

## Next steps (downstream tasks)

* **Inspect/QA**: visualize crops/masks and flag issues.
* **Refine**: run your refinement stage to post‑process masks.
* **Export/Upload**: integrate with CVAT or other tooling for annotation polishing.
