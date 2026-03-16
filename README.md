# Field Segmentation

![Field Segmentation Pipeline](assets/pipeline_diagram.png)

Field Segmentation is a modular, configuration-driven deep learning pipeline for semantic segmentation of field imagery. The repository supports mask generation, preprocessing, training, inference, and evaluation within a unified and reproducible framework.

The system is built using PyTorch Lightning, Hydra, and `segmentation_models_pytorch`, enabling scalable experimentation and clean configuration management.

## Overview

This repository provides an end-to-end segmentation workflow:

* Mask generation
* Data preprocessing, augmentation, and transformation
* Model training with configurable architectures
* Inference and transfer to LTS storage
* Evaluation and visualization
* Structured experiment management per project

All major behaviors are controlled via Hydra configuration groups, enabling reproducible experiments without modifying source code.

## Repository Structure

```text
field_segmentation/
├── README.md
├── LICENSE
├── environment.yaml
├── main.py
├── conf/
├── src/
├── projects/
└── .github/
```

### Configuration (`conf/`)
Hydra configuration groups:

* `paths/` – Filesystem paths
* `model/` – Model architecture configuration
* `train/` – Optimizer, scheduler, training settings
* `augment/` – Data augmentation configuration
* `preprocess/` – Resizing, normalization, mask remapping
* `maskgen/` – Mask generation and refinement settings
* `llm/` – Large Language Model configuration
* `inference/` – Inference and evaluation configuration
* `evaluation/` – Metrics and visualization settings
* `hydra/` – Logging and runtime behavior

### Source Code (`src/`)
* `data/` – Dataset and augmentation pipelines
* `models/` – LightningModule implementations
* `maskgen/` – Mask generation and refinement logic
* `inference/` – Inference pipeline
* `utils/` – Utilities (GPU handling, seeding, logging)
* `train_utils/` – Training helpers and visualizers
* `preprocess_utils/` – Dataset preparation utilities
* `mask_gen_utils/` – Mask post-processing utilities
* `llm_utils/` – LLM integration utilities

Mode entrypoints: `train.py`, `maskgen.py`, `preprocess.py`, `inference.py`, `llm.py`

## Installation

Clone the Repository:

```bash
git clone [https://github.com/](https://github.com/)<your-org>/field_segmentation.git
cd field_segmentation
```

Create the Environment using Conda:

```bash
conda env create -f environment.yaml
conda activate field_segmentation
```
*Alternatively, install dependencies manually from `environment.yaml`.*

# LLM Setup (Ollama)

This project requires Ollama for local model inference. Follow these steps to install and run it without root/sudo access.

---

## 1. Manual User-Only Install

Download and extract the Ollama bundle directly into your home directory:

```bash
# Create local directory structure
mkdir -p ~/.local

# Download and extract the full bundle (no sudo required)
curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -xvf - -C ~/.local

# Add the binary to your PATH
# (Add this line to your ~/.bashrc for a permanent fix)
export PATH=$PATH:$HOME/.local/bin
```

## Usage

The project uses a single Hydra-based entry point:

```bash
python main.py mode=<mode>
```

Available modes: `maskgen`, `preprocess`, `train`, `inference`, `llm`.

### Mask Generation
Generates segmentation masks.

```bash
python main.py mode=maskgen
```
* **Configuration:** `conf/maskgen/default.yaml`
* **Supports:**
    * Classical thresholding with morphological cleanup
    * Model-based refinement
    * Manual quality control overlays

### Preprocessing
Performs resizing, normalization, cropping, mask remapping, train_val_test split and pad_gridcrop_resize.

```bash
python main.py mode=preprocess
```
* **Configuration:** `conf/preprocess/default.yaml`

### Training
Trains a segmentation model using PyTorch Lightning.

```bash
python main.py mode=train
```

Override configuration inline:

```bash
python main.py mode=train model=deeplabv3plus train.max_epochs=50 train.batch_size=16
```
* **Configuration groups:**
    * `conf/model/`
    * `conf/train/`
    * `conf/augment/`

### Inference
Runs model inference using a trained checkpoint.

```bash
python main.py mode=inference inference.checkpoint_path=path/to/checkpoint.ckpt
```
* **Supports:**
    * Normalization
    * Threshold adjustment
    * Mask saving
    * Overlay saving
* **Configuration:** `conf/inference/default.yaml`

## Model Architectures

Model configurations are located in: `conf/model/`

Currently supported:
* UNet
* DeepLabV3+

Models are instantiated via Hydra and wrapped in a LightningModule defined in `src/models/lit_segmentation.py`.

## LLM Integration (Ollama)

Ollama operates as a client-server model. You must have the server running before executing the code.

### Start the Server
In a separate terminal or tmux session, run:

```bash
ollama serve
```

### Pull the Model
Download the required weights:

```bash
ollama pull granite4
```

### Run the Pipeline
Execute your script:

```bash
python main.py
```

## Data Augmentation

Configured in: `conf/augment/default.yaml`

Includes:
* Spatial augmentations (flip, rotate, crop)
* Photometric augmentations (color jitter, blur)
* Batch-level augmentations:
    * MixUp
    * CutMix
    * Mosaic

Augmentations are dynamically composed in `src/data/augmentation.py`.

## Evaluation

Configured in: `conf/evaluation/default.yaml`

Supported metrics:
* Intersection over Union (IoU)
* Dice coefficient
* Accuracy

Evaluation can generate visualizations and CSV reports.

## Project Organization

Experiments are organized under: `projects/{project.name}/`

Example:
```text
projects/my_project/
├── mask_gen/
│   ├── developed-images/
│   ├── cutouts/
│   ├── refined_masks/
│   └── db/
```

This structure ensures that each project maintains its own:
* Raw images
* Generated masks
* Refined masks
* Metadata
* Outputs and reports

## Logging and Outputs

Hydra automatically creates versioned output directories:

```text
projects/<project>/train/version_YYYYMMDD_HHMMSS/
```

Outputs include:
* Model checkpoints
* Logs
* Augmentation visualizations
* Inference results
* Evaluation reports

Custom logging configuration is defined in: `conf/hydra/job_logging/custom.yaml`

Extensive logging through weights and biases (W&B) is supported for training and inference pipelines.

## Reproducibility

The repository supports reproducible experimentation through:
* Global seed control
* Structured Hydra configuration snapshots
* Deterministic training options
* Versioned output directories

## Continuous Integration

CI configuration is defined in: `.github/workflows/ci.yaml`

## Example Training Command

```bash
python main.py \
  mode=train \
  model=unet \
  train.max_epochs=100 \
  train.batch_size=8 \
  augment.train.batch.mixup.enable=True
```

## License

See the `LICENSE` file for licensing details.