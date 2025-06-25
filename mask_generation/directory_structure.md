```sh
field_segmentation/                    # ← top‐level repo
├── README.md                          # 1. High‐level project overview, setup instructions, usage examples
├── LICENSE                            # 2. License file (e.g. MIT, Apache‐2.0, etc.)
├── environment.yaml  or  setup.py     # 3. Conda‐ or pip‐based dependencies, installation instructions
├── requirements.txt                   # 4. Pin versions for “pip install -r requirements.txt”
├── main.py                            # 5a. Entry‐point: @hydra.main(config_path="conf", config_name="config"), dispatches to maskgen, preprocess, train, inference
│
├── conf/                              # 6. All Hydra configs live here
│   ├── config.yaml                    # 6a. Root Hydra config (defaults: [paths, model, train, augment, preprocess, maskgen, inference, hydra/job_logging]). Also holds project‐wide constants
│   │
│   ├── paths/                         # 6b. “paths” group: all filesystem paths
│   │   └── default.yaml               #     • base_dir, data_dir, image_dir, mask_dir, processed dirs, model_save_dir, inference_results_dir, reports_dir, test dirs, external checkpoints, logdir
│   │
│   ├── model/                         # 6c. “model” group: architecture choices
│   │   ├── unet.yaml                  #     • SMP Unet params (encoder_name, encoder_weights, in_channels, classes, activation, etc.)
│   │   └── deeplabv3plus.yaml         #     • SMP DeeplabV3+ params (future use)
│   │
│   ├── train/                         # 6d. “train” group: optimizer, lr, epochs, scheduler, Trainer, callbacks, logger, plus data settings
│   │   └── default.yaml               #     • train.learning_rate, train.batch_size, train.max_epochs, optimizer._target_, scheduler._target_, trainer settings, checkpoint/early_stop/logger settings, train.train_val_split, train.num_workers, train.pin_memory, train.seed
│   │
│   ├── augment/                       # 6e. “augment” group: data augmentation settings
│   │   └── default.yaml               #     • horizontal_flip, vertical_flip, random_rotate, color_jitter, random_resized_crop, gaussian_blur, etc.
│   │
│   ├── preprocess/                    # 6f. “preprocess” group: resizing, normalization, cropping, mask remapping, plus dataset folders
│   │   └── default.yaml               #     • resize.enabled/height/width, normalize.enabled/mean/std, center_crop.enabled/size, remap_mask.enabled/mapping, mask_morphology settings
│   │
│   ├── maskgen/                       # 6g. “maskgen” group: mask generation/refinement settings
│   │   └── default.yaml               #     • classical.enabled/method/min_area/morph settings, sam.enabled/checkpoint/model_type/prompts_csv, refine_with_model.enabled/checkpoint/threshold, manual_qc.enabled/output_qc_dir/overlay_opacity, output_dir
│   │
│   ├── inference/                     # 6h. “inference” group: inference & evaluation settings
│   │   └── default.yaml               #     • inference.checkpoint_path, batch_size, device, tta.enabled/horizontal_flip/vertical_flip, threshold, save_masks, save_overlay, evaluate.enabled/test_image_dir/test_mask_dir/report_csv
│   │
│   ├── evaluation/                    # 6i. “evaluation” group: evaluation and visualization settings
│   │   └── default.yaml               #     • metrics (IoU, Dice, accuracy), evaluation splits, thresholds, visualizations     
│   │
│   └── hydra/                         # 6j. Hydra overrides (job_logging, output dir, etc.)
│       └── job_logging/
│           └── custom.yaml            #     • Custom Python logging config (formatters, handlers, loggers)
│
├── src/                               # 7. All source code under a top‐level Python package
│   ├── __init__.py
│   │
│   ├── data/                          # 7a. Data + transforms
│   │   ├── augmentation.py            #     • Compose augment & preprocess pipelines based on cfg
│   │   └── dataset.py                 #     • Custom Dataset class for field images + masks
│   │
│   ├── models/                        # 7b. Model architecture wrappers / LightningModule
│   │   └── lit_segmentation.py        #     • LightningModule: forward, training_step, validation_step, configure_optimizers (instantiates SMP model via Hydra)
│   │
│   ├── utils/                         # 7c. Utility functions (metrics, logging helpers, callbacks)
│   │   ├── metrics.py                 #     • IoU, Dice coefficient, pixel accuracy, etc.
│   │   ├── logging.py                 #     • Custom logger setup (if needed beyond Hydra’s logging)
│   │   └── callbacks.py               #     • Custom Lightning callbacks (e.g. LR Monitor)
│   │
│   ├── maskgen/                       # 7d. Mask-generation/refinement logic
│   │   ├── threshold.py               #     • Classical CV thresholding, morphological cleanup
│   │   ├── sam_integration.py         #     • Interfaces to Segment Anything Model
│   │   └── refinement.py              #     • Model-assisted proposals and QC overlay
│   │
│   ├── inference/                     # 7e. Inference & evaluation logic
│   │   └── inference.py               #     • Loads checkpoint, runs TTA (optional), thresholds, saves masks, computes IoU vs ground truth
│   │
│   ├── preprocess.py                  # 7f. Secondary entry point for preprocessing mode (calls main())
│   ├── maskgen.py                     # 7g. Secondary entry point for maskgen mode (calls main())
│   ├── train.py                       # 7h. Secondary entry point for training mode (calls main())
│   └── predict.py                     # 7i. Secondary entry point for inference mode (calls main())
│
├── scripts/                           # 8. Shell-level helper scripts (outside Python)
│   ├── train.sh                       # 8a. Example: “bash scripts/train.sh --cfg pipeline.mode=train,model=unet,augment=default,preprocess=default”
│   └── infer.sh                       # 8b. Example: “bash scripts/infer.sh --cfg pipeline.mode=inference,checkpoint=…”
│
├── tests/                             # 9. Unit tests / integration tests
│   ├── test_data.py                   # 9a. Tests for dataset, transforms, DataModule logic (if any)
│   ├── test_models.py                 # 9b. Tests that SMP Unet forward/dimensions are correct
│   ├── test_lit_module.py             # 9c. Tests for Lightning training_step/validation_step logic
│   └── test_maskgen.py                # 9d. (Optional) Tests for mask-generation code
│
├── mask_generation/   
│   └── projects/                      # 11. Project-specific experiment folders
│       └── {project.name}/
│           └── mask_gen/
│               ├── developed-images/
│               │   └── TXC08723.jpg
│               ├── cutouts/
│               │   ├── TXC08723_0.jpg
│               │   ├── TXC08723_0.png
│               │   ├── TXC08723_0.json
│               │   └── TXC08723_0_mask.png
│               ├── refined_masks/
│               │   └── TXC08723_0_mask.png
│               └── db/
│                   └── {project.name}.csv
└── .github/                           # 10. CI/CD workflows, code formatting, etc.
    └── workflows/
        └── ci.yaml                    # 10a. Run pytest, flake8, black, isort, plus any sanity checks (one-epoch train, maskgen sanity)
```
