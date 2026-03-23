```sh
field_segmentation/                    # ← top‐level repo
├── README.md                          # 1. High‐level project overview, setup instructions, usage examples
├── LICENSE                            # 2. License file (e.g. MIT, Apache‐2.0, etc.)
├── environment.yaml                   # 3. Conda‐ or pip‐based dependencies, installation instructions
├── main.py                            # 4. Entry‐point: @hydra.main(config_path="conf", config_name="config"), dispatches to maskgen, preprocess, train, inference
│
├── conf/                              # 5. All Hydra configs live here
│   ├── config.yaml                    # 5a. Root Hydra config (defaults: [paths, model, train, augment, preprocess, maskgen, inference, hydra/job_logging]). Also holds project‐wide constants
│   │
│   ├── paths/                         # 5b. “paths” group: all filesystem paths
│   │   └── default.yaml               #     • base_dir, data_dir, image_dir, mask_dir, processed dirs, model_save_dir, inference_results_dir, reports_dir, test dirs, external checkpoints, logdir
│   │
│   ├── model/                         # 5c. “model” group: architecture choices
│   │   ├── unet.yaml                  #     • SMP Unet params (encoder_name, encoder_weights, in_channels, classes, activation, etc.)
│   │   └── deeplabv3plus.yaml         #     • SMP DeeplabV3+ params (future use)
│   │
│   ├── train/                         # 5d. “train” group: optimizer, lr, epochs, scheduler, Trainer, callbacks, logger, plus data settings
│   │   └── default.yaml               #     • train.learning_rate, train.batch_size, train.max_epochs, optimizer._target_, scheduler._target_, trainer settings, checkpoint/early_stop/logger settings, train.train_val_split, train.num_workers, train.pin_memory, train.seed
│   │
│   ├── augment/                       # 5e. “augment” group: data augmentation settings
│   │   └── default.yaml               #     • horizontal_flip, vertical_flip, random_rotate, color_jitter, random_resized_crop, gaussian_blur, etc.
│   │
│   ├── preprocess/                    # 5f. “preprocess” group: resizing, normalization, cropping, mask remapping, plus dataset folders
│   │   └── default.yaml               #     • resize.enabled/height/width, normalize.enabled/mean/std, center_crop.enabled/size, remap_mask.enabled/mapping, mask_morphology settings
│   │
│   ├── maskgen/                       # 5g. “maskgen” group: mask generation/refinement settings
│   │   └── default.yaml               #     • classical.enabled/method/min_area/morph settings, sam.enabled/checkpoint/model_type/prompts_csv, refine_with_model.enabled/checkpoint/threshold, manual_qc.enabled/output_qc_dir/overlay_opacity, output_dir
│   │
│   ├── llm/                           # 5h. “llm” group: LLM settings for table generation
│   │   └── default.yaml               #     • llm.model_name, llm.max_tokens, llm.temperature, etc.
│   │
│   ├── inference/                     # 5i. “inference” group: inference & evaluation settings
│   │   └── default.yaml               #     • inference.checkpoint_path, batch_size, device, tta.enabled/horizontal_flip/vertical_flip, threshold, save_masks, save_overlay, evaluate.enabled/test_image_dir/test_mask_dir/report_csv
│   │
│   ├── evaluation/                    # 5j. “evaluation” group: evaluation and visualization settings
│   │   └── default.yaml               #     • metrics (IoU, Dice, accuracy), evaluation splits, thresholds, visualizations     
│   │
│   └── hydra/                         # 5k. Hydra overrides (job_logging, output dir, etc.)
│       └── job_logging/
│           └── custom.yaml            #     • Custom Python logging config (formatters, handlers, loggers)
│
├── src/                               # 6. All source code under a top‐level Python package
│   ├── __init__.py
│   │
│   ├── data/                          # 6a. Data + transforms
│   │   ├── augmentation.py            #     • Compose augment & preprocess pipelines based on cfg
│   │   └── dataset.py                 #     • Custom Dataset class for field images + masks
│   │
│   ├── models/                        # 6b. Model architecture wrappers / LightningModule
│   │   └── lit_segmentation.py        #     • LightningModule: forward, training_step, validation_step, configure_optimizers (instantiates SMP model via Hydra)
│   │
│   ├── utils/                         # 6c. Utility functions (metrics, logging helpers, callbacks)
│   │   ├── gpu_utils.py               #     • GPU detection and selection utilities
│   │   ├── seed.py                    #     • Random seed setting for reproducibility
│   │   └── pipeline_log.py            #     • Custom logger for training/inference pipelines (logs to console and file, with timestamps and config context)
│   │
│   ├── maskgen/                       # 6d. Mask-generation/refinement logic
│   │   ├── threshold.py               #     • Classical CV thresholding, morphological cleanup
│   │   ├── sam_integration.py         #     • Interfaces to Segment Anything Model
│   │   └── refinement.py              #     • Model-assisted proposals and QC overlay
│   │
│   ├── inference/                     # 6e. Inference & evaluation logic
│   │   └── inference.py               #     • Loads checkpoint, runs TTA (optional), thresholds, saves masks, computes IoU vs ground truth
│   │
│   ├── inference_utils/               # 6f. Inference-related utilities (e.g. inference_lts, inference_pipeline, weight_loader, etc.)
│   │
│   ├── mask_gen_utils/                # 6g. Mask generation utilities (e.g. mask post-processing, etc.)
│   │
│   ├── preprocess_utils/              # 6h. Preprocessing utilities (e.g. data_stats, train_val_test_split, pad_gridcrop_resize, etc.)
│   │
│   ├── train_utils/                   # 6i. Training utilities (e.g. train_pipeline, augmentation_visualizer, dataloader_visualizer, etc.)
│   │
│   ├── llm_utils/                     # 6j. LLM integration utilities (e.g. prompt engineering, table generation, etc.)
│   │
│   ├── preprocess.py                  # 6k. Secondary entry point for preprocessing mode (calls main())
│   ├── maskgen.py                     # 6l. Secondary entry point for maskgen mode (calls main())
│   ├── train.py                       # 6m. Secondary entry point for training mode (calls main())
│   └── inference.py                   # 6n. Secondary entry point for inference mode (calls main())
│   │
│   ├── scripts/                       # 7. Any standalone scripts for data processing, mask generation, evaluation, etc.
│   │
│   └── projects/                      # 8. Project-specific experiment folders
│       └── {project.name}/
│           └── mask_gen/
│               ├── developed-images/  # 8a. Original developed images (e.g. scanned or processed)
│               │   └── TXC08723.jpg
│               ├── cutouts/           # 8b. Cropped regions, masks, and metadata for each image
│               │   ├── TXC08723_0.jpg
│               │   ├── TXC08723_0.png
│               │   ├── TXC08723_0.json
│               │   └── TXC08723_0_mask.png
│               ├── refined_masks/     # 8c. Post-processed/refined masks
│               │   └── TXC08723_0_mask.png
│               └── db/                # 8d. Project-specific database (e.g. CSV with metadata)
│                   └── {project.name}.csv
└── .github/                           # 9. CI/CD workflows, code formatting, etc.
    └── workflows/
        └── ci.yaml                    # 9a. Run pytest, flake8, black, isort, plus any sanity checks (one-epoch train, maskgen sanity)
```
