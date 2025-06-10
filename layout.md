flowchart TB
  %% Entry point and mode dispatch
  subgraph Entry
    A["conf/config.yaml"]  
    B["main.py\n@hydra.main"]  
  end
  A --> B
  B -->|mode=preprocess| Pre
  B -->|mode=maskgen| Mask
  B -->|mode=train| Train
  B -->|mode=inference| Infer

  %% Preprocess mode
  subgraph Pre [Preprocess]
    Raw["Raw Images & Masks"]
    Raw --> P1["Resize / Normalize / Crop\n(conf/preprocess)"]
    P1 --> Processed["Processed Images & Masks"]
  end

  %% Mask generation mode
  subgraph Mask [MaskGen]
    MRaw["Raw Images"]
    MRaw --> M1["Classical CV\n(conf/maskgen)"]
    MRaw --> M2["SAM Proposals\n(conf/maskgen)"]
    MRaw --> M3["Model Refinement\n(conf/maskgen)"]
    M1 & M2 & M3 --> QC["QC Overlays"]
    QC --> GenMasks["Final Masks\n(${paths.maskgen_output_dir})"]
  end

  %% Training mode
  subgraph Train [Train]
    Processed --> Aug["Augmentation\n(conf/augment)"]
    Aug --> DL["DataLoader\n(data/augmentation.py + data/dataset.py)"]
    DL --> Lit["Lightning Trainer\n(models/lit_segmentation.py\n+ conf/train)"]
    Lit --> Checkpoint["Save Checkpoints & Logs\n(${paths.model_save_dir})"]
  end

  %% Inference mode
  subgraph Infer [Inference]
    Checkpoint --> Inf["Inference Module\n(src/inference/inference.py\n+ conf/inference)"]
    Processed --> Inf
    Inf --> Preds["Predicted Masks & Overlays"]
    Inf --> Reports["Metrics & Reports\n(${paths.reports_dir})"]
  end
