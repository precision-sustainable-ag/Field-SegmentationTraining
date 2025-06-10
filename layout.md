```mermaid
flowchart TB
    %% Top-level Hydra dispatch
    subgraph Entry
      A[Hydra Config\nconf/config.yaml]
      B[main.py\n@hydra.main]
    end

    A --> B
    B -->|mode=preprocess| Pre
    B -->|mode=maskgen| Mask
    B -->|mode=train| Train
    B -->|mode=inference| Infer

    %% Preprocessing mode
    subgraph Preprocess [Preprocess]
      direction LR
      Raw[Raw Images & Masks] --> P1[Resize / Normalize / Crop\n(conf/preprocess)]
      P1 --> Processed[Processed Images & Masks]
    end

    %% Mask generation mode
    subgraph Maskgen [MaskGen]
      direction LR
      Raw --> M1[Classical CV\n(conf/maskgen)]
      Raw --> M2[SAM Proposals\n(conf/maskgen)]
      Raw --> M3[Model-Refinement\n(conf/maskgen)]
      M1 & M2 & M3 --> QC[QC Overlays]
      QC --> GenMasks[Final Masks\n(conf/paths → output_dir)]
    end

    %% Training mode
    subgraph Train [Train]
      direction LR
      Processed --> Aug[Augmentation\n(conf/augment)]
      Aug --> DL[DataLoader\n(src/data/dataset.py + augmentation.py)]
      DL --> Lit[Lightning Trainer\n(src/models/lit_segmentation.py\n+ conf/train)]
      Lit --> Checkpoint[Save Checkpoints & Logs\n(conf/paths)]
    end

    %% Inference mode
    subgraph Infer [Inference]
      direction LR
      Checkpoint --> InfMod[Inference Module\n(src/inference/inference.py\n+ conf/inference)]
      Processed --> InfMod
      InfMod --> Preds[Predicted Masks & Overlays]
      InfMod --> Reports[Compute Metrics & Save\n(conf/paths)]
    end
