# U-Net Segmentation Training Pipeline

This repository provides a pipeline for training a U-Net model for semantic segmentation tasks for the Ag Image repository. It is designed to handle data loading, model training, validation, and inference.

## Included Scripts

1. **`main.py`**: The entry point of the pipeline, allowing task-based modular execution using Hydra for configuration management.
2. **`crop_and_resize.py`**: Utility functions for preprocessing images, including cropping and resizing.
3. **`unet_segmentation.py`**: Script defining the U-Net architecture and training functions.
4. **`inference.py`**: Script for performing inference using the trained U-Net model.

## Configuration

Create a configuration file (e.g., `config.yaml`) in the `conf` directory. Below is an example configuration:

```yaml
unet_conf:
  learning_rate: 0.001
  batch_size: 16
  epochs: 50

paths:
  data_dir: "./data"
  model_save_dir: "./models"

pipeline:
  - crop_and_resize
  - unet_segmentation
  - inference
```

## Running the Pipeline

1. Place your dataset in the specified `data_dir`.
2. Start the pipeline:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python main.py 
   ```
   Modify CUDA_VISIBLE_DEVICES according to availability. 

### Outputs

- **Model Checkpoints**: Saved in the `model_save_dir` with a folder for the current date.
- **Metrics Logs**: Saved in `training_metrics.txt` inside the date-specific folder.
- **Dataset Info**: Saved as `dataset_info.json` in the same directory.

## Metrics

The pipeline evaluates the following metrics on the validation set:
- **Accuracy**
- **Recall**
- **Training Loss**
- **Validation Loss**
- **IOU**
- **Generalized dice index**

## Logging

Metrics are logged in the `training_metrics.txt` file, structured as:

```
Epoch   Train Loss   Val Loss   Accuracy   Recall
1       0.4500      0.3900     0.85       0.88
...
```

