# U-Net Segmentation Training Pipeline

This repository provides a pipeline for training a U-Net model for semantic segmentation tasks for the Ag Image repository. It is designed to handle data loading, model training, validation, and evaluation while offering modularity for customization.

## Features

- **Custom Dataset Handling**: Load and preprocess your datasets seamlessly.
- **Model Training and Validation**: Train a U-Net model with user-defined configurations.
- **Metrics Logging**: Track metrics such as loss, accuracy, recall, IOU, and generalized dice score for each epoch.
- **Save and Resume**: Models and metrics are saved for further use or analysis.

### Configuration

Create a configuration file (e.g., `config.yaml`) in the `conf` directory. Below is an example configuration:

```yaml
unet_conf:
  learning_rate: 0.001
  batch_size: 16
  epochs: 50

paths:
  data_dir: "./data"
  model_save_dir: "./models"
```

### Running the Script

1. Ensure your dataset is placed in the specified `data_dir`.
2. Start training:
   ```bash
   python main.py --config-path /path/to/config.yaml
   ```

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
