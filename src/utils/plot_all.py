from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def plot_all_metrics(data: pd.DataFrame, save_dir: Path, smoothing_sigma: int = 2):

    # Apply smoothing to the metric values if smoothing_sigma is greater than 0
    epochs = data['epoch']
    
    # Apply smoothing or use original data
    def apply_smoothing(series, smoothing_sigma):
        return gaussian_filter1d(series, sigma=smoothing_sigma) if smoothing_sigma > 0 else series

    train_seg_loss = apply_smoothing(data['train/seg_loss'], smoothing_sigma)
    val_seg_loss = apply_smoothing(data['val/seg_loss'], smoothing_sigma)
    train_box_loss = apply_smoothing(data['train/box_loss'], smoothing_sigma)
    val_box_loss = apply_smoothing(data['val/box_loss'], smoothing_sigma)
    train_cls_loss = apply_smoothing(data['train/cls_loss'], smoothing_sigma)
    val_cls_loss = apply_smoothing(data['val/cls_loss'], smoothing_sigma)
    train_dfl_loss = apply_smoothing(data['train/dfl_loss'], smoothing_sigma)
    val_dfl_loss = apply_smoothing(data['val/dfl_loss'], smoothing_sigma)

    precision_b = apply_smoothing(data['metrics/precision(B)'], smoothing_sigma)
    recall_b = apply_smoothing(data['metrics/recall(B)'], smoothing_sigma)
    mAP50_b = apply_smoothing(data['metrics/mAP50(B)'], smoothing_sigma)
    mAP50_95_b = apply_smoothing(data['metrics/mAP50-95(B)'], smoothing_sigma)

    precision_m = apply_smoothing(data['metrics/precision(M)'], smoothing_sigma)
    recall_m = apply_smoothing(data['metrics/recall(M)'], smoothing_sigma)
    mAP50_m = apply_smoothing(data['metrics/mAP50(M)'], smoothing_sigma)
    mAP50_95_m = apply_smoothing(data['metrics/mAP50-95(M)'], smoothing_sigma)

    lr_pg0 = data['lr/pg0']
    lr_pg1 = data['lr/pg1']
    lr_pg2 = data['lr/pg2']

    # Plot all metrics on a single figure with multiple subplots
    fig, axs = plt.subplots(5, 2, figsize=(20, 30))

    # Train and Validation Segmentation Loss
    axs[0, 0].plot(epochs, train_seg_loss, label='Train Segmentation Loss', color='darkblue')
    axs[0, 0].plot(epochs, val_seg_loss, label='Validation Segmentation Loss', color='lightblue')
    axs[0, 0].set_title('Train and Validation Segmentation Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Train and Validation Box Loss
    axs[0, 1].plot(epochs, train_box_loss, label='Train Box Loss', color='darkgreen')
    axs[0, 1].plot(epochs, val_box_loss, label='Validation Box Loss', color='lightgreen')
    axs[0, 1].set_title('Train and Validation Box Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Train and Validation Classification Loss
    axs[1, 0].plot(epochs, train_cls_loss, label='Train Classification Loss', color='darkorange')
    axs[1, 0].plot(epochs, val_cls_loss, label='Validation Classification Loss', color='lightsalmon')
    axs[1, 0].set_title('Train and Validation Classification Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Train and Validation DFL Loss
    axs[1, 1].plot(epochs, train_dfl_loss, label='Train DFL Loss', color='darkred')
    axs[1, 1].plot(epochs, val_dfl_loss, label='Validation DFL Loss', color='lightcoral')
    axs[1, 1].set_title('Train and Validation DFL Loss')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Precision and Recall for B
    axs[2, 0].plot(epochs, precision_b, label='Precision (B)', color='blue')
    axs[2, 0].plot(epochs, recall_b, label='Recall (B)', color='orange')
    axs[2, 0].set_title('Precision and Recall (B)')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Metric Value')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Precision and Recall for M
    axs[2, 1].plot(epochs, precision_m, label='Precision (M)', color='green')
    axs[2, 1].plot(epochs, recall_m, label='Recall (M)', color='red')
    axs[2, 1].set_title('Precision and Recall (M)')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Metric Value')
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # mAP50 and mAP50-95 for B
    axs[3, 0].plot(epochs, mAP50_b, label='mAP50 (B)', color='purple')
    axs[3, 0].plot(epochs, mAP50_95_b, label='mAP50-95 (B)', color='brown')
    axs[3, 0].set_title('mAP50 and mAP50-95 (B)')
    axs[3, 0].set_xlabel('Epoch')
    axs[3, 0].set_ylabel('Metric Value')
    axs[3, 0].legend()
    axs[3, 0].grid(True)

    # mAP50 and mAP50-95 for M
    axs[3, 1].plot(epochs, mAP50_m, label='mAP50 (M)', color='cyan')
    axs[3, 1].plot(epochs, mAP50_95_m, label='mAP50-95 (M)', color='magenta')
    axs[3, 1].set_title('mAP50 and mAP50-95 (M)')
    axs[3, 1].set_xlabel('Epoch')
    axs[3, 1].set_ylabel('Metric Value')
    axs[3, 1].legend()
    axs[3, 1].grid(True)

    # Learning Rates
    axs[4, 0].plot(epochs, lr_pg0, label='Learning Rate PG0', color='blue')
    axs[4, 0].plot(epochs, lr_pg1, label='Learning Rate PG1', color='green')
    axs[4, 0].plot(epochs, lr_pg2, label='Learning Rate PG2', color='red')
    axs[4, 0].set_title('Learning Rates')
    axs[4, 0].set_xlabel('Epoch')
    axs[4, 0].set_ylabel('Learning Rate')
    axs[4, 0].legend()
    axs[4, 0].grid(True)

    # Empty subplot for alignment
    axs[4, 1].axis('off')

    plt.tight_layout()
    plot_path = save_dir / 'all_metrics.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

# Example usage
csv_path = Path('../../runs/segment/train3/results.csv')

# Read the second CSV file
data = pd.read_csv(csv_path)

save_dir = csv_path.parent
plot_all_metrics(data, save_dir, smoothing_sigma=2)
