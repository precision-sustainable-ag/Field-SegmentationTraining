# src/train.py

import os
import sys
from pathlib import Path
from typing import Callable, Dict

# Make `src` importable when running this file directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

# --- Task implementations live in train_utils ---
from src.train_utils.augmentation_visualizer import run_viz_augments
from src.train_utils.train_pipeline import run_train_pipeline
from src.train_utils.gpu_utils import is_launcher, is_rank_zero_worker, select_available_gpus

def _build_task_registry() -> Dict[str, Callable[[DictConfig], None]]:
    """
    Map simple task names to callables that accept only (cfg).
    """
    return {
        "viz_augments": run_viz_augments,   # standalone augment preview
        "train":        run_train_pipeline, # full Lightning training pipeline
    }

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Entrypoint with a tiny task registry.

    Configure in your YAML:
      train:
        vis_augment: true        # run augmentation preview first (optional)
        train_pipeline: true     # run the training pipeline (optional)

    You can also toggle these from the CLI, e.g.:
      python -m src.train train.vis_augment=true train.train_pipeline=false
    """
    # Discover where outputs will go (Hydra 1.3 sets/run dir already)
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # # If running in the launcher, print the output directory
    # if is_launcher(cfg) or not getattr(cfg.train, "use_multi_gpu", False):
    #     print(f"Hydra output dir: {out_dir}")

    # If you want to auto-pick GPUs, do it ONCE in the launcher and freeze the env
    if getattr(cfg.train, "use_multi_gpu", False) and getattr(cfg.train, "num_gpus", 1) > 1 and is_launcher(cfg):
        picked = select_available_gpus(max_gpus=min(cfg.train.num_gpus, 8),
                                       exclude_ids=getattr(cfg.train, "exclude_gpu_ids", [0]),
                                       verbose=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, picked))

    tasks = _build_task_registry()
    ran_any = False

    # 1) Optional: visualize augments as a lightweight, standalone step
    # Run ONCE in launcher (no DDP children yet)
    if bool(cfg.tasks.train.get("vis_augment", False)) and is_launcher(cfg):
        print("[task] viz_augments")
        tasks["viz_augments"](cfg)
        ran_any = True

    # 2) Optional: run the full training pipeline
    # Let Lightning spawn children; avoid duplicate prints here.
    if bool(cfg.tasks.train.get("train_pipeline", False)):
        # Optional: print tag only in the rank-0 worker (won't print in launcher now)
        if is_rank_zero_worker(cfg):
            print("[task] train")
        tasks["train"](cfg)
        ran_any = True

    if not ran_any and (is_launcher(cfg) or is_rank_zero_worker(cfg)):
        print("Nothing to do: both train.vis_augment and train.train_pipeline are False.")


if __name__ == "__main__":
    train()
