# src/inference.py

import os
import sys
from pathlib import Path
from typing import Callable, Dict

# Make `src` importable when running this file directly (same trick as train.py)
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

# --- Task implementations live in inference_utils ---
from src.inference_utils.inference_pipeline import run_inference_pipeline
from src.inference_utils.inference_lts import run_inference_lts
from src.utils.gpu_utils import select_available_gpus



def _build_task_registry() -> Dict[str, Callable[[DictConfig], None]]:
    """
    Map simple task names to callables that accept only (cfg).
    Extend as needed: e.g. "lts": run_inference_lts
    """
    return {
        "local": run_inference_pipeline,   # local inference path
        "lts": run_inference_lts,        # lts inference path
    }


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def inference(cfg: DictConfig) -> None:
    """
    Entrypoint with a tiny task registry, matching train.py style.

    In YAML:
      tasks:
        inference:
          local: true
          lts: false
    """
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    gpu_cfg = getattr(getattr(cfg, "inference", None), "gpu", None)

    if gpu_cfg is not None and bool(getattr(gpu_cfg, "enable", True)):
        max_gpus = int(getattr(gpu_cfg, "max_gpus", 1))
        exclude  = list(getattr(gpu_cfg, "exclude_gpu_ids", []))

        try:
            picked = select_available_gpus(
                max_gpus=max_gpus,
                exclude_ids=exclude,
                verbose=True,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, picked))
            print(
                f"[inference] Using physical GPUs {picked} → "
                f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
            )
        except Exception as e:
            print(f"[inference] GPU auto-selection failed ({e}). Falling back to default device.")
    else:
        print("[inference] GPU auto-selection disabled or not configured; using default CUDA/CPU.")


    tasks = _build_task_registry()
    ran_any = False

    # Iterate flags under cfg.tasks.inference and run the enabled ones if present.
    inf_cfg = getattr(cfg.tasks, "inference", {})
    for name, enabled in inf_cfg.items():
        if bool(enabled) and name in tasks:
            print(f"[task] inference.{name} -> {out_dir}")
            tasks[name](cfg)
            ran_any = True

    if not ran_any:
        print("Nothing to do: all tasks.inference.* are False (or no known tasks).")


if __name__ == "__main__":
    inference()
