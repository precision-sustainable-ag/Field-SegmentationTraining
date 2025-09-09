# src/train_utils/gpu_utils.py

import os
import GPUtil
from typing import List


def select_available_gpus(
    max_gpus: int = 3, exclude_ids: List[int] = [0], verbose: bool = True
) -> List[int]:
    """
    Select available GPU IDs, excluding specified ones.

    Args:
        max_gpus (int): Maximum number of GPUs to return.
        exclude_ids (List[int]): List of GPU IDs to skip (e.g., [0] to skip GPU 0).
        If `verbose` False, suppress printing (useful in workers).

    Returns:
        List[int]: List of selected GPU device IDs.
    """
    available = GPUtil.getAvailable(order="memory", limit=8, maxLoad=0.5, maxMemory=0.5)
    filtered = [gpu for gpu in available if gpu not in exclude_ids]

    if not filtered:
        raise RuntimeError(f"No suitable GPUs available after excluding: {exclude_ids}")
    chosen = filtered[:max_gpus]
    if verbose:
        print(f"Selected GPU IDs: {chosen}")
    return chosen

def is_launcher(cfg=None) -> bool:
    """
    True only in the parent 'launcher' process when multi-GPU is requested,
    before Lightning spawns children (no rank envs yet).
    """
    use_multi = False
    if cfg is not None:
        t = getattr(cfg, "train", None)
        if t:
            use_multi = bool(getattr(t, "use_multi_gpu", False) and getattr(t, "num_gpus", 1) > 1)
    no_rank_env = os.environ.get("LOCAL_RANK") is None and os.environ.get("RANK") is None
    return use_multi and no_rank_env

def is_rank_zero_worker(cfg=None) -> bool:
    """
    True exactly in the worker that should do side-effects (W&B, prints).
    - single GPU: True
    - multi GPU: only when LOCAL_RANK/RANK == 0
    """
    t = getattr(cfg, "train", None)
    use_multi = bool(t and getattr(t, "use_multi_gpu", False) and getattr(t, "num_gpus", 1) > 1)
    if use_multi:
        lr = os.environ.get("LOCAL_RANK")
        r  = os.environ.get("RANK")
        if lr is not None:
            return lr == "0"
        if r is not None:
            return r == "0"
        return False  # launcher, not a worker
    return True  # single-process