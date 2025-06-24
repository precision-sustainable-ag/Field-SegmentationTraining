import GPUtil
from typing import List


def select_available_gpus(max_gpus: int = 3, exclude_ids: List[int] = [0]) -> List[int]:
    """
    Select available GPU IDs, excluding specified ones.

    Args:
        max_gpus (int): Maximum number of GPUs to return.
        exclude_ids (List[int]): List of GPU IDs to skip (e.g., [0] to skip GPU 0).

    Returns:
        List[int]: List of selected GPU device IDs.
    """
    available = GPUtil.getAvailable(order="memory", limit=8, maxLoad=0.5, maxMemory=0.5)
    filtered = [gpu for gpu in available if gpu not in exclude_ids]

    if not filtered:
        raise RuntimeError(f"No suitable GPUs available after excluding: {exclude_ids}")
    print(f"Selected GPU IDs: {filtered[:max_gpus]}")
    return filtered[:max_gpus]
