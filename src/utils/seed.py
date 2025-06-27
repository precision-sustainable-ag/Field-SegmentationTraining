import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    Also configures PyTorch to use deterministic operations.
    
    Args:
        seed (int): The seed value to use.
    """

    # Set seed for Python's built-in random module
    # Affects functions like random.random(), random.shuffle(), etc.
    random.seed(seed)

    # Set seed for NumPy's random number generator
    # Useful when data augmentations or sampling use numpy (e.g., np.random.rand)
    np.random.seed(seed)

    # Set seed for PyTorch's CPU operations
    # Affects torch.rand, torch.randn, and other tensor initializations on CPU
    torch.manual_seed(seed)

    # Set seed for all CUDA devices (GPU operations in PyTorch)
    # Needed when training on GPU for reproducible results
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN operations deterministic (e.g., convolutions)
    # Ensures same results across runs but might reduce performance
    torch.backends.cudnn.deterministic = True

    # Disable cuDNN benchmark which selects best algorithms per input shape
    # Prevents non-deterministic behavior due to auto-selection of fast kernels
    torch.backends.cudnn.benchmark = False

    # Enforce that only deterministic algorithms are used throughout PyTorch
    # If a non-deterministic algorithm is called, it raises an error (strict mode)
    torch.use_deterministic_algorithms(True, warn_only=False)

def seed_worker(worker_id: int) -> None:
    """
    Seed function for each DataLoader worker process.
    Ensures deterministic behavior across all workers by assigning each
    a unique but reproducible seed derived from the global seed.
    
    Args:
        worker_id (int): Worker index passed by the DataLoader.
    """
    # Derive a seed for this worker from PyTorch's global seed
    worker_seed = torch.initial_seed() % 2**32

    # Seed NumPy RNG for this worker
    np.random.seed(worker_seed)

    # Seed Python's built-in random module for this worker
    random.seed(worker_seed)
