import shutil
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

from src.utils.seed import set_seed

log = logging.getLogger(__name__)

def _copy_one(
    img_path: Path,
    mask_path: Path,
    img_dest: Path,
    mask_dest: Path,
    remove_src: bool
) -> None:
    """
    Copy one image & mask into its destination folder,
    then optionally remove the sources.

    Args:
        img_path:    Path to image file (e.g., TXC08723_0.jpg)
        mask_path:   Path to corresponding mask file (e.g., TXC08723_0_mask.png)
        img_dest:    Path to directory where image should be copied
        mask_dest:   Path to directory where mask should be copied
        remove_src:  If True, remove the original source files after copying
    """
    shutil.copy2(img_path,  img_dest  / img_path.name)
    shutil.copy2(mask_path, mask_dest / mask_path.name)

    # Optionally remove the original source files
    if remove_src:
        img_path.unlink()
        mask_path.unlink()

def train_val_test_split(
    images_dir: Path,
    masks_dir: Path,
    cfg: any,
) -> None:
    """
    Entry point for train/val/test split stage:
      1) Deterministically shuffle images (using cfg.train.seed)
      2) Partition into train/val/test sets by cfg.split ratios
      3) Copy into ${paths.train/val/test}_{images,masks} directories
      4) Optionally remove original preprocessed files
    Supports optional multithreading (use_concurrency) for faster I/O.
    Args:
        images_dir: Path to directory with preprocessed images
        masks_dir:  Path to directory with corresponding masks
        cfg:        Hydra config containing paths and split ratios
    """
    # 1) Deterministic shuffle
    set_seed(cfg.train.seed)
    all_imgs = sorted([p for p in images_dir.iterdir() if p.is_file()])
    random.shuffle(all_imgs)

    # 2) Compute split counts
    total   = len(all_imgs)
    n_train = int(cfg.preprocess.split.train * total)
    n_val   = int(cfg.preprocess.split.val   * total)
    # rest goes to test
    
    splits = {
        "train": all_imgs[:n_train],
        "val":   all_imgs[n_train : n_train + n_val],
        "test":  all_imgs[n_train + n_val :],
    }

    # 3) Resolve output directories from cfg.paths
    paths = cfg.paths
    outs = {
        "train": (Path(paths.train_images_dir), Path(paths.train_masks_dir)),
        "val":   (Path(paths.val_images_dir),   Path(paths.val_masks_dir)),
        "test":  (Path(paths.test_images_dir),  Path(paths.test_masks_dir)),
    }

    # 4) Make sure output directories exist
    for img_d, mask_d in outs.values():
        img_d.mkdir(parents=True, exist_ok=True)
        mask_d.mkdir(parents=True, exist_ok=True)

    # 5) Copy files with optional multithreading
    remove_src = bool(cfg.preprocess.split.remove_src)
    use_cc     = bool(cfg.preprocess.split.use_concurrency)
    threads    = int(getattr(cfg.preprocess.split, "num_workers", 1)) if use_cc else 1

    if use_cc and threads > 1:
        # ─── multithreaded copy ───────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=threads) as exe:
            for split, imgs in splits.items():
                img_dest, mask_dest = outs[split]
                for img_path in imgs:
                    stem      = img_path.stem
                    mask_path = masks_dir / f"{stem}_mask.png"
                    if not mask_path.exists():
                        log.warning(f"No mask for {stem}, skipping")
                        continue
                    exe.submit(_copy_one, img_path, mask_path, img_dest, mask_dest, remove_src)
    else:
        # ─── sequential fallback ────────────────────────────────────────────────
        for split, imgs in splits.items():
            img_dest, mask_dest = outs[split]
            for img_path in imgs:
                stem      = img_path.stem
                mask_path = masks_dir / f"{stem}_mask.png"
                if not mask_path.exists():
                    log.warning(f"No mask for {stem}, skipping")
                    continue
                _copy_one(img_path, mask_path, img_dest, mask_dest, remove_src)
