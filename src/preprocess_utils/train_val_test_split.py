# src/preprocess_utils/train_val_test_split.py

import shutil
import random
from pathlib import Path
from src.utils.seed import set_seed

def train_val_test_split(
    images_dir: Path,
    masks_dir: Path,
    cfg: any,
) -> None:
    """
    Split images & masks into train/val/test using cfg.preprocess_split & paths from cfg.paths.
    """
    # 1) Deterministic shuffle
    set_seed(cfg.train.seed)
    all_imgs = sorted([p for p in images_dir.iterdir() if p.is_file()])
    random.shuffle(all_imgs)

    # 2) compute counts
    total = len(all_imgs)
    n_train = int(cfg.preprocess.split.train * total)
    n_val   = int(cfg.preprocess.split.val   * total)
    
    # rest goes to test
    train_imgs = all_imgs[:n_train]
    val_imgs   = all_imgs[n_train : n_train + n_val]
    test_imgs  = all_imgs[n_train + n_val :]

    # 3) output dirs from your paths config
    train_img_out = Path(cfg.paths.train_images_dir)
    train_mask_out= Path(cfg.paths.train_masks_dir)
    val_img_out   = Path(cfg.paths.val_images_dir)
    val_mask_out  = Path(cfg.paths.val_masks_dir)
    test_img_out  = Path(cfg.paths.test_images_dir)
    test_mask_out = Path(cfg.paths.test_masks_dir)

    # 4) make directories
    for d in [train_img_out, train_mask_out,
              val_img_out,   val_mask_out,
              test_img_out,  test_mask_out]:
        d.mkdir(parents=True, exist_ok=True)

    # 5) copy files
    remove_src = bool(getattr(cfg.preprocess.split, "remove_src", False))
    for img_list, img_dest, mask_dest in [
        (train_imgs, train_img_out, train_mask_out),
        (val_imgs,   val_img_out,   val_mask_out),
        (test_imgs,  test_img_out,  test_mask_out),
    ]:
        for img_path in img_list:
            shutil.copy2(img_path, img_dest / img_path.name)
            mask_name = f"{img_path.stem}_mask.png"
            shutil.copy2(masks_dir / mask_name, mask_dest / mask_name)

            # Optionally remove the original source files
            if remove_src:
                img_path.unlink()
                (masks_dir / mask_name).unlink()