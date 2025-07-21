# src/preprocess_utils/pad_gridcrop_resize.py

from pathlib import Path
from PIL import Image

def pad_image(img: Image.Image, target_h: int, target_w: int, mode: str, fill: int) -> Image.Image:
    """Center-pad an image to (target_h, target_w) with specified background."""
    padded = Image.new(img.mode, (target_w, target_h), color=fill)
    w, h = img.size
    offset_x = (target_w - w) // 2
    offset_y = (target_h - h) // 2
    padded.paste(img, (offset_x, offset_y))
    return padded


def pad_gridcrop_resize(
    cutouts_dir: Path,
    masks_dir: Path,
    out_images: Path,
    out_masks: Path,
    cfg: any
) -> None:
    """
    Standardize cutout images and corresponding masks to a fixed size.

    - Pads if smaller than cfg.size
    - Grid-crops if > cfg.grid_crop.threshold × target size
    - Resizes (preserving aspect ratio) and pads otherwise (if resize=True)

    Args:
        cutouts_dir: Path to directory with JPEG/PNG cutouts (e.g., TXC08723_0.jpg)
        masks_dir:   Path to directory with refined masks (e.g., TXC08723_0_mask.png)
        out_images:  Path to write standardized images
        out_masks:   Path to write standardized masks
        cfg:         Hydra config with fields:
                       size: [height, width]
                       grid_crop:
                         enabled: bool
                         stride: Optional[int]
                       resize: bool
    """
    target_h = int(cfg.size.height)
    target_w = int(cfg.size.width)
    stride = int(cfg.grid_crop.stride) if cfg.grid_crop.stride else target_h
    threshold = float(cfg.grid_crop.threshold)

    img_interp  = getattr(Image, cfg.resize.interpolation.image)
    mask_interp = getattr(Image, cfg.resize.interpolation.mask)
    remove_src  = bool(getattr(cfg, "remove_src", False))
    ignore_empty = bool(getattr(cfg, "ignore_empty_data", False))
    pad_fill    = int(cfg.pad.fill)

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    for img_path in cutouts_dir.glob("*.jpg"):
        if not img_path.is_file():
            continue
        mask_path = masks_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            continue

        img  = Image.open(img_path)
        mask = Image.open(mask_path)
        w, h = img.size

        # 1) PAD if smaller than target
        if cfg.pad.enabled and (w < target_w or h < target_h):
            img_out  = pad_image(img,  target_h, target_w, cfg.pad.mode, cfg.pad.fill)
            mask_out = pad_image(mask, target_h, target_w, cfg.pad.mode, cfg.pad.fill)

            # If the image is now completely empty, skip saving
            if ignore_empty and mask_out.getextrema() == (pad_fill, pad_fill):
                continue

            img_out.save(out_images / img_path.name)
            mask_out.save(out_masks / mask_path.name)

            # Optionally remove the original source files
            if remove_src:
                img_path.unlink()
                mask_path.unlink()
            continue

        # 2) GRID-CROP if *significantly* larger
        elif (w  >= threshold * target_w) or (h >= threshold * target_h):

            # compute origins so last tile aligns to edge
            x_starts = list(range(0, max(w - target_w + 1, 1), stride))
            y_starts = list(range(0, max(h - target_h + 1, 1), stride))
            if x_starts[-1] != max(w - target_w, 0):
                x_starts.append(max(w - target_w, 0))
            if y_starts[-1] != max(h - target_h, 0):
                y_starts.append(max(h - target_h, 0))

            # generate multiple tiles
            for top in y_starts:
                for left in x_starts:
                    box       = (left, top, left + target_w, top + target_h)
                    tile_img  = img.crop(box)
                    tile_mask = mask.crop(box)

                    # pad partial tiles
                    if tile_img.size != (target_w, target_h):
                        tile_img  = pad_image(tile_img,  target_h, target_w, cfg.pad.mode, cfg.pad.fill)
                        tile_mask = pad_image(tile_mask, target_h, target_w, cfg.pad.mode, cfg.pad.fill)
                    
                    # If the image is now completely empty, skip saving
                    if ignore_empty and tile_mask.getextrema() == (pad_fill, pad_fill):
                        continue

                    stem = f"{img_path.stem}_{left}_{top}"
                    tile_img.save(out_images / f"{stem}{img_path.suffix}")
                    tile_mask.save(out_masks / f"{stem}_mask{mask_path.suffix}")

            # Optionally remove the original source files
            if remove_src:
                img_path.unlink()
                mask_path.unlink()
            continue

        # 3) RESIZE if just a bit over target
        elif cfg.resize.enabled:
            scale     = min(target_w / w, target_h / h)
            new_w     = int(w * scale)
            new_h     = int(h * scale)
            img_res   = img.resize((new_w, new_h),  resample=img_interp)
            mask_res  = mask.resize((new_w, new_h), resample=mask_interp)
            img_out   = pad_image(img_res,  target_h, target_w, cfg.pad.mode, cfg.pad.fill)
            mask_out  = pad_image(mask_res, target_h, target_w, cfg.pad.mode, cfg.pad.fill)

            # If the image is now completely empty, skip saving
            if ignore_empty and mask_out.getextrema() == (pad_fill, pad_fill):
                continue

            img_out.save(out_images / img_path.name)
            mask_out.save(out_masks / mask_path.name)

            # Optionally remove the original source files
            if remove_src:
                img_path.unlink()
                mask_path.unlink()
            continue

        else:
            # This should never happen — neither pad, grid-crop, nor resize applied.
            msg = (
                f"Unexpected image size for {img_path.name}: "
                f"{w}x{h} px (target {target_w}x{target_h}), "
                f"pad.enabled={cfg.pad.enabled}, "
                f"grid_crop.enabled={cfg.grid_crop.enabled}, "
                f"resize.enabled={cfg.resize.enabled}"
            )
            raise RuntimeError(msg)