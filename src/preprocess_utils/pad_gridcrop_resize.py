from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

log = logging.getLogger(__name__)

def _process_one(
    img_path: Path,
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
    # ─── load per-image config ────────────────────────────────────────────────────
    target_h     = int(cfg.size.height)
    target_w     = int(cfg.size.width)
    stride       = int(cfg.grid_crop.stride) if cfg.grid_crop.stride else target_h
    threshold    = float(cfg.grid_crop.threshold)
    img_interp   = getattr(Image, cfg.resize.interpolation.image.upper())
    mask_interp  = getattr(Image, cfg.resize.interpolation.mask.upper())
    remove_src   = bool(cfg.remove_src)
    ignore_empty = bool(cfg.ignore_empty_data)
    pad_fill     = int(cfg.pad.fill)

    # ─── locate files ─────────────────────────────────────────────────────────────
    mask_path = masks_dir / f"{img_path.stem}_mask.png"
    img  = Image.open(img_path)
    mask = Image.open(mask_path)
    w, h = img.size

    # 1) PAD if smaller than target
    #    Center-pad any image < target size to exactly (target_h, target_w).
    if cfg.pad.enabled and (w < target_w or h < target_h):
        img_out  = Image.new(img.mode,  (target_w, target_h), color=pad_fill)
        mask_out = Image.new(mask.mode, (target_w, target_h), color=pad_fill)
        img_out.paste(img,  ((target_w - w)//2, (target_h - h)//2))
        mask_out.paste(mask,((target_w - w)//2, (target_h - h)//2))

        # If the mask is entirely pad_fill and we're ignoring empty, skip saving.
        if ignore_empty and mask_out.getextrema() == (pad_fill, pad_fill):
            if remove_src:
                img_path.unlink(); mask_path.unlink()
            return

        img_out.save(out_images / img_path.name)
        mask_out.save(out_masks   / mask_path.name)

        # Optionally remove the original source files
        if remove_src:
            img_path.unlink(); mask_path.unlink()
        return

    # 2) GRID-CROP if *significantly* larger
    #    Break images > threshold×target into overlapping tiles of size (target_h, target_w).
    if cfg.grid_crop.enabled and ((w >= threshold*target_w) or (h >= threshold*target_h)):

        # compute scale & new size just like in branch 3
        scale    = min(target_w / w, target_h / h)
        new_w    = int(w * scale)
        new_h    = int(h * scale)
        img_res  = img.resize((new_w, new_h), resample=img_interp)
        mask_res = mask.resize((new_w, new_h), resample=mask_interp)
        full_img = Image.new(img.mode,  (target_w, target_h), color=pad_fill)
        full_mask= Image.new(mask.mode, (target_w, target_h), color=pad_fill)
        full_img.paste(img_res,  ((target_w - new_w)//2, (target_h - new_h)//2))
        full_mask.paste(mask_res,((target_w - new_w)//2, (target_h - new_h)//2))
        # save that resized “full” image & mask
        full_img.save(out_images / img_path.name)
        full_mask.save(out_masks   / mask_path.name)

        # compute starts so last tile aligns at edge
        x_starts = list(range(0, max(w - target_w + 1, 1), stride))
        y_starts = list(range(0, max(h - target_h + 1, 1), stride))
        if x_starts[-1] != max(w - target_w, 0):
            x_starts.append(max(w - target_w, 0))
        if y_starts[-1] != max(h - target_h, 0):
            y_starts.append(max(h - target_h, 0))

        # generate multiple tiles
        for top in y_starts:
            for left in x_starts:
                box       = (left, top, left+target_w, top+target_h)
                tile_img  = img.crop(box)
                tile_mask = mask.crop(box)

                # pad partial tiles to exact size
                if tile_img.size != (target_w, target_h):
                    padded_img  = Image.new(tile_img.mode,  (target_w, target_h), color=pad_fill)
                    padded_mask = Image.new(tile_mask.mode, (target_w, target_h), color=pad_fill)
                    padded_img.paste(tile_img,  ((target_w - tile_img.width)//2, (target_h - tile_img.height)//2))
                    padded_mask.paste(tile_mask,((target_w - tile_mask.width)//2, (target_h - tile_mask.height)//2))
                    tile_img, tile_mask = padded_img, padded_mask

                # skip empty tiles if configured
                if ignore_empty and tile_mask.getextrema() == (pad_fill, pad_fill):
                    continue

                stem = f"{img_path.stem}_{left}_{top}"
                tile_img.save(out_images / f"{stem}{img_path.suffix}")
                tile_mask.save(out_masks / f"{stem}_mask.png")

        # Optionally remove the original source files
        if remove_src:
            img_path.unlink(); mask_path.unlink()
        return

    # 3) RESIZE if just a bit over target
    #    Downscale (preserving aspect ratio) then pad to target size.
    if cfg.resize.enabled:
        scale    = min(target_w / w, target_h / h)
        new_w    = int(w * scale)
        new_h    = int(h * scale)
        img_res  = img.resize((new_w, new_h), resample=img_interp)
        mask_res = mask.resize((new_w, new_h), resample=mask_interp)
        img_out  = Image.new(img.mode,  (target_w, target_h), color=pad_fill)
        mask_out = Image.new(mask.mode, (target_w, target_h), color=pad_fill)
        img_out.paste(img_res,  ((target_w - new_w)//2, (target_h - new_h)//2))
        mask_out.paste(mask_res,((target_w - new_w)//2, (target_h - new_h)//2))

        # If the mask is entirely pad_fill and we're ignoring empty, skip saving.
        if ignore_empty and mask_out.getextrema() == (pad_fill, pad_fill):
            if remove_src:
                img_path.unlink(); mask_path.unlink()
            return

        img_out.save(out_images / img_path.name)
        mask_out.save(out_masks   / mask_path.name)
        
        # Optionally remove the original source files
        if remove_src:
            img_path.unlink(); mask_path.unlink()
        return

    # This should never happen — neither pad, grid-crop, nor resize applied.
    msg = (
        f"Unexpected image size for {img_path.name}: "
        f"{w}x{h} px (target {target_w}x{target_h}), "
        f"pad.enabled={cfg.pad.enabled}, "
        f"grid_crop.enabled={cfg.grid_crop.enabled}, "
        f"resize.enabled={cfg.resize.enabled}"
    )
    raise RuntimeError(msg)

def pad_gridcrop_resize(
    images_dir: Path,
    masks_dir: Path,
    out_images: Path,
    out_masks: Path,
    cfg: any
) -> None:
    """
    Entry point for pad/gridcrop/resize stage.
    Dispatches to either a process pool (if use_concurrency=True)
    or sequentially processes each image via _process_one().

    Args:
        images_dir:  Path to directory with JPEG/PNG images (e.g., TXC08723_0.jpg)
        masks_dir:   Path to directory with refined masks (e.g., TXC08723_0_mask.png)
        out_images:  Path to write standardized images
        out_masks:   Path to write standardized masks
        cfg:         Hydra config
    """
    # ensure output dirs exist
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    img_paths = list(images_dir.glob("*.jpg"))
    use_cc    = bool(getattr(cfg, "use_concurrency", False))
    workers   = int(getattr(cfg, "num_workers", 1)) if use_cc else 1

    if use_cc and workers > 1:
        # ─── multiprocess dispatch ─────────────────────────────────────────────
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {
                exe.submit(_process_one, p, masks_dir, out_images, out_masks, cfg): p
                for p in img_paths
            }
            for fut in as_completed(futures):
                img_p = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    log.error(f"[pad_gridcrop] {img_p.name} failed: {e}")
    else:
        # ─── sequential fallback ────────────────────────────────────────────────
        for p in img_paths:
            try:
                _process_one(p, masks_dir, out_images, out_masks, cfg)
            except Exception as e:
                log.error(f"[pad_gridcrop] {p.name} failed: {e}")
