# Explore Image Selection

A simple Python script to manually browse and select images from a folder. It can be used to select images with the current desired trait.

---

## How it Works

- Shows each image in a window.
- Press:
  - `'1'` to **select**
  - `'0'` to **skip**
  - `'q'` to **quit early**
- Selected image names are saved to `selected_images.txt` in the same folder.

---

## Usage

1. Set your image directory in the script (`main()` function).
2. Run:

```bash
python explore_image.py
```

3. Follow on-screen instructions.

---

## Notes

- Supports `.jpg`, `.jpeg`, `.png` files.
- Images are resized for easy viewing.
- `selected_images.txt` will **overwrite** each run.

