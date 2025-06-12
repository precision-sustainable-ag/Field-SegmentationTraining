import cv2
import os
from pathlib import Path

# Load and downscale the image
image_dir = Path("/home/nsingh27/Field-AnnotationPipeline/organize_sam/images/del_final_images_cutouts_for_fine_tuning_unet_for_dirty_mat/full_image_and_plant_bbox")
for image_path in image_dir.glob("*.jpg"):
    print(f"Processing image: {image_path}")
    image = cv2.imread(str(image_path))
    orig_h, orig_w = image.shape[:2]

    scale_percent = 10  # Resize to 50% of original size
    w = int(orig_w * scale_percent / 100)
    h = int(orig_h * scale_percent / 100)
    image_resized = cv2.resize(image, (w, h))
    clone = image_resized.copy()

    drawing = False
    start_point = None
    boxes = []

    def draw_rectangle(event, x, y, flags, param):
        global start_point, drawing, image_resized

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            image_resized = clone.copy()
            cv2.rectangle(image_resized, start_point, (x, y), (0, 255, 0), 3)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = start_point
            x2, y2 = x, y
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)

    while True:
        cv2.imshow("Image", image_resized)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Escape key to exit
            break
        elif key == ord("y") and not drawing:
            # Accept the current bounding box
            if boxes:
                print(f"Accepted box: {boxes[-1]}")
        elif key == ord("n") and not drawing:
            # Redo the last bounding box
            if boxes:
                print(f"Redoing box: {boxes[-1]}")
                boxes.pop()
                image_resized = clone.copy()
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.destroyAllWindows()

    # Scale boxes back to original image size
    rescale_factor_x = orig_w / w
    rescale_factor_y = orig_h / h

    scaled_boxes = []
    for (x1, y1, x2, y2) in boxes:
        sx1 = int(x1 * rescale_factor_x)
        sy1 = int(y1 * rescale_factor_y)
        sx2 = int(x2 * rescale_factor_x)
        sy2 = int(y2 * rescale_factor_y)
        scaled_boxes.append((sx1, sy1, sx2, sy2))

    # Draw boxes on original image and save
    for (x1, y1, x2, y2) in scaled_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    output_img_name = image_path.stem + "_bbox.jpg"
    output_img_path = Path(image_path.parent / output_img_name)

    # cv2.imwrite(str(output_img_path), image)
    # print(f"Saved image with boxes: {output_img_path}")

    # Save YOLO format
    yolo_box_file = image_path.stem + ".txt"
    yolo_txt_path = Path(image_path.parent / yolo_box_file)
    print(f"Saving YOLO format to: {yolo_txt_path}")

    with open(yolo_txt_path, "w") as f:
        for (x1, y1, x2, y2) in scaled_boxes:
            x_center = ((x1 + x2) / 2) / orig_w
            y_center = ((y1 + y2) / 2) / orig_h
            box_width = (x2 - x1) / orig_w
            box_height = (y2 - y1) / orig_h
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    print(f"Saved YOLO-format boxes: {yolo_txt_path}")