from ultralytics.data.annotator import auto_annotate
data = "/home/nsingh27/AgIR-FieldSegmentation/data/YOLO_dataset/custom_data_fine_tuning/masks"
det_model = "/home/nsingh27/AgIR-FieldSegmentation/data/model_YOLO_det/yolov8x.pt"
sam_model = "/home/nsingh27/AgIR-FieldSegmentation/data/model_SAM/sam_b.pt"

auto_annotate(data=data, det_model=det_model, sam_model=sam_model)