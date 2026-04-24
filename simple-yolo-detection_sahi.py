from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("./models/yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# SAHI sliced inference
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="./models/yolo11n.pt",
    confidence_threshold=0.25,
    device="cpu",
)

result = get_sliced_prediction(
    "https://ultralytics.com/images/bus.jpg",
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

print(f"SAHI detections: {len(result.object_prediction_list)}")

# Export the model to ONNX format
success = model.export(format="onnx")
