import argparse

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="./models/yolo11n.pt", help="Path to pretrained model")
    parser.add_argument("--data", default="coco8.yaml", help="Path to training dataset YAML")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--slice_height", type=int, default=640, help="Slice height")
    parser.add_argument("--slice_width", type=int, default=640, help="Slice width")
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2, help="Slice overlap ratio on height")
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2, help="Slice overlap ratio on width")
    parser.add_argument("--predict_source", default="https://ultralytics.com/images/bus.jpg", help="Input source for prediction")
    parser.add_argument("--export_format", default="onnx", help="Format used for model export")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs)
    model.val()

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )
    result = get_sliced_prediction(
        args.predict_source,
        detection_model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
    )
    print(f"SAHI detections: {len(result.object_prediction_list)}")
    model.export(format=args.export_format)


if __name__ == "__main__":
    main()
