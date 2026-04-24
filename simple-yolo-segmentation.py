import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="./models/yolo11n-seg.pt", help="Path to pretrained segmentation model")
    parser.add_argument("--data", default="coco8-seg.yaml", help="Path to training dataset YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--predict_source", default="https://ultralytics.com/images/bus.jpg", help="Input source for prediction")
    parser.add_argument("--export_format", default="onnx", help="Format used for model export")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    model.val()
    model(args.predict_source)
    model.export(format=args.export_format)


if __name__ == "__main__":
    main()