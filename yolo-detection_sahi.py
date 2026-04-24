import argparse
import os
import time
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO, settings


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_datasets_yaml_path", help="input annotated dataset yaml path")
    parser.add_argument("--predict_datasets_folder", help="predict folder")
    parser.add_argument("--name", default="dl", help="project name")
    parser.add_argument("--epochs", default=50, help="epochs")
    parser.add_argument("--batch", default=2, help="batch")
    parser.add_argument("--models", default="yolo11n", help="models name")
    parser.add_argument("--device", default="cpu", help="inference device, e.g. cpu / cuda:0")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--slice_height", type=int, default=640, help="slice height")
    parser.add_argument("--slice_width", type=int, default=640, help="slice width")
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2, help="slice overlap height ratio")
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2, help="slice overlap width ratio")
    return parser.parse_args()


def yolo_bbox_line(obj_pred, image_w, image_h):
    bbox = obj_pred.bbox
    x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
    x_c = ((x1 + x2) / 2.0) / image_w
    y_c = ((y1 + y2) / 2.0) / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    return f"{obj_pred.category.id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def main():
    args = parse_args()
    input_datasets_yaml_path = args.input_datasets_yaml_path
    predict_datasets_folder = args.predict_datasets_folder

    settings.reset()
    epochs_num = int(args.epochs)
    batch_num = int(args.batch)
    project_name = args.name
    models_name = args.models

    model_mapping = {
        "yolo11n": "yolo11n.pt",
        "yolo11s": "yolo11s.pt",
        "yolo11m": "yolo11m.pt",
        "yolo11l": "yolo11l.pt",
        "yolo11x": "yolo11x.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8x": "yolov8x.pt",
        "yolov10b": "yolov10b.pt",
        "yolov10l": "yolov10l.pt",
        "yolov10m": "yolov10m.pt",
        "yolov10n": "yolov10n.pt",
        "yolov10s": "yolov10s.pt",
        "yolov10x": "yolov10x.pt",
        "yolov9c": "yolov9c.pt",
        "yolov9e": "yolov9e.pt",
        "yolov9m": "yolov9m.pt",
        "yolov9s": "yolov9s.pt",
        "yolov9t": "yolov9t.pt",
        "yolo12n": "yolo12n.pt",
        "yolo12s": "yolo12s.pt",
        "yolo12m": "yolo12m.pt",
        "yolo12l": "yolo12l.pt",
        "yolo12x": "yolo12x.pt",
    }

    models_key = "./models/" + model_mapping.get(models_name, "yolo11n.pt")
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)

    t = time.strftime("%Y%m%d%H%M%S", time.localtime())
    p = os.getcwd()
    runs_dir = os.path.join(p, f"yolo_runs_{models_name}_{project_name}_{t}")
    settings.update({"runs_dir": runs_dir})

    image_paths = []
    image_names = []
    for filename in os.listdir(predict_datasets_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_paths.append(os.path.join(predict_datasets_folder, filename))
            image_names.append(filename)

    info_log_files = "INFO. Files : " + str(image_names)
    info_log_the_file_of_number = "INFO. The File Of Number : " + str(len(image_names))
    print(info_log_files)
    print(info_log_the_file_of_number)

    model_detection = YOLO(models_key)
    results_ydetection = model_detection.train(
        data=input_datasets_yaml_path,
        epochs=epochs_num,
        imgsz=640,
        batch=batch_num,
    )

    best_model_path = str(results_ydetection.save_dir) + "/weights/best.pt"
    if not os.path.exists(best_model_path):
        info_log_model = "INFO. Model training failed : " + best_model_path
    else:
        info_log_model = "INFO. The Model training successful : " + best_model_path

    log_file_path = os.path.dirname(str(results_ydetection.save_dir)) + "/yolo_training_log.txt"
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            info_log_files
            + "\n"
            + info_log_the_file_of_number
            + "\n"
            + info_log_model
            + "\n"
            + info_log_model_type
        )

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=best_model_path,
        confidence_threshold=args.conf,
        device=args.device,
    )

    predict_dir = Path(os.path.dirname(str(results_ydetection.save_dir))) / "predict_sahi"
    labels_dir = predict_dir / "labels"
    predict_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image_stem = Path(image_path).stem
        image_ext = Path(image_path).suffix.lower()
        result = get_sliced_prediction(
            image_path,
            sahi_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
        )

        result.export_visuals(export_dir=str(predict_dir), file_name=image_stem)
        saved_visual = predict_dir / f"{image_stem}.png"
        final_visual = predict_dir / f"{image_stem}{image_ext}"
        if saved_visual.exists():
            saved_visual.replace(final_visual)

        image_w = result.image_width
        image_h = result.image_height
        label_lines = [yolo_bbox_line(pred, image_w, image_h) for pred in result.object_prediction_list]
        (labels_dir / f"{image_stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")

    print(f"INFO. SAHI prediction outputs: {predict_dir}")


if __name__ == "__main__":
    main()
