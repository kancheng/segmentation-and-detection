import argparse
import csv
import os
import time
from pathlib import Path

# Workaround for duplicated OpenMP runtime on some Windows/Conda setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from function.fdash import report_function_d


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_datasets_yaml_path", help="input annotated dataset yaml path")
    parser.add_argument("--predict_datasets_folder", help="predict folder")
    parser.add_argument("--name", default="dl", help="project name")
    parser.add_argument("--epochs", default=50, help="epochs")
    parser.add_argument("--batch", default=2, help="batch")
    parser.add_argument("--models", default="yolo11n", help="models name")
    parser.add_argument("--device", default="cpu", help="inference device, e.g. cpu / cuda:0")
    parser.add_argument("--imgsz", type=int, default=512, help="inference image size for SAHI model")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--slice_height", type=int, default=512, help="slice height")
    parser.add_argument("--slice_width", type=int, default=512, help="slice width")
    parser.add_argument("--overlap_height_ratio", type=float, default=0.1, help="slice overlap height ratio")
    parser.add_argument("--overlap_width_ratio", type=float, default=0.1, help="slice overlap width ratio")
    parser.add_argument("--output_root", default="./yolo_runs", help="root directory for all Ultralytics outputs")
    return parser.parse_args()


def yolo_bbox_line(obj_pred, image_w, image_h):
    bbox = obj_pred.bbox
    x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
    x_c = ((x1 + x2) / 2.0) / image_w
    y_c = ((y1 + y2) / 2.0) / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    return f"{obj_pred.category.id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def summarize_detection_labels(label_dir, image_names):
    per_image = []
    total_detections = 0
    for image_name in image_names:
        stem = Path(image_name).stem
        label_path = label_dir / f"{stem}.txt"
        det_count = 0
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                det_count = sum(1 for line in f if line.strip())
        per_image.append((image_name, det_count))
        total_detections += det_count
    return per_image, total_detections


def release_memory(device):
    if str(device).lower().startswith("cpu"):
        torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    release_memory(args.device)
    input_datasets_yaml_path = args.input_datasets_yaml_path
    predict_datasets_folder = args.predict_datasets_folder

    epochs_num = int(args.epochs)
    batch_num = int(args.batch)
    project_name = args.name
    models_name = args.models
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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
    run_name = f"{models_name}_{project_name}_{t}"

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
        project=str(output_root),
        name=run_name,
    )

    best_model_path = str(results_ydetection.save_dir) + "/weights/best.pt"
    if not os.path.exists(best_model_path):
        info_log_model = "INFO. Model training failed : " + best_model_path
    else:
        info_log_model = "INFO. The Model training successful : " + best_model_path

    run_dir = Path(results_ydetection.save_dir)
    log_file_path = str(run_dir / "yolo_training_log.txt")
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
        image_size=args.imgsz,
        confidence_threshold=args.conf,
        device=args.device,
    )

    predict_dir = run_dir / "predict"
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
        release_memory(args.device)

    per_image, total_detections = summarize_detection_labels(labels_dir, image_names)
    mean_detections = (total_detections / len(image_names)) if image_names else 0.0

    res_dir = run_dir
    csv_file_path = res_dir / "result.csv"
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "detections"])
        for filename, det_count in per_image:
            w.writerow([filename, det_count])
        w.writerow(["Total", total_detections])
        w.writerow(["Mean per image", f"{mean_detections:.6f}"])
    print(f"CSV 已寫入: {csv_file_path}")

    res_file_path = res_dir / "result.txt"
    with open(res_file_path, "w", encoding="utf-8") as file:
        file.write("Total Detections: " + str(total_detections) + "\n")
        file.write("Mean Detections Per Image: " + str(mean_detections) + "\n")
    print(f"result.txt 已寫入: {res_file_path}")

    yaml_path = input_datasets_yaml_path
    original_image_dir = predict_datasets_folder
    train_dir = str(res_dir)
    html_file = str(res_dir / "index.html")
    pout_dir = str(res_dir / "yolo2images")
    report_function_d(yaml_path, original_image_dir, str(predict_dir), train_dir, html_file, pout_dir)

    print(f"INFO. SAHI prediction outputs: {predict_dir}")


if __name__ == "__main__":
    main()
