import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO, settings

from function.fdash import report_function_d
from function.feval import evaluate_miou_mdice


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_datasets_yaml_path", help="input annotated directory")
    parser.add_argument("--predict_datasets_folder", help="predict folder")
    parser.add_argument("--name", default="dl", help="project name")
    parser.add_argument("--epochs", default=50, help="epochs")
    parser.add_argument("--batch", default=2, help="batch")
    parser.add_argument("--models", default="yolo11n-seg", help="models name")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes for Dice calculation")
    parser.add_argument("--no_filter_low_scores", action="store_true", help="disable filter low scores")
    parser.add_argument("--min_iou", type=float, default=0.5, help="min IoU when filtering")
    parser.add_argument("--min_dice", type=float, default=0.5, help="min DSC when filtering")
    parser.add_argument("--device", default="cpu", help="inference device, e.g. cpu / cuda:0")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--slice_height", type=int, default=640, help="slice height")
    parser.add_argument("--slice_width", type=int, default=640, help="slice width")
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2, help="slice overlap height ratio")
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2, help="slice overlap width ratio")
    return parser.parse_args()


def write_prediction_txt(result, txt_path):
    lines = []
    for pred in result.object_prediction_list:
        cls_id = pred.category.id
        if pred.mask is not None and pred.mask.segmentation:
            poly = pred.mask.segmentation[0]
            poly_str = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"{cls_id} {poly_str}")
        else:
            bbox = pred.bbox
            x_c = ((bbox.minx + bbox.maxx) / 2.0) / result.image_width
            y_c = ((bbox.miny + bbox.maxy) / 2.0) / result.image_height
            w = (bbox.maxx - bbox.minx) / result.image_width
            h = (bbox.maxy - bbox.miny) / result.image_height
            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def read_txt_labels(txt_file):
    labels = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            values = [float(x) for x in parts[1:]]
            labels.append([class_id, values])
    return labels


def draw_labels(mask, labels):
    h, w = mask.shape[:2]
    for _, values in labels:
        if len(values) >= 6 and len(values) % 2 == 0:
            points = [(int(values[i] * w), int(values[i + 1] * h)) for i in range(0, len(values), 2)]
        elif len(values) == 4:
            x_c, y_c, bw, bh = values
            x1 = int((x_c - bw / 2.0) * w)
            y1 = int((y_c - bh / 2.0) * h)
            x2 = int((x_c + bw / 2.0) * w)
            y2 = int((y_c + bh / 2.0) * h)
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            continue

        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))


def yolo_txt_to_mask(image_path, txt_path, out_path):
    image = cv2.imread(image_path)
    if image is None:
        return
    mask = np.zeros_like(image, dtype=np.uint8)
    labels = read_txt_labels(txt_path)
    draw_labels(mask, labels)
    cv2.imwrite(out_path, mask)


def get_prefix(filename):
    return filename.split(".")[0]


def filter_common_prefix(list1, list2):
    prefixes1 = {get_prefix(f) for f in list1}
    prefixes2 = {get_prefix(f) for f in list2}
    common_prefixes = prefixes1.intersection(prefixes2)
    filtered_list1 = [f for f in list1 if get_prefix(f) in common_prefixes]
    filtered_list2 = [f for f in list2 if get_prefix(f) in common_prefixes]
    return filtered_list1, filtered_list2


def yolo2maskdir_all(label_dir, images_dir, output_mask_dir):
    files = []
    txts = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".JPEG", ".PNG")):
            files.append(filename)
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            txts.append(filename)

    filtered_files, filtered_txts = filter_common_prefix(files, txts)
    for image_name, txt_name in zip(filtered_files, filtered_txts):
        image_path = os.path.join(images_dir, image_name)
        txt_path = os.path.join(label_dir, txt_name)
        out_path = os.path.join(output_mask_dir, image_name)
        yolo_txt_to_mask(image_path, txt_path, out_path)


def main():
    args = parse_args()
    input_datasets_yaml_path = args.input_datasets_yaml_path
    predict_datasets_folder = args.predict_datasets_folder
    num_classes = args.num_classes
    filter_low_scores = not args.no_filter_low_scores

    settings.reset()
    epochs_num = int(args.epochs)
    batch_num = int(args.batch)
    project_name = args.name
    models_name = args.models

    model_mapping = {
        "yolov8n-seg": "yolov8n-seg.pt",
        "yolov8l-seg": "yolov8l-seg.pt",
        "yolov8m-seg": "yolov8m-seg.pt",
        "yolov8s-seg": "yolov8s-seg.pt",
        "yolov8x-seg": "yolov8x-seg.pt",
        "yolov9c-seg": "yolov9c-seg.pt",
        "yolov9e-seg": "yolov9e-seg.pt",
        "yolo11l-seg": "yolo11l-seg.pt",
        "yolo11m-seg": "yolo11m-seg.pt",
        "yolo11n-seg": "yolo11n-seg.pt",
        "yolo11s-seg": "yolo11s-seg.pt",
        "yolo11x-seg": "yolo11x-seg.pt",
        "yolo26n-seg": "yolo26n-seg.pt",
    }
    models_key = "./models/" + model_mapping.get(models_name, "yolo11n-seg.pt")
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

    model_seg = YOLO(models_key)
    results_yseg = model_seg.train(data=input_datasets_yaml_path, epochs=epochs_num, imgsz=640, batch=batch_num)
    best_model_path = str(results_yseg.save_dir) + "/weights/best.pt"

    if not os.path.exists(best_model_path):
        info_log_model = "INFO. Model training failed : " + best_model_path
    else:
        info_log_model = "INFO. The Model training successful : " + best_model_path

    log_file_path = os.path.dirname(str(results_yseg.save_dir)) + "/yolo_training_log.txt"
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

    res_dir = Path(os.path.dirname(str(results_yseg.save_dir)))
    predict_dir = res_dir / "predict_sahi"
    labels_dir = predict_dir / "labels"
    output_mask_dir = predict_dir / "masks"
    predict_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

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
        write_prediction_txt(result, labels_dir / f"{image_stem}.txt")

    yolo2maskdir_all(str(labels_dir), str(predict_dir), str(output_mask_dir))

    per_image, total_iou, total_dice = evaluate_miou_mdice(
        str(output_mask_dir),
        predict_datasets_folder,
        pred_ext=".jpg",
        gt_ext=".png",
        num_classes=num_classes,
        filter_low_scores=filter_low_scores,
        min_iou=args.min_iou,
        min_dice=args.min_dice,
    )

    for filename, iou, dice in per_image:
        print(f"{filename}: IoU={iou:.4f}, DSC={dice:.4f}")
    print(f"Total Mean IoU: {total_iou:.4f}")
    print(f"Total Mean DSC: {total_dice:.4f}")

    csv_file_path = res_dir / "result.csv"
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "IoU", "DSC"])
        for filename, iou, dice in per_image:
            w.writerow([filename, f"{iou:.6f}", f"{dice:.6f}"])
        w.writerow(["Total", f"{total_iou:.6f}", f"{total_dice:.6f}"])

    res_file_path = res_dir / "result.txt"
    with open(res_file_path, "w", encoding="utf-8") as file:
        file.write("Mean IoU: " + str(total_iou) + "\n")
        file.write("Mean DSC: " + str(total_dice) + "\n")

    yaml_path = input_datasets_yaml_path
    original_image_dir = predict_datasets_folder
    train_dir = str(res_dir / "train")
    html_file = str(res_dir / "index.html")
    pout_dir = str(res_dir / "yolo2images")
    report_function_d(yaml_path, original_image_dir, str(predict_dir), train_dir, html_file, pout_dir)

    print(f"INFO. SAHI segmentation outputs: {predict_dir}")


if __name__ == "__main__":
    main()
