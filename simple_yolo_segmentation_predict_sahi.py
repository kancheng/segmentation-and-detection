import argparse
import os
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# EX: python simple_yolo_segmentation_predict_sahi.py --predict_path="./datasets/default_data/dataset_predict/" --model="yolo11n-seg"
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--predict_path", type=str, required=True, help="預測圖片所在的資料夾路徑")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="YOLO 模型路徑")
    parser.add_argument("--device", type=str, default="cpu", help="推論裝置，例如 cpu 或 cuda:0")
    parser.add_argument("--conf", type=float, default=0.25, help="信心分數門檻")
    parser.add_argument("--slice_height", type=int, default=640, help="切片高度")
    parser.add_argument("--slice_width", type=int, default=640, help="切片寬度")
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2, help="切片高度重疊比例")
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2, help="切片寬度重疊比例")
    parser.add_argument("--save_visual", action="store_true", help="是否儲存可視化預測影像")
    parser.add_argument("--save_txt", action="store_true", help="是否儲存 YOLO txt 標註")
    return parser.parse_args()


def to_yolo_line(obj_pred, image_w, image_h):
    bbox = obj_pred.bbox
    x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
    x_c = ((x1 + x2) / 2.0) / image_w
    y_c = ((y1 + y2) / 2.0) / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    cls = obj_pred.category.id
    return f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def main():
    args = parse_args()
    predict_path = Path(args.predict_path)
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = sorted([p for p in predict_path.iterdir() if p.suffix.lower() in valid_extensions])

    if not image_files:
        print(f"沒有在資料夾 {args.predict_path} 中找到圖片檔案。")
        return

    print("INFO. Files :", [f.name for f in image_files])
    print("INFO. The File Of Number :", len(image_files))

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    out_dir = predict_path / "sahi_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_file in image_files:
        print(f"Processing {image_file} ...")
        result = get_sliced_prediction(
            str(image_file),
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
        )

        stem = image_file.stem
        if args.save_visual:
            visual_path = out_dir / f"{stem}_sahi.jpg"
            result.export_visuals(export_dir=str(out_dir), file_name=f"{stem}_sahi")
            if visual_path.exists():
                print(f"Saved visual: {visual_path}")

        if args.save_txt:
            img = cv2.imread(str(image_file))
            h, w = img.shape[:2]
            txt_path = out_dir / f"{stem}.txt"
            lines = [to_yolo_line(obj_pred, w, h) for obj_pred in result.object_prediction_list]
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Saved labels: {txt_path}")


if __name__ == "__main__":
    main()
