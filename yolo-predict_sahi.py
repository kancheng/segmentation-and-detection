import argparse
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--txt", required=True, help="labels directory")
    parser.add_argument("--img", required=True, help="images directory")
    parser.add_argument("--out", default="./outputs/masks", help="mask output directory")
    return parser.parse_args()


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
        # SAHI/YOLO segmentation: cls x1 y1 x2 y2 ... ; detection: cls xc yc w h
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
    os.makedirs(args.out, exist_ok=True)
    yolo2maskdir_all(args.txt, args.img, args.out)


if __name__ == "__main__":
    main()
