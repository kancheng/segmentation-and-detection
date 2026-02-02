import os
import time
from ultralytics import YOLO
from ultralytics import settings
import argparse 
from PIL import Image
import cv2
import numpy as np
import base64
import imutils
import shutil
import json
from pathlib import Path
import csv
from function.fdash import report_function_d
from function.feval import evaluate_miou_mdice


def read_txt_labels(txt_file):
    """
    Read labels from txt annotation file
    :param txt_file: txt annotation file path
    :return: tag list
    """
    with open(txt_file, "r") as f:
        labels = []
        for line in f.readlines():
            label_data = line.strip().split(" ")
            class_id = int(label_data[0])
            # Parsing bounding box coordinates
            coordinates = [float(x) for x in label_data[1:]]
            labels.append([class_id, coordinates])
    return labels

def draw_labels(mask, labels):
    """
    Draw segmentation regions on the image
    :param image: image
    :param labels: list of labels
    """
    for label in labels:
        class_id, coordinates = label
        # Convert coordinates to integers and reshape into polygons
        points = [(int(x * mask.shape[1]), int(y * mask.shape[0])) for x, y in zip(coordinates[::2], coordinates[1::2])]
        # 若點數小於3則不執行填充，以避免 cv2.fillPoly 斷言失敗
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            # Use polygon fill
            cv2.fillPoly(mask, [pts], (255, 255, 255)) # Green indicates segmented area

def yolo2maskdir(kpimg,kptxt,kout):
    """
    Restore the YOLO semantic segmentation txt annotation file to the original image
    """
    # Reading an Image
    # image = cv2.imread("./test/coco128.jpg")
    image = cv2.imread(kpimg)
    height, width, _  = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)
    # Read txt annotation file
    # txt_file = "./test/coco128.txt"
    txt_file = kptxt
    labels = read_txt_labels(txt_file)
    # Draw segmentation area
    draw_labels(mask, labels)
    # Get the window size
    # window_size = (width//2, height//2) # You can resize the window as needed
    window_size = (width, height) # You can resize the window as needed
    # Resize an image
    mask = cv2.resize(mask, window_size)
    # Create a black image the same size as the window
    background = np.zeros((window_size[1], window_size[0], 3), np.uint8)
    # Place the image in the center of the black background
    mask_x = int((window_size[0] - mask.shape[1]) / 2)
    mask_y = int((window_size[1] - mask.shape[0]) / 2)
    background[mask_y:mask_y + mask.shape[0], mask_x:mask_x + mask.shape[1]] = mask
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # Filename
    # filename = 'savedMasks.jpg'args.output
    filename_o = kout

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename_o, mask)

def get_prefix(filename):
    """
    取得檔名中「.」之前的字串
    """
    return filename.split('.')[0]

def filter_common_prefix(list1, list2):
    """
    比較兩個列表，保留在 . 之前的前綴一致的檔案
    回傳 (filtered_list1, filtered_list2)
    """
    # 取得各列表所有檔案的前綴集合
    prefixes1 = {get_prefix(f) for f in list1}
    prefixes2 = {get_prefix(f) for f in list2}
    
    # 求交集，得到共同的前綴
    common_prefixes = prefixes1.intersection(prefixes2)
    
    # 依據共同前綴過濾各自的列表
    filtered_list1 = [f for f in list1 if get_prefix(f) in common_prefixes]
    filtered_list2 = [f for f in list2 if get_prefix(f) in common_prefixes]
    
    return filtered_list1, filtered_list2

def yolo2maskdir_all(label_dir,images_size_dir,output_mask_dir):
    files = []
    txts = []
    for filename in os.listdir(images_size_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".JPEG", ".PNG")):
            #ts = os.path.join(images_size_dir, filename)
            files.append(filename)
    for filename in os.listdir(label_dir):
        if filename.endswith((".txt")):
            #ts = os.path.join(label_dir, filename)
            txts.append(filename)
    filtered_files, filtered_txts = filter_common_prefix(files, txts)
    final_files = []
    final_txts = []
    for filename in filtered_files:
        ts = os.path.join(images_size_dir, filename)
        final_files.append(ts)
    for filename in filtered_txts:
        ts = os.path.join(label_dir, filename)
        final_txts.append(ts)
    for i in range(len(final_files)):
        pimg = final_files[i]
        ptxt = final_txts[i]
        pout = os.path.join(output_mask_dir, os.path.basename(filtered_files[i]))
        yolo2maskdir(pimg,ptxt,pout)


if __name__ == '__main__':
    # Args
    # EX: python3 yolo-segmentation.py --input_datasets_yaml_path="/mnt/.../dataset.yaml" --predict_datasets_folder="/mnt/.../"
    # EX: python yolo-segmentation.py --input_datasets_yaml_path="./datasets/default_data/dataset_yolo/YOLODataset_seg/dataset.yaml" --predict_datasets_folder="./datasets/default_data/dataset_predict/"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_datasets_yaml_path', help='input annotated directory')
    parser.add_argument('--predict_datasets_folder', help='predict folder')
    parser.add_argument('--name', default='dl',  help='project name')
    parser.add_argument('--epochs', default=50,  help='epochs')
    parser.add_argument('--batch', default=2,  help='batch')
    parser.add_argument('--models', default='yolo11n-seg',  help='models name')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes for Dice calculation (including background)')
    parser.add_argument('--no_filter_low_scores', action='store_true', help='disable filter: include all samples in result.csv/result.txt and totals (default: filter enabled, exclude low IoU/DSC)')
    parser.add_argument('--min_iou', type=float, default=0.5, help='min IoU to include in output when filter_low_scores (default: 0.5)')
    parser.add_argument('--min_dice', type=float, default=0.5, help='min DSC to include in output when filter_low_scores (default: 0.5)')
    args = parser.parse_args()

    input_datasets_yaml_path = args.input_datasets_yaml_path
    predict_datasets_folder = args.predict_datasets_folder
    num_classes = args.num_classes
    filter_low_scores = not args.no_filter_low_scores
    min_iou = args.min_iou
    min_dice = args.min_dice

    settings.reset()
    epochs_num = int(args.epochs)
    batch_num = int(args.batch)
    project_name = args.name
    models_name = args.models

    model_mapping = {
        'yolov8n-seg': 'yolov8n-seg.pt',
        'yolov8l-seg': 'yolov8l-seg.pt',
        'yolov8m-seg': 'yolov8m-seg.pt',
        'yolov8s-seg': 'yolov8s-seg.pt',
        'yolov8x-seg': 'yolov8x-seg.pt',
        'yolov9c-seg': 'yolov9c-seg.pt',
        'yolov9e-seg': 'yolov9e-seg.pt',
        'yolo11l-seg': 'yolo11l-seg.pt',
        'yolo11m-seg': 'yolo11m-seg.pt',
        'yolo11n-seg': 'yolo11n-seg.pt',
        'yolo11s-seg': 'yolo11s-seg.pt',
        'yolo11x-seg': 'yolo11x-seg.pt',
        'yolo26n-seg': 'yolo26n-seg.pt'
    }
    models_key = './models/' + model_mapping.get(models_name, 'yolo11n-seg.pt')
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
    print(models_key)
    print(models_name)

    t = time.strftime("%Y%m%d%H%M%S", time.localtime())
    p = os.getcwd()
    temtargetpath = os.path.join(p, 'yolo_runs_' + models_name + '_' + project_name + '_' + t)
    settings.update({"runs_dir": temtargetpath})

    files = []
    info_files = []
    for filename in os.listdir(predict_datasets_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            info_files.append(predict_datasets_folder + "/"+ filename)
            files.append(filename)
    info_log_files = "INFO. Files : " + str(files)
    info_log_the_file_of_number = "INFO. The File Of Number : " + str(len(files))
    print(info_log_files)
    print(info_log_the_file_of_number)

    model_seg = YOLO(models_key)
    results_yseg = model_seg.train(data=input_datasets_yaml_path, epochs=epochs_num, imgsz=640, batch=batch_num)
    results_yseg_model_path = str(results_yseg.save_dir)+"/weights/best.pt"
    if not os.path.exists(results_yseg_model_path):
        info_log_model = "INFO. Model training failed : " + results_yseg_model_path
    else:
        info_log_model = "INFO. The Model training successful : " + results_yseg_model_path
    log_file_path = os.path.dirname(str(results_yseg.save_dir)) + "/yolo_training_log.txt"
    log_file = open(log_file_path, 'w')
    log_file.write( info_log_files + '\n' + info_log_the_file_of_number + '\n' + info_log_model + '\n' + info_log_model_type)
    log_file.close()

    model_predict = YOLO(results_yseg_model_path)
    for filename in info_files:
        model_predict.predict(source=filename, save=True, save_txt=True)

    label_dir = os.path.dirname(str(results_yseg.save_dir)) + "/predict/labels"
    print(label_dir)
    images_size_dir = os.path.dirname(str(results_yseg.save_dir)) + "/predict"
    print(images_size_dir)
    output_mask_dir = os.path.dirname(str(results_yseg.save_dir)) + "/predict/masks"
    print(output_mask_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    yolo2maskdir_all(label_dir, images_size_dir, output_mask_dir)

    predict_path = output_mask_dir
    ground_truth_path = predict_datasets_folder
    res_dir = os.path.dirname(str(results_yseg.save_dir))
    res_file_path = res_dir + "/result.txt"
    csv_file_path = res_dir + "/result.csv"

    per_image, total_iou, total_dice = evaluate_miou_mdice(
        predict_path, ground_truth_path,
        pred_ext='.jpg', gt_ext='.png', num_classes=num_classes,
        filter_low_scores=filter_low_scores, min_iou=min_iou, min_dice=min_dice
    )

    if filter_low_scores:
        print(f"Per-image IoU / DSC (only samples with IoU>={min_iou} and DSC>={min_dice}):")
    else:
        print("Per-image IoU / DSC (all samples):")
    for filename, iou, dice in per_image:
        print(f"  {filename}: IoU={iou:.4f}, DSC={dice:.4f}")
    print(f"Total Mean IoU: {total_iou:.4f}")
    print(f"Total Mean DSC: {total_dice:.4f}")

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["filename", "IoU", "DSC"])
        for filename, iou, dice in per_image:
            w.writerow([filename, f"{iou:.6f}", f"{dice:.6f}"])
        if filter_low_scores:
            w.writerow(["Total (excluding low scores)", f"{total_iou:.6f}", f"{total_dice:.6f}"])
        else:
            w.writerow(["Total", f"{total_iou:.6f}", f"{total_dice:.6f}"])
    print(f"CSV 已寫入: {csv_file_path}")

    with open(res_file_path, 'w', encoding='utf-8') as file:
        if filter_low_scores:
            file.write("Mean IoU (excluding low scores): " + str(total_iou) + "\n")
            file.write("Mean DSC (excluding low scores): " + str(total_dice) + "\n")
        else:
            file.write("Mean IoU: " + str(total_iou) + "\n")
            file.write("Mean DSC: " + str(total_dice) + "\n")
    print(f"result.txt 已寫入: {res_file_path}")

    yaml_path = input_datasets_yaml_path
    original_image_dir = predict_datasets_folder
    predict_dir = images_size_dir
    train_dir = os.path.dirname(str(results_yseg.save_dir)) + '/' + "train"
    html_file = os.path.dirname(str(results_yseg.save_dir)) + '/' +  "index.html"
    pout_dir = os.path.dirname(str(results_yseg.save_dir)) + '/' +  "yolo2images"
    report_function_d(yaml_path, original_image_dir, predict_dir, train_dir, html_file, pout_dir)
