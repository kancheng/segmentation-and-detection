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
from function.fdash import report_function_d
from function.fmiou import *
from function.fmdice import *

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
args = parser.parse_args()

# Settings Path.
# input_datasets_yaml_path = '/mnt/ ... /dataset.yaml'
input_datasets_yaml_path = args.input_datasets_yaml_path
# predict_datasets_folder = '/mnt/ ... /'
predict_datasets_folder = args.predict_datasets_folder

num_classes = args.num_classes  # 类别数量，默认为 2
# Update a setting
# datasets_dir = os.path.abspath(os.path.dirname(input_datasets_yaml_path))
# print(datasets_dir)
# settings.update({"datasets_dir": datasets_dir})

# Reset settings to default values
settings.reset()

# epochs
epochs_num = int(args.epochs)
# batch
batch_num = int(args.batch)
# name 
project_name = args.name
# models name
models_name = args.models
models_key = ""
info_log_model_type = ""

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
    'yolo11x-seg': 'yolo11x-seg.pt'
}

# 根據模型名稱選擇對應的模型文件，若不存在則默認使用 'yolo11n-seg.pt'
models_key = './models/' + model_mapping.get(models_name, 'yolo11n-seg.pt')
info_log_model_type = "INFO. Model Type : " + models_key
print(info_log_model_type)


print(models_key)
print(models_name)
# Build Dir.
t = time.strftime("%Y%m%d%H%M%S", time.localtime())
# temtargetpath = './yolo_runs_'+t
p = os.getcwd()
temtargetpath = p + '/yolo_runs_'+ models_name +'_'+ project_name +'_'+ t
command = "yolo settings runs_dir='"+ temtargetpath +"'" 
os.system(command)
# print(temtargetpath)

files = []
info_files = []
# input_folder = args.input_dir
for filename in os.listdir(predict_datasets_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        info_files.append(predict_datasets_folder + "/"+ filename)
        files.append(filename)
info_log_files = "INFO. Files : " + str(files)
info_log_the_file_of_number = "INFO. The File Of Number : " + str(len(files))
print(info_log_files)
print(info_log_the_file_of_number)

# Train the model
# Load a model
# model_seg = YOLO("yolo11n-seg.pt")
model_seg = YOLO(models_key)

## EX: results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
results_yseg = model_seg.train(data=input_datasets_yaml_path, epochs=epochs_num, imgsz=640, batch=batch_num)
results_yseg_model_path = str(results_yseg.save_dir)+"/weights/best.pt"
if not os.path.exists(results_yseg_model_path):
    info_log_model = "INFO. Model training failed : " + results_yseg_model_path
else :
    info_log_model = "INFO. The Model training successful : " + results_yseg_model_path
# log_file_path = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir)) + "/yolo_training_log.txt"
log_file_path = os.path.dirname(str(results_yseg.save_dir)) + "/yolo_training_log.txt"
log_file = open(log_file_path, 'w')
log_file.write( info_log_files + '\n' + info_log_the_file_of_number + '\n' + info_log_model + '\n' + info_log_model_type)
log_file.close()
# Predict
## EX : yolo segment predict model='/mnt/../../yolov8/runs/segment/train/weights/best.pt' source='/mnt/../... .png' save_txt=True

model_predict = YOLO(results_yseg_model_path)

for filename in info_files:
    results_ypred = model_predict.predict(source=filename, save=True, save_txt=True)


## 將預測結果轉換為mask

label_dir = os.path.dirname(str(results_yseg.save_dir)) + "/predict/labels"
print(label_dir)
images_size_dir = os.path.dirname(str(results_yseg.save_dir)) + "/predict"
print(images_size_dir)
output_mask_dir = os.path.dirname(str(results_yseg.save_dir)) + "/predict/masks"
print(output_mask_dir)
# 判断路径是否存在，不存在则创建
if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

'''
Read txt annotation files and original images
'''

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

yolo2maskdir_all(label_dir,images_size_dir,output_mask_dir)

# 使用範例
predict_path = output_mask_dir  # 預測結果的路徑
# predict_path = './datasets/default_data/dataset_masks/masks'  # 預測結果的路徑
# ground_truth_path = './datasets/default_data/dataset_mask/masks'  # Ground Truth 的路徑
ground_truth_path = predict_datasets_folder
miou_value = calculate_miou(predict_path, ground_truth_path, pred_ext='.jpg', gt_ext='.png')
tem_miou_value = "Mean IoU: " + str(miou_value)
print(f'Mean IoU: {miou_value:.4f}')

res_file_path = os.path.dirname(str(results_yseg.save_dir)) + "/result.txt"
# 開啟檔案，如果檔案不存在會自動創建
with open(res_file_path, 'w', encoding='utf-8') as file:
    # 寫入內容
    file.write(tem_miou_value)

print(f"檔案 '{res_file_path}' 已建立並寫入成功。")

mdice_value = calculate_mdice(predict_path, ground_truth_path, pred_ext='.jpg', gt_ext='.png', num_classes=num_classes)
tem_mdice_value = "Mean Dice: " + str(mdice_value)
print(f'Mean Dice: {mdice_value:.4f}')

with open(res_file_path, 'a', encoding='utf-8') as file:
    # 寫入內容
    file.write(tem_mdice_value)

print(f"檔案 '{res_file_path}' 已建立並寫入成功。")

# Report
# 統一基本路徑
# base_report_path = Path("D:/default_data")
# base_yolo_path = Path("D:/YOLO/yolo_runs_")

# 定義各個路徑
yaml_path = input_datasets_yaml_path
original_image_dir = predict_datasets_folder
predict_dir = images_size_dir
train_dir = os.path.dirname(str(results_yseg.save_dir)) + '/' + "train"
html_file = os.path.dirname(str(results_yseg.save_dir)) + '/' +  "index.html"
pout_dir = os.path.dirname(str(results_yseg.save_dir)) + '/' +  "yolo2images"

# 呼叫報告函數
report_function_d(yaml_path, original_image_dir, predict_dir, train_dir, html_file, pout_dir)
