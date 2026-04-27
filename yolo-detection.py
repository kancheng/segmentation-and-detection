import os
import time
from ultralytics import YOLO
import argparse 
from PIL import Image
import argparse
from pathlib import Path
import csv
import torch
from function.fdash import report_function_d


def summarize_detection_labels(label_dir, image_files):
    per_image = []
    total_detections = 0
    for image_name in image_files:
        stem = os.path.splitext(image_name)[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        det_count = 0
        if os.path.exists(label_path):
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

# Args
# EX: python3 yolo-detection.py --input_datasets_yaml_path="/mnt/.../dataset.yaml" --predict_datasets_folder="/mnt/.../"
# EX: python yolo-detection.py --input_datasets_yaml_path="./datasets/default_data/dataset_yolo/YOLODataset/dataset.yaml" --predict_datasets_folder="./datasets/default_data/dataset_predict/"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_datasets_yaml_path', help='input annotated directory')
parser.add_argument('--predict_datasets_folder', help='predict folder')
parser.add_argument('--name', default='dl',  help='project name')
parser.add_argument('--epochs', default=50,  help='epochs')
parser.add_argument('--batch', default=2,  help='batch')
parser.add_argument('--models', default='yolo11n',  help='models name')
parser.add_argument('--output_root', default='./yolo_runs', help='root directory for all Ultralytics outputs')
parser.add_argument('--device', default='cpu', help='inference device, e.g. cpu / cuda:0')
parser.add_argument('--predict_imgsz', type=int, default=512, help='prediction image size (lower uses less RAM)')
args = parser.parse_args()
release_memory(args.device)

# Settings Path.
# input_datasets_yaml_path = '/mnt/ ... /dataset.yaml'
input_datasets_yaml_path = args.input_datasets_yaml_path
# predict_datasets_folder = '/mnt/ ... /'
predict_datasets_folder = args.predict_datasets_folder

# # Update a setting
# datasets_dir = os.path.abspath(os.path.dirname(input_datasets_yaml_path))
# print(datasets_dir)
# settings.update({"datasets_dir": datasets_dir})

# epochs
epochs_num = int(args.epochs)
# batch
batch_num = int(args.batch)
# name 
project_name = args.name
# models name
models_name = args.models
output_root = os.path.abspath(os.path.expanduser(args.output_root))
os.makedirs(output_root, exist_ok=True)
models_key = ""
info_log_model_type = ""

model_mapping = {
    'yolo11n': 'yolo11n.pt',
    'yolo11s': 'yolo11s.pt',
    'yolo11m': 'yolo11m.pt',
    'yolo11l': 'yolo11l.pt',
    'yolo11x': 'yolo11x.pt',
    'yolov8l': 'yolov8l.pt',
    'yolov8m': 'yolov8m.pt',
    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolov8x': 'yolov8x.pt',
    'yolov10b': 'yolov10b.pt',
    'yolov10l': 'yolov10l.pt',
    'yolov10m': 'yolov10m.pt',
    'yolov10n': 'yolov10n.pt',
    'yolov10s': 'yolov10s.pt',
    'yolov10x': 'yolov10x.pt',
    'yolov9c': 'yolov9c.pt',
    'yolov9e': 'yolov9e.pt',
    'yolov9m': 'yolov9m.pt',
    'yolov9s': 'yolov9s.pt',
    'yolov9t': 'yolov9t.pt',
    'yolo12n': 'yolo12n.pt',
    'yolo12s': 'yolo12s.pt',
    'yolo12m': 'yolo12m.pt',
    'yolo12l': 'yolo12l.pt',
    'yolo12x': 'yolo12x.pt'
}

# 如果模型名稱在字典中，則選擇對應的模型文件，否則使用默認模型
models_key = './models/' + model_mapping.get(models_name, 'yolo11n.pt')
info_log_model_type = "INFO. Model Type : " + models_key
print(info_log_model_type)

print(models_key)
print(models_name)
# Build Dir.
t = time.strftime("%Y%m%d%H%M%S", time.localtime())
run_name = f"{models_name}_{project_name}_{t}"

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
model_detection = YOLO(models_key)

## EX: results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
results_ydetection = model_detection.train(
    data=input_datasets_yaml_path,
    epochs=epochs_num,
    imgsz=640,
    batch=batch_num,
    project=output_root,
    name=run_name,
)
results_ydetection_model_path = str(results_ydetection.save_dir)+"/weights/best.pt"
if not os.path.exists(results_ydetection_model_path):
    info_log_model = "INFO. Model training failed : " + results_ydetection_model_path
else :
    info_log_model = "INFO. The Model training successful : " + results_ydetection_model_path
# log_file_path = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir)) + "/yolo_training_log.txt"
run_dir = Path(results_ydetection.save_dir)
log_file_path = str(run_dir / "yolo_training_log.txt")
log_file = open(log_file_path, 'w')
log_file.write( info_log_files + '\n' + info_log_the_file_of_number + '\n' + info_log_model + '\n' + info_log_model_type)
log_file.close()
# Predict
## EX : yolo detect predict model='/mnt/../../yolov8/runs/detection/train/weights/best.pt' source='/mnt/../... .png' save_txt=True

model_predict = YOLO(results_ydetection_model_path)
for image_path in info_files:
    predict_results = model_predict.predict(
        source=image_path,
        save=True,
        save_txt=True,
        imgsz=args.predict_imgsz,
        device=args.device,
        batch=1,
        stream=True,
        project=str(run_dir),
        name="predict",
        exist_ok=True,
    )
    for _ in predict_results:
        pass
    release_memory(args.device)

predict_dir = run_dir / "predict"
label_dir = str(predict_dir / "labels")
print(label_dir)
images_size_dir = str(predict_dir)
print(images_size_dir)

per_image, total_detections = summarize_detection_labels(label_dir, files)
mean_detections = (total_detections / len(files)) if files else 0.0

res_dir = str(run_dir)
res_file_path = res_dir + "/result.txt"
csv_file_path = res_dir + "/result.csv"

with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(["filename", "detections"])
    for filename, det_count in per_image:
        w.writerow([filename, det_count])
    w.writerow(["Total", total_detections])
    w.writerow(["Mean per image", f"{mean_detections:.6f}"])
print(f"CSV 已寫入: {csv_file_path}")

with open(res_file_path, 'w', encoding='utf-8') as file:
    file.write("Total Detections: " + str(total_detections) + "\n")
    file.write("Mean Detections Per Image: " + str(mean_detections) + "\n")
print(f"result.txt 已寫入: {res_file_path}")

yaml_path = input_datasets_yaml_path
original_image_dir = predict_datasets_folder
predict_dir_for_report = images_size_dir
train_dir = str(run_dir)
html_file = str(run_dir / "index.html")
pout_dir = str(run_dir / "yolo2images")
report_function_d(yaml_path, original_image_dir, predict_dir_for_report, train_dir, html_file, pout_dir)
