import os
import time
from ultralytics import YOLO
from ultralytics import settings
import argparse 
from PIL import Image
import argparse

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
args = parser.parse_args()

# Settings Path.
# input_datasets_yaml_path = '/mnt/ ... /dataset.yaml'
input_datasets_yaml_path = args.input_datasets_yaml_path
# predict_datasets_folder = '/mnt/ ... /'
predict_datasets_folder = args.predict_datasets_folder

# # Update a setting
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
    'yolo11n': 'yolo11n.pt',
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
    'yolov9t': 'yolov9t.pt'
}

# 如果模型名稱在字典中，則選擇對應的模型文件，否則使用默認模型
models_key = './models/' + model_mapping.get(models_name, 'yolo11n.pt')
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
model_detection = YOLO(models_key)

## EX: results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
results_ydetection = model_detection.train(data=input_datasets_yaml_path, epochs=epochs_num, imgsz=640, batch=batch_num)
results_ydetection_model_path = str(results_ydetection.save_dir)+"/weights/best.pt"
if not os.path.exists(results_ydetection_model_path):
    info_log_model = "INFO. Model training failed : " + results_ydetection_model_path
else :
    info_log_model = "INFO. The Model training successful : " + results_ydetection_model_path
# log_file_path = os.path.dirname(os.getcwd()+"/"+str(results_yseg.save_dir)) + "/yolo_training_log.txt"
log_file_path = os.path.dirname(str(results_ydetection.save_dir)) + "/yolo_training_log.txt"
log_file = open(log_file_path, 'w')
log_file.write( info_log_files + '\n' + info_log_the_file_of_number + '\n' + info_log_model + '\n' + info_log_model_type)
log_file.close()
# Predict
## EX : yolo detect predict model='/mnt/../../yolov8/runs/detection/train/weights/best.pt' source='/mnt/../... .png' save_txt=True

model_predict = YOLO(results_ydetection_model_path)

for filename in info_files:
    results_ypred = model_predict.predict(source=filename, save=True, save_txt=True)

