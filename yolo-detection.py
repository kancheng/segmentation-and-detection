import os
import time
from ultralytics import YOLO
from ultralytics import settings
import argparse 
from PIL import Image
import argparse

# Args
# EX: python3 yolo-segmentation.py --input_datasets_yaml_path="/mnt/.../dataset.yaml" --predict_datasets_folder="/mnt/.../"
# EX: python yolo-segmentation.py --input_datasets_yaml_path="./datasets/default_data/dataset_yolo/YOLODataset_seg/dataset.yaml" --predict_datasets_folder="./datasets/default_data/dataset_predict/"
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
if models_name == 'yolo11n' :
    models_key = './models/' + 'yolo11n.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8l' :
    models_key = './models/' + 'yolov8l.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8m' :
    models_key = './models/' + 'yolov8m.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8n' :
    models_key = './models/' + 'yolov8n.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8s' :
    models_key = './models/' + 'yolov8s.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov8x' :
    models_key = './models/' + 'yolov8x.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov10b' :
    models_key = './models/' + 'yolov10b.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov10l' :
    models_key = './models/' + 'yolov10l.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov10m' :
    models_key = './models/' + 'yolov10m.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov10n' :
    models_key = './models/' + 'yolov10n.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov10s' :
    models_key = './models/' + 'yolov10s.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov10x' :
    models_key = './models/' + 'yolov10x.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov9c' :
    models_key = './models/' + 'yolov9c.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov9e' :
    models_key = './models/' + 'yolov9e.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov9m' :
    models_key = './models/' + 'yolov9m.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov9s' :
    models_key = './models/' + 'yolov9s.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
elif models_name == 'yolov9t' :
    models_key = './models/' + 'yolov9t.pt'
    info_log_model_type = "INFO. Model Type : " + models_key
    print(info_log_model_type)
else :
    models_key = './models/' + 'yolo11n.pt'
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
model_detection = YOLO(models_key)

## EX: results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
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
## EX : yolo segment predict model='/mnt/../../yolov8/runs/segment/train/weights/best.pt' source='/mnt/../... .png' save_txt=True

model_predict = YOLO(results_ydetection_model_path)

for filename in info_files:
    results_ypred = model_predict.predict(source=filename, save=True, save_txt=True)

# YOLO Predict Label to Labelme JSON

files = []
info_files = []
files_check = []
input_folder = os.path.dirname(str(results_ydetection.save_dir)) +'/predict'
input_folder_labels = input_folder + '/labels'
for filename in os.listdir(input_folder_labels):
    if filename.endswith((".txt")):
        info_files.append(input_folder_labels + "/" + filename)
        files.append(filename) 
        for con in files:
            files_check.append(con.split(".")[0])
print("INFO. Files : ", files)
print("INFO. The File Of Number : ", len(files))
# print("INFO. TXT Path : ", info_files)
