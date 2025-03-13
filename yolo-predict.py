import os
import numpy as np
import cv2

import argparse

# Args
# EX: python3 yolo-predict.py --txt="./yolo_runs_/segment/predict/labels"  --img="./yolo_runs_/segment/predict" --out="./yolo_runs_/segment/predict/masks"
# python yolo-predict.py --txt="./yolo_runs_yolo11n-seg_dl_20250219095038/segment/predict/labels"  --img="./yolo_runs_yolo11n-seg_dl_20250219095038/segment/predict" --out="./yolo_runs_yolo11n-seg_dl_20250219095038/segment/predict/masks"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--txt')
parser.add_argument('--img')
parser.add_argument('--out', default='./outputs/masks')
args = parser.parse_args()

# YOLO 標註檔案的目錄
# label_dir = r'\yolo_runs\segment\predict\labels'
# label_dir = r'yolo_runs_\segment\predict\labels'
# label_dir = r'yolo_runs_yolo11n-seg_dl_20250219095038\segment\predict\labels'
label_dir = args.txt
# images_size_dir = r'yolo_runs_\segment\predict'
# images_size_dir = r'yolo_runs_yolo11n-seg_dl_20250219095038\segment\predict'
images_size_dir = args.img
# 輸出 Mask 圖像的目錄
# output_mask_dir = r'\yolo_runs\segment\predict\masks'
# output_mask_dir = r'yolo_runs_\segment\predict\masks'
# output_mask_dir = r'yolo_runs_yolo11n-seg_dl_20250219095038\segment\predict\masks'
output_mask_dir = args.out
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
