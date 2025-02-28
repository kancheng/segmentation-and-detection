import os, sys
import numpy as np
import cv2
import argparse

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--pp', default=None, help='The predict path')
parser.add_argument('--gdp', default=None, help='The ground truth path')
args = parser.parse_args()

# Check if the paths are provided
if not args.pp or not args.gdp:
    print("Error: Missing required paths. Please provide both prediction and ground truth paths.")
    sys.exit(1)  # Exit the script if paths are not provided

def calculate_iou(pred_mask_path, gt_mask_path, threshold=127):
    # 讀取圖像，使用灰階模式
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 檢查是否成功讀取檔案
    if pred_mask is None:
        raise ValueError(f"無法讀取預測的 Mask 圖檔: {pred_mask_path}")
    if gt_mask is None:
        raise ValueError(f"無法讀取原先標柱的 Mask 圖檔: {gt_mask_path}")
    
    # 將圖像轉成二值圖 (0 與 255)
    _, pred_mask_bin = cv2.threshold(pred_mask, threshold, 255, cv2.THRESH_BINARY)
    _, gt_mask_bin = cv2.threshold(gt_mask, threshold, 255, cv2.THRESH_BINARY)
    
    # 將二值圖轉換為 0 與 1（方便計算）
    pred_mask_bin = pred_mask_bin // 255
    gt_mask_bin = gt_mask_bin // 255
    
    # 計算交集與聯集
    intersection = np.logical_and(pred_mask_bin, gt_mask_bin).sum()
    union = np.logical_or(pred_mask_bin, gt_mask_bin).sum()
    
    # 避免除以 0 的狀況
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou

def calculate_miou(predict_dir, ground_truth_dir, pred_ext='.jpg', gt_ext='.png'):
    """
    計算多張預測結果和多張 Ground Truth 的 mIoU，處理附檔名不同的情況
    """
    predict_files = sorted(os.listdir(predict_dir))
    print("predict_files",predict_files)
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    print("ground_truth_files",ground_truth_files)
    iou_list = []

    # 遍歷所有的 Ground Truth 文件
    for gt_file in ground_truth_files:
        gt_path = os.path.join(ground_truth_dir, gt_file)
        print("gt_path",gt_path)
        # 嘗試找尋對應的預測文件，假設檔名一致，但附檔名不同
        pred_file = os.path.splitext(gt_file)[0] + pred_ext  # 使用 Ground Truth 檔名並修改為預測圖像的附檔名
        # print(pred_file)
        pred_path = os.path.join(predict_dir, pred_file)
        print("pred_path",pred_path)
        if os.path.exists(pred_path):
            # 讀取二值化圖像
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # 計算 IoU
            iou = calculate_iou(pred_mask_path=pred_path, gt_mask_path=gt_path)
            iou_list.append(iou)
        else:
            print(f"Warning: No prediction file for Ground Truth {gt_file}, skipping...")

    # 計算 mIoU
    if iou_list:
        miou = np.mean(iou_list)
        return miou
    else:
        print("No valid predictions found.")
        return 0

# 使用範例
# predict_path = './yolo_runs_yolo11n-seg_dl_/segment/predict/masks'  # 預測結果的路徑
# ground_truth_path = './datasets/default_data/dataset_masks/masks'  # Ground Truth 的路徑

predict_path = args.pp  # 預測結果的路徑
ground_truth_path = args.gdp  # Ground Truth 的路徑

miou_value = calculate_miou(predict_path, ground_truth_path, pred_ext='.jpg', gt_ext='.png')
print(f'Mean IoU: {miou_value:.4f}')