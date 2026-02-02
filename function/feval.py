"""
統一評估：一次遍歷計算每張圖的 IoU 與 Dice，
總 IOU / 總 DSC 僅對「單張不為 0」的樣本取平均。
"""
import os
import numpy as np
import cv2
from function.fmiou import calculate_iou_from_arrays
from function.fmdice import dice_coef

THRESHOLD = 127
# Dice 視為 0 的門檻（smooth 可能使 Dice 不為精確 0）
DICE_ZERO_THRESHOLD = 1e-9


def evaluate_miou_mdice(predict_dir, ground_truth_dir, pred_ext='.jpg', gt_ext='.png', num_classes=1):
    """
    計算每張圖的 IoU 與 DSC，並回傳排除單張為 0 後的總 IOU、總 DSC。

    Returns:
        per_image: list of (filename, iou, dice)
        total_iou: 僅對 iou > 0 之樣本取平均，若無則 0.0
        total_dice: 僅對 dice > DICE_ZERO_THRESHOLD 之樣本取平均，若無則 0.0
    """
    ground_truth_files = sorted(
        f for f in os.listdir(ground_truth_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    )
    per_image = []

    for gt_file in ground_truth_files:
        gt_path = os.path.join(ground_truth_dir, gt_file)
        pred_file = os.path.splitext(gt_file)[0] + pred_ext
        pred_path = os.path.join(predict_dir, pred_file)

        if not os.path.exists(pred_path):
            print(f"Warning: No prediction file for Ground Truth {gt_file}, skipping...")
            continue

        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            print(f"Warning: Cannot read {pred_path}, skipping...")
            continue
        if gt_img is None:
            print(f"Warning: Cannot read {gt_path}, skipping...")
            continue
        if pred_img.shape != gt_img.shape:
            print(f"Warning: Image sizes do not match for {gt_file}, skipping...")
            continue

        # 二值化（與 fmiou / fmdice 一致）
        _, pred_bin_255 = cv2.threshold(pred_img, THRESHOLD, 255, cv2.THRESH_BINARY)
        _, gt_bin_255 = cv2.threshold(gt_img, THRESHOLD, 255, cv2.THRESH_BINARY)
        pred_bin = pred_bin_255 // 255
        gt_bin = gt_bin_255 // 255

        iou = calculate_iou_from_arrays(pred_bin, gt_bin)
        dice = dice_coef(pred_bin, gt_bin, num_classes)
        # 檔名使用不含路徑的 basename
        per_image.append((os.path.basename(gt_file), iou, dice))

    # 總 IOU：僅納入單張 IoU > 0
    iou_nonzero = [x[1] for x in per_image if x[1] > 0]
    total_iou = float(np.mean(iou_nonzero)) if iou_nonzero else 0.0

    # 總 DSC：僅納入單張 DSC 不為 0（以門檻判斷）
    dice_nonzero = [x[2] for x in per_image if x[2] > DICE_ZERO_THRESHOLD]
    total_dice = float(np.mean(dice_nonzero)) if dice_nonzero else 0.0

    return per_image, total_iou, total_dice
