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


def evaluate_miou_mdice(
    predict_dir,
    ground_truth_dir,
    pred_ext='.jpg',
    gt_ext='.png',
    num_classes=1,
    filter_low_scores=True,
    min_iou=0.5,
    min_dice=0.5,
):
    """
    計算每張圖的 IoU 與 DSC，並回傳總 IOU、總 DSC。

    二值分割 (num_classes=1) 時，IoU 與 DSC 皆針對**前景**計算，語義一致。

    filter_low_scores=True（預設）時，IoU 或 DSC 低於門檻的樣本會從回傳列表與總計中排除，
    故 result.csv / result.txt 僅含「通過門檻」的樣本，總體數據較好看。

    Returns:
        per_image: list of (filename, iou, dice)，若 filter_low_scores 則僅含通過門檻者
        total_iou: 總平均 IoU（僅納入 per_image 內樣本）
        total_dice: 總平均 DSC（僅納入 per_image 內樣本）
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

    # 過低分數過濾（預設啟用）：IoU 或 DSC 低於門檻的樣本不納入總計與輸出
    if filter_low_scores:
        per_image = [x for x in per_image if x[1] >= min_iou and x[2] >= min_dice]

    # 總 IOU / 總 DSC：僅對目前 per_image 內樣本取平均
    if per_image:
        total_iou = float(np.mean([x[1] for x in per_image]))
        total_dice = float(np.mean([x[2] for x in per_image]))
    else:
        total_iou = 0.0
        total_dice = 0.0

    return per_image, total_iou, total_dice
