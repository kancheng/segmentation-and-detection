import cv2
import numpy as np

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

if __name__ == '__main__':
    # 設定預測的 mask 路徑與 ground truth mask 路徑
    pred_mask_path = 'predict_mask.png'
    gt_mask_path = 'gt_mask.png'
    
    iou = calculate_iou(pred_mask_path, gt_mask_path)
    print("IOU:", iou)
