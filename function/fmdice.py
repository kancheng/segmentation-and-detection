import os
import sys
import numpy as np
import cv2
# import argparse

# # 设置参数解析器
# parser = argparse.ArgumentParser()
# parser.add_argument('--pp', default=None, help='The predict path')
# parser.add_argument('--gdp', default=None, help='The ground truth path')
# parser.add_argument('--num_classes', type=int, default=1, help='Number of classes for Dice calculation (including background)')
# args = parser.parse_args()

# # 检查是否提供了路径
# if not args.pp or not args.gdp:
#     print("Error: Missing required paths. Please provide both prediction and ground truth paths.")
#     sys.exit(1)  # 如果路径没有提供，则退出脚本

def dice_coef(pred, target, num_classes):
    """
    计算 Dice 系数
    """
    smooth = 1.0
    dice = 0.0
    for cls in range(num_classes):
        # 将 NumPy 数组转换为 float 类型
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        
        # 计算交集与并集
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        # 更新 Dice 系数
        dice += (2. * intersection + smooth) / (union + smooth)
    
    return dice / num_classes

def calculate_mdice(predict_dir, ground_truth_dir, pred_ext='.jpg', gt_ext='.png', num_classes=2):
    """
    计算多张预测结果和多张 Ground Truth 的 mDice，处理附档名不同的情况
    """
    predict_files = sorted(os.listdir(predict_dir))
    print("predict_files", predict_files)
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    print("ground_truth_files", ground_truth_files)
    dice_list = []

    # 遍历所有的 Ground Truth 文件
    for gt_file in ground_truth_files:
        gt_path = os.path.join(ground_truth_dir, gt_file)
        print("gt_path", gt_path)
        # 尝试找到对应的预测文件，假设文件名一致，但附档名不同
        pred_file = os.path.splitext(gt_file)[0] + pred_ext  # 使用 Ground Truth 文件名并修改为预测图像的附档名
        pred_path = os.path.join(predict_dir, pred_file)
        print("pred_path", pred_path)
        
        if os.path.exists(pred_path):
            # 读取预测和真实标签图像
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # 确保预测和真实标签图像的尺寸一致
            if pred_img.shape != gt_img.shape:
                print(f"Warning: Image sizes do not match for {gt_file}, skipping...")
                continue

            # 计算 Dice
            pred_img_bin = pred_img // 255  # 将预测图像转换为二值图
            gt_img_bin = gt_img // 255  # 将真实标签图像转换为二值图

            dice = dice_coef(pred_img_bin, gt_img_bin, num_classes)
            dice_list.append(dice)
        else:
            print(f"Warning: No prediction file for Ground Truth {gt_file}, skipping...")

    # 计算 mDice
    if dice_list:
        mdice = np.mean(dice_list)
        return mdice
    else:
        print("No valid predictions found.")
        return 0

# 使用例子
# predict_path = './yolo_runs_yolo11n-seg_dl_/segment/predict/masks'  # 预测结果的路径
# ground_truth_path = './datasets/default_data/dataset_masks/masks'  # Ground Truth 的路径

# predict_path = args.pp  # 预测结果的路径
# ground_truth_path = args.gdp  # Ground Truth 的路径
# num_classes = args.num_classes  # 类别数量，默认为 2

# mdice_value = calculate_mdice(predict_path, ground_truth_path, pred_ext='.jpg', gt_ext='.png', num_classes=num_classes)
# print(f'Mean Dice: {mdice_value:.4f}')