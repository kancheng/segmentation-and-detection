import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from unet import UNET
from unet import U2NET

import argparse

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 載入 U-Net & U2Net 模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # 定義 U-Net 結構 (簡化版本)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 載入模型
model_path = "model.pth"  # 請修改成你的模型路徑
model = UNet(in_channels=3, out_channels=1)  # 依據你的模型架構修改
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 2. 預測整個資料夾
image_dir = "test_images"  # 修改為你的圖片資料夾
mask_dir = "test_masks"    # 修改為你的 Ground Truth Mask 資料夾 (可選)

image_paths = sorted(glob(os.path.join(image_dir, "*.png")))  # 支援 PNG，請修改副檔名
mask_paths = sorted(glob(os.path.join(mask_dir, "*.png"))) if os.path.exists(mask_dir) else None

# 影像預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # 依據你的模型修改大小
])

# 計算 IoU
def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else 1.0  # 避免除以零

iou_scores = []

with torch.no_grad():
    for i, img_path in enumerate(tqdm(image_paths, desc="Processing Images")):
        # 讀取圖片
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(device)

        # 預測
        output = model(image)
        pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)  # 二值化

        # 若有 Ground Truth Mask，則計算 IoU
        if mask_paths:
            mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))  # 依據你的模型大小調整
            true_mask = (mask > 127).astype(np.uint8)  # 二值化
            iou = compute_iou(pred_mask, true_mask)
            iou_scores.append(iou)

            # 儲存預測結果 (可選)
            save_path = os.path.join("predictions", os.path.basename(img_path))
            os.makedirs("predictions", exist_ok=True)
            cv2.imwrite(save_path, (pred_mask * 255))

# 計算 mIoU
if iou_scores:
    miou = np.mean(iou_scores)
    print(f"Mean IoU (mIoU): {miou:.4f}")
else:
    print("沒有 Ground Truth，無法計算 IoU")