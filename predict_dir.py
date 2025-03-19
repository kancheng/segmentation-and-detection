import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset
from unet import UNET
from utils.utils import plot_img_and_mask

def dice_coef(pred, target, num_classes):
    """
    計算 Dice 系數
    """
    smooth = 1.0
    dice = 0.0
    for cls in range(num_classes):
        pred_cls = (torch.tensor(pred) == cls).float()  # 轉換為 torch 張量並使用 float()
        target_cls = (torch.tensor(target) == cls).float()  # 轉換為 torch 張量並使用 float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return dice / num_classes

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False).copy())
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        outputs = net(img)
        # 如果返回的是元組，則取第一個輸出
        if isinstance(outputs, tuple):
            output = outputs[0]
        else:
            output = outputs
        output = output.cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()



def mask_to_image(mask: np.ndarray, mask_values):
    # 根據 mask_values 決定合適的資料型態
    if isinstance(mask_values[0], list):
        dtype = np.uint16 if max(mask_values[0]) > 255 else np.uint8
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=dtype)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        dtype = np.uint16 if max(mask_values) > 255 else np.uint8
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=dtype)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from a directory of input images and compute mIoU and mDice')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='指定存放模型的檔案')
    parser.add_argument('--input-dir', '-i', metavar='INPUT_DIR',
                        help='包含輸入圖像的目錄', required=True)
    parser.add_argument('--output-dir', '-o', metavar='OUTPUT_DIR',
                        help='存放預測結果的目錄', required=True)
    parser.add_argument('--gt-dir', '-g', metavar='GT_DIR',
                        help='包含 Ground Truth Mask 的目錄 (選填)', default=None)
    parser.add_argument('--viz', '-v', action='store_true',
                        help='在處理時視覺化圖像')
    parser.add_argument('--no-save', '-n', action='store_true', help='不儲存輸出結果')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='判定 mask 像素為正的最小機率')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='輸入圖像的縮放比例')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用 bilinear 上採樣')
    parser.add_argument('--classes', '-c', type=int, default=1, help='類別數量')
    return parser.parse_args()

def compute_miou_and_mdice(gt_dir, pred_dir, num_classes):
    gt_files = [f for f in sorted(os.listdir(gt_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    total_dice = 0.0
    num_images = 0
    
    for file in gt_files:
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)
        if not os.path.exists(pred_path):
            logging.warning(f'預測結果中找不到與 {gt_path} 對應的檔案')
            continue
        gt_mask = np.array(Image.open(gt_path))
        pred_mask = np.array(Image.open(pred_path))

        # Calculate IoU
        for cls in range(num_classes):
            gt_cls = (gt_mask == cls)
            pred_cls = (pred_mask == cls)
            intersection = np.logical_and(gt_cls, pred_cls).sum()
            union = np.logical_or(gt_cls, pred_cls).sum()
            total_intersection[cls] += intersection
            total_union[cls] += union

        # Calculate Dice Coefficient
        total_dice += dice_coef(pred_mask, gt_mask, num_classes)
        num_images += 1

    # Calculate IoU
    ious = []
    for cls in range(num_classes):
        if total_union[cls] == 0:
            logging.info(f'類別 {cls} 在 ground truth 中不存在，跳過 IoU 計算')
            continue
        iou = total_intersection[cls] / total_union[cls]
        ious.append(iou)
        logging.info(f'類別 {cls} 的 IoU: {iou:.4f}')

    # Calculate mIoU and mDice
    if len(ious) > 0:
        miou = sum(ious) / len(ious)
        mdice = total_dice / num_images
        logging.info(f'平均 mIoU: {miou:.4f}')
        logging.info(f'平均 mDice: {mdice:.4f}')
        return miou, mdice
    else:
        logging.warning('無法計算 mIoU，請確認 ground truth 中至少有一個類別存在')
        return None, None

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 檢查輸出目錄是否存在，若無則建立
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f'建立輸出目錄: {args.output_dir}')

    # 取得輸入目錄下所有圖檔路徑
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    in_files = [os.path.join(args.input_dir, f) for f in sorted(os.listdir(args.input_dir))
                if f.lower().endswith(valid_extensions)]
    out_files = [os.path.join(args.output_dir, os.path.basename(f)) for f in in_files]

    net = UNET(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'載入模型 {args.model}')
    logging.info(f'使用的裝置: {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('模型載入完成！')

    # 對輸入目錄內的每張圖做預測
    for i, in_file in enumerate(in_files):
        logging.info(f'預測圖像 {in_file} ...')
        img = Image.open(in_file).convert('RGB')
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask 已儲存至 {out_filename}')

        if args.viz:
            logging.info(f'視覺化 {in_file} 的結果，關閉視窗以繼續...')
            plot_img_and_mask(img, mask)

    # 如果有提供 ground truth 目錄，則計算 mIoU 和 mDice
    if args.gt_dir:
        logging.info('開始計算 mIoU 和 mDice...')
        miou, mdice = compute_miou_and_mdice(args.gt_dir, args.output_dir, args.classes)
