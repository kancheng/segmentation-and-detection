import torch
import torch.nn.functional as F
from tqdm import tqdm

def dice_coef(pred, target, num_classes):
    """
    计算 Dice 系数
    """
    smooth = 1.0
    dice = 0.0
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return dice / num_classes

def evaluate(model, loader, device, amp, num_classes):
    model.eval()
    dice_score_total = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', unit='batch'):
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                outputs = model(images)
                
                if isinstance(outputs, tuple):
                    # 对于多输出模型（如 U2NET），选择主输出进行评估
                    mask_pred = outputs[-1]  # 假设最后一个输出是主输出
                else:
                    # 对于单输出模型（如 UNET）
                    mask_pred = outputs
                
                if num_classes == 1:
                    # 二分类目标
                    mask_pred = torch.sigmoid(mask_pred.squeeze(1))
                    pred = (mask_pred > 0.5).float()
                else:
                    # 多分类目标
                    pred = torch.argmax(mask_pred, dim=1)
                
                # 计算 Dice 分数
                dice_score = dice_coef(pred, true_masks, num_classes)
                dice_score_total += dice_score
                num_batches += 1
    
    average_dice = dice_score_total / num_batches
    return average_dice

