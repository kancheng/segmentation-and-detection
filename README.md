# Segmentation and Detection

本專案提供影像分割與物件偵測的 AI 框架，整合 **YOLO 分割**、**UNet**、**U2Net** 等模型，支援訓練、預測與評估流程。

## 支援方法

| 方法 | 說明 |
|------|------|
| **YOLO** | YOLOv8 / YOLOv9 / YOLO11 語義分割，支援多種 backbone（n/s/m/l/x） |
| **UNet** | 經典 U-Net 架構，用於影像分割 |
| **U2Net** | U²-Net 架構，用於顯著性檢測與分割 |

## 環境需求

- Python 3.8+
- CUDA（可選，用於 GPU 訓練）

## 安裝

```bash
# 使用 requirements 一次安裝依賴
pip install -r requirements.txt

# 或僅安裝 YOLO 分割所需
pip install ultralytics
pip install -U onnx
pip install opencv-python numpy
```

## YOLO 分割：快速開始

主程式為 `yolo-segmentation.py`，會依序執行：**訓練 → 預測 → 產生 mask → 評估 IoU / DSC → 輸出結果與報表**。

### 基本用法

```bash
python yolo-segmentation.py ^
  --input_datasets_yaml_path="<資料集 yaml 路徑>" ^
  --predict_datasets_folder="<預測/評估用圖片或 Ground Truth 目錄>"
```

### 參數說明

| 參數 | 必填 | 預設值 | 說明 |
|------|------|--------|------|
| `--input_datasets_yaml_path` | 是 | — | 訓練用資料集 YAML 路徑（如 `dataset.yaml`） |
| `--predict_datasets_folder` | 是 | — | 預測用圖片目錄；評估時亦作為 **Ground Truth** 目錄（需與預測結果檔名對齊） |
| `--name` | 否 | `dl` | 專案名稱，用於輸出資料夾命名 |
| `--epochs` | 否 | `50` | 訓練輪數 |
| `--batch` | 否 | `2` | 批次大小 |
| `--models` | 否 | `yolo11n-seg` | 模型名稱 |
| `--num_classes` | 否 | `1` | 類別數（含背景），用於 Dice 計算 |

### 支援的 YOLO 分割模型

- **YOLOv8**：`yolov8n-seg`, `yolov8s-seg`, `yolov8m-seg`, `yolov8l-seg`, `yolov8x-seg`
- **YOLOv9**：`yolov9c-seg`, `yolov9e-seg`
- **YOLO11**：`yolo11n-seg`, `yolo11s-seg`, `yolo11m-seg`, `yolo11l-seg`, `yolo11x-seg`
- **YOLO26**：`yolo26n-seg`

### 範例

```bash
# Windows
python yolo-segmentation.py --input_datasets_yaml_path="./datasets/default_data/dataset_yolo/YOLODataset_seg/dataset.yaml" --predict_datasets_folder="./demo_data/imgs"

# Linux / macOS
python3 yolo-segmentation.py --input_datasets_yaml_path="/path/to/dataset.yaml" --predict_datasets_folder="/path/to/predict_folder"
```

## 輸出說明

執行完成後，會在專案目錄下產生以時間戳命名的資料夾（如 `yolo_runs_yolo11n-seg_dl_<時間>`），內含：

| 項目 | 說明 |
|------|------|
| `weights/best.pt` | 最佳權重 |
| `predict/` | 預測結果（圖片、labels、masks） |
| `result.csv` | 每張圖的 IoU、DSC，以及總平均 |
| `result.txt` | 總平均 Mean IoU、Mean DSC |
| `yolo_training_log.txt` | 訓練與模型路徑紀錄 |
| `index.html` | 報表頁面（透過 `report_function_d` 產生） |

評估時會比對 **預測 mask** 與 **Ground Truth**（由 `--predict_datasets_folder` 指定目錄內的對應檔名），計算每張圖的 **IoU** 與 **DSC (Dice)**，並輸出排除無效樣本後的總平均。

## 測試支援（Simple Scripts）

為了快速驗證環境、模型與基本流程，專案提供四支可直接測試的參數化腳本：

- `simple-yolo-detection.py`
- `simple-yolo-detection_sahi.py`
- `simple-yolo-segmentation.py`
- `simple-yolo-segmentation_sahi.py`

> 這四支腳本目前都支援 `--epochs` 參數，不再使用寫死的訓練輪數。

### 1) Detection（不含 SAHI）

```bash
python simple-yolo-detection.py --epochs 10 --data coco8.yaml --model ./models/yolo11n.pt
```

常用參數：

- `--model`：Path to pretrained model
- `--data`：Path to training dataset YAML
- `--epochs`：Number of training epochs
- `--predict_source`：Input source for prediction
- `--export_format`：Format used for model export

### 2) Detection（SAHI 切片推論）

```bash
python simple-yolo-detection_sahi.py --epochs 10 --data coco8.yaml --model ./models/yolo11n.pt --device cpu
```

額外支援 SAHI 參數：

- `--conf`
- `--slice_height`
- `--slice_width`
- `--overlap_height_ratio`
- `--overlap_width_ratio`

### 3) Segmentation（不含 SAHI）

```bash
python simple-yolo-segmentation.py --epochs 10 --data coco8-seg.yaml --model ./models/yolo11n-seg.pt --imgsz 640
```

常用參數：

- `--model`：Path to pretrained segmentation model
- `--data`：Path to training dataset YAML
- `--epochs`：Number of training epochs
- `--imgsz`：Training image size
- `--predict_source`：Input source for prediction
- `--export_format`：Format used for model export

### 4) Segmentation（SAHI 切片推論）

```bash
python simple-yolo-segmentation_sahi.py --epochs 10 --data coco8-seg.yaml --model ./models/yolo11n-seg.pt --imgsz 640 --device cpu
```

建議先以預設參數跑通，再依資料集大小調整 `--epochs`、`--imgsz`、切片大小與重疊比例。

## 專案結構概覽

```
segmentation-and-detection/
├── yolo-segmentation.py   # YOLO 分割：訓練、預測、評估主程式
├── yolo-detection.py     # YOLO 偵測
├── yolo-detection_sahi.py # YOLO 偵測（含 SAHI）
├── simple-yolo-detection.py
├── simple-yolo-detection_sahi.py
├── simple-yolo-segmentation.py
├── simple-yolo-segmentation_sahi.py
├── yolo-predict.py       # YOLO 預測
├── train_unet.py         # UNet 訓練
├── train_u2net.py        # U2Net 訓練
├── function/
│   ├── feval.py          # 統一評估：IoU + DSC（evaluate_miou_mdice）
│   ├── fmiou.py          # IoU 計算
│   ├── fmdice.py         # Dice 係數
│   └── fdash.py          # 報表產生
├── datasets/             # 資料集與 yaml 配置
├── demo_data/            # 範例圖片與 mask
├── models/               # 模型權重存放目錄
└── requirements.txt
```

```
python simple-yolo-detection.py --epochs 10 --data coco8.yaml --model ./models/yolo11n.pt

python simple-yolo-detection_sahi.py --epochs 10 --data coco8.yaml --model ./models/yolo11n.pt --device cpu

python simple-yolo-segmentation.py --epochs 10 --data coco8-seg.yaml --model ./models/yolo11n-seg.pt --imgsz 640

python simple-yolo-segmentation_sahi.py --epochs 10 --data coco8-seg.yaml --model ./models/yolo11n-seg.pt --imgsz 640 --device cpu

```


## 授權

見 [LICENSE](LICENSE)。
