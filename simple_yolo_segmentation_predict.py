import os
import argparse
from ultralytics import YOLO

# EX: python3 simple_yolo_segmentation_predict.py --predict_path="./datasets/default_data/dataset_predict/" --model="yolo11n-seg"
# EX: python simple_yolo_segmentation_predict.py --predict_path="./datasets/default_data/dataset_predict/" --model="yolo11n-seg"
# EX: python simple_yolo_segmentation_predict.py --predict_path="E:/images" --model="E:/segment/train/weights/best.pt"
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--predict_path', type=str, required=True,
                        help='預測圖片所在的資料夾路徑')
    parser.add_argument('--model', type=str, default="yolo11n-seg",
                        help='YOLO 模型的路徑，若不提供則使用預設模型 yolo11n-seg')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='類別數量，預設為 2')
    return parser.parse_args()

def main():
    args = parse_args()

    # 載入模型 (使用傳入的模型路徑或預設模型)
    model = YOLO(args.model)

    # 取得資料夾中所有圖片檔案 (支援多種格式)
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = [
        os.path.join(args.predict_path, filename)
        for filename in os.listdir(args.predict_path)
        if filename.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(f"沒有在資料夾 {args.predict_path} 中找到圖片檔案。")
        return

    # 列印資訊
    print("INFO. Files :", [os.path.basename(f) for f in image_files])
    print("INFO. The File Of Number :", len(image_files))

    # 對每個圖片進行預測
    for image_file in image_files:
        print(f"Processing {image_file} ...")
        results = model.predict(source=image_file, save=True, save_txt=True)
        # 若有需要進一步處理 results 可在此加入

if __name__ == '__main__':
    main()
