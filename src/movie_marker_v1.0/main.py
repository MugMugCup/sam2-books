import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

# 入力フォルダ
INPUT_DIR = "../image_resizer_v1.1/aerial_full_25"
# 出力動画ファイル名
OUTPUT_FILE = "aeirial_full_25.mp4"
# FPS（1秒あたりのフレーム数）
FPS = 10

def get_image_datetime(path):
    """画像のEXIFから撮影日時を取得。なければファイルの更新日時を返す"""
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    tag_name = TAGS.get(tag)
                    if tag_name == "DateTimeOriginal":
                        # "2023:09:10 14:23:59" → datetime型に変換
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    # EXIFがなければファイルの更新時刻を使う
    return datetime.fromtimestamp(os.path.getmtime(path))

def images_to_video(input_dir, output_file, fps):
    files = [f for f in os.listdir(input_dir) if not f.startswith(".")]

    if not files:
        print("フォルダに画像がありません")
        return

    # ファイルの撮影日時順にソート
    files.sort(key=lambda x: get_image_datetime(os.path.join(input_dir, x)))

    # 最初の画像でサイズを決定
    first_img_path = os.path.join(input_dir, files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print("最初の画像が読み込めませんでした")
        return

    height, width, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for filename in files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"{filename} を読み込めませんでした。スキップします。")
            continue

        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))

        out.write(img)
        print(f"{filename} を追加しました")

    out.release()
    print(f"動画を出力しました → {output_file}")

if __name__ == "__main__":
    images_to_video(INPUT_DIR, OUTPUT_FILE, FPS)
