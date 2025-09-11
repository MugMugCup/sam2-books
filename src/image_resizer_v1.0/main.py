import os
from PIL import Image

# 入力フォルダと出力フォルダ
INPUT_DIR = "../../data/images/aerial_full"
OUTPUT_DIR = "aerial_full_25"

# リサイズ比率（例: 0.5 で半分）
SCALE = 0.25

def resize_images(input_dir, output_dir, scale):
    # 出力フォルダがなければ作成
    os.makedirs(output_dir, exist_ok=True)

    # フォルダ内のファイルを走査
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # 画像ファイルのみ処理
        try:
            with Image.open(input_path) as img:
                # 新しいサイズを計算
                new_size = (int(img.width * scale), int(img.height * scale))
                resized_img = img.resize(new_size, Image.LANCZOS)

                # 出力先パス
                output_path = os.path.join(output_dir, filename)
                resized_img.save(output_path)

                print(f"{filename} を {new_size} にリサイズしました → {output_path}")
        except Exception as e:
            print(f"{filename} は画像として処理できませんでした: {e}")

if __name__ == "__main__":
    resize_images(INPUT_DIR, OUTPUT_DIR, SCALE)
