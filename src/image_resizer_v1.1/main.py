import os
from PIL import Image

INPUT_DIR = "../../data/images/aerial_full"
SCALE = 0.125
OUTPUT_DIR = f"aerial_full_{int(SCALE * 100)}"

def resize_images(input_dir, output_dir, scale):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        try:
            with Image.open(input_path) as img:
                exif = img.info.get("exif")  # 元画像のEXIFを取得

                new_size = (int(img.width * scale), int(img.height * scale))
                resized_img = img.resize(new_size, Image.LANCZOS)

                output_path = os.path.join(output_dir, filename)
                # exifを引き継いで保存
                if exif:
                    resized_img.save(output_path, exif=exif)
                else:
                    resized_img.save(output_path)

                print(f"{filename} をリサイズしました → {output_path}")
        except Exception as e:
            print(f"{filename} は画像として処理できませんでした: {e}")

if __name__ == "__main__":
    resize_images(INPUT_DIR, OUTPUT_DIR, SCALE)
