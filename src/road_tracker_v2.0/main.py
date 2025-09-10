import os
import cv2
import glob
import time
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import numpy as np
import torch

# SAM2
from sam2.build_sam import build_sam2_video_predictor

# ===================== ユーザー設定（ここを書き換えるだけでも運用可） =====================
DEFAULT_CFG = "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"  # 公式のsam2.1設定ファイル
DEFAULT_CKPT = "checkpoints/sam2.1_hiera_small.pt"       # 公式チェックポイントの例
ALPHA = 0.35                                             # 緑/赤の半透明度(0-1)
HOLE_AREA_MAX = 300                                      # 赤で強調する穴の最大面積(px^2)
WRITE_VIDEO = True                                       # MP4も保存するか
FPS = 30                                                 # 動画出力FPS
VOS_OPTIMIZED = True                                     # 速度最適化（torch.compile等）
# =======================================================================================

def auto_out_dir(base="outputs"):
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = os.path.join(base, f"tracked_overlay_{ts}")
  os.makedirs(out_dir, exist_ok=True)
  return out_dir

def ensure_jpeg_sequence(src_dir, work_dir):
  """
  任意拡張子のフレーム群を SAM2 が読む JPEG 連番 (0.jpg, 1.jpg, ...) に整形。
  """
  if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
  os.makedirs(work_dir, exist_ok=True)

  exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
  paths = []
  for ex in exts:
    paths.extend(glob.glob(os.path.join(src_dir, ex)))
  if not paths:
    raise FileNotFoundError(f"No images in: {src_dir}")

  def sort_key(p):
    base = os.path.splitext(os.path.basename(p))[0]
    nums = "".join([c if c.isdigit() else " " for c in base]).split()
    return (int(nums[-1]) if nums else 1 << 30, base, p)

  paths = sorted(paths, key=sort_key)

  for i, p in enumerate(paths):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
      continue
    dst = os.path.join(work_dir, f"{i}.jpg")
    cv2.imwrite(dst, im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

  frame_count = len(glob.glob(os.path.join(work_dir, "*.jpg")))
  if frame_count == 0:
    raise RuntimeError("Failed to prepare JPEG sequence.")
  return work_dir, frame_count

def select_roi_box(img_bgr):
  """
  1枚目で矩形ドラッグ選択。戻り値 [x_min, y_min, x_max, y_max] (float)
  """
  h, w = img_bgr.shape[:2]
  win = "Select ROI - drag, Enter=OK, Esc=Cancel"
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win, min(1280, w), min(720, h))
  box = cv2.selectROI(win, img_bgr, fromCenter=False, showCrosshair=True)
  cv2.destroyWindow(win)
  if box is None or box == (0, 0, 0, 0):
    raise RuntimeError("ROI not selected.")
  x, y, bw, bh = box
  return [float(x), float(y), float(x + bw), float(y + bh)]

def find_small_holes(mask_bool, area_max=300):
  """
  マスク内部の「穴」を OpenCV の floodFill + connectedComponents で抽出し、
  面積が area_max 以下のものだけ True にする。
  """
  mask_u8 = (mask_bool.astype(np.uint8) * 255)
  inv = cv2.bitwise_not(mask_u8)

  h, w = mask_bool.shape
  ff_mask = np.zeros((h + 2, w + 2), np.uint8)
  flood = inv.copy()
  # 画像外周と連結な背景を 255 に塗る
  cv2.floodFill(flood, ff_mask, (0, 0), 255)

  # inv のうち flood で塗られなかった領域 = マスク内部の穴
  holes = cv2.bitwise_and(inv, cv2.bitwise_not(flood))
  holes_bin = (holes > 0).astype(np.uint8)

  n_labels, labels = cv2.connectedComponents(holes_bin, connectivity=8)
  small = np.zeros_like(mask_bool, dtype=bool)
  for lid in range(1, n_labels):
    area = np.sum(labels == lid)
    if area <= area_max:
      small[labels == lid] = True
  return small

def overlay_green_and_holes(img_bgr, mask_bool, hole_area_max=300, alpha=0.35):
  """
  緑: 追跡マスク（半透明）
  赤: マスク内部の穴 (面積 <= hole_area_max)
  """
  mask = mask_bool.astype(bool)

  # 緑オーバーレイ
  out = img_bgr.copy()
  green_layer = out.copy()
  green_layer[mask] = (0, 255, 0)
  out = cv2.addWeighted(out, 1.0 - alpha, green_layer, alpha, 0.0)

  # 赤（小さな穴のみ）
  small_holes = find_small_holes(mask, area_max=hole_area_max)
  if np.any(small_holes):
    yy, xx = np.where(small_holes)
    base = out[yy, xx].astype(np.float32)
    red = np.array([0, 0, 255], dtype=np.float32)
    blended = (1.0 - alpha) * base + alpha * red
    out[yy, xx] = blended.astype(np.uint8)

  return out, small_holes

def pick_dir(title="フレームフォルダを選択してください"):
  root = tk.Tk()
  root.withdraw()
  path = filedialog.askdirectory(title=title)
  root.update()
  root.destroy()
  if not path:
    raise RuntimeError("フォルダが選択されませんでした。")
  return path

def pick_file(title="ファイルを選択", filetypes=(("All", "*.*"),)):
  root = tk.Tk()
  root.withdraw()
  path = filedialog.askopenfilename(title=title, filetypes=filetypes)
  root.update()
  root.destroy()
  if not path:
    raise RuntimeError("ファイルが選択されませんでした。")
  return path

def main():
  # フレームフォルダ選択
  frames_dir = pick_dir("動画フレーム（画像群）のフォルダを選択")
  work_jpg_dir = os.path.join("tmp_jpeg_seq")

  # JPEG連番を準備
  jpg_dir, frame_count = ensure_jpeg_sequence(frames_dir, work_jpg_dir)
  first_path = os.path.join(jpg_dir, "0.jpg")
  first_img = cv2.imread(first_path, cv2.IMREAD_COLOR)
  if first_img is None:
    raise RuntimeError("Failed to read first frame.")

  # ROI選択
  box = select_roi_box(first_img)

  # CFG/CKPT の存在確認。なければダイアログで選ばせる。
  cfg_path = DEFAULT_CFG if os.path.exists(DEFAULT_CFG) else pick_file(
    "SAM2のconfig(.yaml)を選択",
    (("YAML", "*.yaml;*.yml"), ("All", "*.*")),
  )
  ckpt_path = DEFAULT_CKPT if os.path.exists(DEFAULT_CKPT) else pick_file(
    "SAM2のチェックポイント(.pt)を選択",
    (("PyTorch", "*.pt"), ("All", "*.*")),
  )

  out_dir = auto_out_dir()
  frames_out_dir = os.path.join(out_dir, "frames")
  os.makedirs(frames_out_dir, exist_ok=True)

  device = "cuda" if torch.cuda.is_available() else "cpu"

  # 予測器の構築（VOS最適化ON）
  predictor = build_sam2_video_predictor(
    cfg_path,
    ckpt_path,
    vos_optimized=VOS_OPTIMIZED
  )

  # 追跡
  with torch.inference_mode():
    if device == "cuda":
      autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
      # CPU では autocast なし
      class Dummy:
        def __enter__(self): return None
        def __exit__(self, a, b, c): return False
      autocast_ctx = Dummy()

    with autocast_ctx:
      state = predictor.init_state(video_path=jpg_dir)

      # 1フレーム目に矩形プロンプト
      _fidx, _obj_ids, _masks = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        box=np.array(box, dtype=np.float32),
      )

      # 全フレームへ伝播
      video_segments = {}
      for out_frame_idx, out_obj_ids, out_masks in predictor.propagate_in_video(state):
        per_frame = {}
        for i, oid in enumerate(out_obj_ids):
          mb = out_masks[i]
          if isinstance(mb, torch.Tensor):
            mb = mb.detach().cpu().numpy()
          mask_bool = np.squeeze(mb) > 0.0  # ロジット0を閾値（sigmoid0.5相当）
          per_frame[int(oid)] = mask_bool
        video_segments[out_frame_idx] = per_frame

  # 合成・保存
  h, w = first_img.shape[:2]
  for idx in range(frame_count):
    img = cv2.imread(os.path.join(jpg_dir, f"{idx}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
      continue
    mask = video_segments.get(idx, {}).get(1, None)
    if mask is None:
      out = img
    else:
      out, _ = overlay_green_and_holes(
        img, mask, hole_area_max=HOLE_AREA_MAX, alpha=ALPHA
      )
    cv2.imwrite(os.path.join(frames_out_dir, f"overlay_{idx:05d}.png"), out)

  # 動画出力
  if WRITE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = os.path.join(out_dir, f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    writer = cv2.VideoWriter(vid_path, fourcc, FPS, (w, h))
    frame_paths = sorted(glob.glob(os.path.join(frames_out_dir, "overlay_*.png")))
    for p in frame_paths:
      im = cv2.imread(p, cv2.IMREAD_COLOR)
      if im is not None and im.shape[1] == w and im.shape[0] == h:
        writer.write(im)
    writer.release()

  print(f"[DONE] outputs -> {out_dir}")

  # 中間JPEGを消したい場合は以下を有効化
  # shutil.rmtree(jpg_dir, ignore_errors=True)

if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    tk.Tk().withdraw()
    messagebox.showerror("エラー", str(e))
    raise
