import os
import cv2
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

# ===================== ユーザー設定 =====================
DEFAULT_CFG = "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_CKPT = "checkpoints/sam2.1_hiera_small.pt"
ALPHA = 0.35            # 緑/赤の半透明度(0-1)
HOLE_AREA_MAX = 300     # 赤で強調する穴の最大面積(px^2)
WRITE_VIDEO = True      # MP4も保存
FPS = 30
VOS_OPTIMIZED = True
# ======================================================

def auto_out_dir(base="outputs"):
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = os.path.join(base, f"tracked_overlay_{ts}")
  os.makedirs(out_dir, exist_ok=True)
  return out_dir

def select_roi_box(img_bgr):
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
  mask_u8 = (mask_bool.astype(np.uint8) * 255)
  inv = cv2.bitwise_not(mask_u8)

  h, w = mask_bool.shape
  ff_mask = np.zeros((h + 2, w + 2), np.uint8)
  flood = inv.copy()
  cv2.floodFill(flood, ff_mask, (0, 0), 255)

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
  mask = mask_bool.astype(bool)

  out = img_bgr.copy()
  green_layer = out.copy()
  green_layer[mask] = (0, 255, 0)
  out = cv2.addWeighted(out, 1.0 - alpha, green_layer, alpha, 0.0)

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
  frames_dir = pick_dir("動画フレーム（0.jpg,1.jpg,...) のフォルダを選択")
  first_path = os.path.join(frames_dir, "0.jpg")
  first_img = cv2.imread(first_path, cv2.IMREAD_COLOR)
  if first_img is None:
    raise RuntimeError("0.jpg が読めません。フォルダがJPEG連番(0.jpg,1.jpg,...)になっているか確認してください。")

  box = select_roi_box(first_img)

  cfg_path = DEFAULT_CFG if os.path.exists(DEFAULT_CFG) else pick_file(
    "SAM2のconfig(.yaml)を選択",
    (("YAML", "*.yaml;*.yml"), ("All", "*.*")),
  )
  print(cfg_path)
  ckpt_path = DEFAULT_CKPT if os.path.exists(DEFAULT_CKPT) else pick_file(
    "SAM2のチェックポイント(.pt)を選択",
    (("PyTorch", "*.pt"), ("All", "*.*")),
  )
  print(ckpt_path)

  out_dir = auto_out_dir()
  frames_out_dir = os.path.join(out_dir, "frames")
  os.makedirs(frames_out_dir, exist_ok=True)

  device = "cuda" if torch.cuda.is_available() else "cpu"

  predictor = build_sam2_video_predictor(
    cfg_path,
    ckpt_path,
    vos_optimized=VOS_OPTIMIZED
  )

  class Dummy:
    def __enter__(self): return None
    def __exit__(self, a, b, c): return False

  with torch.inference_mode():
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else Dummy()
    with autocast_ctx:
      state = predictor.init_state(video_path=frames_dir)

      _fidx, _obj_ids, _masks = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        box=np.array(box, dtype=np.float32),
      )

      video_segments = {}
      for out_frame_idx, out_obj_ids, out_masks in predictor.propagate_in_video(state):
        per_frame = {}
        for i, oid in enumerate(out_obj_ids):
          mb = out_masks[i]
          if isinstance(mb, torch.Tensor):
            mb = mb.detach().cpu().numpy()
          mask_bool = np.squeeze(mb) > 0.0
          per_frame[int(oid)] = mask_bool
        video_segments[out_frame_idx] = per_frame

  h, w = first_img.shape[:2]
  frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
  for idx, p in enumerate(frame_paths):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
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

  if WRITE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = os.path.join(out_dir, f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    writer = cv2.VideoWriter(vid_path, fourcc, FPS, (w, h))
    out_frames = sorted(glob.glob(os.path.join(frames_out_dir, "overlay_*.png")))
    for p in out_frames:
      im = cv2.imread(p, cv2.IMREAD_COLOR)
      if im is not None and im.shape[1] == w and im.shape[0] == h:
        writer.write(im)
    writer.release()

  print(f"[DONE] outputs -> {out_dir}")

if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    tk.Tk().withdraw()
    messagebox.showerror("エラー", str(e))
    raise
