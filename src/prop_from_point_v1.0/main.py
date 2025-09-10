# ========================
# 1) 設定（ここだけ触ればOK）
# ========================
# 解析したい動画を /content にアップロードして、下のパスを設定してください。
SOURCE_VIDEO = "../../notebooks/videos/bedroom.mp4"   # ← ここに自分の動画パス
SCALE_FACTOR = 1.0                    # リサイズ倍率（0.5で半分など）
CUT_START = 100                       # 切り出し開始フレーム
CUT_END = 300                         # 切り出し終了フレーム（非含む）
ANN_FRAME_IDX = 0                     # どのフレームで最初のプロンプトを置くか（切り出し後の先頭=0）
ANN_OBJ_ID = 1                        # 追跡対象のオブジェクトID（任意の整数）
POINT_XY = (550, 290)                 # 最初のクリック座標（x, y）
USE_BOX = False                       # Trueならボックスで初期化（下のBOX_XYXYを使う）
BOX_XYXY = (500, 250, 620, 360)       # (x1, y1, x2, y2)

# ========================
# 2) 連番フレームへ分解
# ========================
from pathlib import Path
import time
import numpy as np
import cv2
import supervision as sv

ts = time.strftime("%Y%m%d_%H%M%S")
ROOT = Path("/content/sam2_demo_" + ts)
ROOT.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = ROOT / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# 動画→フレーム（指定範囲のみ）
frames_generator = sv.get_video_frames_generator(
  source_path=SOURCE_VIDEO, start=CUT_START, end=CUT_END
)
images_sink = sv.ImageSink(
  target_dir_path=FRAMES_DIR.as_posix(),
  overwrite=True,
  image_name_pattern="{:05d}.jpeg"
)
with images_sink:
  for frame in frames_generator:
    if SCALE_FACTOR != 1.0:
      frame = sv.scale_image(frame, SCALE_FACTOR)
    images_sink.save_image(frame)

# ========================
# 3) SAM2 の読み込み
# ========================
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "../../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # smallに対応するyaml

predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# ========================
# 4) 推論ステート初期化＆対象登録
# ========================
# SAM2は動画全体に対する「推論ステート」を持ちます。
# 今回は連番フレームのディレクトリを渡します。
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
  state = predictor.init_state(FRAMES_DIR.as_posix())

  # クリック or ボックスで最初の対象を登録
  if USE_BOX:
    box = np.array(BOX_XYXY, dtype=np.float32)
    _ = predictor.add_new_points_or_box(
      inference_state=state,
      frame_idx=ANN_FRAME_IDX,
      obj_id=ANN_OBJ_ID,
      box=box
    )
  else:
    points = np.array([[POINT_XY[0], POINT_XY[1]]], dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)  # 正例=1
    _ = predictor.add_new_points_or_box(
      inference_state=state,
      frame_idx=ANN_FRAME_IDX,
      obj_id=ANN_OBJ_ID,
      points=points,
      labels=labels
    )

# ========================
# 5) 順方向・逆方向の伝播
# ========================
# 返ってくる masks の型は環境により logit/確率/ブールの可能性があるので安全に処理
def to_bool_mask(arr):
  import numpy as np
  m = arr
  if hasattr(m, "detach"):  # torch.Tensor
    m = m.detach().cpu().numpy()
  m = np.asarray(m)
  if m.dtype == bool:
    return m
  # 2値化（0より大きければマスク）
  return (m > 0).astype(bool)

# フレームパス一覧
frame_paths = sorted(sv.list_files_with_extensions(FRAMES_DIR.as_posix(), extensions=["jpeg"]))
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
video_info.width = int(video_info.width * SCALE_FACTOR)
video_info.height = int(video_info.height * SCALE_FACTOR)

mask_annotator = sv.MaskAnnotator()

f_frames, b_frames = [], []

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
  # Forward (→)
  for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    frame = cv2.imread(frame_paths[frame_idx])
    # 単一オブジェクト想定：先頭だけ使う
    m = to_bool_mask(masks[0])
    det = sv.Detections(
      xyxy=sv.mask_to_xyxy(masks=m),
      mask=m.astype(bool)
    )
    annotated = mask_annotator.annotate(scene=frame.copy(), detections=det)
    f_frames.append(annotated)

  # Backward (←)
  for frame_idx, object_ids, masks in predictor.propagate_in_video(state, reverse=True):
    frame = cv2.imread(frame_paths[frame_idx])
    m = to_bool_mask(masks[0])
    det = sv.Detections(
      xyxy=sv.mask_to_xyxy(masks=m),
      mask=m.astype(bool)
    )
    annotated = mask_annotator.annotate(scene=frame.copy(), detections=det)
    b_frames.append(annotated)

# ========================
# 6) 書き出し（順・逆・結合）
# ========================
out_forward = str(ROOT / f"sam2_forward_{ts}.mp4")
out_backward = str(ROOT / f"sam2_backward_{ts}.mp4")
out_merged  = str(ROOT / f"sam2_merged_{ts}.mp4")

# 順方向
with sv.VideoSink(out_forward, video_info=video_info) as sink:
  for f in f_frames:
    sink.write_frame(f)

# 逆方向（そのまま＝逆再生のまま）
with sv.VideoSink(out_backward, video_info=video_info) as sink:
  for f in b_frames:
    sink.write_frame(f)

# 結合：逆方向を反転して時系列を戻し、先頭の重複1枚を除いて連結
merged = b_frames[::-1] + f_frames[1:]
with sv.VideoSink(out_merged, video_info=video_info) as sink:
  for f in merged:
    sink.write_frame(f)

print("出力ファイル:")
print(out_forward)
print(out_backward)
print(out_merged)
