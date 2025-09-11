#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 Video GUI (PySide6) – single‑click mask tracking

このスクリプトは **コマンドライン引数なし**で動くようにしてあります。
起動前に下の「USER EDITABLE DEFAULTS」を編集してから、

    python sam2_gui.py

だけで実行してください（引数は不要）。

■ 操作
- 起動 →（VIDEO を設定していない場合）ファイル選択ダイアログが自動で開きます。
- 1フレーム目で左クリック = 正例点（選択）、右クリック = 負例点（除外）。
- クリック直後に半透明マスクで選択が可視化されます。
- ▶Play で動画全体にマスクを伝播・追跡します。

■ 注意
- SAM 2 の Video Predictor はフレーム画像ディレクトリを受け取るため、
  内部で動画から JPEG を一時ディレクトリへ抽出します。
- CUDA GPU 推奨。CPU でも動きますがとても遅いです。
- チェックポイントと YAML のパスはご自分の環境に合わせて編集してください。

License: Apache-2.0 (follows upstream SAM 2 demo license for example usage patterns)
"""

from __future__ import annotations
import os
import sys
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from contextlib import nullcontext

# PySide6 for GUI
from PySide6 import QtCore, QtGui, QtWidgets

# SAM 2
from sam2.build_sam import build_sam2_video_predictor


def resolve_for_hydra(path_like: str) -> str:
  """
  相対パスを Hydra 環境でも安全に絶対パスへ解決する。
  優先度:
    1) Hydra が居れば hydra.utils.to_absolute_path()
    2) このファイルの場所基準 (__file__)
    3) 現在の作業ディレクトリ
  """
  if not path_like:
    return path_like
  p = os.path.expanduser(path_like)
  if os.path.isabs(p):
    return p
  # Hydra があれば「元の cwd」を基準に解決
  try:
    from hydra.utils import to_absolute_path  # type: ignore
    return str(Path(to_absolute_path(p)))
  except Exception:
    pass
  # このスクリプトの場所を基準に解決
  try:
    here = Path(__file__).resolve().parent
    cand = here.joinpath(p)
    if cand.exists():
      return str(cand)
  except Exception:
    pass
  # 最後に現在の cwd 基準
  return str(Path(p).resolve())

# =============================================================
# USER EDITABLE DEFAULTS  ←←← ここを編集してから実行してください
# =============================================================
VIDEO: Optional[str] = None  # 例: r"C:\\path\\to\\video.mp4"。None のままだと起動時にファイル選択ダイアログが開きます
# CHECKPOINT: str = r"./checkpoints/sam2.1_hiera_large.pt"
# CONFIG_YAML: str = r"configs/sam2.1/sam2.1_hiera_l.yaml"
# CHECKPOINT: str = resolve_for_hydra("../../checkpoints/sam2.1_hiera_large.pt")
CHECKPOINT: str = resolve_for_hydra("../../checkpoints/sam2.1_hiera_tiny.pt")
# CONFIG_YAML: str = resolve_for_hydra("../../configs/sam2.1/sam2.1_hiera_l.yaml")
CONFIG_YAML: str = resolve_for_hydra("../../configs/sam2.1/sam2.1_hiera_t.yaml")

# 速度やメモリに関するオプション
VOS_OPTIMIZED: bool = False       # torch.compile 最適化（PyTorch 2.5.1+）
MAX_LONG_SIDE: int = 1280         # フレーム抽出時に長辺を縮小（<=0 で無効）
OFFLOAD_VIDEO: bool = False       # 動画フレームを CPU に退避（VRAM 節約）
OFFLOAD_STATE: bool = False       # 推論状態を CPU に退避（VRAM 節約）

# 出力動画（半透明マスク重畳）の保存
SAVE_OUT: Optional[str] = None    # 例: "overlay_out.mp4"（None で保存しない）
SAVE_FPS: float = 30.0

# 起動時のふるまい
AUTO_OPEN_DIALOG_ON_START: bool = True  # VIDEO が None のとき、起動直後にファイル選択を自動表示
# =============================================================


# -----------------------------
# Utility: video frame extraction
# -----------------------------

def extract_frames_to_dir(
    video_path: str,
    out_dir: str,
    max_long_side: Optional[int] = None,
    jpeg_quality: int = 92,
) -> Tuple[List[str], Tuple[int, int]]:
    """Extract video frames to JPEGs in out_dir and return list of frame paths and (H, W) of first frame.
    Optionally resizes each frame so that max(height, width) <= max_long_side.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    first_size = None
    frame_paths: List[str] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_long_side is not None and max_long_side > 0:
            h, w = frame.shape[:2]
            long_side = max(h, w)
            if long_side > max_long_side:
                scale = max_long_side / float(long_side)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if first_size is None:
            first_size = frame.shape[:2]  # (H, W)
        out_path = os.path.join(out_dir, f"{idx:05d}.jpg")
        ok = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError(f"Failed to write frame to {out_path}")
        frame_paths.append(out_path)
        idx += 1

    cap.release()
    if not frame_paths:
        raise RuntimeError("No frames extracted from video.")
    return frame_paths, (first_size[0], first_size[1])  # H, W


# -----------------------------
# Utility: image <-> QPixmap
# -----------------------------

def cv_bgr_to_qpixmap(bgr: np.ndarray) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)


def _normalize_mask_to_frame(mask_like: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    """Return a 2D boolean mask (H, W) aligned to frame_shape.
    Accepts masks shaped like (H,W), (1,H,W), (H,W,1), (O,H,W), etc., and resizes if needed.
    """
    m = np.asarray(mask_like)
    # Squeeze obvious singletons first
    while m.ndim > 2 and (m.shape[0] == 1 or m.shape[-1] == 1):
        if m.shape[0] == 1:
            m = np.squeeze(m, axis=0)
        elif m.shape[-1] == 1:
            m = np.squeeze(m, axis=-1)
        else:
            break
    # If still 3D, take the first channel (O,H,W) or (H,W,O)
    if m.ndim == 3:
        # Choose along the dim that looks like object channels
        if m.shape[0] < min(m.shape[1:]):
            m = m[0]
        else:
            m = m[..., 0]
    # Final squeeze just in case
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask_like.shape} -> {m.shape}")

    H, W = frame_shape[:2]
    if m.shape != (H, W):
        # Resize with nearest to keep mask crisp
        m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    if m.dtype != np.bool_:
        # Threshold >0 to bool
        m = m > 0.5
    return m


def overlay_mask_on_frame(
    frame_bgr: np.ndarray,
    mask_like: np.ndarray,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Return a new image with semi-transparent mask overlay.
    Accepts mask in various shapes (H,W), (1,H,W), (O,H,W), etc.
    """
    mask_bool = _normalize_mask_to_frame(mask_like, frame_bgr.shape)
    overlay = np.zeros_like(frame_bgr)
    overlay[mask_bool] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
    return blended


# --------------------------------------
# SAM2 Predictor wrapper for convenience
# --------------------------------------

@dataclass
class Sam2Config:
    checkpoint: str
    config_yaml: str
    vos_optimized: bool = False
    offload_video: bool = False
    offload_state: bool = False


class Sam2VideoHelper:
    def __init__(self, cfg: Sam2Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build predictor (optionally VOS optimized)
        self.predictor = build_sam2_video_predictor(
            cfg.config_yaml, cfg.checkpoint, vos_optimized=cfg.vos_optimized
        )
        self.inference_state = None
        self.amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        )

    def init_state(self, frames_dir: str):
        # According to upstream API, pass directory of JPEG frames via video_path
        with torch.inference_mode(), self.amp_ctx:
            self.inference_state = self.predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=self.cfg.offload_video,
                offload_state_to_cpu=self.cfg.offload_state,
            )
        return self.inference_state

    def click_points(
        self,
        frame_idx: int,
        obj_id: int,
        points_xy: np.ndarray,  # shape (N, 2), absolute pixel coords (x, y)
        labels: np.ndarray,     # shape (N,), 1=positive, 0=negative
    ) -> Tuple[int, List[int], np.ndarray]:
        """Send clicks to predictor and get immediate masks for the same frame.
        Returns (frame_idx, object_ids, masks_of_shape_[O, H, W]).
        """
        assert self.inference_state is not None
        if points_xy.ndim == 1:
            points_xy = points_xy[None, :]
        if labels.ndim == 0:
            labels = labels[None]
        with torch.inference_mode(), self.amp_ctx:
            # Prefer the modern API if available; otherwise fall back to legacy.
            if hasattr(self.predictor, "add_new_points_or_box"):
                fidx, obj_ids, masks = self.predictor.add_new_points_or_box(
                    self.inference_state,
                    frame_idx,
                    obj_id,
                    points=points_xy.astype(np.float32),
                    labels=labels.astype(np.int32),
                    clear_old_points=False,
                    normalize_coords=True,
                    box=None,
                )
            else:
                fidx, obj_ids, masks = self.predictor.add_new_points(
                    self.inference_state,
                    frame_idx,
                    obj_id,
                    points=points_xy.astype(np.float32),
                    labels=labels.astype(np.int32),
                    clear_old_points=False,
                    normalize_coords=True,
                )
        # Convert to numpy; keep original shape (O, H, W) if present
        if torch.is_tensor(masks):
            masks_np = masks.detach().to("cpu").numpy()
        else:
            masks_np = np.asarray(masks)
        # Normalize to boolean in later steps; keep raw here.
        return int(fidx), list(obj_ids), masks_np

    def track_all(self, start_frame_idx: Optional[int] = None):
        assert self.inference_state is not None
        with torch.inference_mode(), self.amp_ctx:
            yield from self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=start_frame_idx
            )

    def reset(self):
        if self.inference_state is not None:
            with torch.inference_mode():
                self.predictor.reset_state(self.inference_state)


# -----------------------------
# GUI widgets
# -----------------------------

class ClickableLabel(QtWidgets.QLabel):
    # Emits (x, y) in image pixel coordinates
    clicked = QtCore.Signal(int, int, object)  # (x, y, mouse_button)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self._display_img: Optional[np.ndarray] = None  # the currently shown frame (BGR)

    def set_frame(self, frame_bgr: np.ndarray):
        self._display_img = frame_bgr.copy()
        self._update_pixmap(frame_bgr)

    def _update_pixmap(self, bgr: np.ndarray):
        pix = cv_bgr_to_qpixmap(bgr)
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))

    def overlay_and_update(self, overlay_bgr: np.ndarray):
        self._update_pixmap(overlay_bgr)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self._display_img is None:
            return
        # Map from label coords to image coords
        pixmap = self.pixmap()
        if pixmap is None:
            return
        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = pixmap.width(), pixmap.height()
        # Compute top-left offset of the scaled pixmap within the label
        x_off = (label_w - pix_w) // 2
        y_off = (label_h - pix_h) // 2
        x = e.position().x() - x_off
        y = e.position().y() - y_off
        if x < 0 or y < 0 or x >= pix_w or y >= pix_h:
            return  # clicked outside the image area
        # Map back to original image coordinates
        img_h, img_w = self._display_img.shape[:2]
        img_x = int(x * img_w / pix_w)
        img_y = int(y * img_h / pix_h)
        button = e.button()
        self.clicked.emit(img_x, img_y, button)


class TrackerWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(int, object)  # frame_idx, masks_np (numpy array, usually [O,H,W])
    finished_ok = QtCore.Signal()

    def __init__(self, helper: Sam2VideoHelper, start_frame_idx: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.helper = helper
        self._stop = False
        self.start_frame_idx = start_frame_idx

    def run(self):
        try:
            for fidx, obj_ids, masks in self.helper.track_all(start_frame_idx=self.start_frame_idx):
                if self._stop:
                    break
                # Convert to numpy, but keep shape; normalize later in GUI
                if torch.is_tensor(masks):
                    masks_np = masks.detach().to("cpu").numpy()
                else:
                    masks_np = np.asarray(masks)
                self.frame_ready.emit(int(fidx), masks_np)
            self.finished_ok.emit()
        except Exception as e:
            print(f"Tracking error: {e}")
            # Let GUI continue (no re-raise)

    def stop(self):
        self._stop = True


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Video Tracker – single-click mask tracking")
        self.resize(1200, 800)

        # State (from USER DEFAULTS)
        self.args = type("Args", (), {
            "video": VIDEO,
            "checkpoint": CHECKPOINT,
            "config": CONFIG_YAML,
            "vos_optimized": VOS_OPTIMIZED,
            "max_long_side": MAX_LONG_SIDE,
            "save_out": SAVE_OUT,
            "fps": SAVE_FPS,
            "offload_video": OFFLOAD_VIDEO,
            "offload_state": OFFLOAD_STATE,
        })()

        self.helper: Optional[Sam2VideoHelper] = None
        self.frames: List[str] = []  # file paths
        self.frame0_bgr: Optional[np.ndarray] = None
        self.selected_obj_id: int = 1
        self.points_xy: List[Tuple[int, int]] = []
        self.labels: List[int] = []  # 1 for positive (LMB), 0 for negative (RMB)
        self.worker: Optional[TrackerWorker] = None
        self.temp_frames_dir: Optional[str] = None
        self.writer: Optional[cv2.VideoWriter] = None

        # UI
        self.image_label = ClickableLabel()
        self.image_label.setStyleSheet("background-color: #111;")
        self.setCentralWidget(self.image_label)

        open_btn = QtWidgets.QPushButton("Open Video…")
        play_btn = QtWidgets.QPushButton("▶ Play")
        stop_btn = QtWidgets.QPushButton("⏹ Stop")
        reset_btn = QtWidgets.QPushButton("🔁 Reset")
        clear_pts_btn = QtWidgets.QPushButton("⌫ Clear Points")

        for b in (play_btn, stop_btn, reset_btn, clear_pts_btn):
            b.setEnabled(False)

        info = QtWidgets.QLabel("Left-click: select (+). Right-click: refine (−). Then press Play.")
        info.setStyleSheet("color:#ddd; padding:4px;")

        # Layout
        toolbar = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(toolbar)
        hl.setContentsMargins(8, 8, 8, 8)
        hl.addWidget(open_btn)
        hl.addWidget(play_btn)
        hl.addWidget(stop_btn)
        hl.addWidget(reset_btn)
        hl.addWidget(clear_pts_btn)
        hl.addStretch(1)
        hl.addWidget(info)
        dock = QtWidgets.QDockWidget()
        dock.setTitleBarWidget(QtWidgets.QWidget())  # hide titlebar
        dock.setWidget(toolbar)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        # Signals
        open_btn.clicked.connect(self.on_open)
        play_btn.clicked.connect(self.on_play)
        stop_btn.clicked.connect(self.on_stop)
        reset_btn.clicked.connect(self.on_reset)
        clear_pts_btn.clicked.connect(self.on_clear_points)
        self.image_label.clicked.connect(self.on_click)

        self._btns = {
            "open": open_btn,
            "play": play_btn,
            "stop": stop_btn,
            "reset": reset_btn,
            "clear": clear_pts_btn,
        }

        # Auto-open
        if self.args.video:
            QtCore.QTimer.singleShot(0, lambda: self.load_video(self.args.video))
        elif AUTO_OPEN_DIALOG_ON_START:
            QtCore.QTimer.singleShot(0, self.on_open)

    # ----------
    # Callbacks
    # ----------

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a video", "", "Video files (*.mp4 *.mov *.avi *.mkv)")
        if path:
            self.load_video(path)

    def load_video(self, path: str):
        self.cleanup_temp()
        # Extract frames
        base = Path(path).stem
        out_dir = Path("._sam2_frames") / f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.parent.mkdir(exist_ok=True)
        frame_paths, (H, W) = extract_frames_to_dir(
            path, str(out_dir), max_long_side=self.args.max_long_side
        )
        self.temp_frames_dir = str(out_dir)
        self.frames = frame_paths

        # Show first frame
        frame0 = cv2.imread(self.frames[0])
        if frame0 is None:
            raise RuntimeError("Failed to read first frame.")
        self.frame0_bgr = frame0
        self.image_label.set_frame(frame0)

        # Prepare SAM2 helper/predictor
        # Basic runtime checks/warnings
        if not torch.cuda.is_available():
            QtWidgets.QMessageBox.warning(
                None,
                "CUDA not found",
                "CUDA GPU not detected. The app will run on CPU and be extremely slow.\n"
                "Install a CUDA-enabled PyTorch build for practical speed.",
            )

        for pth in (self.args.checkpoint, self.args.config):
            if not os.path.exists(pth):
                QtWidgets.QMessageBox.critical(None, "Missing file", f"Not found: {pth}")
                return

        self.helper = Sam2VideoHelper(
            Sam2Config(
                checkpoint=self.args.checkpoint,
                config_yaml=self.args.config,
                vos_optimized=self.args.vos_optimized,
                offload_video=self.args.offload_video,
                offload_state=self.args.offload_state,
            )
        )
        self.helper.init_state(self.temp_frames_dir)

        # Enable buttons
        self._btns["play"].setEnabled(True)
        self._btns["reset"].setEnabled(True)
        self._btns["clear"].setEnabled(True)
        self._btns["stop"].setEnabled(False)

        # Prepare writer if requested
        if self.args.save_out:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = self.frame0_bgr.shape[:2]
            self.writer = cv2.VideoWriter(self.args.save_out, fourcc, self.args.fps, (w, h))

        self.points_xy.clear()
        self.labels.clear()

    def on_click(self, x: int, y: int, mouse_button):
        if self.helper is None or self.frame0_bgr is None:
            return
        # Left button -> positive (1); Right button -> negative (0)
        if mouse_button == QtCore.Qt.MouseButton.LeftButton:
            label = 1
        elif mouse_button == QtCore.Qt.MouseButton.RightButton:
            label = 0
        else:
            return  # ignore other buttons (e.g., MiddleButton)
        self.points_xy.append((x, y))
        self.labels.append(label)

        # Send clicks to predictor on frame 0 and overlay immediate mask
        pts = np.array(self.points_xy, dtype=np.float32)
        lbs = np.array(self.labels, dtype=np.int32)
        fidx, obj_ids, masks_np = self.helper.click_points(frame_idx=0, obj_id=self.selected_obj_id, points_xy=pts, labels=lbs)

        # Choose the mask for the selected object id if available
        mask_like = None
        if masks_np.ndim >= 3:
            try:
                # obj_ids is a list like [1, 2, ...]; pick the index of selected_obj_id
                obj_pos = list(obj_ids).index(self.selected_obj_id)
            except ValueError:
                obj_pos = 0
            mask_like = masks_np[obj_pos]
        else:
            mask_like = masks_np

        over = overlay_mask_on_frame(self.frame0_bgr, mask_like, color_bgr=(0, 255, 0), alpha=0.5)
        # Draw click markers
        over2 = over.copy()
        for (px, py), lb in zip(self.points_xy, self.labels):
            color = (0, 255, 0) if lb == 1 else (0, 0, 255)
            cv2.circle(over2, (px, py), 6, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(over2, (px, py), 10, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        self.image_label.overlay_and_update(over2)

    def on_play(self):
        if self.helper is None:
            return
        # Disable play; enable stop
        self._btns["play"].setEnabled(False)
        self._btns["stop"].setEnabled(True)
        self._btns["open"].setEnabled(False)
        self._btns["clear"].setEnabled(False)

        # Start tracker worker
        self.worker = TrackerWorker(self.helper, start_frame_idx=0)
        self.worker.frame_ready.connect(self.on_tracked_frame)
        self.worker.finished_ok.connect(self.on_track_done)
        self.worker.start()

    @QtCore.Slot(int, object)
    def on_tracked_frame(self, frame_idx: int, masks_np):
        # Load frame and overlay the selected object's mask
        frame = cv2.imread(self.frames[frame_idx])
        if frame is None:
            return
        if masks_np.ndim >= 3:
            mask_like = masks_np[0]  # best-effort: first object channel
        else:
            mask_like = masks_np
        over = overlay_mask_on_frame(frame, mask_like, (0, 255, 0), alpha=0.5)
        self.image_label.overlay_and_update(over)
        if self.writer is not None:
            self.writer.write(over)

    def on_track_done(self):
        self._btns["play"].setEnabled(True)
        self._btns["stop"].setEnabled(False)
        self._btns["open"].setEnabled(True)
        self._btns["clear"].setEnabled(True)
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def on_stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self._btns["play"].setEnabled(True)
        self._btns["stop"].setEnabled(False)
        self._btns["open"].setEnabled(True)

    def on_reset(self):
        # Stop worker
        self.on_stop()
        # Reset predictor state and UI overlay back to first frame
        if self.helper is not None:
            self.helper.reset()
        if self.frame0_bgr is not None:
            self.image_label.set_frame(self.frame0_bgr)
        self.points_xy.clear()
        self.labels.clear()

    def on_clear_points(self):
        self.points_xy.clear()
        self.labels.clear()
        if self.frame0_bgr is not None:
            self.image_label.set_frame(self.frame0_bgr)

    def cleanup_temp(self):
        # Stop worker
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None
        # Close writer
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        # Remove previous temp frames dir
        if self.temp_frames_dir and os.path.isdir(self.temp_frames_dir):
            try:
                shutil.rmtree(self.temp_frames_dir)
            except Exception:
                pass
        self.temp_frames_dir = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cleanup_temp()
        return super().closeEvent(event)


# -----------------------------
# Main
# -----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    ret = app.exec()
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
