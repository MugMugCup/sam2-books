# filename: road_seg_proto_sam2_gui.py
# -*- coding: utf-8 -*-
# 要: Python 3.10+, PySide6, numpy, pillow, opencv-python, torch>=2.5.1, sam2 (facebookresearch/sam2)
# 保存機能は実装していません（RAM内のみ）。SAM2の仕様上、連番JPEGを一時作成しますが終了時に削除します。

import sys
import os
import shutil
import tempfile
import threading
import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ExifTags

import torch

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, QSize, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QBrush, QAction
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QListWidget,
    QListWidgetItem,
    QGraphicsView,
    QGraphicsScene,
    QStyle,
    QSpinBox,
    QFrame,
    QMessageBox,
    QSplitter,
)

# SAM2: 安定API（2024/12 README準拠）
# - Video: SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
# - state = predictor.init_state(video_path=<dir>)
# - クリック追加入力: predictor.add_new_points_or_box(inference_state=state, frame_idx=i, obj_id=j, points=..., labels=...)
# - 伝播: for frame_idx, obj_ids, out in predictor.propagate_in_video(state): ...
# 参考: https://github.com/facebookresearch/sam2 （READMEの「Video prediction」）,
#      add_new_points_or_boxのポイントラベル: 1=Positive, 0=Negative（例: Roboflowブログ）
from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore


# ========== ユーティリティ ==========


def read_exif_datetime(path: str) -> Optional[str]:
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None
        # DateTimeOriginal tag id を取得
        tag_map = {ExifTags.TAGS.get(k, k): k for k in exif.keys()}
        dto_tag = tag_map.get("DateTimeOriginal", None)
        if dto_tag is None:
            return None
        return exif.get(dto_tag)
    except Exception:
        return None


def file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def sort_images_windows_chrono(paths: List[str]) -> List[str]:
    # EXIF(DateTimeOriginal) -> mtime -> ナチュラル（数字優先）
    def natural_key(p: str):
        base = os.path.splitext(os.path.basename(p))[0]
        nums = []
        cur = ""
        for ch in base:
            if ch.isdigit():
                cur += ch
            else:
                if cur:
                    nums.append(int(cur))
                    cur = ""
        if cur:
            nums.append(int(cur))
        return nums

    exif_pairs = []
    no_exif = []
    for p in paths:
        dto = read_exif_datetime(p)
        if dto:
            exif_pairs.append((p, dto))
        else:
            no_exif.append(p)

    exif_pairs.sort(key=lambda x: x[1])
    no_exif.sort(key=lambda p: (file_mtime(p), natural_key(p)))

    return [p for p, _ in exif_pairs] + no_exif


def to_qimage_rgb(img_bgr: np.ndarray) -> QImage:
    if img_bgr.ndim == 2:
        h, w = img_bgr.shape
        qimg = QImage(img_bgr.data, w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return qimg.copy()


def color_from_index(i: int) -> QColor:
    # 10色程度を循環
    palette = [
        QColor(80, 180, 60, 150),  # greenish
        QColor(60, 120, 220, 150),  # blue
        QColor(220, 80, 60, 150),  # red
        QColor(200, 160, 20, 150),  # yellow
        QColor(160, 60, 200, 150),  # purple
        QColor(20, 180, 180, 150),  # cyan
        QColor(240, 120, 40, 150),  # orange
        QColor(120, 200, 80, 150),  # lime
        QColor(200, 80, 120, 150),  # pink
        QColor(100, 100, 100, 150),  # gray
    ]
    return palette[i % len(palette)]


# ========== データモデル ==========


@dataclass
class ObjectTrack:
    obj_id: int
    name: str
    color: QColor
    visible: bool = True
    # フレームごとの存在フラグ（マスクが十分な面積）と2Dマスクキャッシュ(必要時のみ)
    exists: List[bool] = field(default_factory=list)
    masks: Dict[int, np.ndarray] = field(default_factory=dict)


# ========== SAM2バックエンド ==========


class Sam2Engine(QObject):
    masks_updated = Signal(int)  # frame_idx
    state_ready = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor: Optional[SAM2VideoPredictor] = None
        self.inference_state = None
        self.num_frames = 0
        self.video_dir = None
        self.temp_root = None
        self._lock = threading.Lock()
        self._autocast_dtype = torch.bfloat16  # 失敗時はfp16にフォールバック
        self._stop_propagation = threading.Event()

    def load_model(self):
        if not torch.cuda.is_available():
            QMessageBox.warning(
                None,
                "CUDA未検出",
                "CUDA対応GPUが見つかりませんでした。CPU実行は非常に遅くなります。",
            )
        try:
            self.predictor = SAM2VideoPredictor.from_pretrained(
                "facebook/sam2-hiera-large"
            )  # 公式のHF経由
        except Exception as e:
            QMessageBox.critical(
                None, "SAM2ロード失敗", f"SAM2の読み込みに失敗しました: {e}"
            )
            raise

        # autocast dtype確認（bfloat16非対応GPUならfp16へ）
        try:
            with torch.autocast("cuda", dtype=self._autocast_dtype):
                pass
        except Exception:
            self._autocast_dtype = torch.float16

    def prepare_video_from_images(self, image_paths_sorted: List[str]):
        # SAM2のinit_stateは「連番JPEGのディレクトリ」を受け取る実装が一般的
        # → 一時作業フォルダに 000000.jpg 形式で再エンコード（終了時に削除）
        self.cleanup_temp()
        self.temp_root = tempfile.mkdtemp(prefix="sam2_work_")
        video_dir = os.path.join(self.temp_root, "frames")
        os.makedirs(video_dir, exist_ok=True)

        for i, p in enumerate(image_paths_sorted):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            # SAM2はJPEG推奨、長辺スケールの自動抑制はここでは行わず、元解像度で保存
            cv2.imwrite(
                os.path.join(video_dir, f"{i:05d}.jpg"),
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )

        self.video_dir = video_dir

        assert self.predictor is not None
        try:
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=self._autocast_dtype
            ):
                # メモリ節約のため videoをCPUへオフロード
                self.inference_state = self.predictor.init_state(
                    video_path=self.video_dir
                )
            self.num_frames = self.inference_state.get("num_frames", 0)
            self.state_ready.emit()
        except Exception as e:
            QMessageBox.critical(None, "SAM2初期化失敗", f"init_stateで失敗: {e}")
            raise

    def add_click(
        self, frame_idx: int, obj_id: int, x: float, y: float, positive: bool
    ) -> Optional[np.ndarray]:
        if self.predictor is None or self.inference_state is None:
            return None
        try:
            labels = np.array([1 if positive else 0], np.int32)
            points = np.array([[x, y]], dtype=np.float32)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=self._autocast_dtype
            ):
                # APIの戻りは版で差異があることがあるため両対応
                out = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
            # 戻り値パース
            mask = self._extract_mask_from_add(out)
            return mask
        except Exception as e:
            print("add_click error:", e)
            return None

    def _extract_mask_from_add(self, out) -> Optional[np.ndarray]:
        # 例1: _, out_obj_ids, out_mask_logits
        # 例2: frame_idx, object_ids, masks(bool)
        try:
            if isinstance(out, tuple) and len(out) == 3:
                last = out[2]
                if isinstance(last, torch.Tensor):
                    # ロジット -> 2値
                    mask = (last[0] > 0.0).detach().cpu().numpy().astype(np.uint8)
                    return mask
                else:
                    # すでにbool numpyの可能性
                    arr = np.array(last)
                    if arr.dtype == bool or arr.dtype == np.bool_:
                        return arr.astype(np.uint8)
                    # 形状が (1, H, W) のfloatなど
                    if arr.ndim == 3:
                        return (arr[0] > 0).astype(np.uint8)
            # 不明な形式
            return None
        except Exception:
            return None

    def start_propagation_forward(self, callback_per_frame):
        # 伝播スレッド（現在の状態から最終フレームに向けて）
        if self.predictor is None or self.inference_state is None:
            return
        self._stop_propagation.clear()

        def run():
            try:
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=self._autocast_dtype
                ):
                    for out in self.predictor.propagate_in_video(self.inference_state):
                        if self._stop_propagation.is_set():
                            break
                        # out は (frame_idx, object_ids, out_masks/ logits)
                        frame_idx, obj_ids, last = self._parse_propagate_out(out)
                        if frame_idx is None:
                            continue
                        callback_per_frame(frame_idx, obj_ids, last)
                        self.masks_updated.emit(frame_idx)
            except Exception as e:
                print("propagation error:", e)

        th = threading.Thread(target=run, daemon=True)
        th.start()

    def stop_propagation(self):
        self._stop_propagation.set()

    def _parse_propagate_out(
        self, out
    ) -> Tuple[Optional[int], List[int], Dict[int, np.ndarray]]:
        try:
            if isinstance(out, tuple) and len(out) == 3:
                frame_idx = int(out[0])
                obj_ids = [int(x) for x in out[1]]
                last = out[2]
                masks_map: Dict[int, np.ndarray] = {}
                if isinstance(last, dict):
                    # 新しめの実装では obj_id -> mask(logits or bool) のdict
                    for k, v in last.items():
                        if isinstance(v, torch.Tensor):
                            m = (v > 0.0).detach().cpu().numpy().astype(np.uint8)
                        else:
                            a = np.array(v)
                            m = (
                                (a > 0).astype(np.uint8)
                                if a.dtype != bool
                                else a.astype(np.uint8)
                            )
                        masks_map[int(k)] = m
                elif isinstance(last, torch.Tensor):
                    # 単一オブジェクト想定
                    m = (last > 0.0).detach().cpu().numpy().astype(np.uint8)
                    if len(obj_ids) == 1:
                        masks_map[obj_ids[0]] = m
                return frame_idx, obj_ids, masks_map
        except Exception:
            pass
        return None, [], {}

    def cleanup_temp(self):
        try:
            if self.temp_root and os.path.isdir(self.temp_root):
                shutil.rmtree(self.temp_root, ignore_errors=True)
        except Exception:
            pass
        self.temp_root = None
        self.video_dir = None


# ========== ビュー（画像＋オーバーレイ） ==========


class ImageView(QGraphicsView):
    clicked = Signal(float, float, int)  # x,y, button(1:left 2:right)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(QPixmap())
        self._overlay = None
        self._scale = 1.0
        self._pan = False
        self._last_pos = None

    def set_image(self, qimg: QImage):
        self._scene.setSceneRect(QRectF(0, 0, qimg.width(), qimg.height()))
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def draw_overlay(self, overlay_img: Optional[QImage]):
        if self._overlay:
            self._scene.removeItem(self._overlay)
            self._overlay = None
        if overlay_img is None:
            return
        pix = QPixmap.fromImage(overlay_img)
        self._overlay = self._scene.addPixmap(pix)
        self._overlay.setZValue(10)

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            pos = self.mapToScene(event.pos())
            self.clicked.emit(
                pos.x(), pos.y(), 1 if event.button() == Qt.LeftButton else 2
            )
        elif event.button() == Qt.MiddleButton:
            self._pan = True
            self._last_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan = False
            self._last_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Ctrl+ホイールでズーム
        if event.modifiers() & Qt.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.15 if angle > 0 else 1.0 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)


# ========== タイムライン ==========


class TimelineWidget(QWidget):
    frame_changed = Signal(int)
    row_height = 18
    margin = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.objects: List[ObjectTrack] = []
        self.num_frames = 0
        self.current_frame = 0
        self.setMinimumHeight(120)

    def set_objects(self, objs: List[ObjectTrack], num_frames: int):
        self.objects = objs
        self.num_frames = num_frames
        self.update()

    def set_current(self, f: int):
        self.current_frame = max(0, min(f, self.num_frames - 1))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        if self.num_frames <= 0 or not self.objects:
            return
        # スケール
        left = self.margin * 2 + 120
        usable_w = max(1, w - left - self.margin)
        px_per_frame = usable_w / max(1, self.num_frames - 1)

        # 行ラベル
        painter.setPen(QPen(QColor(220, 220, 220)))
        y = self.margin
        for obj in self.objects:
            painter.drawText(self.margin, y + self.row_height - 4, obj.name)
            # 帯
            if obj.exists:
                # ランレングス化して描画
                start = None
                for i, flag in enumerate(obj.exists):
                    if flag and start is None:
                        start = i
                    if (not flag or i == len(obj.exists) - 1) and start is not None:
                        end_i = i if not flag else i
                        x0 = int(left + px_per_frame * start)
                        x1 = int(left + px_per_frame * end_i)
                        band_color = QColor(obj.color)
                        band_color.setAlpha(180 if obj.visible else 80)
                        painter.fillRect(
                            x0,
                            y + 3,
                            max(2, x1 - x0 + 2),
                            self.row_height - 6,
                            band_color,
                        )
                        start = None
            y += self.row_height

        # 現在フレームの赤線
        x = int(left + px_per_frame * self.current_frame)
        painter.setPen(QPen(QColor(220, 80, 80), 2))
        painter.drawLine(x, self.margin, x, h - self.margin)

    def mousePressEvent(self, event):
        if self.num_frames <= 0 or not self.objects:
            return
        w = self.width()
        left = self.margin * 2 + 120
        usable_w = max(1, w - left - self.margin)
        x = max(0, min(event.position().x(), w)) - left
        if x < 0:
            return
        f = int(round((x / usable_w) * (self.num_frames - 1)))
        self.frame_changed.emit(f)


# ========== メインウィンドウ ==========


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Masking Prototype (SAM2 Large)")
        self.resize(1280, 800)

        # 状態
        self.image_paths: List[str] = []
        self.frames_bgr: List[np.ndarray] = []
        self.current_frame: int = 0
        self.playing: bool = False
        self.fps: int = 10
        self.exists_thresh_ratio = 0.002  # 総画素の0.2%既定
        self.exists_thresh_minpx = 2000

        # SAM2
        self.engine = Sam2Engine()
        self.engine.masks_updated.connect(self.on_masks_updated)
        self.engine.state_ready.connect(self.on_state_ready)
        self.engine.load_model()

        # UI
        self.viewer = ImageView()
        self.timeline = TimelineWidget()
        self.timeline.frame_changed.connect(self.on_timeline_seek)

        self.objects: List[ObjectTrack] = []
        self.active_obj_index = -1

        # 左パネル（オブジェクトリスト）
        self.obj_list = QListWidget()
        self.obj_list.currentRowChanged.connect(self.on_obj_switched)
        self.btn_add_obj = QPushButton("＋オブジェクト")
        self.btn_add_obj.clicked.connect(self.on_add_object)

        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("オブジェクト"))
        left_panel.addWidget(self.obj_list, 1)
        left_panel.addWidget(self.btn_add_obj)
        left_w = QWidget()
        left_w.setLayout(left_panel)

        # 再生コントロール
        self.btn_open = QPushButton("フォルダを開く")
        self.btn_play = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "")
        self.btn_prev = QPushButton(
            self.style().standardIcon(QStyle.SP_MediaSkipBackward), ""
        )
        self.btn_next = QPushButton(
            self.style().standardIcon(QStyle.SP_MediaSkipForward), ""
        )
        self.sld_seek = QSlider(Qt.Horizontal)
        self.sld_seek.setMinimum(0)
        self.sld_seek.valueChanged.connect(self.on_seek_slider)
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 60)
        self.spn_fps.setValue(self.fps)
        self.spn_fps.valueChanged.connect(self.on_fps_change)

        self.btn_open.clicked.connect(self.on_open_folder)
        self.btn_play.clicked.connect(self.on_play_toggle)
        self.btn_prev.clicked.connect(self.on_prev_frame)
        self.btn_next.clicked.connect(self.on_next_frame)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_open)
        top_bar.addWidget(self.btn_prev)
        top_bar.addWidget(self.btn_play)
        top_bar.addWidget(self.btn_next)
        top_bar.addWidget(QLabel("FPS"))
        top_bar.addWidget(self.spn_fps)
        top_bar.addWidget(QLabel("シーク"))
        top_bar.addWidget(self.sld_seek, 1)

        # レイアウト
        splitter = QSplitter()
        left_container = QWidget()
        left_container.setLayout(left_panel)
        right_panel = QVBoxLayout()
        right_panel.addLayout(top_bar)
        right_panel.addWidget(self.viewer, 1)
        right_panel.addWidget(self.timeline, 0)
        right_container = QWidget()
        right_container.setLayout(right_panel)

        splitter.addWidget(left_w)
        splitter.addWidget(right_container)
        splitter.setSizes([250, 1000])

        root = QVBoxLayout()
        root.addWidget(splitter, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # タイマー
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        # クリックイベント
        self.viewer.clicked.connect(self.on_view_click)

        # ステータス
        self.statusBar().showMessage(
            "準備完了（保存機能なし・一時ファイルは終了時に削除）"
        )

    # --------- フォルダ読み込み ---------

    def on_open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if not d:
            return
        cand = []
        for name in os.listdir(d):
            ext = os.path.splitext(name)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                cand.append(os.path.join(d, name))
        if not cand:
            QMessageBox.warning(self, "画像なし", "jpg/png画像が見つかりませんでした。")
            return

        self.image_paths = sort_images_windows_chrono(cand)
        # 表示用に最初のフレームだけデコード
        first = cv2.imread(self.image_paths[0], cv2.IMREAD_COLOR)
        if first is None:
            QMessageBox.critical(
                self, "読込失敗", "最初の画像の読み込みに失敗しました。"
            )
            return
        self.frames_bgr = [first]  # 必要になったら随時読む（I/O最小化）
        self.current_frame = 0
        self.sld_seek.blockSignals(True)
        self.sld_seek.setMaximum(len(self.image_paths) - 1)
        self.sld_seek.setValue(0)
        self.sld_seek.blockSignals(False)

        # SAM2初期化（連番JPEGを一時生成）
        self.engine.prepare_video_from_images(self.image_paths)

        qimg = to_qimage_rgb(first)
        self.viewer.set_image(qimg)
        self.update_overlay()
        self.refresh_timeline()
        self.statusBar().showMessage(f"読み込み完了: {len(self.image_paths)} 枚")

    def on_state_ready(self):
        # オブジェクトは初期空
        self.objects = []
        self.obj_list.clear()
        self.active_obj_index = -1
        self.refresh_timeline()

    # --------- 再生コントロール ---------

    def on_play_toggle(self):
        if not self.image_paths:
            return
        self.playing = not self.playing
        self.btn_play.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaPause if self.playing else QStyle.SP_MediaPlay
            )
        )
        if self.playing:
            self.engine.start_propagation_forward(self.on_propagated)
            self.timer.start(int(1000 / max(1, self.fps)))
        else:
            self.engine.stop_propagation()
            self.timer.stop()

    def on_prev_frame(self):
        if not self.image_paths:
            return
        self.current_frame = max(0, self.current_frame - 1)
        self.show_frame(self.current_frame)

    def on_next_frame(self):
        if not self.image_paths:
            return
        self.current_frame = min(len(self.image_paths) - 1, self.current_frame + 1)
        self.show_frame(self.current_frame)

    def on_seek_slider(self, v: int):
        if not self.image_paths:
            return
        self.current_frame = v
        self.show_frame(self.current_frame)

    def on_fps_change(self, v: int):
        self.fps = v
        if self.playing:
            self.timer.start(int(1000 / max(1, self.fps)))

    def on_tick(self):
        if not self.image_paths:
            return
        self.current_frame += 1
        if self.current_frame >= len(self.image_paths):
            self.current_frame = len(self.image_paths) - 1
            self.on_play_toggle()  # 停止
            return
        self.sld_seek.blockSignals(True)
        self.sld_seek.setValue(self.current_frame)
        self.sld_seek.blockSignals(False)
        self.show_frame(self.current_frame)

    def on_timeline_seek(self, f: int):
        if not self.image_paths:
            return
        self.current_frame = f
        self.sld_seek.blockSignals(True)
        self.sld_seek.setValue(f)
        self.sld_seek.blockSignals(False)
        self.show_frame(f)

    # --------- 表示 ---------

    def show_frame(self, idx: int):
        # 必要ならデコード
        if idx >= len(self.frames_bgr):
            img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
            if img is not None:
                self.frames_bgr.append(img)
            else:
                # 失敗時は前のものを再利用
                self.frames_bgr.append(self.frames_bgr[-1])

        qimg = to_qimage_rgb(self.frames_bgr[idx])
        self.viewer.set_image(qimg)
        self.update_overlay()
        self.timeline.set_current(idx)

    def update_overlay(self):
        if not self.objects or not self.frames_bgr:
            self.viewer.draw_overlay(None)
            return
        h, w, _ = self.frames_bgr[self.current_frame].shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        for obj in self.objects:
            if not obj.visible:
                continue
            m = obj.masks.get(self.current_frame, None)
            if m is None:
                continue
            # m は (H,W) 0/1
            color = obj.color
            rgba = np.array(
                [color.red(), color.green(), color.blue(), color.alpha()],
                dtype=np.uint8,
            )
            mask = (m > 0).astype(np.uint8)
            overlay[mask == 1] = rgba
        qimg = QImage(overlay.data, w, h, w * 4, QImage.Format_RGBA8888)
        self.viewer.draw_overlay(qimg.copy())

    def refresh_timeline(self):
        # オブジェクトごとにexists長さをnum_framesに合わせる
        nf = len(self.image_paths)
        for obj in self.objects:
            if len(obj.exists) != nf:
                obj.exists = [False] * nf
        self.timeline.set_objects(self.objects, nf)
        self.timeline.set_current(self.current_frame)

    # --------- クリック編集 ---------

    def on_view_click(self, x: float, y: float, button: int):
        if not self.image_paths or self.engine.inference_state is None:
            return
        if self.active_obj_index < 0:
            QMessageBox.information(
                self,
                "オブジェクト未選択",
                "左パネルの「＋オブジェクト」で追加し、選択してください。",
            )
            return
        obj = self.objects[self.active_obj_index]
        pos = (x, y)
        positive = button == 1
        mask = self.engine.add_click(
            self.current_frame, obj.obj_id, pos[0], pos[1], positive=positive
        )
        if mask is not None:
            obj.masks[self.current_frame] = mask
            obj.exists[self.current_frame] = self._mask_exists(
                mask, self.frames_bgr[self.current_frame].shape[:2]
            )
            self.update_overlay()
            self.refresh_timeline()
            # クリック後に前方伝播（エンジンのスレッドで進行、到着次第 on_propagated が更新）

    def _mask_exists(self, mask: np.ndarray, hw: Tuple[int, int]) -> bool:
        h, w = hw
        area = int(mask.sum())
        thresh = max(int(h * w * self.exists_thresh_ratio), self.exists_thresh_minpx)
        return area >= thresh

    def on_propagated(
        self, frame_idx: int, obj_ids: List[int], masks_map: Dict[int, np.ndarray]
    ):
        # 伝播から受け取った結果を反映
        if not self.objects:
            return
        id_to_obj = {o.obj_id: o for o in self.objects}
        for oid in obj_ids:
            if oid in id_to_obj and oid in masks_map:
                m = masks_map[oid]
                id_to_obj[oid].masks[frame_idx] = m
                # デコード済み画像が必要なら読む
                if frame_idx >= len(self.frames_bgr):
                    img = cv2.imread(self.image_paths[frame_idx], cv2.IMREAD_COLOR)
                    if img is not None:
                        self.frames_bgr.append(img)
                base_shape = self.frames_bgr[
                    min(frame_idx, len(self.frames_bgr) - 1)
                ].shape[:2]
                id_to_obj[oid].exists[frame_idx] = self._mask_exists(m, base_shape)
        if frame_idx == self.current_frame:
            self.update_overlay()
        self.refresh_timeline()

    def on_masks_updated(self, f: int):
        # 将来のフック（今は不要）
        pass

    # --------- オブジェクト操作 ---------

    def on_add_object(self):
        new_id = 1
        used = {o.obj_id for o in self.objects}
        while new_id in used:
            new_id += 1
        name = f"obj{new_id:02d}"
        color = color_from_index(len(self.objects))
        nf = len(self.image_paths)
        track = ObjectTrack(obj_id=new_id, name=name, color=color, exists=[False] * nf)
        self.objects.append(track)
        item = QListWidgetItem(name)
        # 色プレビュー
        pix = QPixmap(16, 16)
        pix.fill(color)
        item.setIcon(pix)
        self.obj_list.addItem(item)
        self.obj_list.setCurrentRow(len(self.objects) - 1)
        self.refresh_timeline()

    def on_obj_switched(self, row: int):
        if row < 0 or row >= len(self.objects):
            self.active_obj_index = -1
            return
        self.active_obj_index = row
        self.statusBar().showMessage(
            f"選択中: {self.objects[row].name} (ID={self.objects[row].obj_id})"
        )

    # --------- 終了処理 ---------

    def closeEvent(self, e):
        try:
            self.engine.stop_propagation()
            self.engine.cleanup_temp()
        except Exception:
            pass
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
