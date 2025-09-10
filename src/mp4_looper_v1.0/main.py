# ファイル名: mp4_looper_safe_v1.py
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from moviepy import VideoFileClip, concatenate_videoclips

def main():
  root = tk.Tk()
  root.withdraw()

  filepath = filedialog.askopenfilename(
    title="MP4ファイルを選択してください",
    filetypes=[("MP4 files", "*.mp4")]
  )
  if not filepath:
    messagebox.showerror("エラー", "ファイルが選択されませんでした。")
    return

  loop_count = simpledialog.askinteger("ループ回数", "動画を何回繰り返しますか？", minvalue=1)
  if not loop_count:
    return

  clips = []
  try:
    # ① with で元クリップを開く（自動クローズ）
    with VideoFileClip(filepath) as base:
      # ② 必要回数だけサブクリップを作る（メモリ節約のため元の参照を保持）
      for _ in range(loop_count):
        clips.append(base.copy())

      # ③ 連結（異なる解像度/コーデックの可能性がある場合は method="compose"）
      final_clip = concatenate_videoclips(clips, method="chain")

      # ④ 出力（Windows 互換の一般的設定）
      out_name = "looped_output_001.mp4"
      final_clip.write_videofile(
        out_name,
        codec="libx264",
        audio_codec="aac"
      )
  finally:
    # ⑤ 念のため後片付け
    for c in clips:
      try:
        c.close()
      except Exception:
        pass
    try:
      final_clip.close()
    except Exception:
      pass

  messagebox.showinfo("完了", "保存しました: looped_output_001.mp4")

if __name__ == "__main__":
  main()
