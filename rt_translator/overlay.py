"""透過オーバーレイ字幕ウィンドウ (tkinter)。

- 枠なし・常に最前面・半透明の黒帯に日本語字幕を表示
- 確定訳は白、翻訳中の暫定訳はグレー
- 左ドラッグで移動、右クリックメニューでフォントサイズ変更・終了
"""
import queue
import threading
import tkinter as tk
from collections import deque

from .config import Config

BG = "#101010"


class SubtitleOverlay:
    def __init__(self, cfg: Config, ui_queue: queue.Queue, stop_event: threading.Event):
        self.cfg = cfg
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.final_texts = deque(maxlen=cfg.final_lines)

        self.root = tk.Tk()
        self.root.title("RT Translator")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", cfg.overlay_alpha)
        self.root.configure(bg=BG)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = int(screen_w * cfg.overlay_width_ratio)
        x = (screen_w - width) // 2
        # 帯は下端を固定し、行数が増えたら上方向に伸ばす(タスクバーに被らない)
        self._anchor_bottom = int(screen_h * 0.92)
        self.root.geometry(f"{width}x10+{x}+{self._anchor_bottom - 10}")
        self._wraplength = width - 40

        self.font_size = cfg.font_size
        self.label_source = tk.Label(
            self.root, text="", fg="#9fbfdf", bg=BG, justify="left", anchor="w",
            wraplength=self._wraplength)
        if cfg.show_source:
            self.label_source.pack(fill="x", padx=20, pady=(8, 0))
        self.label_final = tk.Label(
            self.root, text="", fg="white", bg=BG, justify="left", anchor="w",
            wraplength=self._wraplength)
        self.label_final.pack(fill="x", padx=20, pady=(8, 0))
        self.label_partial = tk.Label(
            self.root, text="", fg="#a0a0a0", bg=BG, justify="left", anchor="w",
            wraplength=self._wraplength)
        self.label_partial.pack(fill="x", padx=20, pady=(0, 8))
        self._apply_font()

        for widget in (self.root, self.label_source, self.label_final, self.label_partial):
            widget.bind("<Button-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
            widget.bind("<Button-3>", self._show_menu)

        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="フォント拡大", command=lambda: self._resize_font(+2))
        self.menu.add_command(label="フォント縮小", command=lambda: self._resize_font(-2))
        self.menu.add_command(label="字幕クリア", command=self._clear)
        self.menu.add_separator()
        self.menu.add_command(label="終了", command=self.close)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._poll)

    def _apply_font(self):
        font = (self.cfg.font_family, self.font_size)
        small = (self.cfg.font_family, max(10, int(self.font_size * 0.6)))
        self.label_final.configure(font=font)
        self.label_partial.configure(font=font)
        self.label_source.configure(font=small)

    def _resize_font(self, delta: int):
        self.font_size = max(10, self.font_size + delta)
        self._apply_font()

    def _clear(self):
        self.final_texts.clear()
        self._render(partial="", source="")

    def _drag_start(self, event):
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _drag_move(self, event):
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{event.x_root - self._drag_x}+{y}")
        self._anchor_bottom = y + self.root.winfo_height()

    def _show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    def _poll(self):
        if self.stop_event.is_set():
            self.root.destroy()
            return
        updated = False
        partial = None
        source = None
        while True:
            try:
                item = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            updated = True
            if item["kind"] == "final":
                self.final_texts.append(item["ja"])
                partial = ""      # 確定が来たら暫定行は消す
                source = item["en"]
            else:
                partial = item["ja"]
                source = item["en"]
        if updated:
            self._render(partial=partial, source=source)
        self.root.after(50, self._poll)

    def _render(self, partial=None, source=None):
        self.label_final.configure(text="\n".join(self.final_texts))
        if partial is not None:
            self.label_partial.configure(text=partial)
        if source is not None and self.cfg.show_source:
            self.label_source.configure(text=source)
        self.root.update_idletasks()
        height = self.root.winfo_reqheight()
        self.root.geometry(
            f"{self.root.winfo_width()}x{height}"
            f"+{self.root.winfo_x()}+{self._anchor_bottom - height}")

    def close(self):
        self.stop_event.set()

    def run(self):
        self.root.mainloop()
