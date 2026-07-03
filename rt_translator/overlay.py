"""透過オーバーレイ字幕ウィンドウ (tkinter)。

- 枠なし・常に最前面・半透明の黒帯に日本語字幕を表示
- 確定訳は白、翻訳中の暫定訳はグレー
- 左ドラッグで移動、右クリックメニューでフォントサイズ変更・終了
- マウスホイールで過去の字幕に遡れる。右端のシークバーでも操作可能
  (遡り中は新しい字幕が来ても表示位置を固定し、最下部に戻ると追従再開)
"""
import queue
import threading
import tkinter as tk
from collections import deque

from .config import Config

BG = "#101010"
MAX_HISTORY = 200
SCROLLBAR_W = 12
THUMB_MIN_H = 24


class SubtitleOverlay:
    def __init__(self, cfg: Config, ui_queue: queue.Queue, stop_event: threading.Event):
        self.cfg = cfg
        self.ui_queue = ui_queue
        self.stop_event = stop_event

        self.history: deque[str] = deque(maxlen=MAX_HISTORY)  # 確定訳の全履歴
        self.view_offset = 0        # 0=最新に追従 / n=末尾から n 件遡った位置
        self._live_partial = ""     # 遡り中も裏で保持し、最新に戻ったら表示
        self._live_source = ""

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
        self._wraplength = width - 40 - SCROLLBAR_W

        content = tk.Frame(self.root, bg=BG)
        content.pack(side="left", fill="both", expand=True)
        self.scrollbar = tk.Canvas(
            self.root, width=SCROLLBAR_W, bg="#282828", highlightthickness=0)
        self.scrollbar.pack(side="right", fill="y")

        self.font_size = cfg.font_size
        self.label_source = tk.Label(
            content, text="", fg="#9fbfdf", bg=BG, justify="left", anchor="w",
            wraplength=self._wraplength)
        if cfg.show_source:
            self.label_source.pack(fill="x", padx=20, pady=(8, 0))
        self.label_final = tk.Label(
            content, text="", fg="white", bg=BG, justify="left", anchor="w",
            wraplength=self._wraplength)
        self.label_final.pack(fill="x", padx=20, pady=(8, 0))
        self.label_partial = tk.Label(
            content, text="", fg="#a0a0a0", bg=BG, justify="left", anchor="w",
            wraplength=self._wraplength)
        self.label_partial.pack(fill="x", padx=20, pady=(0, 8))
        self._apply_font()

        for widget in (self.root, content, self.label_source, self.label_final,
                       self.label_partial):
            widget.bind("<Button-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
            widget.bind("<Button-3>", self._show_menu)
            widget.bind("<MouseWheel>", self._on_wheel)
        self.scrollbar.bind("<MouseWheel>", self._on_wheel)
        self.scrollbar.bind("<Button-1>", self._on_scrollbar_click)
        self.scrollbar.bind("<B1-Motion>", self._on_scrollbar_click)

        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="フォント拡大", command=lambda: self._resize_font(+2))
        self.menu.add_command(label="フォント縮小", command=lambda: self._resize_font(-2))
        self.menu.add_command(label="最新に戻る", command=self._jump_to_live)
        self.menu.add_command(label="字幕クリア", command=self._clear)
        self.menu.add_separator()
        self.menu.add_command(label="終了", command=self.close)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._poll)

    # ---------- 表示 ----------

    def _apply_font(self):
        font = (self.cfg.font_family, self.font_size)
        small = (self.cfg.font_family, max(10, int(self.font_size * 0.6)))
        self.label_final.configure(font=font)
        self.label_partial.configure(font=font)
        self.label_source.configure(font=small)

    def _resize_font(self, delta: int):
        self.font_size = max(10, self.font_size + delta)
        self._apply_font()
        self._render()

    def _clear(self):
        self.history.clear()
        self.view_offset = 0
        self._live_partial = ""
        self._live_source = ""
        self._render()

    def _visible_finals(self) -> list:
        n = self.cfg.final_lines
        end = len(self.history) - self.view_offset
        return list(self.history)[max(0, end - n):end]

    def _render(self):
        live = self.view_offset == 0
        self.label_final.configure(text="\n".join(self._visible_finals()))
        self.label_partial.configure(text=self._live_partial if live else "‹ 過去の字幕 ›")
        if self.cfg.show_source:
            self.label_source.configure(text=self._live_source if live else "")
        self.root.update_idletasks()
        height = self.root.winfo_reqheight()
        self.root.geometry(
            f"{self.root.winfo_width()}x{height}"
            f"+{self.root.winfo_x()}+{self._anchor_bottom - height}")
        self.root.update_idletasks()
        self._draw_scrollbar()

    # ---------- スクロール ----------

    def _max_offset(self) -> int:
        return max(0, len(self.history) - self.cfg.final_lines)

    def _set_offset(self, offset: int):
        offset = max(0, min(self._max_offset(), offset))
        if offset != self.view_offset:
            self.view_offset = offset
            self._render()

    def _on_wheel(self, event):
        step = 1 if event.delta > 0 else -1  # 上に回す=過去へ
        self._set_offset(self.view_offset + step)

    def _jump_to_live(self):
        self._set_offset(0)

    def _draw_scrollbar(self):
        self.scrollbar.delete("all")
        h = self.scrollbar.winfo_height()
        total = len(self.history)
        n = self.cfg.final_lines
        if total <= n:
            thumb_top, thumb_h = 0, h
        else:
            thumb_h = max(THUMB_MIN_H, int(h * n / total))
            max_off = self._max_offset()
            # offset=max_off が最上部(最古)、offset=0 が最下部(最新)
            ratio = 1.0 - self.view_offset / max_off
            thumb_top = int((h - thumb_h) * ratio)
        color = "#e0e0e0" if self.view_offset else "#707070"
        self.scrollbar.create_rectangle(
            2, thumb_top + 1, SCROLLBAR_W - 2, thumb_top + thumb_h - 1,
            fill=color, outline="")

    def _on_scrollbar_click(self, event):
        h = self.scrollbar.winfo_height()
        max_off = self._max_offset()
        if h <= 0 or max_off == 0:
            return
        ratio = min(1.0, max(0.0, event.y / h))
        self._set_offset(round((1.0 - ratio) * max_off))

    # ---------- ウィンドウ操作 ----------

    def _drag_start(self, event):
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _drag_move(self, event):
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{event.x_root - self._drag_x}+{y}")
        self._anchor_bottom = y + self.root.winfo_height()

    def _show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    # ---------- キューからの更新 ----------

    def _poll(self):
        if self.stop_event.is_set():
            self.root.destroy()
            return
        updated = False
        while True:
            try:
                item = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            updated = True
            if item["kind"] == "final":
                at_capacity = len(self.history) == MAX_HISTORY
                self.history.append(item["ja"])
                self._live_partial = ""
                self._live_source = item["en"]
                # 遡り中は表示位置を固定する(deque が満杯のときは全体が
                # 1 つずれるので補正不要)
                if self.view_offset > 0 and not at_capacity:
                    self.view_offset = min(self._max_offset(), self.view_offset + 1)
            else:
                self._live_partial = item["ja"]
                self._live_source = item["en"]
        if updated:
            self._render()
        self.root.after(50, self._poll)

    def close(self):
        self.stop_event.set()

    def run(self):
        self.root.mainloop()
