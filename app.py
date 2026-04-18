import tkinter as tk
from tkinter import filedialog, font
import threading
import time
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cover_song_system import (
    attach_confidence,
    dtw_with_transposition,
    mean_chroma_similarity,
    load_or_compute_beat_chroma,
    load_dataset,
)

BG          = "#0a0a0a"
BG2         = "#111111"
BG3         = "#1a1a1a"
BORDER      = "#2a2a2a"
ACCENT      = "#00ff9c"      
ACCENT2     = "#00cc7a"
AMBER       = "#ffb300"
RED         = "#ff4455"
DIM         = "#3a3a3a"
TEXT        = "#e0e0e0"
TEXT_DIM    = "#666666"
LCD_BG      = "#0d1a0d"
LCD_TEXT    = "#00ff9c"
LCD_DIM     = "#004422"
FONT_MONO   = "Courier"       
FONT_SIZE_S = 9
FONT_SIZE_M = 11
FONT_SIZE_L = 13
FONT_SIZE_XL= 16



def build_demo_data():
    import numpy as np
    np.random.seed(42)

    def normalize(ch):
        norms = np.linalg.norm(ch, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (ch / norms).astype(np.float32)

    def make_cover(base, noise=0.05, shift=0, tempo_stretch=1.0,
                   drop_section=False, blur=False):
        ch = np.roll(base.copy(), shift, axis=0)
        if blur:
            blurred = ch.copy()
            for i in range(12):
                blurred[i] += 0.3 * ch[(i-1)%12] + 0.3 * ch[(i+1)%12]
            ch = blurred
        ch = ch + np.random.randn(*ch.shape) * noise
        if tempo_stretch != 1.0:
            old_len = ch.shape[1]
            new_len = max(10, int(old_len * tempo_stretch))
            indices = np.linspace(0, old_len-1, new_len).astype(int)
            ch = ch[:, indices]
        if drop_section and ch.shape[1] > 30:
            t = ch.shape[1]
            start  = np.random.randint(t//4, t//2)
            length = np.random.randint(t//6, t//4)
            ch = np.concatenate([ch[:, :start], ch[:, start+length:]], axis=1)
        return normalize(ch)

    songs = [
        {"song_id": "001", "title": "Halo",      "artist": "Beyonce"},
        {"song_id": "002", "title": "Jolene",     "artist": "Dolly Parton"},
        {"song_id": "003", "title": "Creep",      "artist": "Radiohead"},
        {"song_id": "004", "title": "Hallelujah", "artist": "Leonard Cohen"},
        {"song_id": "005", "title": "Mad World",  "artist": "Tears for Fears"},
    ]

    cover_configs = [
        {"noise": 0.08, "shift": 2, "tempo_stretch": 1.0,  "drop_section": False, "blur": False},
        {"noise": 0.18, "shift": 5, "tempo_stretch": 1.15, "drop_section": False, "blur": True},
        {"noise": 0.30, "shift": 7, "tempo_stretch": 0.85, "drop_section": True,  "blur": True},
    ]

    dataset, chromas = [], {}
    for song in songs:
        base = normalize(np.random.rand(12, 80).astype(np.float32))
        entries = [{"version": "original", "chroma": base, "artist": song["artist"]}]
        for i, cfg in enumerate(cover_configs, 1):
            entries.append({
                "version": f"cover_{i}",
                "chroma":  make_cover(base, **cfg),
                "artist":  f"Cover Artist {chr(64+i)}",
            })
        for e in entries:
            fn = f"{song['song_id']}_{e['version']}.wav"
            dataset.append({
                "file_name": fn, "artist": e["artist"],
                "title": song["title"], "song_version": e["version"],
                "song_id": song["song_id"],
            })
            chromas[fn] = e["chroma"]

    return dataset, chromas


def run_fake_query(query_file, dataset, chromas, method="dtw", top_k=5):
    import numpy as np
    q_ch = chromas[query_file]
    results = []
    for row in dataset:
        fn = row["file_name"]
        if fn == query_file:
            continue
        c_ch = chromas[fn]
        if method == "dtw":
            cost  = dtw_with_transposition(q_ch, c_ch)
            score = float(1.0 / (1.0 + cost))
        else:
            score = mean_chroma_similarity(q_ch, c_ch)
        results.append({**row, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return attach_confidence(results)[:top_k]


# WIDGETS
class RetroButton(tk.Canvas):
    def __init__(self, parent, text, command=None, color=ACCENT,
                 text_color=BG, width=140, height=38, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=parent["bg"], highlightthickness=0, **kwargs)
        self.command  = command
        self.color    = color
        self.pressed  = False
        self.txt      = text
        self.w        = width
        self.h        = height
        self.txt_color = text_color
        self._draw(False)
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self, pressed):
        self.delete("all")
        ox, oy = (2, 2) if pressed else (0, 0)
        # shadow
        if not pressed:
            self.create_rectangle(4, 4, self.w, self.h,
                                  fill=ACCENT2, outline="")
        # main body
        self.create_rectangle(ox, oy, self.w-4+ox, self.h-4+oy,
                               fill=self.color, outline="")
        # top highlight
        self.create_line(ox+1, oy+1, self.w-5+ox, oy+1,
                         fill="#558866", width=1)
        # label
        self.create_text(self.w//2 - 2 + ox, self.h//2 - 2 + oy,
                         text=self.txt,
                         font=(FONT_MONO, FONT_SIZE_M, "bold"),
                         fill=self.txt_color)

    def _on_press(self, _):
        self.pressed = True
        self._draw(True)

    def _on_release(self, _):
        self.pressed = False
        self._draw(False)
        if self.command:
            self.command()

    def configure_color(self, color):
        self.color = color
        self._draw(False)


class LCDScreen(tk.Canvas):
    """LCD-style display panel with scanlines."""
    def __init__(self, parent, width=440, height=110, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=LCD_BG, highlightthickness=2,
                         highlightbackground=ACCENT2, **kwargs)
        self.w = width
        self.h = height
        self._draw_scanlines()

    def _draw_scanlines(self):
        for y in range(0, self.h, 3):
            self.create_line(0, y, self.w, y,
                             fill="#001108", width=1, tags="scanline")

    def set_text(self, line1="", line2="", line3="", blink=False):
        self.delete("text")
        lines = [line1, line2, line3]
        y_positions = [22, 50, 78]
        sizes = [FONT_SIZE_L, FONT_SIZE_M, FONT_SIZE_S]
        for i, (line, y, sz) in enumerate(zip(lines, y_positions, sizes)):
            if line:
                self.create_text(
                    self.w // 2, y,
                    text=line.upper(),
                    font=(FONT_MONO, sz, "bold"),
                    fill=LCD_TEXT,
                    tags="text"
                )
        # pixel glow effect on left edge
        self.create_rectangle(0, 0, 3, self.h,
                               fill=ACCENT2, outline="", tags="text")


class ResultRow(tk.Frame):
    def __init__(self, parent, rank, artist, title, score,
                 conf_label, conf_score, **kwargs):
        super().__init__(parent, bg=BG2, **kwargs)
        self.configure(pady=2)

        conf_colors = {"high": ACCENT, "medium": AMBER, "low": RED}
        conf_color  = conf_colors.get(conf_label, TEXT_DIM)

        # rank
        tk.Label(self, text=f"{rank:02d}", font=(FONT_MONO, FONT_SIZE_M, "bold"),
                 bg=BG2, fg=TEXT_DIM, width=3).pack(side=tk.LEFT, padx=(8, 4))

        # artist / title
        name = f"{artist[:16]} — {title[:12]}"
        tk.Label(self, text=name.upper(), font=(FONT_MONO, FONT_SIZE_S),
                 bg=BG2, fg=TEXT, anchor="w", width=28).pack(side=tk.LEFT, padx=4)

        # score
        tk.Label(self, text=f"{score:.3f}", font=(FONT_MONO, FONT_SIZE_S),
                 bg=BG2, fg=ACCENT2, width=6).pack(side=tk.LEFT, padx=4)

        # confidence bar
        bar_canvas = tk.Canvas(self, width=60, height=14,
                               bg=BG2, highlightthickness=0)
        bar_canvas.pack(side=tk.LEFT, padx=4)
        bar_w = int(conf_score * 58)
        bar_canvas.create_rectangle(0, 3, 58, 11, fill=DIM, outline="")
        if bar_w > 0:
            bar_canvas.create_rectangle(0, 3, bar_w, 11,
                                        fill=conf_color, outline="")

        # label
        tk.Label(self, text=conf_label.upper(),
                 font=(FONT_MONO, FONT_SIZE_S, "bold"),
                 bg=BG2, fg=conf_color, width=7).pack(side=tk.LEFT, padx=(2, 8))



class CoverDetectionApp:
    def __init__(self, root):
        self.root        = root
        self.query_file  = None
        self.results     = []
        self.animating   = False
        self._anim_dots  = 0
        self._anim_job   = None
        self.method      = tk.StringVar(value="dtw")

        # demo data (always available)
        self.demo_dataset, self.demo_chromas = build_demo_data()
        self.demo_files = [r["file_name"] for r in self.demo_dataset
                           if r["song_version"] == "original"]
        self._demo_idx  = 0

        self._setup_window()
        self._build_ui()
        self._lcd_idle()


    def _setup_window(self):
        self.root.title("COVER SONG DETECTOR")
        self.root.geometry("480x780")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)



    def _build_ui(self):
        pad = dict(padx=20)

        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill=tk.X, padx=20, pady=(18, 0))

        tk.Label(hdr, text="COVER SONG DETECTOR",
                 font=(FONT_MONO, FONT_SIZE_XL, "bold"),
                 bg=BG, fg=ACCENT).pack()
        tk.Label(hdr, text=" ",
                 font=(FONT_MONO, FONT_SIZE_S),
                 bg=BG, fg=TEXT_DIM).pack(pady=(2, 0))

        tk.Canvas(self.root, height=1, bg=ACCENT2,
                  highlightthickness=0).pack(fill=tk.X, padx=20, pady=10)

        lcd_outer = tk.Frame(self.root, bg=BORDER, padx=3, pady=3)
        lcd_outer.pack(padx=20, pady=(0, 12))
        self.lcd = LCDScreen(lcd_outer, width=430, height=100)
        self.lcd.pack()


        self.wave_canvas = tk.Canvas(self.root, height=36, bg=BG,
                                     highlightthickness=0)
        self.wave_canvas.pack(fill=tk.X, padx=20, pady=(0, 10))
        self._draw_wave(active=False)

        method_frame = tk.Frame(self.root, bg=BG)
        method_frame.pack(**pad, pady=(0, 8))

        tk.Label(method_frame, text="METHOD:",
                 font=(FONT_MONO, FONT_SIZE_S, "bold"),
                 bg=BG, fg=TEXT_DIM).pack(side=tk.LEFT, padx=(0, 8))

        for val, label in [("dtw", "DTW"), ("baseline", "MEAN CHROMA")]:
            rb = tk.Radiobutton(
                method_frame, text=label, variable=self.method, value=val,
                font=(FONT_MONO, FONT_SIZE_S),
                bg=BG, fg=ACCENT, selectcolor=BG3,
                activebackground=BG, activeforeground=ACCENT,
                indicatoron=True
            )
            rb.pack(side=tk.LEFT, padx=6)

# buttons
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(**pad, pady=(0, 6))

        self.btn_load = RetroButton(btn_frame, "LOAD FILE",
                                    command=self._load_file,
                                    color=DIM, text_color=ACCENT,
                                    width=130, height=40)
        self.btn_load.grid(row=0, column=0, padx=6)

        self.btn_query = RetroButton(btn_frame, "QUERY",
                                     command=self._run_query,
                                     color=ACCENT, text_color=BG,
                                     width=100, height=40)
        self.btn_query.grid(row=0, column=1, padx=6)

        self.btn_demo = RetroButton(btn_frame, "DEMO MODE",
                                    command=self._run_demo,
                                    color=AMBER, text_color=BG,
                                    width=130, height=40)
        self.btn_demo.grid(row=0, column=2, padx=6)

        self.btn_clear = RetroButton(btn_frame, "CLR",
                                     command=self._clear,
                                     color=BG3, text_color=TEXT_DIM,
                                     width=52, height=40)
        self.btn_clear.grid(row=0, column=3, padx=6)

        tk.Canvas(self.root, height=1, bg=BORDER,
                  highlightthickness=0).pack(fill=tk.X, padx=20, pady=(6, 0))

        results_header = tk.Frame(self.root, bg=BG)
        results_header.pack(fill=tk.X, padx=20, pady=(6, 0))

        tk.Label(results_header, text="##  ARTIST — TITLE",
                 font=(FONT_MONO, FONT_SIZE_S, "bold"),
                 bg=BG, fg=TEXT_DIM, anchor="w").pack(side=tk.LEFT)
        tk.Label(results_header, text="SCORE  CONFIDENCE",
                 font=(FONT_MONO, FONT_SIZE_S, "bold"),
                 bg=BG, fg=TEXT_DIM, anchor="e").pack(side=tk.RIGHT)

        list_frame = tk.Frame(self.root, bg=BG2, bd=1, relief=tk.SUNKEN)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(4, 0))

        self.results_canvas = tk.Canvas(list_frame, bg=BG2,
                                        highlightthickness=0)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical",
                                 command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.results_inner = tk.Frame(self.results_canvas, bg=BG2)
        self.results_canvas.create_window((0, 0), window=self.results_inner,
                                          anchor="nw")
        self.results_inner.bind("<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")))

        self._show_placeholder()

# ── status bar ────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="READY  //  NO FILE LOADED")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              font=(FONT_MONO, FONT_SIZE_S),
                              bg=BG3, fg=TEXT_DIM, anchor="w", padx=8, pady=4)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _draw_wave(self, active=False):
        self.wave_canvas.delete("all")
        w = 440
        cx = w // 2
        color = ACCENT if active else DIM
        for x in range(w):
            t = x / w
            if active:
                amp   = 12 * math.sin(t * math.pi)
                freq  = 6 + 4 * math.sin(t * math.pi * 3)
                y     = 18 + amp * math.sin(t * freq * math.pi * 2)
            else:
                y = 18 + 4 * math.sin(t * 8 * math.pi)
            self.wave_canvas.create_line(x, 18, x, int(y),
                                         fill=color, width=1)

# LCD helpers

    def _lcd_idle(self):
        self.lcd.set_text("COVER SONG DETECTOR",
                          "ready — load file or run demo")

    def _lcd_set(self, l1="", l2="", l3=""):
        self.lcd.set_text(l1, l2, l3)

    def _start_animation(self, msg="analysing"):
        self.animating = True
        self._draw_wave(active=True)
        self._animate_lcd(msg)

    def _animate_lcd(self, msg):
        if not self.animating:
            return
        dots = "." * (self._anim_dots % 4)
        self._lcd_set(f"{msg}{dots}",
                      "scanning library",
                      "beat-sync dtw in progress")
        self._anim_dots += 1
        self._anim_job = self.root.after(400, lambda: self._animate_lcd(msg))

    def _stop_animation(self):
        self.animating = False
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
        self._draw_wave(active=False)

# results

    def _show_placeholder(self):
        for w in self.results_inner.winfo_children():
            w.destroy()
        tk.Label(self.results_inner,
                 text="\n  no results yet\n  load a file and press query\n  or press demo mode\n",
                 font=(FONT_MONO, FONT_SIZE_S),
                 bg=BG2, fg=TEXT_DIM, justify="left").pack(anchor="w", padx=8)

    def _animate_results(self, results, idx=0):
        if idx >= len(results):
            self._stop_animation()
            self.status_var.set(
                f"METHOD: {self.method.get().upper()}  //  "
                f"{len(results)} RESULTS  //  QUERY COMPLETE"
            )
            # beep
            try:
                self.root.bell()
            except Exception:
                pass
            return

        r = results[idx]
        row = ResultRow(
            self.results_inner,
            rank=idx + 1,
            artist=r["artist"],
            title=r["title"],
            score=r["score"],
            conf_label=r["confidence_label"],
            conf_score=r["confidence"],
        )
        row.pack(fill=tk.X, pady=1)
        self.results_canvas.update_idletasks()
        self.results_canvas.yview_moveto(1.0)

        self.root.after(180, lambda: self._animate_results(results, idx + 1))

    def _display_results(self, results, query_name):
        for w in self.results_inner.winfo_children():
            w.destroy()
        short = os.path.basename(query_name)[:30]
        self._lcd_set(f"query: {short}",
                      f"top match: {results[0]['artist'][:18]}",
                      f"score: {results[0]['score']:.3f}  conf: {results[0]['confidence_label']}")
        self._animate_results(results)

# actions

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        self.query_file = path
        name = os.path.basename(path)
        self._lcd_set(f"loaded: {name[:28]}",
                      "press query to search",
                      "method: " + self.method.get())
        self.status_var.set(f"FILE LOADED  //  {name}")

    def _run_query(self):
        if not self.query_file:
            self._lcd_set("!! no file loaded",
                          "press load file first",
                          "or use demo mode")
            return

        self._start_animation("querying")
        t_start = time.time()

        def worker():
            try:
                dataset  = load_dataset()
                q_chroma = load_or_compute_beat_chroma(
                    os.path.basename(self.query_file))
                results  = []
                method   = self.method.get()
                for row in dataset:
                    fn = row["file_name"].strip()
                    if fn == os.path.basename(self.query_file):
                        continue
                    try:
                        from cover_song_system import load_or_compute_beat_chroma as lbc
                        c_ch = lbc(fn)
                    except FileNotFoundError:
                        continue
                    if method == "dtw":
                        cost  = dtw_with_transposition(q_chroma, c_ch)
                        score = float(1.0 / (1.0 + cost))
                    else:
                        score = mean_chroma_similarity(q_chroma, c_ch)
                    results.append({**row, "score": score})

                results.sort(key=lambda x: x["score"], reverse=True)
                results = attach_confidence(results)[:10]
                elapsed = time.time() - t_start
                self.root.after(0, lambda: self._on_query_done(
                    results, self.query_file, elapsed))
            except Exception as e:
                self.root.after(0, lambda: self._on_query_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _run_demo(self):
        self._start_animation("demo mode")
        t_start = time.time()

        def worker():
            method  = self.method.get()
            qfile   = self.demo_files[self._demo_idx % len(self.demo_files)]
            self._demo_idx += 1
            results = run_fake_query(qfile, self.demo_dataset,
                                     self.demo_chromas, method=method, top_k=8)
            elapsed = time.time() - t_start
            self.root.after(0, lambda: self._on_query_done(
                results, qfile, elapsed, demo=True))

        threading.Thread(target=worker, daemon=True).start()

    def _on_query_done(self, results, query_file, elapsed, demo=False):
        self._stop_animation()
        if not results:
            self._lcd_set("!! no results",
                          "check feature files exist",
                          "try demo mode instead")
            return
        tag = " [DEMO]" if demo else ""
        self.status_var.set(
            f"METHOD: {self.method.get().upper()}{tag}  //  "
            f"{len(results)} RESULTS  //  {elapsed:.2f}s"
        )
        self._display_results(results, query_file)

    def _on_query_error(self, err):
        self._stop_animation()
        self._lcd_set("!! error", err[:36], "try demo mode instead")
        self.status_var.set(f"ERROR: {err[:50]}")

    def _clear(self):
        self._stop_animation()
        self.query_file = None
        self._show_placeholder()
        self._lcd_idle()
        self.status_var.set("READY  //  NO FILE LOADED")
        self._draw_wave(active=False)


#main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app  = CoverDetectionApp(root)
    root.mainloop()