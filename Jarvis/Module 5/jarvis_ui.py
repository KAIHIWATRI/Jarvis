"""
jarvis_ui.py — JARVIS Desktop UI  (CustomTkinter)
Complete cyberpunk-style assistant interface.

Layout
──────
┌─ Header (logo + status bar) ──────────────────────────────┐
│ ┌─ Sidebar ──┐  ┌─ Center ──────────────┐  ┌─ Right ───┐ │
│ │  Nav       │  │  Orb + Waveform       │  │  Sysmon   │ │
│ │  Voice     │  │  Chat window          │  │  Settings │ │
│ │  Controls  │  │  Input bar            │  │  Logs     │ │
│ └────────────┘  └───────────────────────┘  └───────────┘ │
└─ Status bar ──────────────────────────────────────────────┘
"""

import os
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from typing import Optional

import customtkinter as ctk

from theme import COLORS, FONTS, LAYOUT, apply_theme
from components import (
    AIOrb, ListeningPulse, RingGauge, StatusLED, VoiceWaveform,
)

# Optional system stats
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _C(key: str) -> str:
    """Shorthand for COLORS dict lookup."""
    return COLORS[key]


def _ctk_frame(parent, **kwargs) -> ctk.CTkFrame:
    defaults = dict(
        fg_color=_C("bg_panel"),
        border_color=_C("border_normal"),
        border_width=1,
        corner_radius=LAYOUT["corner_radius"],
    )
    defaults.update(kwargs)
    return ctk.CTkFrame(parent, **defaults)


def _ctk_label(parent, text="", color="text_primary", font_key="body", **kwargs) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent, text=text,
        text_color=_C(color),
        font=FONTS[font_key],
        **kwargs,
    )


def _ctk_button(parent, text="", command=None, color="cyan_core", **kwargs) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent, text=text, command=command,
        fg_color=_C("bg_card"),
        hover_color=_C("bg_hover"),
        border_color=_C("cyan_dim"),
        text_color=_C(color),
        border_width=1,
        corner_radius=LAYOUT["corner_radius_sm"],
        font=FONTS["heading"],
        **kwargs,
    )


# ─────────────────────────────────────────────
# Header Bar
# ─────────────────────────────────────────────
class HeaderBar(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color=_C("bg_void"),
            border_color=_C("border_normal"),
            border_width=0,
            corner_radius=0,
            height=LAYOUT["header_height"],
            **kwargs,
        )
        self.grid_propagate(False)
        self._build()

    def _build(self):
        # Logo
        ctk.CTkLabel(
            self, text="[ J.A.R.V.I.S ]",
            text_color=_C("cyan_bright"),
            font=FONTS["title_lg"],
        ).pack(side="left", padx=16, pady=8)

        ctk.CTkLabel(
            self, text="Just A Rather Very Intelligent System",
            text_color=_C("text_dim"),
            font=FONTS["label"],
        ).pack(side="left", padx=0)

        # Right side: time + version
        self._time_lbl = ctk.CTkLabel(
            self, text="",
            text_color=_C("text_secondary"),
            font=FONTS["mono"],
        )
        self._time_lbl.pack(side="right", padx=16)

        ctk.CTkLabel(
            self, text="v2.0  |  ONLINE",
            text_color=_C("cyan_dim"),
            font=FONTS["status"],
        ).pack(side="right", padx=8)

        self._update_time()

    def _update_time(self):
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self._time_lbl.configure(text=now)
        self.after(1000, self._update_time)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
class Sidebar(ctk.CTkFrame):
    """Left sidebar: navigation, voice controls, quick actions."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(
            parent,
            fg_color=_C("bg_void"),
            border_color=_C("border_normal"),
            border_width=1,
            corner_radius=0,
            width=LAYOUT["sidebar_width"],
            **kwargs,
        )
        self.grid_propagate(False)
        self._app = app
        self._build()

    def _build(self):
        pad = LAYOUT["pad_x"]

        # ── System section ────────────────────
        _ctk_label(self, "◈  SYSTEM", color="cyan_dim", font_key="status").pack(
            anchor="w", padx=pad, pady=(12, 4)
        )
        self._separator()

        # Status rows
        self._status_rows = {}
        status_items = [
            ("STT Engine",  "Whisper",    "idle"),
            ("LLM",         "Ollama",     "idle"),
            ("TTS Engine",  "Edge-TTS",   "idle"),
            ("MIC",         "Ready",      "idle"),
        ]
        for label, value, state in status_items:
            row = _ctk_frame(self, fg_color="transparent", border_width=0)
            row.pack(fill="x", padx=pad, pady=2)

            led = StatusLED(row, size=8)
            led.pack(side="left", padx=(0, 6))
            led.set_status(state)

            _ctk_label(row, label, color="text_secondary", font_key="label").pack(side="left")
            val_lbl = _ctk_label(row, value, color="cyan_core", font_key="label")
            val_lbl.pack(side="right")

            self._status_rows[label] = (led, val_lbl)

        # ── Voice controls ────────────────────
        _ctk_label(self, "◈  VOICE", color="cyan_dim", font_key="status").pack(
            anchor="w", padx=pad, pady=(14, 4)
        )
        self._separator()

        # Voice profile selector
        _ctk_label(self, "Profile", color="text_secondary", font_key="label").pack(
            anchor="w", padx=pad, pady=(4, 0)
        )
        self._voice_menu = ctk.CTkOptionMenu(
            self,
            values=["JARVIS (British)", "FRIDAY (US)", "ATLAS (UK-F)", "KAREN (AU)", "EDWIN (US-M)"],
            fg_color=_C("bg_card"),
            button_color=_C("cyan_dim"),
            text_color=_C("text_primary"),
            font=FONTS["body_sm"],
            width=LAYOUT["sidebar_width"] - pad * 2,
        )
        self._voice_menu.pack(padx=pad, pady=(0, 6))

        # Volume slider
        _ctk_label(self, "Volume", color="text_secondary", font_key="label").pack(
            anchor="w", padx=pad
        )
        self._vol_slider = ctk.CTkSlider(
            self, from_=0, to=100, number_of_steps=20,
            button_color=_C("cyan_core"),
            progress_color=_C("cyan_dim"),
            fg_color=_C("bg_card"),
            width=LAYOUT["sidebar_width"] - pad * 2,
        )
        self._vol_slider.set(80)
        self._vol_slider.pack(padx=pad, pady=(0, 6))

        # Speed slider
        _ctk_label(self, "Speech speed", color="text_secondary", font_key="label").pack(
            anchor="w", padx=pad
        )
        self._speed_slider = ctk.CTkSlider(
            self, from_=0, to=100, number_of_steps=10,
            button_color=_C("cyan_core"),
            progress_color=_C("cyan_dim"),
            fg_color=_C("bg_card"),
            width=LAYOUT["sidebar_width"] - pad * 2,
        )
        self._speed_slider.set(50)
        self._speed_slider.pack(padx=pad, pady=(0, 8))

        # ── Quick actions ─────────────────────
        _ctk_label(self, "◈  CONTROLS", color="cyan_dim", font_key="status").pack(
            anchor="w", padx=pad, pady=(8, 4)
        )
        self._separator()

        btns = [
            ("⬤  Start Listening",  self._app.start_listening),
            ("◼  Stop / Interrupt",  self._app.stop_listening),
            ("✕  Clear Chat",        self._app.clear_chat),
            ("⚙  Settings",          self._app.open_settings),
        ]
        for label, cmd in btns:
            _ctk_button(self, text=label, command=cmd, height=28).pack(
                fill="x", padx=pad, pady=2
            )

        # ── Listening pulse ───────────────────
        self._listen_pulse = ListeningPulse(
            self, size=50,
            bg=_C("bg_void"),
        )
        self._listen_pulse.pack(pady=12)
        self._listen_pulse.start()

    def _separator(self):
        tk.Frame(self, bg=_C("border_normal"), height=1).pack(fill="x", padx=LAYOUT["pad_x"], pady=2)

    def set_status(self, key: str, value: str, state: str = "idle"):
        if key in self._status_rows:
            led, val_lbl = self._status_rows[key]
            led.set_status(state, blink=(state in ("listening", "thinking")))
            val_lbl.configure(text=value)

    def set_listening(self, active: bool):
        self._listen_pulse.set_active(active)


# ─────────────────────────────────────────────
# Center Panel: Orb + Waveform + Chat
# ─────────────────────────────────────────────
class CenterPanel(ctk.CTkFrame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(
            parent,
            fg_color=_C("bg_primary"),
            border_color=_C("border_normal"),
            border_width=0,
            corner_radius=0,
            **kwargs,
        )
        self._app = app
        self._build()

    def _build(self):
        pad = LAYOUT["pad_x"]

        # ── Top: Orb + status strip ───────────
        top = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        top.pack(fill="x", padx=pad, pady=(pad, 0))

        orb_frame = _ctk_frame(top, fg_color=_C("bg_void"))
        orb_frame.pack(side="left", padx=(0, 8))

        self.orb = AIOrb(
            orb_frame,
            size=LAYOUT["orb_size"],
            bg=_C("bg_void"),
        )
        self.orb.pack(padx=4, pady=4)
        self.orb.start()

        # Status text panel next to orb
        status_col = ctk.CTkFrame(top, fg_color="transparent", corner_radius=0)
        status_col.pack(side="left", fill="both", expand=True)

        _ctk_label(status_col, "ASSISTANT STATUS", color="cyan_dim", font_key="status").pack(
            anchor="w", pady=(4, 2)
        )
        self._status_text = _ctk_label(status_col, "Systems nominal. Awaiting input.",
                                        color="text_primary", font_key="body")
        self._status_text.pack(anchor="w")

        self._context_text = _ctk_label(
            status_col, "Context: 0 tokens  |  Session: 0:00",
            color="text_dim", font_key="label"
        )
        self._context_text.pack(anchor="w", pady=(4, 0))

        # Waveform
        wf_frame = _ctk_frame(status_col, fg_color=_C("bg_void"), height=60)
        wf_frame.pack(fill="x", pady=(8, 0))
        wf_frame.pack_propagate(False)

        self.waveform = VoiceWaveform(
            wf_frame,
            bars=LAYOUT["waveform_bars"],
            height=LAYOUT["waveform_height"],
            bg=_C("bg_void"),
        )
        self.waveform.pack(fill="both", expand=True, padx=4, pady=4)
        self.waveform.start()

        # ── Chat window ───────────────────────
        chat_outer = _ctk_frame(self, fg_color=_C("bg_panel"))
        chat_outer.pack(fill="both", expand=True, padx=pad, pady=(8, 0))

        _ctk_label(chat_outer, "◈  CONVERSATION LOG", color="cyan_dim", font_key="status").pack(
            anchor="w", padx=8, pady=(6, 2)
        )

        self._chat_box = tk.Text(
            chat_outer,
            bg=_C("bg_void"),
            fg=_C("text_primary"),
            font=FONTS["chat_ai"],
            wrap="word",
            state="disabled",
            relief="flat",
            bd=0,
            highlightthickness=0,
            insertbackground=_C("cyan_bright"),
            selectbackground=_C("cyan_ghost"),
            padx=8, pady=6,
        )
        scrollbar = ctk.CTkScrollbar(chat_outer, command=self._chat_box.yview)
        self._chat_box.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y", padx=(0, 2), pady=2)
        self._chat_box.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 4))

        # Tag configs
        self._chat_box.tag_config("user_label",   foreground=_C("blue_bright"), font=FONTS["chat_label"])
        self._chat_box.tag_config("user_text",    foreground=_C("text_primary"),  font=FONTS["chat_user"])
        self._chat_box.tag_config("ai_label",     foreground=_C("cyan_bright"),  font=FONTS["chat_label"])
        self._chat_box.tag_config("ai_text",      foreground=_C("text_primary"),  font=FONTS["chat_ai"])
        self._chat_box.tag_config("system_text",  foreground=_C("text_dim"),      font=FONTS["label"])
        self._chat_box.tag_config("timestamp",    foreground=_C("text_dim"),      font=FONTS["label"])
        self._chat_box.tag_config("error_text",   foreground=_C("red_alert"),     font=FONTS["body_sm"])

        # ── Input bar ─────────────────────────
        input_row = ctk.CTkFrame(self, fg_color=_C("bg_void"), corner_radius=0)
        input_row.pack(fill="x", padx=pad, pady=(4, pad))

        self._input = ctk.CTkEntry(
            input_row,
            placeholder_text="Type a message or press mic to speak…",
            fg_color=_C("bg_input"),
            border_color=_C("border_normal"),
            text_color=_C("text_primary"),
            placeholder_text_color=_C("text_dim"),
            font=FONTS["body"],
            corner_radius=LAYOUT["corner_radius_sm"],
            height=34,
        )
        self._input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._input.bind("<Return>", self._on_enter)

        _ctk_button(input_row, text="SEND", command=self._on_send,
                    width=60, height=34).pack(side="left", padx=(0, 4))
        _ctk_button(input_row, text="MIC", command=self._app.toggle_mic,
                    width=50, height=34, color="green_active").pack(side="left")

    def _on_enter(self, _event=None):
        self._on_send()

    def _on_send(self):
        text = self._input.get().strip()
        if text:
            self._input.delete(0, "end")
            self._app.on_user_input(text)

    def append_message(self, role: str, text: str):
        ts  = datetime.now().strftime("%H:%M:%S")
        box = self._chat_box
        box.configure(state="normal")
        box.insert("end", f"\n")

        if role == "user":
            box.insert("end", f"YOU  ", "user_label")
            box.insert("end", f"[{ts}]\n", "timestamp")
            box.insert("end", f"  {text}\n", "user_text")
        elif role == "assistant":
            box.insert("end", f"JARVIS  ", "ai_label")
            box.insert("end", f"[{ts}]\n", "timestamp")
            box.insert("end", f"  {text}\n", "ai_text")
        elif role == "system":
            box.insert("end", f"  ⟩ {text}\n", "system_text")
        elif role == "error":
            box.insert("end", f"  ✕ ERROR: {text}\n", "error_text")

        box.configure(state="disabled")
        box.see("end")

    def set_status(self, text: str):
        self._status_text.configure(text=text)

    def set_context_info(self, tokens: int, session_secs: int):
        mins = session_secs // 60
        secs = session_secs % 60
        self._context_text.configure(
            text=f"Context: {tokens} tokens  |  Session: {mins}:{secs:02d}"
        )


# ─────────────────────────────────────────────
# Right Panel: System Monitor + Settings
# ─────────────────────────────────────────────
class RightPanel(ctk.CTkFrame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(
            parent,
            fg_color=_C("bg_void"),
            border_color=_C("border_normal"),
            border_width=1,
            corner_radius=0,
            width=200,
            **kwargs,
        )
        self.grid_propagate(False)
        self._app = app
        self._build()

    def _build(self):
        pad = LAYOUT["pad_x"]

        # ── System Monitor ────────────────────
        _ctk_label(self, "◈  SYSTEM MONITOR", color="cyan_dim", font_key="status").pack(
            anchor="w", padx=pad, pady=(10, 4)
        )
        tk.Frame(self, bg=_C("border_normal"), height=1).pack(fill="x", padx=pad)

        gauges_row = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        gauges_row.pack(fill="x", padx=pad, pady=6)

        self._cpu_gauge = RingGauge(
            gauges_row, "CPU", size=72,
            color=_C("cyan_core"), bg=_C("bg_void"),
        )
        self._cpu_gauge.pack(side="left", padx=(0, 4))
        self._cpu_gauge.start()

        self._ram_gauge = RingGauge(
            gauges_row, "RAM", size=72,
            color=_C("blue_bright"), bg=_C("bg_void"),
        )
        self._ram_gauge.pack(side="left", padx=(0, 4))
        self._ram_gauge.start()

        # Stat rows
        self._stat_labels = {}
        for key in ("CPU Freq", "RAM Used", "Disk", "Uptime", "Temp"):
            row = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
            row.pack(fill="x", padx=pad, pady=1)
            _ctk_label(row, key, color="text_secondary", font_key="label").pack(side="left")
            lbl = _ctk_label(row, "—", color="text_primary", font_key="label")
            lbl.pack(side="right")
            self._stat_labels[key] = lbl

        # ── Session Log ───────────────────────
        _ctk_label(self, "◈  SESSION LOG", color="cyan_dim", font_key="status").pack(
            anchor="w", padx=pad, pady=(12, 4)
        )
        tk.Frame(self, bg=_C("border_normal"), height=1).pack(fill="x", padx=pad)

        self._log_box = tk.Text(
            self,
            bg=_C("bg_void"),
            fg=_C("text_dim"),
            font=FONTS["mono"],
            wrap="word",
            state="disabled",
            relief="flat",
            bd=0,
            highlightthickness=0,
            height=8,
        )
        self._log_box.pack(fill="x", padx=pad, pady=(4, 8))

        # ── Model settings ────────────────────
        _ctk_label(self, "◈  MODEL", color="cyan_dim", font_key="status").pack(
            anchor="w", padx=pad, pady=(4, 4)
        )
        tk.Frame(self, bg=_C("border_normal"), height=1).pack(fill="x", padx=pad)

        _ctk_label(self, "Ollama model", color="text_secondary", font_key="label").pack(
            anchor="w", padx=pad, pady=(4, 0)
        )
        self._model_menu = ctk.CTkOptionMenu(
            self,
            values=["mistral", "llama3", "gemma2", "phi3", "deepseek-r1"],
            fg_color=_C("bg_card"),
            button_color=_C("cyan_dim"),
            text_color=_C("text_primary"),
            font=FONTS["body_sm"],
            width=178,
        )
        self._model_menu.pack(padx=pad, pady=(0, 6))

        _ctk_label(self, "Whisper model", color="text_secondary", font_key="label").pack(
            anchor="w", padx=pad
        )
        self._whisper_menu = ctk.CTkOptionMenu(
            self,
            values=["tiny", "base", "small", "medium"],
            fg_color=_C("bg_card"),
            button_color=_C("cyan_dim"),
            text_color=_C("text_primary"),
            font=FONTS["body_sm"],
            width=178,
        )
        self._whisper_menu.pack(padx=pad, pady=(0, 6))

        _ctk_button(self, text="Apply settings", height=26,
                    command=self._apply_settings).pack(fill="x", padx=pad, pady=(0, 8))

    def _apply_settings(self):
        self._app.log("Settings applied.")

    def update_system_stats(self, stats: dict):
        cpu = stats.get("cpu", 0)
        ram = stats.get("ram", 0)
        self._cpu_gauge.set_value(cpu)
        self._ram_gauge.set_value(ram)
        self._stat_labels["CPU Freq"].configure(text=stats.get("cpu_freq", "—"))
        self._stat_labels["RAM Used"].configure(text=stats.get("ram_used", "—"))
        self._stat_labels["Disk"].configure(text=stats.get("disk", "—"))
        self._stat_labels["Uptime"].configure(text=stats.get("uptime", "—"))
        self._stat_labels["Temp"].configure(text=stats.get("temp", "—"))

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_box.configure(state="normal")
        self._log_box.insert("end", f"[{ts}] {msg}\n")
        self._log_box.configure(state="disabled")
        self._log_box.see("end")


# ─────────────────────────────────────────────
# Status Bar
# ─────────────────────────────────────────────
class StatusBar(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color=_C("bg_void"),
            border_color=_C("border_normal"),
            border_width=0,
            corner_radius=0,
            height=LAYOUT["statusbar_height"],
            **kwargs,
        )
        self.grid_propagate(False)
        self._build()

    def _build(self):
        self._led = StatusLED(self, size=10)
        self._led.pack(side="left", padx=(12, 4), pady=8)
        self._led.set_status("online")

        self._main_lbl = _ctk_label(self, "JARVIS online — all systems nominal",
                                    color="text_secondary", font_key="label")
        self._main_lbl.pack(side="left")

        _ctk_label(self, f"Python {sys.version.split()[0]}  |  CustomTkinter",
                   color="text_dim", font_key="label").pack(side="right", padx=12)

    def set_message(self, msg: str, status: str = "online"):
        self._main_lbl.configure(text=msg)
        self._led.set_status(status)


# ─────────────────────────────────────────────
# Settings Panel (modal overlay)
# ─────────────────────────────────────────────
class SettingsPanel(ctk.CTkToplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("JARVIS — Settings")
        self.geometry("460x520")
        self.resizable(False, False)
        self.configure(fg_color=_C("bg_primary"))
        self._build()

    def _build(self):
        pad = 14
        _ctk_label(self, "[ JARVIS SETTINGS ]", color="cyan_bright", font_key="title_lg").pack(
            anchor="w", padx=pad, pady=(14, 4)
        )
        tk.Frame(self, bg=_C("border_normal"), height=1).pack(fill="x", padx=pad)

        sections = [
            ("HOTKEYS",   [("Push-to-talk key", "F4"), ("Interrupt key", "F5"), ("Clear chat", "F6")]),
            ("BEHAVIOUR", [("Auto-listen on startup", "ON"), ("Show tray icon", "ON"),
                           ("Background mode", "OFF"), ("Typing indicator", "ON")]),
            ("AUDIO",     [("Input device", "Default"), ("Output device", "Default"),
                           ("Noise suppression", "ON")]),
        ]

        for section, rows in sections:
            _ctk_label(self, f"◈  {section}", color="cyan_dim", font_key="status").pack(
                anchor="w", padx=pad, pady=(10, 2)
            )
            tk.Frame(self, bg=_C("border_dim"), height=1).pack(fill="x", padx=pad)
            for label, default in rows:
                row = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
                row.pack(fill="x", padx=pad, pady=2)
                _ctk_label(row, label, color="text_secondary", font_key="body_sm").pack(side="left")
                _ctk_label(row, default, color="cyan_core", font_key="body_sm").pack(side="right")

        _ctk_button(self, text="Save & Close", command=self.destroy,
                    height=32).pack(fill="x", padx=pad, pady=14)


# ─────────────────────────────────────────────
# Main JARVIS Application
# ─────────────────────────────────────────────
class JARVISApp(ctk.CTk):
    def __init__(self):
        apply_theme()
        ctk.set_appearance_mode("dark")
        super().__init__()

        self.title("J.A.R.V.I.S — Assistant Interface")
        self.geometry(f"{LAYOUT['window_width']}x{LAYOUT['window_height']}")
        self.minsize(LAYOUT["window_min_w"], LAYOUT["window_min_h"])
        self.configure(fg_color=_C("bg_primary"))

        self._listening      = False
        self._session_start  = time.time()
        self._token_count    = 0
        self._settings_win: Optional[SettingsPanel] = None

        self._build_layout()
        self._start_monitors()
        self._welcome()

    def _build_layout(self):
        # Top header
        self._header = HeaderBar(self)
        self._header.pack(fill="x", side="top")

        # Bottom status bar
        self._statusbar = StatusBar(self)
        self._statusbar.pack(fill="x", side="bottom")

        # Main body
        body = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        body.pack(fill="both", expand=True)

        # Three columns
        self._sidebar = Sidebar(body, app=self)
        self._sidebar.pack(side="left", fill="y")

        self._right = RightPanel(body, app=self)
        self._right.pack(side="right", fill="y")

        self._center = CenterPanel(body, app=self)
        self._center.pack(side="left", fill="both", expand=True)

    def _start_monitors(self):
        self._poll_system_stats()
        self._update_session_info()

    def _poll_system_stats(self):
        if HAS_PSUTIL:
            try:
                cpu    = psutil.cpu_percent(interval=None)
                ram    = psutil.virtual_memory()
                disk   = psutil.disk_usage("/")
                freq   = psutil.cpu_freq()
                uptime = time.time() - psutil.boot_time()

                stats = {
                    "cpu":      cpu,
                    "ram":      ram.percent,
                    "cpu_freq": f"{freq.current:.0f} MHz" if freq else "—",
                    "ram_used": f"{ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB",
                    "disk":     f"{disk.percent:.0f}%",
                    "uptime":   f"{int(uptime//3600)}h {int(uptime%3600//60)}m",
                    "temp":     "—",
                }

                # Try CPU temp (Linux / macOS)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        first = next(iter(temps.values()))
                        stats["temp"] = f"{first[0].current:.0f}°C"
                except Exception:
                    pass

                self._right.update_system_stats(stats)
            except Exception:
                pass
        else:
            import random
            self._right.update_system_stats({
                "cpu": random.uniform(5, 35), "ram": random.uniform(30, 60),
                "cpu_freq": "3400 MHz", "ram_used": "6.2/16.0 GB",
                "disk": "44%", "uptime": "2h 14m", "temp": "52°C",
            })

        self.after(LAYOUT["sys_poll_ms"], self._poll_system_stats)

    def _update_session_info(self):
        elapsed = int(time.time() - self._session_start)
        self._center.set_context_info(self._token_count, elapsed)
        self.after(5000, self._update_session_info)

    def _welcome(self):
        self._center.append_message("system", "JARVIS interface initialised.")
        self._center.append_message("assistant",
            "Good day. All systems are nominal. How may I assist you today?")
        self._statusbar.set_message("JARVIS online — all systems nominal.", "online")

    # ── Public API ────────────────────────────

    def on_user_input(self, text: str):
        """Called when user sends a text message."""
        self._center.append_message("user", text)
        self._token_count += len(text.split())
        self.set_orb_state("thinking")
        self._sidebar.set_status("LLM", "Processing…", "thinking")
        self._statusbar.set_message("Processing…", "thinking")
        # Simulate async response (replace with real pipeline)
        self.after(1200, lambda: self._mock_response(text))

    def _mock_response(self, _user_text: str):
        resp = "I'm processing your request. In a real deployment, this would call Ollama and stream the response token-by-token through the speech pipeline."
        self._center.append_message("assistant", resp)
        self._token_count += len(resp.split())
        self.set_orb_state("speaking")
        self._sidebar.set_status("TTS Engine", "Speaking", "speaking")
        self.after(2000, lambda: (
            self.set_orb_state("idle"),
            self._sidebar.set_status("TTS Engine", "Ready", "idle"),
            self._sidebar.set_status("LLM", "Ollama", "idle"),
            self._statusbar.set_message("Ready.", "online"),
        ))

    def start_listening(self):
        self._listening = True
        self.set_orb_state("listening")
        self._center.orb.set_energy(0.7)
        self._center.waveform.set_active(True)
        self._sidebar.set_listening(True)
        self._sidebar.set_status("MIC", "Active", "listening")
        self._statusbar.set_message("Listening…", "listening")
        self.log("Microphone activated.")

    def stop_listening(self):
        self._listening = False
        self.set_orb_state("idle")
        self._center.orb.set_energy(0.0)
        self._center.waveform.set_active(False)
        self._sidebar.set_listening(False)
        self._sidebar.set_status("MIC", "Ready", "idle")
        self._statusbar.set_message("Standby.", "online")
        self.log("Microphone deactivated.")

    def toggle_mic(self):
        if self._listening:
            self.stop_listening()
        else:
            self.start_listening()

    def set_orb_state(self, state: str):
        self._center.orb.set_state(state)
        self._center.set_status({
            "idle":      "Systems nominal. Awaiting input.",
            "listening": "Listening — speak now…",
            "thinking":  "Processing request…",
            "speaking":  "Speaking…",
            "error":     "Error detected. Check logs.",
        }.get(state, ""))

    def clear_chat(self):
        box = self._center._chat_box
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.configure(state="disabled")
        self._token_count = 0
        self.log("Chat cleared.")
        self._center.append_message("system", "Conversation cleared.")

    def open_settings(self):
        if self._settings_win is None or not self._settings_win.winfo_exists():
            self._settings_win = SettingsPanel(self)
        self._settings_win.lift()
        self._settings_win.focus()

    def log(self, msg: str):
        self._right.log(msg)

    # ── TTS engine integration ────────────────
    def connect_tts(self, engine):
        """Attach a TTSEngine instance (from tts_engine.py)."""
        from ui_bridge import TTSUIBridge
        self._tts_bridge = TTSUIBridge(root=self, engine=engine)
        self._tts_bridge.on_speak_start = lambda _: self.set_orb_state("speaking")
        self._tts_bridge.on_speak_end   = lambda _: self.set_orb_state("idle")
        self.log("TTS engine connected.")

    def connect_speech_manager(self, manager):
        """Attach a SpeechManager for streaming TTS."""
        self._speech_manager = manager
        self.log("Speech manager connected.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def run():
    app = JARVISApp()

    # Optional: system tray (requires pystray + Pillow)
    try:
        import pystray
        from PIL import Image, ImageDraw

        def _make_icon():
            img = Image.new("RGB", (64, 64), color=(5, 8, 16))
            d = ImageDraw.Draw(img)
            d.ellipse((8, 8, 56, 56), outline=(0, 212, 212), width=3)
            d.ellipse((24, 24, 40, 40), fill=(0, 212, 212))
            return img

        def _tray_show(_icon, _item):
            app.deiconify()
            app.lift()

        def _tray_quit(_icon, _item):
            _icon.stop()
            app.quit()

        icon = pystray.Icon(
            "JARVIS",
            _make_icon(),
            "JARVIS Assistant",
            menu=pystray.Menu(
                pystray.MenuItem("Show", _tray_show),
                pystray.MenuItem("Quit", _tray_quit),
            ),
        )
        tray_thread = threading.Thread(target=icon.run, daemon=True)
        tray_thread.start()
    except ImportError:
        pass  # pystray not installed, skip tray

    app.mainloop()


if __name__ == "__main__":
    run()
