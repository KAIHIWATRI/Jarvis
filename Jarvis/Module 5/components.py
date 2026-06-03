"""
components.py — JARVIS Animated UI Components
Canvas-based animated widgets: AI Orb, Voice Waveform, System Ring Gauges.
All components use tkinter.Canvas for smooth, low-CPU animation.
"""

import math
import random
import time
import tkinter as tk
from typing import Callable, Optional

from theme import COLORS, FONTS, LAYOUT


# ─────────────────────────────────────────────
# Base animated canvas widget
# ─────────────────────────────────────────────
class AnimatedCanvas(tk.Canvas):
    """Canvas with a built-in animation loop. Override _draw_frame()."""

    def __init__(self, parent, fps: int = 20, **kwargs):
        defaults = dict(
            bg=COLORS["bg_primary"],
            highlightthickness=0,
            bd=0,
        )
        defaults.update(kwargs)
        super().__init__(parent, **defaults)
        self._fps       = fps
        self._interval  = max(16, 1000 // fps)
        self._running   = False
        self._job       = None
        self.bind("<Destroy>", self._on_destroy)

    def start(self):
        if not self._running:
            self._running = True
            self._tick()

    def stop(self):
        self._running = False
        if self._job:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass

    def _tick(self):
        if not self._running:
            return
        try:
            self._draw_frame()
            self._job = self.after(self._interval, self._tick)
        except tk.TclError:
            pass

    def _draw_frame(self):
        pass  # override

    def _on_destroy(self, _event=None):
        self.stop()


# ─────────────────────────────────────────────
# AI Orb
# ─────────────────────────────────────────────
class AIOrb(AnimatedCanvas):
    """
    Animated central AI orb with:
    - Pulsing outer ring
    - Rotating arc segments
    - Inner energy core glow
    - State: idle / listening / thinking / speaking
    """

    STATES = {
        "idle":      {"color": COLORS["cyan_dim"],    "pulse_speed": 0.02, "rotation_speed": 0.008},
        "listening": {"color": COLORS["green_active"],"pulse_speed": 0.06, "rotation_speed": 0.03},
        "thinking":  {"color": COLORS["purple_ai"],   "pulse_speed": 0.04, "rotation_speed": 0.06},
        "speaking":  {"color": COLORS["cyan_bright"], "pulse_speed": 0.08, "rotation_speed": 0.02},
        "error":     {"color": COLORS["red_alert"],   "pulse_speed": 0.1,  "rotation_speed": 0.01},
    }

    def __init__(self, parent, size: int = 180, **kwargs):
        super().__init__(parent, width=size, height=size, fps=20, **kwargs)
        self._size     = size
        self._cx       = size / 2
        self._cy       = size / 2
        self._state    = "idle"
        self._phase    = 0.0
        self._rot      = 0.0
        self._energy   = 0.0           # 0..1 visualised as inner glow

    def set_state(self, state: str):
        if state in self.STATES:
            self._state = state

    def set_energy(self, value: float):
        """0.0 = silent, 1.0 = maximum energy (from mic/tts levels)."""
        self._energy = max(0.0, min(1.0, value))

    def _draw_frame(self):
        cfg = self.STATES[self._state]
        self._phase += cfg["pulse_speed"]
        self._rot   += cfg["rotation_speed"]

        self.delete("all")

        cx, cy, s = self._cx, self._cy, self._size
        r_outer = s * 0.46
        r_mid   = s * 0.36
        r_inner = s * 0.24
        r_core  = s * 0.14

        pulse   = (math.sin(self._phase) * 0.5 + 0.5)           # 0..1
        energy  = self._energy * 0.7 + pulse * 0.3

        color   = cfg["color"]

        # ── Outer ghost rings ─────────────────
        for i in range(3):
            alpha = 0.15 + i * 0.05
            rad   = r_outer + i * 6 + pulse * 4
            self._draw_ring(cx, cy, rad, color, width=1, dash=(4, 6), stipple="gray25")

        # ── Rotating arc segments ─────────────
        num_arcs  = 6
        arc_span  = 35
        for i in range(num_arcs):
            angle = self._rot * 180 / math.pi + i * (360 / num_arcs)
            # Alternate bright/dim
            c = color if i % 2 == 0 else COLORS["cyan_ghost"]
            w = 2 if i % 2 == 0 else 1
            self._draw_arc(cx, cy, r_outer, angle, arc_span, c, width=w)

        # ── Middle ring ───────────────────────
        self._draw_ring(cx, cy, r_mid, color, width=1)

        # ── Rotating inner arcs ───────────────
        for i in range(4):
            angle = -self._rot * 180 / math.pi * 1.5 + i * 90
            self._draw_arc(cx, cy, r_mid, angle, 50, COLORS["blue_dim"], width=1)

        # ── Inner glow fill ───────────────────
        glow_r = r_inner * (0.8 + energy * 0.35)
        self._draw_filled_circle(cx, cy, glow_r, color, stipple="gray50")
        self._draw_ring(cx, cy, r_inner, color, width=1)

        # ── Core ─────────────────────────────
        core_r = r_core * (0.9 + pulse * 0.2)
        self._draw_filled_circle(cx, cy, core_r, color)

        # ── Center text ──────────────────────
        state_glyphs = {
            "idle":      "◈",
            "listening": "◉",
            "thinking":  "◎",
            "speaking":  "◈",
            "error":     "✕",
        }
        self.create_text(
            cx, cy,
            text=state_glyphs.get(self._state, "◈"),
            fill=COLORS["bg_primary"],
            font=FONTS["orb_center"],
        )

        # ── State label ───────────────────────
        self.create_text(
            cx, s - 12,
            text=self._state.upper(),
            fill=color,
            font=FONTS["status"],
        )

    def _draw_ring(self, cx, cy, r, color, width=1, **kwargs):
        self.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=color, width=width, fill="", **kwargs
        )

    def _draw_filled_circle(self, cx, cy, r, color, **kwargs):
        self.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=color, outline="", **kwargs
        )

    def _draw_arc(self, cx, cy, r, start_deg, span_deg, color, width=1):
        pad = self._size * 0.02
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r
        self.create_arc(
            x0, y0, x1, y1,
            start=start_deg, extent=span_deg,
            style=tk.ARC, outline=color, width=width,
        )


# ─────────────────────────────────────────────
# Voice Waveform
# ─────────────────────────────────────────────
class VoiceWaveform(AnimatedCanvas):
    """
    Animated voice waveform bar visualiser.
    Feed amplitude values via update_amplitudes() or auto-animate.
    """

    def __init__(self, parent, bars: int = 40, height: int = 48, color=None, **kwargs):
        super().__init__(parent, width=400, height=height, fps=20, **kwargs)
        self._bars      = bars
        self._h         = height
        self._color     = color or COLORS["cyan_core"]
        self._amplitudes= [0.0] * bars
        self._targets   = [0.0] * bars
        self._phase     = 0.0
        self._active    = False
        self._smoothing = 0.35   # lerp factor per frame

    def set_active(self, active: bool):
        """True = animate, False = decay to silence."""
        self._active = active

    def update_amplitudes(self, values: list):
        """Push a list of 0..1 amplitude values (auto-resampled to bar count)."""
        n = len(values)
        if n == 0:
            return
        for i in range(self._bars):
            src_i = int(i * n / self._bars)
            self._targets[i] = max(0.0, min(1.0, values[src_i]))

    def _draw_frame(self):
        self._phase += 0.08

        # Smooth targets
        for i in range(self._bars):
            if self._active:
                # Organic idle wave when no real data
                wave = (math.sin(self._phase + i * 0.4) * 0.5 + 0.5) * 0.45
                self._targets[i] = max(self._targets[i], wave * 0.3)

            # Lerp towards target, then decay
            self._amplitudes[i] += (self._targets[i] - self._amplitudes[i]) * self._smoothing
            self._targets[i] *= 0.88   # decay

        self.delete("all")
        w     = self.winfo_width() or 400
        gap   = 2
        bar_w = max(2, (w - gap * (self._bars + 1)) / self._bars)
        mid   = self._h / 2

        for i, amp in enumerate(self._amplitudes):
            x    = gap + i * (bar_w + gap)
            half = max(2, amp * (self._h / 2 - 2))
            y0   = mid - half
            y1   = mid + half

            # Color gradient: dim at edges, bright at center
            center_dist = abs(i - self._bars / 2) / (self._bars / 2)
            alpha_color = COLORS["cyan_dim"] if center_dist > 0.7 else self._color

            self.create_rectangle(
                x, y0, x + bar_w, y1,
                fill=alpha_color, outline="",
            )

        # Center line
        self.create_line(0, mid, w, mid, fill=COLORS["border_dim"], width=1)


# ─────────────────────────────────────────────
# Ring Gauge (CPU / RAM / GPU)
# ─────────────────────────────────────────────
class RingGauge(AnimatedCanvas):
    """
    Animated ring gauge for system metrics.
    Shows percentage as arc fill with animated needle.
    """

    def __init__(self, parent, label: str = "CPU", size: int = 80,
                 color=None, warn_at: float = 75, crit_at: float = 90, **kwargs):
        super().__init__(parent, width=size, height=size, fps=10, **kwargs)
        self._label    = label
        self._size     = size
        self._color    = color or COLORS["cyan_core"]
        self._warn_at  = warn_at
        self._crit_at  = crit_at
        self._value    = 0.0   # current display value (smoothed)
        self._target   = 0.0   # real value
        self._pulse    = 0.0

    def set_value(self, pct: float):
        """Set percentage 0..100."""
        self._target = max(0.0, min(100.0, pct))

    def _draw_frame(self):
        self._value += (self._target - self._value) * 0.15
        self._pulse  = (self._pulse + 0.1) % (math.pi * 2)

        self.delete("all")
        s  = self._size
        cx = s / 2
        cy = s / 2
        r  = s * 0.38
        r2 = s * 0.30

        pct   = self._value
        color = (self._color if pct < self._warn_at
                 else COLORS["amber_warn"] if pct < self._crit_at
                 else COLORS["red_alert"])

        # Background ring
        self.create_arc(
            cx-r, cy-r, cx+r, cy+r,
            start=135, extent=270,
            style=tk.ARC, outline=COLORS["border_dim"], width=4,
        )

        # Value arc
        sweep = (pct / 100.0) * 270
        if sweep > 1:
            self.create_arc(
                cx-r, cy-r, cx+r, cy+r,
                start=135, extent=sweep,
                style=tk.ARC, outline=color, width=4,
            )

        # Inner ring
        self.create_arc(
            cx-r2, cy-r2, cx+r2, cy+r2,
            start=0, extent=360,
            style=tk.ARC, outline=COLORS["border_normal"], width=1,
        )

        # Value text
        self.create_text(
            cx, cy - 4,
            text=f"{int(pct)}%",
            fill=color,
            font=("Courier New", max(7, int(s * 0.13)), "bold"),
        )
        self.create_text(
            cx, cy + 10,
            text=self._label,
            fill=COLORS["text_secondary"],
            font=("Courier New", max(6, int(s * 0.10))),
        )


# ─────────────────────────────────────────────
# Listening Animation (radial pulse rings)
# ─────────────────────────────────────────────
class ListeningPulse(AnimatedCanvas):
    """Concentric expanding rings that animate when JARVIS is listening."""

    def __init__(self, parent, size: int = 60, color=None, **kwargs):
        super().__init__(parent, width=size, height=size, fps=20, **kwargs)
        self._size   = size
        self._color  = color or COLORS["green_active"]
        self._active = False
        self._rings  = [{"r": size * 0.12, "alpha": 1.0}]
        self._tick_c = 0

    def set_active(self, active: bool):
        self._active = active

    def _draw_frame(self):
        self._tick_c += 1
        cx = cy = self._size / 2
        max_r = self._size * 0.46

        # Spawn new ring every 12 frames when active
        if self._active and self._tick_c % 12 == 0:
            self._rings.append({"r": self._size * 0.12, "alpha": 0.9})

        # Update rings
        updated = []
        for ring in self._rings:
            ring["r"]     += 1.4
            ring["alpha"] -= 0.025
            if ring["alpha"] > 0 and ring["r"] < max_r:
                updated.append(ring)
        self._rings = updated

        self.delete("all")

        # Static center dot
        self.create_oval(
            cx - 5, cy - 5, cx + 5, cy + 5,
            fill=self._color if self._active else COLORS["cyan_ghost"],
            outline="",
        )

        # Animated rings
        for ring in self._rings:
            r = ring["r"]
            # Approximate alpha with stipple
            stipple = "gray75" if ring["alpha"] > 0.6 else "gray50" if ring["alpha"] > 0.3 else "gray25"
            self.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=self._color, width=1, stipple=stipple,
            )


# ─────────────────────────────────────────────
# Status Indicator LED
# ─────────────────────────────────────────────
class StatusLED(tk.Canvas):
    """Small colored LED dot with optional blink."""

    COLORS_MAP = {
        "online":      COLORS["green_active"],
        "listening":   COLORS["green_active"],
        "thinking":    COLORS["purple_ai"],
        "speaking":    COLORS["cyan_bright"],
        "offline":     COLORS["red_alert"],
        "warning":     COLORS["amber_warn"],
        "idle":        COLORS["cyan_dim"],
    }

    def __init__(self, parent, size: int = 10, **kwargs):
        super().__init__(parent, width=size, height=size,
                         bg=COLORS["bg_panel"], highlightthickness=0, bd=0, **kwargs)
        self._size   = size
        self._status = "idle"
        self._blink  = False
        self._bright = True
        self._job    = None
        self._draw()

    def set_status(self, status: str, blink: bool = False):
        self._status = status
        self._blink  = blink
        if blink and self._job is None:
            self._start_blink()
        elif not blink:
            if self._job:
                self.after_cancel(self._job)
                self._job = None
            self._bright = True
            self._draw()

    def _start_blink(self):
        self._bright = not self._bright
        self._draw()
        self._job = self.after(500, self._start_blink)

    def _draw(self):
        self.delete("all")
        color = self.COLORS_MAP.get(self._status, COLORS["text_dim"])
        r = self._size / 2 - 1
        cx = cy = self._size / 2
        # Outer ring (dim)
        self.create_oval(cx-r, cy-r, cx+r, cy+r,
                         fill=COLORS["bg_card"], outline=color, width=1)
        # Inner fill
        if self._bright:
            ir = r * 0.6
            self.create_oval(cx-ir, cy-ir, cx+ir, cy+ir,
                             fill=color, outline="")
