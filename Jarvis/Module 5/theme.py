"""
theme.py — JARVIS Cyberpunk Design System
All colors, fonts, and layout constants for the JARVIS UI.
"""

# ─────────────────────────────────────────────
# Color Palette  (JARVIS cyan-on-dark aesthetic)
# ─────────────────────────────────────────────
COLORS = {
    # Backgrounds
    "bg_void":        "#050810",   # Deepest background
    "bg_primary":     "#080D1A",   # Main window bg
    "bg_panel":       "#0C1424",   # Panel surfaces
    "bg_card":        "#0F1A2E",   # Card surfaces
    "bg_input":       "#0A1220",   # Input fields
    "bg_hover":       "#162038",   # Hover state

    # Primary — JARVIS cyan
    "cyan_bright":    "#00FFFF",   # Active glow / accent
    "cyan_core":      "#00D4D4",   # Primary interactive
    "cyan_dim":       "#008080",   # Inactive / muted
    "cyan_ghost":     "#003333",   # Subtle tint

    # Secondary — Arc blue
    "blue_bright":    "#4DC3FF",   # Info highlights
    "blue_core":      "#1E90FF",   # Secondary interactive
    "blue_dim":       "#0D4A80",   # Inactive

    # Status
    "green_active":   "#00FF88",   # Online / active
    "amber_warn":     "#FFB800",   # Warning
    "red_alert":      "#FF3D3D",   # Error / alert
    "purple_ai":      "#A855F7",   # AI processing

    # Text
    "text_primary":   "#E0F0FF",   # Main readable text
    "text_secondary": "#7BA8CC",   # Muted label text
    "text_dim":       "#3D6080",   # Very muted / hint text
    "text_accent":    "#00FFFF",   # Cyan accent text

    # Borders
    "border_bright":  "#00CCCC",   # Active borders
    "border_normal":  "#0D3D4D",   # Normal borders
    "border_dim":     "#071828",   # Subtle separators
}

# ─────────────────────────────────────────────
# Typography
# ─────────────────────────────────────────────
FONTS = {
    "title":        ("Courier New", 13, "bold"),
    "title_lg":     ("Courier New", 16, "bold"),
    "heading":      ("Courier New", 11, "bold"),
    "body":         ("Courier New", 10),
    "body_sm":      ("Courier New", 9),
    "mono":         ("Courier New", 9),
    "status":       ("Courier New", 8, "bold"),
    "label":        ("Courier New", 8),
    "orb_center":   ("Courier New", 11, "bold"),

    # Fallback sans for chat
    "chat_user":    ("Arial", 10),
    "chat_ai":      ("Arial", 10),
    "chat_label":   ("Arial", 8, "bold"),
}

# ─────────────────────────────────────────────
# Layout Constants
# ─────────────────────────────────────────────
LAYOUT = {
    "window_width":      1100,
    "window_height":     720,
    "window_min_w":      800,
    "window_min_h":      560,

    "sidebar_width":     220,
    "header_height":     52,
    "statusbar_height":  28,

    "orb_size":          180,
    "orb_ring_gap":      8,

    "corner_radius":     8,
    "corner_radius_sm":  4,
    "border_width":      1,

    "pad_x":             12,
    "pad_y":             8,
    "gap":               8,

    "waveform_bars":     40,
    "waveform_height":   48,

    "update_ms":         50,    # animation frame budget (20 fps)
    "sys_poll_ms":       2000,  # system stats refresh
}

# ─────────────────────────────────────────────
# CustomTkinter appearance overrides
# ─────────────────────────────────────────────
CTK_THEME = {
    "CTk": {
        "fg_color": [COLORS["bg_primary"], COLORS["bg_primary"]],
    },
    "CTkFrame": {
        "fg_color": [COLORS["bg_panel"], COLORS["bg_panel"]],
        "border_color": [COLORS["border_normal"], COLORS["border_normal"]],
        "border_width": 1,
        "corner_radius": LAYOUT["corner_radius"],
    },
    "CTkButton": {
        "fg_color":          [COLORS["bg_card"], COLORS["bg_card"]],
        "hover_color":       [COLORS["bg_hover"], COLORS["bg_hover"]],
        "border_color":      [COLORS["cyan_dim"], COLORS["cyan_dim"]],
        "text_color":        [COLORS["cyan_core"], COLORS["cyan_core"]],
        "border_width":      1,
        "corner_radius":     LAYOUT["corner_radius_sm"],
    },
    "CTkLabel": {
        "text_color": [COLORS["text_primary"], COLORS["text_primary"]],
        "fg_color":   "transparent",
    },
    "CTkEntry": {
        "fg_color":      [COLORS["bg_input"], COLORS["bg_input"]],
        "border_color":  [COLORS["border_normal"], COLORS["border_normal"]],
        "text_color":    [COLORS["text_primary"], COLORS["text_primary"]],
        "placeholder_text_color": [COLORS["text_dim"], COLORS["text_dim"]],
    },
    "CTkScrollbar": {
        "fg_color":    [COLORS["bg_panel"], COLORS["bg_panel"]],
        "button_color":[COLORS["cyan_dim"], COLORS["cyan_dim"]],
    },
    "CTkSlider": {
        "fg_color":       [COLORS["bg_card"], COLORS["bg_card"]],
        "progress_color": [COLORS["cyan_dim"], COLORS["cyan_dim"]],
        "button_color":   [COLORS["cyan_core"], COLORS["cyan_core"]],
    },
    "CTkOptionMenu": {
        "fg_color":    [COLORS["bg_card"], COLORS["bg_card"]],
        "button_color":[COLORS["cyan_dim"], COLORS["cyan_dim"]],
        "text_color":  [COLORS["text_primary"], COLORS["text_primary"]],
    },
}

def apply_theme():
    """Apply JARVIS theme to CustomTkinter."""
    import customtkinter as ctk
    import json, os, tempfile

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Write temp theme JSON
    theme_data = {
        "CTkButton": {
            "corner_radius": LAYOUT["corner_radius_sm"],
            "border_width": 1,
            "fg_color": [COLORS["bg_card"], COLORS["bg_card"]],
            "hover_color": [COLORS["bg_hover"], COLORS["bg_hover"]],
            "border_color": [COLORS["cyan_dim"], COLORS["cyan_dim"]],
            "text_color": [COLORS["cyan_core"], COLORS["cyan_core"]],
            "text_color_disabled": [COLORS["text_dim"], COLORS["text_dim"]],
        },
    }
    # CustomTkinter color theme is applied at widget level; we store globally
    return COLORS
