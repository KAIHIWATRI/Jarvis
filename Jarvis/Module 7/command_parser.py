"""
command_parser.py — JARVIS Command Parser
Rule-based NLP parser: maps natural-language strings to structured
(Intent, slots) objects without any external NLP library.

Supports fuzzy matching, synonyms, and slot extraction via regex.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger("JARVIS.Parser")


# ─────────────────────────────────────────────
# Intent catalogue
# ─────────────────────────────────────────────
class Intent(Enum):
    OPEN_APP       = auto()
    CLOSE_APP      = auto()
    GOOGLE_SEARCH  = auto()
    YOUTUBE_SEARCH = auto()
    OPEN_URL       = auto()
    TYPE_TEXT      = auto()
    PRESS_KEY      = auto()
    HOTKEY         = auto()
    SCREENSHOT     = auto()
    SET_VOLUME     = auto()
    GET_VOLUME     = auto()
    SYSTEM_STATS   = auto()
    KILL_PROCESS   = auto()
    SCROLL         = auto()
    CLICK          = auto()
    MOVE_MOUSE     = auto()
    CLOSE_BROWSER  = auto()
    UNKNOWN        = auto()


# ─────────────────────────────────────────────
# Parsed command
# ─────────────────────────────────────────────
@dataclass
class ParsedCommand:
    raw:    str
    intent: Intent
    slots:  dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


# ─────────────────────────────────────────────
# App name aliases
# ─────────────────────────────────────────────
APP_ALIASES: dict[str, str] = {
    # Browsers
    "chrome":    "chrome",
    "google chrome": "chrome",
    "firefox":   "firefox",
    "mozilla":   "firefox",
    "edge":      "msedge",
    "microsoft edge": "msedge",
    "safari":    "safari",
    "brave":     "brave",
    # Editors
    "notepad":   "notepad",
    "notepad++": "notepad++",
    "vscode":    "code",
    "vs code":   "code",
    "visual studio code": "code",
    "sublime":   "sublime_text",
    "vim":       "vim",
    # Office
    "word":      "WINWORD",
    "excel":     "EXCEL",
    "powerpoint":"POWERPNT",
    "outlook":   "OUTLOOK",
    # Media
    "vlc":       "vlc",
    "spotify":   "spotify",
    "itunes":    "iTunes",
    # System
    "task manager": "taskmgr",
    "explorer":  "explorer",
    "file explorer": "explorer",
    "terminal":  "cmd",
    "cmd":       "cmd",
    "command prompt": "cmd",
    "powershell":"powershell",
    "calculator":"calc",
    "paint":     "mspaint",
    "snipping tool": "SnippingTool",
    # Comms
    "discord":   "discord",
    "slack":     "slack",
    "teams":     "teams",
    "zoom":      "zoom",
    "skype":     "skype",
    "whatsapp":  "WhatsApp",
    # Dev
    "pycharm":   "pycharm",
    "intellij":  "idea",
    "postman":   "Postman",
    "docker":    "docker",
}

# Key name normalisation
KEY_ALIASES: dict[str, str] = {
    "enter":     "enter",
    "return":    "enter",
    "space":     "space",
    "spacebar":  "space",
    "backspace":  "backspace",
    "delete":    "delete",
    "del":       "delete",
    "escape":    "esc",
    "esc":       "esc",
    "tab":       "tab",
    "up":        "up",
    "down":      "down",
    "left":      "left",
    "right":     "right",
    "home":      "home",
    "end":       "end",
    "page up":   "pageup",
    "page down": "pagedown",
    "print screen": "printscreen",
    "prtsc":     "printscreen",
    "f1":"f1","f2":"f2","f3":"f3","f4":"f4","f5":"f5","f6":"f6",
    "f7":"f7","f8":"f8","f9":"f9","f10":"f10","f11":"f11","f12":"f12",
}


# ─────────────────────────────────────────────
# Rule engine
# ─────────────────────────────────────────────
class CommandParser:
    """
    Pattern-based command parser.
    Rules are evaluated in priority order; first match wins.
    """

    def __init__(self):
        # Each rule: (compiled_regex, intent, slot_extractor_fn)
        self._rules = self._build_rules()
        logger.debug("CommandParser ready with %d rules.", len(self._rules))

    # ── Public ────────────────────────────────

    def parse(self, raw: str) -> ParsedCommand:
        text = raw.strip().lower()

        for pattern, intent, extractor in self._rules:
            m = pattern.search(text)
            if m:
                slots = extractor(m, text) if extractor else {}
                cmd = ParsedCommand(raw=raw, intent=intent, slots=slots)
                logger.debug("Matched rule → %s %s", intent.name, slots)
                return cmd

        return ParsedCommand(raw=raw, intent=Intent.UNKNOWN, confidence=0.0)

    # ── Rule builders ─────────────────────────

    def _build_rules(self):
        """Return list of (pattern, intent, extractor) tuples, priority-ordered."""
        return [
            # ── Browser close ─────────────────
            (
                re.compile(r"\b(close|quit|exit)\s+(browser|chrome|firefox|edge|safari)\b"),
                Intent.CLOSE_BROWSER,
                None,
            ),

            # ── YouTube search ────────────────
            (
                re.compile(
                    r"\b(search\s+youtube|youtube\s+search|play\s+on\s+youtube|"
                    r"find\s+on\s+youtube|youtube)\b"
                ),
                Intent.YOUTUBE_SEARCH,
                self._extract_youtube_query,
            ),

            # ── Google search ─────────────────
            (
                re.compile(
                    r"\b(google|search\s+(google|the\s+web|online|for)|"
                    r"look\s+up|find\s+online|search)\b"
                ),
                Intent.GOOGLE_SEARCH,
                self._extract_google_query,
            ),

            # ── Open URL ──────────────────────
            (
                re.compile(r"\b(open|go\s+to|navigate\s+to|visit|browse\s+to)\s+"
                           r"(https?://|www\.)\S+"),
                Intent.OPEN_URL,
                self._extract_url,
            ),
            (
                re.compile(r"\bhttps?://\S+"),
                Intent.OPEN_URL,
                self._extract_url_bare,
            ),

            # ── Open app ──────────────────────
            (
                re.compile(r"\b(open|launch|start|run)\s+(.+)"),
                Intent.OPEN_APP,
                self._extract_open_app,
            ),

            # ── Close app ─────────────────────
            (
                re.compile(r"\b(close|kill|quit|exit|stop)\s+(.+)"),
                Intent.CLOSE_APP,
                self._extract_close_app,
            ),

            # ── Type text ─────────────────────
            (
                re.compile(r"\b(type|write|input|enter\s+text|paste)\s+(.+)"),
                Intent.TYPE_TEXT,
                self._extract_type_text,
            ),

            # ── Hotkey combos ─────────────────
            (
                re.compile(
                    r"\b(press|hit|use)\s+(ctrl|control|alt|shift|win|cmd)\s*\+\s*\S+"
                ),
                Intent.HOTKEY,
                self._extract_hotkey,
            ),

            # ── Press single key ─────────────
            (
                re.compile(r"\b(press|hit|push)\s+(.+?)\s*(key|button)?\s*$"),
                Intent.PRESS_KEY,
                self._extract_press_key,
            ),

            # ── Screenshot ────────────────────
            (
                re.compile(r"\b(take\s+a?\s*(screenshot|screen\s+shot|screen\s+grab)|"
                           r"capture\s+(screen|desktop)|screenshot)\b"),
                Intent.SCREENSHOT,
                self._extract_screenshot,
            ),

            # ── Set volume ────────────────────
            (
                re.compile(r"\b(set|change|put|turn)\s+volume\s+(to|at)?\s*(\d+)"),
                Intent.SET_VOLUME,
                self._extract_set_volume,
            ),
            (
                re.compile(r"\bvolume\s+(\d+)\b"),
                Intent.SET_VOLUME,
                lambda m, t: {"level": int(m.group(1))},
            ),
            (
                re.compile(r"\b(mute|unmute|volume\s+up|increase\s+volume|"
                           r"volume\s+down|decrease\s+volume|lower\s+volume)\b"),
                Intent.SET_VOLUME,
                self._extract_volume_cmd,
            ),

            # ── Get volume ────────────────────
            (
                re.compile(r"\b(what.s\s+the\s+volume|get\s+volume|current\s+volume|"
                           r"volume\s+level)\b"),
                Intent.GET_VOLUME,
                None,
            ),

            # ── System stats ──────────────────
            (
                re.compile(
                    r"\b(system\s+stats|cpu|memory|ram|disk\s+usage|"
                    r"system\s+(info|status|monitor|report)|"
                    r"how\s+(much|many)\s+(memory|ram|cpu|disk))\b"
                ),
                Intent.SYSTEM_STATS,
                None,
            ),

            # ── Kill process ──────────────────
            (
                re.compile(r"\b(kill|terminate|end|force\s+close)\s+(process\s+)?(.+)"),
                Intent.KILL_PROCESS,
                self._extract_kill_process,
            ),

            # ── Scroll ────────────────────────
            (
                re.compile(r"\b(scroll)\s+(up|down|left|right)(\s+(\d+))?\b"),
                Intent.SCROLL,
                self._extract_scroll,
            ),

            # ── Click ─────────────────────────
            (
                re.compile(r"\b(click|left\s+click|right\s+click|double\s+click)"
                           r"(\s+at\s+(\d+)\s*,\s*(\d+))?\b"),
                Intent.CLICK,
                self._extract_click,
            ),

            # ── Move mouse ────────────────────
            (
                re.compile(r"\b(move\s+mouse|move\s+cursor)\s+(to\s+)?(\d+)\s*,\s*(\d+)\b"),
                Intent.MOVE_MOUSE,
                self._extract_move_mouse,
            ),
        ]

    # ── Slot extractors ───────────────────────

    @staticmethod
    def _extract_google_query(m: re.Match, text: str) -> dict:
        patterns = [
            r"(?:search\s+(?:google|the\s+web|online)?\s*for\s+)(.+)",
            r"(?:google|look\s+up|find\s+online|search)\s+(.+)",
            r"search\s+(.+)",
        ]
        for p in patterns:
            mm = re.search(p, text)
            if mm:
                return {"query": mm.group(1).strip()}
        return {"query": text}

    @staticmethod
    def _extract_youtube_query(m: re.Match, text: str) -> dict:
        patterns = [
            r"(?:search\s+youtube\s+for|youtube\s+search\s+for|"
            r"play\s+on\s+youtube|find\s+on\s+youtube|youtube)\s+(.+)",
        ]
        for p in patterns:
            mm = re.search(p, text)
            if mm:
                return {"query": mm.group(1).strip()}
        return {"query": text.replace("youtube", "").strip()}

    @staticmethod
    def _extract_url(m: re.Match, text: str) -> dict:
        mm = re.search(r"(https?://\S+|www\.\S+)", text)
        url = mm.group(1) if mm else text
        if url.startswith("www."):
            url = "https://" + url
        return {"url": url}

    @staticmethod
    def _extract_url_bare(m: re.Match, text: str) -> dict:
        return {"url": m.group(0)}

    @staticmethod
    def _extract_open_app(m: re.Match, text: str) -> dict:
        raw = m.group(2).strip()
        # Remove trailing noise words
        raw = re.sub(r"\s+(please|now|app|application|program|software)$", "", raw)
        normalised = APP_ALIASES.get(raw.lower(), raw)
        return {"app_name": normalised, "raw_name": raw}

    @staticmethod
    def _extract_close_app(m: re.Match, text: str) -> dict:
        raw = m.group(2).strip()
        raw = re.sub(r"\s+(please|now|app|application|program|software)$", "", raw)
        # Strip "close browser" → handled by separate rule
        normalised = APP_ALIASES.get(raw.lower(), raw)
        return {"app_name": normalised, "raw_name": raw}

    @staticmethod
    def _extract_type_text(m: re.Match, text: str) -> dict:
        raw_text = m.group(2).strip()
        # Strip surrounding quotes if present
        raw_text = re.sub(r'^["\']|["\']$', '', raw_text)
        return {"text": raw_text}

    @staticmethod
    def _extract_hotkey(m: re.Match, text: str) -> dict:
        # Extract all modifier + key parts
        combo = re.sub(r"\b(press|hit|use)\s+", "", text)
        parts = [p.strip() for p in re.split(r"\s*\+\s*", combo) if p.strip()]
        key_map = {"control": "ctrl", "cmd": "ctrl", "win": "winleft"}
        normalised = [key_map.get(k.lower(), k.lower()) for k in parts]
        return {"keys": normalised}

    @staticmethod
    def _extract_press_key(m: re.Match, text: str) -> dict:
        raw_key = m.group(2).strip()
        normalised = KEY_ALIASES.get(raw_key.lower(), raw_key.lower())
        return {"key": normalised}

    @staticmethod
    def _extract_screenshot(m: re.Match, text: str) -> dict:
        # Check if filename mentioned
        fn_match = re.search(r"(?:save\s+as|named?|file\s*name)\s+(\S+)", text)
        return {"filename": fn_match.group(1) if fn_match else None}

    @staticmethod
    def _extract_set_volume(m: re.Match, text: str) -> dict:
        level = int(m.group(3))
        return {"level": max(0, min(100, level))}

    @staticmethod
    def _extract_volume_cmd(m: re.Match, text: str) -> dict:
        cmd = m.group(1).lower()
        if "mute" in cmd:
            return {"level": 0, "action": "mute"}
        if "unmute" in cmd:
            return {"action": "unmute"}
        if "up" in cmd or "increase" in cmd:
            return {"action": "up", "delta": 10}
        if "down" in cmd or "decrease" in cmd or "lower" in cmd:
            return {"action": "down", "delta": 10}
        return {}

    @staticmethod
    def _extract_kill_process(m: re.Match, text: str) -> dict:
        name = m.group(3).strip()
        name = re.sub(r"\s+(please|now)$", "", name)
        return {"process_name": name}

    @staticmethod
    def _extract_scroll(m: re.Match, text: str) -> dict:
        direction = m.group(2).lower()
        amount    = int(m.group(4)) if m.group(4) else 3
        return {"direction": direction, "amount": amount}

    @staticmethod
    def _extract_click(m: re.Match, text: str) -> dict:
        button = "right" if "right" in m.group(1) else "left"
        double = "double" in m.group(1)
        x = int(m.group(3)) if m.group(3) else None
        y = int(m.group(4)) if m.group(4) else None
        return {"button": button, "double": double, "x": x, "y": y}

    @staticmethod
    def _extract_move_mouse(m: re.Match, text: str) -> dict:
        return {"x": int(m.group(3)), "y": int(m.group(4))}
