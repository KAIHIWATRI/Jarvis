"""
jarvis_automation_demo.py — JARVIS Automation Demo
Shows how to connect the automation engine to the full JARVIS pipeline
(Whisper STT → intent → automation → TTS response).
"""

from __future__ import annotations

import asyncio
import logging
import sys

# ── Demo-mode flag: set False to use real engine ──────────────────────────────
DEMO_MODE = True   # True = mock actions (no real browser/GUI needed)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("JARVIS.Demo")


# ─────────────────────────────────────────────
# Mock engine for demonstration without deps
# ─────────────────────────────────────────────
class MockAutomationEngine:
    """Simulates the real AutomationEngine for testing."""

    class MockResult:
        def __init__(self, success, message, intent=None, data=None):
            self.success  = success
            self.message  = message
            self.intent   = intent
            self.data     = data
            self.duration = 0.05
            self.error    = None

        def __str__(self):
            return f"[{'✓' if self.success else '✗'}] {self.message}"

    MOCK_RESPONSES = {
        "open":       ("OPEN_APP",      True,  "Launched {}"),
        "close":      ("CLOSE_APP",     True,  "Closed {}"),
        "search":     ("GOOGLE_SEARCH", True,  "Searched Google for: {}"),
        "youtube":    ("YOUTUBE_SEARCH",True,  "Searched YouTube for: {}"),
        "type":       ("TYPE_TEXT",     True,  "Typed: {}"),
        "screenshot": ("SCREENSHOT",    True,  "Screenshot saved: jarvis_screenshot.png"),
        "volume":     ("SET_VOLUME",    True,  "Volume set to {}%"),
        "system":     ("SYSTEM_STATS",  True,  "CPU 12% | RAM 41% | Disk 44%"),
    }

    def execute(self, command: str):
        cmd_lower = command.lower()
        for keyword, (intent, ok, tmpl) in self.MOCK_RESPONSES.items():
            if keyword in cmd_lower:
                # Extract payload for template
                words = cmd_lower.split()
                try:
                    idx = words.index(keyword)
                    payload = " ".join(words[idx+1:]) if idx + 1 < len(words) else keyword
                except ValueError:
                    payload = keyword
                try:
                    msg = tmpl.format(payload)
                except Exception:
                    msg = tmpl
                return self.MockResult(ok, msg, intent=intent)
        return self.MockResult(False, f"Unknown command: {command}", intent="UNKNOWN")

    def shutdown(self):
        pass


# ─────────────────────────────────────────────
# JARVIS Automation Controller
# ─────────────────────────────────────────────
class JARVISAutomationController:
    """
    Bridges the AutomationEngine with JARVIS's TTS + STT pipeline.
    Dispatches voice/text commands and speaks the result.
    """

    # Map automation intents → spoken response templates
    RESPONSE_TEMPLATES = {
        "OPEN_APP":       "Done. {}",
        "CLOSE_APP":      "Closed. {}",
        "GOOGLE_SEARCH":  "{}",
        "YOUTUBE_SEARCH": "{}",
        "OPEN_URL":       "Navigating. {}",
        "TYPE_TEXT":      "Text entered.",
        "PRESS_KEY":      "Key pressed.",
        "HOTKEY":         "Hotkey executed.",
        "SCREENSHOT":     "{}",
        "SET_VOLUME":     "{}",
        "GET_VOLUME":     "{}",
        "SYSTEM_STATS":   "{}",
        "KILL_PROCESS":   "{}",
        "SCROLL":         "Scrolled.",
        "CLICK":          "Clicked.",
        "MOVE_MOUSE":     "Mouse moved.",
        "CLOSE_BROWSER":  "Browser closed.",
        "UNKNOWN":        "I'm sorry, I don't know how to do that.",
    }

    def __init__(self, engine=None, tts_engine=None):
        if engine is None:
            if DEMO_MODE:
                self._engine = MockAutomationEngine()
            else:
                from automation_engine import AutomationEngine
                self._engine = AutomationEngine(on_event=self._on_automation_event)
        else:
            self._engine = engine

        self._tts = tts_engine   # optional TTSEngine from tts_engine.py
        logger.info("JARVISAutomationController ready (demo=%s).", DEMO_MODE)

    def handle_command(self, command: str) -> str:
        """
        Execute an automation command and return a spoken response string.
        Designed to be called from the JARVIS chat/voice loop.
        """
        logger.info("Handling automation command: %r", command)
        result = self._engine.execute(command)

        if self._tts:
            self._speak(result.message)

        return result.message

    async def handle_command_async(self, command: str) -> str:
        """Async wrapper for use in async JARVIS pipelines."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.handle_command, command)

    def _speak(self, text: str):
        if self._tts:
            try:
                self._tts.speak(text)
            except Exception as exc:
                logger.debug("TTS error: %s", exc)

    def _on_automation_event(self, event: str, data: dict):
        logger.debug("Automation event: %s %s", event, data)

    def shutdown(self):
        self._engine.shutdown()


# ─────────────────────────────────────────────
# Command detection helper
# ─────────────────────────────────────────────
AUTOMATION_TRIGGERS = {
    "open", "launch", "start", "close", "search", "google",
    "youtube", "type", "press", "hotkey", "screenshot", "volume",
    "cpu", "ram", "memory", "system", "kill", "scroll", "click",
    "move mouse", "navigate", "go to", "look up", "find",
}

def is_automation_command(text: str) -> bool:
    """
    Quick check: does this text likely require desktop automation?
    Used by the JARVIS chat loop to route commands.
    """
    lower = text.lower()
    return any(trigger in lower for trigger in AUTOMATION_TRIGGERS)


# ─────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────
def run_demo():
    controller = JARVISAutomationController()

    demo_commands = [
        "open notepad",
        "search google for Python automation tutorials",
        "search youtube for lo-fi coding music",
        "set volume to 65",
        "system stats",
        "take a screenshot",
        "type Hello, I am JARVIS",
        "press enter",
        "scroll down 3",
        "close notepad",
        "what's the volume",
    ]

    print("\n" + "="*55)
    print("  JARVIS Automation Engine — Demo")
    print("="*55 + "\n")

    for cmd in demo_commands:
        print(f"  > {cmd}")
        response = controller.handle_command(cmd)
        print(f"    → {response}\n")

    print("─"*55)
    print("  Interactive mode (type 'quit' to exit)")
    print("─"*55)

    while True:
        try:
            user_input = input("\n  JARVIS > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input:
            response = controller.handle_command(user_input)
            print(f"    → {response}")

    controller.shutdown()
    print("\n  JARVIS automation engine shut down.")


if __name__ == "__main__":
    run_demo()
