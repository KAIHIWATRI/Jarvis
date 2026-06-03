"""
browser_manager.py — JARVIS Selenium Browser Manager
Manages Chrome/Firefox via Selenium with auto-retry, wait strategies,
and high-level helpers for Google, YouTube, and general browsing.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import quote_plus

logger = logging.getLogger("JARVIS.Browser")


# ─────────────────────────────────────────────
# Selenium imports (graceful degradation)
# ─────────────────────────────────────────────
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.common.exceptions import (
        WebDriverException, TimeoutException, NoSuchElementException,
        ElementNotInteractableException,
    )
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("selenium not installed — browser automation unavailable.")


# ─────────────────────────────────────────────
# Browser Manager
# ─────────────────────────────────────────────
class BrowserManager:
    """
    Selenium-based browser controller.

    Supports Chrome and Firefox with automatic ChromeDriver/GeckoDriver
    management via selenium-manager (Selenium 4.6+) or webdriver-manager.

    Usage
    -----
    bm = BrowserManager(browser="chrome", headless=False)
    bm.google_search("Python tutorials")
    bm.youtube_search("lo-fi beats")
    bm.open_url("https://github.com")
    bm.quit()
    """

    # Common CSS / XPath selectors (updated periodically)
    _SELECTORS = {
        "google_search_input": [
            (By.NAME, "q"),
            (By.CSS_SELECTOR, "input[type='search']"),
            (By.CSS_SELECTOR, "textarea[name='q']"),
        ],
        "youtube_search_input": [
            (By.NAME, "search_query"),
            (By.CSS_SELECTOR, "input#search"),
            (By.CSS_SELECTOR, "ytd-searchbox input"),
        ],
        "youtube_search_btn": [
            (By.ID, "search-icon-legacy"),
            (By.CSS_SELECTOR, "button[aria-label='Search']"),
        ],
    }

    def __init__(
        self,
        browser:    str  = "chrome",
        headless:   bool = False,
        timeout:    int  = 15,
        window_size: tuple = (1280, 900),
    ):
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("selenium is not installed. Run: pip install selenium")

        self._browser_name = browser.lower()
        self._headless     = headless
        self._timeout      = timeout
        self._window_size  = window_size
        self._driver: Optional[webdriver.Remote] = None
        self._init_driver()

    # ── Driver init ───────────────────────────

    def _init_driver(self):
        logger.info("Starting %s browser (headless=%s)…", self._browser_name, self._headless)
        try:
            if self._browser_name == "chrome":
                self._driver = self._make_chrome()
            elif self._browser_name in ("firefox", "gecko"):
                self._driver = self._make_firefox()
            elif self._browser_name == "edge":
                self._driver = self._make_edge()
            else:
                raise ValueError(f"Unsupported browser: {self._browser_name}")

            self._driver.set_window_size(*self._window_size)
            logger.info("Browser started: %s", self._browser_name)

        except Exception as exc:
            logger.error("Failed to start browser: %s", exc)
            raise

    def _make_chrome(self):
        opts = ChromeOptions()
        if self._headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--disable-extensions")
        opts.add_argument(f"--window-size={self._window_size[0]},{self._window_size[1]}")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        # Try selenium-manager (Selenium 4.6+) first, fall back to webdriver-manager
        try:
            return webdriver.Chrome(options=opts)
        except Exception:
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                svc = Service(ChromeDriverManager().install())
                return webdriver.Chrome(service=svc, options=opts)
            except ImportError:
                raise RuntimeError(
                    "ChromeDriver not found. Install webdriver-manager: "
                    "pip install webdriver-manager"
                )

    def _make_firefox(self):
        opts = FirefoxOptions()
        if self._headless:
            opts.add_argument("--headless")
        try:
            return webdriver.Firefox(options=opts)
        except Exception:
            try:
                from webdriver_manager.firefox import GeckoDriverManager
                from selenium.webdriver.firefox.service import Service
                svc = Service(GeckoDriverManager().install())
                return webdriver.Firefox(service=svc, options=opts)
            except ImportError:
                raise RuntimeError(
                    "GeckoDriver not found. Install webdriver-manager: "
                    "pip install webdriver-manager"
                )

    def _make_edge(self):
        from selenium.webdriver.edge.options import Options as EdgeOptions
        opts = EdgeOptions()
        if self._headless:
            opts.add_argument("--headless=new")
        return webdriver.Edge(options=opts)

    # ── High-level actions ────────────────────

    def open_url(self, url: str) -> tuple[bool, str]:
        """Navigate to a URL."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            self._ensure_driver()
            self._driver.get(url)
            self._wait_for_load()
            title = self._driver.title
            logger.info("Opened URL: %s (title: %s)", url, title)
            return True, f"Opened {url} — {title}"
        except Exception as exc:
            logger.error("open_url failed: %s", exc)
            return False, f"Failed to open URL: {exc}"

    def google_search(self, query: str) -> tuple[bool, str]:
        """Search Google for a query (URL method — most reliable)."""
        url = f"https://www.google.com/search?q={quote_plus(query)}"
        try:
            self._ensure_driver()
            self._driver.get(url)
            self._wait_for_load()
            logger.info("Google search: %r", query)
            return True, f"Searched Google for: {query}"
        except Exception as exc:
            logger.error("google_search failed: %s", exc)
            return False, f"Google search failed: {exc}"

    def youtube_search(self, query: str) -> tuple[bool, str]:
        """Search YouTube for a query."""
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        try:
            self._ensure_driver()
            self._driver.get(url)
            self._wait_for_load()
            logger.info("YouTube search: %r", query)
            return True, f"Searched YouTube for: {query}"
        except Exception as exc:
            logger.error("youtube_search failed: %s", exc)
            return False, f"YouTube search failed: {exc}"

    def find_and_click(self, selector: str, by: By = By.CSS_SELECTOR) -> tuple[bool, str]:
        """Wait for an element and click it."""
        try:
            el = WebDriverWait(self._driver, self._timeout).until(
                EC.element_to_be_clickable((by, selector))
            )
            el.click()
            return True, f"Clicked element: {selector}"
        except TimeoutException:
            return False, f"Element not found (timeout): {selector}"
        except Exception as exc:
            return False, f"Click failed: {exc}"

    def find_and_type(self, selector: str, text: str,
                      by: By = By.CSS_SELECTOR, clear: bool = True) -> tuple[bool, str]:
        """Wait for an input element and type into it."""
        try:
            el = WebDriverWait(self._driver, self._timeout).until(
                EC.element_to_be_clickable((by, selector))
            )
            if clear:
                el.clear()
            el.send_keys(text)
            return True, f"Typed into {selector}"
        except TimeoutException:
            return False, f"Input not found (timeout): {selector}"
        except Exception as exc:
            return False, f"Type failed: {exc}"

    def get_page_text(self) -> str:
        """Return visible text of current page."""
        try:
            body = self._driver.find_element(By.TAG_NAME, "body")
            return body.text
        except Exception:
            return ""

    def take_browser_screenshot(self, path: str) -> tuple[bool, str]:
        """Capture the browser viewport."""
        try:
            self._driver.save_screenshot(path)
            return True, f"Browser screenshot saved: {path}"
        except Exception as exc:
            return False, f"Browser screenshot failed: {exc}"

    def get_current_url(self) -> str:
        try:
            return self._driver.current_url
        except Exception:
            return ""

    def get_title(self) -> str:
        try:
            return self._driver.title
        except Exception:
            return ""

    def back(self):
        self._driver.back()

    def forward(self):
        self._driver.forward()

    def refresh(self):
        self._driver.refresh()

    def scroll_page(self, pixels: int = 500):
        self._driver.execute_script(f"window.scrollBy(0, {pixels});")

    def execute_js(self, script: str, *args):
        """Execute arbitrary JavaScript in the browser context."""
        return self._driver.execute_script(script, *args)

    def quit(self):
        """Close and clean up the browser."""
        if self._driver:
            try:
                self._driver.quit()
                logger.info("Browser quit.")
            except Exception as exc:
                logger.debug("Browser quit error: %s", exc)
            finally:
                self._driver = None

    # ── Internals ─────────────────────────────

    def _ensure_driver(self):
        if self._driver is None:
            self._init_driver()

    def _wait_for_load(self, timeout: int = 10):
        """Wait until document.readyState == 'complete'."""
        try:
            WebDriverWait(self._driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except TimeoutException:
            logger.warning("Page load timed out after %ds.", timeout)

    def _find_element_multi(self, selector_list: list) -> Optional[object]:
        """Try multiple (By, selector) pairs; return first match."""
        for by, sel in selector_list:
            try:
                return self._driver.find_element(by, sel)
            except NoSuchElementException:
                continue
        return None

    def __del__(self):
        self.quit()
