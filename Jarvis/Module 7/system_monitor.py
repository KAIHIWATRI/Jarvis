"""
system_monitor.py — JARVIS System Monitor
psutil-based resource monitor with alerting, history, and formatted reports.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("JARVIS.SysMonitor")

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False
    logger.warning("psutil not installed — system monitoring unavailable.")


# ─────────────────────────────────────────────
# Alert thresholds
# ─────────────────────────────────────────────
@dataclass
class AlertThresholds:
    cpu_warn:    float = 75.0
    cpu_crit:    float = 90.0
    ram_warn:    float = 80.0
    ram_crit:    float = 95.0
    disk_warn:   float = 85.0
    disk_crit:   float = 95.0
    temp_warn:   float = 75.0
    temp_crit:   float = 90.0


# ─────────────────────────────────────────────
# System Monitor
# ─────────────────────────────────────────────
class SystemMonitor:
    """
    Continuous system resource monitor with:
    - Snapshot (get_full_report)
    - Background polling with configurable interval
    - Rolling history (last N samples)
    - Threshold-based alerting
    - JARVIS-formatted text summaries

    Usage
    -----
    monitor = SystemMonitor()
    report  = monitor.get_full_report()
    monitor.start_background(interval=5, on_alert=print)
    monitor.stop_background()
    """

    HISTORY_SIZE = 60   # Keep last 60 samples

    def __init__(self, thresholds: Optional[AlertThresholds] = None):
        self.thresholds = thresholds or AlertThresholds()
        self._history: deque[dict] = deque(maxlen=self.HISTORY_SIZE)
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_alert: Optional[Callable[[str, dict], None]] = None

    # ── Snapshot ──────────────────────────────

    def get_cpu(self) -> dict:
        if not PSUTIL_OK:
            return {"percent": 0, "cores": 0, "freq_mhz": 0}
        try:
            freq    = psutil.cpu_freq()
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            return {
                "percent":     psutil.cpu_percent(interval=0.1),
                "per_core":    per_cpu,
                "cores_phys":  psutil.cpu_count(logical=False),
                "cores_logic": psutil.cpu_count(logical=True),
                "freq_mhz":    round(freq.current, 1) if freq else 0,
                "freq_max":    round(freq.max, 1) if freq else 0,
            }
        except Exception as exc:
            logger.debug("CPU read error: %s", exc)
            return {"percent": 0, "error": str(exc)}

    def get_memory(self) -> dict:
        if not PSUTIL_OK:
            return {"percent": 0, "used_gb": 0, "total_gb": 0}
        try:
            vm  = psutil.virtual_memory()
            swp = psutil.swap_memory()
            return {
                "percent":    vm.percent,
                "used_gb":    round(vm.used / 1024**3, 2),
                "total_gb":   round(vm.total / 1024**3, 2),
                "available_gb": round(vm.available / 1024**3, 2),
                "swap_percent": swp.percent,
                "swap_used_gb": round(swp.used / 1024**3, 2),
            }
        except Exception as exc:
            return {"percent": 0, "error": str(exc)}

    def get_disk(self, path: str = "/") -> dict:
        if not PSUTIL_OK:
            return {"percent": 0}
        if platform.system() == "Windows":
            path = "C:\\"
        try:
            usage = psutil.disk_usage(path)
            io    = psutil.disk_io_counters()
            return {
                "percent":   usage.percent,
                "used_gb":   round(usage.used / 1024**3, 2),
                "total_gb":  round(usage.total / 1024**3, 2),
                "free_gb":   round(usage.free / 1024**3, 2),
                "read_mb":   round(io.read_bytes / 1024**2, 1) if io else 0,
                "write_mb":  round(io.write_bytes / 1024**2, 1) if io else 0,
            }
        except Exception as exc:
            return {"percent": 0, "error": str(exc)}

    def get_network(self) -> dict:
        if not PSUTIL_OK:
            return {}
        try:
            io = psutil.net_io_counters()
            return {
                "bytes_sent_mb":  round(io.bytes_sent / 1024**2, 2),
                "bytes_recv_mb":  round(io.bytes_recv / 1024**2, 2),
                "packets_sent":   io.packets_sent,
                "packets_recv":   io.packets_recv,
                "errors_out":     io.errout,
                "errors_in":      io.errin,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_temperatures(self) -> dict:
        if not PSUTIL_OK or not hasattr(psutil, "sensors_temperatures"):
            return {}
        try:
            temps = psutil.sensors_temperatures()
            result = {}
            for chip, readings in (temps or {}).items():
                result[chip] = [
                    {"label": r.label or f"sensor_{i}", "current": r.current,
                     "high": r.high, "critical": r.critical}
                    for i, r in enumerate(readings)
                ]
            return result
        except Exception:
            return {}

    def get_top_processes(self, n: int = 8) -> list[dict]:
        if not PSUTIL_OK:
            return []
        try:
            procs = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
                try:
                    info = proc.info
                    procs.append({
                        "pid":    info["pid"],
                        "name":   info["name"],
                        "cpu":    round(info["cpu_percent"] or 0, 1),
                        "mem":    round(info["memory_percent"] or 0, 1),
                        "status": info["status"],
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            procs.sort(key=lambda p: p["cpu"], reverse=True)
            return procs[:n]
        except Exception as exc:
            logger.debug("Top processes error: %s", exc)
            return []

    def get_battery(self) -> Optional[dict]:
        if not PSUTIL_OK or not hasattr(psutil, "sensors_battery"):
            return None
        try:
            bat = psutil.sensors_battery()
            if bat is None:
                return None
            return {
                "percent":     round(bat.percent, 1),
                "plugged":     bat.power_plugged,
                "secs_left":   bat.secsleft if bat.secsleft != psutil.POWER_TIME_UNLIMITED else -1,
            }
        except Exception:
            return None

    def get_boot_time(self) -> float:
        if not PSUTIL_OK:
            return 0
        return psutil.boot_time()

    def get_uptime_str(self) -> str:
        elapsed = time.time() - self.get_boot_time()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        return f"{h}h {m}m {s}s"

    def get_full_report(self) -> dict:
        """Collect all stats into a single dict."""
        report = {
            "timestamp": time.time(),
            "cpu":       self.get_cpu(),
            "memory":    self.get_memory(),
            "disk":      self.get_disk(),
            "network":   self.get_network(),
            "temps":     self.get_temperatures(),
            "battery":   self.get_battery(),
            "processes": self.get_top_processes(),
            "uptime":    self.get_uptime_str(),
            "platform":  platform.platform(),
        }
        self._history.append(report)
        return report

    def format_report(self, report: Optional[dict] = None) -> str:
        """Return a JARVIS-style formatted string summary."""
        r = report or self.get_full_report()
        cpu = r.get("cpu", {})
        mem = r.get("memory", {})
        dsk = r.get("disk", {})
        net = r.get("network", {})
        bat = r.get("battery")

        lines = [
            "── SYSTEM STATUS ──────────────────────────",
            f"  CPU:     {cpu.get('percent', 0):.1f}%  |  {cpu.get('freq_mhz', 0)} MHz",
            f"  RAM:     {mem.get('percent', 0):.1f}%  |  {mem.get('used_gb', 0):.1f} / {mem.get('total_gb', 0):.1f} GB",
            f"  DISK:    {dsk.get('percent', 0):.1f}%  |  {dsk.get('used_gb', 0):.1f} / {dsk.get('total_gb', 0):.1f} GB",
            f"  UPTIME:  {r.get('uptime', '—')}",
        ]
        if net:
            lines.append(f"  NET:     ↑ {net.get('bytes_sent_mb', 0):.1f} MB  ↓ {net.get('bytes_recv_mb', 0):.1f} MB")
        if bat:
            plugged = "⚡" if bat["plugged"] else "🔋"
            lines.append(f"  BATTERY: {bat['percent']}% {plugged}")

        # Temperature
        temps = r.get("temps", {})
        if temps:
            for chip, readings in temps.items():
                for sensor in readings[:1]:
                    lines.append(f"  TEMP:    {sensor['current']}°C  ({chip})")
                break

        # Top processes
        procs = r.get("processes", [])
        if procs:
            lines.append("── TOP PROCESSES ───────────────────────────")
            for p in procs[:5]:
                lines.append(f"  {p['name'][:20]:<20} CPU:{p['cpu']:5.1f}%  RAM:{p['mem']:4.1f}%")

        lines.append("────────────────────────────────────────────")
        return "\n".join(lines)

    # ── Alerting ──────────────────────────────

    def check_alerts(self, report: dict) -> list[str]:
        """Return list of alert strings for any breached thresholds."""
        alerts = []
        t = self.thresholds
        cpu = report.get("cpu", {}).get("percent", 0)
        ram = report.get("memory", {}).get("percent", 0)
        dsk = report.get("disk", {}).get("percent", 0)

        if cpu >= t.cpu_crit:
            alerts.append(f"CRITICAL: CPU at {cpu:.0f}%")
        elif cpu >= t.cpu_warn:
            alerts.append(f"WARNING: CPU at {cpu:.0f}%")

        if ram >= t.ram_crit:
            alerts.append(f"CRITICAL: RAM at {ram:.0f}%")
        elif ram >= t.ram_warn:
            alerts.append(f"WARNING: RAM at {ram:.0f}%")

        if dsk >= t.disk_crit:
            alerts.append(f"CRITICAL: Disk at {dsk:.0f}%")
        elif dsk >= t.disk_warn:
            alerts.append(f"WARNING: Disk at {dsk:.0f}%")

        # Temperature alerts
        for chip, readings in report.get("temps", {}).items():
            for sensor in readings:
                temp = sensor.get("current", 0)
                if temp >= t.temp_crit:
                    alerts.append(f"CRITICAL: {chip} temp at {temp}°C")
                elif temp >= t.temp_warn:
                    alerts.append(f"WARNING: {chip} temp at {temp}°C")

        return alerts

    # ── Background polling ────────────────────

    def start_background(
        self,
        interval: float = 10.0,
        on_alert: Optional[Callable[[str, dict], None]] = None,
    ):
        """Start a background thread that polls stats and fires alerts."""
        self._on_alert = on_alert
        self._stop_event.clear()
        self._bg_thread = threading.Thread(
            target=self._bg_loop,
            args=(interval,),
            name="SysMonitor-BG",
            daemon=True,
        )
        self._bg_thread.start()
        logger.info("System monitor background polling started (interval=%.0fs).", interval)

    def stop_background(self):
        self._stop_event.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=5)
        logger.info("System monitor background polling stopped.")

    def _bg_loop(self, interval: float):
        while not self._stop_event.is_set():
            try:
                report = self.get_full_report()
                alerts = self.check_alerts(report)
                if alerts and self._on_alert:
                    for alert in alerts:
                        self._on_alert(alert, report)
            except Exception as exc:
                logger.debug("BG poll error: %s", exc)
            self._stop_event.wait(timeout=interval)

    def get_history(self) -> list[dict]:
        return list(self._history)

    def get_averages(self) -> dict:
        """Compute averages over collected history."""
        history = self.get_history()
        if not history:
            return {}
        n = len(history)
        return {
            "cpu_avg":  sum(r["cpu"].get("percent", 0) for r in history) / n,
            "ram_avg":  sum(r["memory"].get("percent", 0) for r in history) / n,
            "disk_avg": sum(r["disk"].get("percent", 0) for r in history) / n,
            "samples":  n,
        }
