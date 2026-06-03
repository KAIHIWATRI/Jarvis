"""
voice_queue.py — JARVIS Voice Queue System
Thread-safe priority queue with deduplication, rate-limiting, and inspection.
"""

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

from tts_engine import Priority, SpeechItem, VoiceProfile

logger = logging.getLogger("JARVIS.VoiceQueue")


@dataclass(order=True)
class _TimedItem:
    """Internal wrapper that preserves insertion order for equal-priority items."""
    priority:  int         = field(compare=True)
    sequence:  int         = field(compare=True)   # tie-break: lower = earlier
    item:      SpeechItem  = field(compare=False)

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority   # higher Priority value → higher urgency
        return self.sequence < other.sequence        # earlier insertion wins


class VoiceQueue:
    """
    Thread-safe priority queue for SpeechItems.

    Features
    --------
    - Priority scheduling (URGENT > HIGH > NORMAL > LOW)
    - FIFO within equal priority
    - Deduplication by item_id
    - Max-depth cap with oldest-low-priority eviction
    - Inspection / iteration (non-destructive)
    - Rate limit: minimum gap between NORMAL/LOW items
    """

    def __init__(self, maxsize: int = 50, min_gap_ms: int = 0):
        self._heap:     List[_TimedItem] = []
        self._ids:      set[str]         = set()
        self._lock      = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._seq       = 0
        self._maxsize   = maxsize
        self._min_gap   = min_gap_ms / 1000.0
        self._last_put  = 0.0

    # ── Put ───────────────────────────────────

    def put(self, item: SpeechItem, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Add a SpeechItem to the queue.
        Returns False if duplicate or queue full and can't evict.
        """
        with self._not_empty:
            # Dedup
            if item.item_id and item.item_id in self._ids:
                logger.debug("Duplicate item_id=%s, skipped.", item.item_id)
                return False

            # Rate limiting (LOW/NORMAL only)
            if item.priority <= Priority.NORMAL.value and self._min_gap > 0:
                now = time.monotonic()
                wait = self._min_gap - (now - self._last_put)
                if wait > 0:
                    if block:
                        time.sleep(wait)
                    else:
                        return False

            # Evict if over capacity
            if len(self._heap) >= self._maxsize:
                evicted = self._evict_lowest()
                if not evicted:
                    logger.warning("Queue full, item dropped: %.40s", item.text)
                    return False

            self._seq += 1
            ti = _TimedItem(priority=item.priority, sequence=self._seq, item=item)
            heapq.heappush(self._heap, ti)
            if item.item_id:
                self._ids.add(item.item_id)
            self._last_put = time.monotonic()
            self._not_empty.notify()
            logger.debug("Queued seq=%d pri=%d text=%.40s…", self._seq, item.priority, item.text)
            return True

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[SpeechItem]:
        """Pop and return the highest-priority item."""
        with self._not_empty:
            if block:
                deadline = time.monotonic() + timeout if timeout else None
                while not self._heap:
                    remaining = deadline - time.monotonic() if deadline else None
                    if remaining is not None and remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining)
            if not self._heap:
                return None
            ti = heapq.heappop(self._heap)
            if ti.item.item_id:
                self._ids.discard(ti.item.item_id)
            return ti.item

    def get_nowait(self) -> Optional[SpeechItem]:
        return self.get(block=False)

    # ── Inspection ────────────────────────────

    def peek(self) -> Optional[SpeechItem]:
        """Return next item without removing it."""
        with self._lock:
            return self._heap[0].item if self._heap else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def items(self) -> List[SpeechItem]:
        """Return a snapshot of all queued items, highest priority first."""
        with self._lock:
            return [ti.item for ti in sorted(self._heap)]

    def clear(self):
        with self._not_empty:
            self._heap.clear()
            self._ids.clear()
            logger.info("Queue cleared.")

    def remove_by_id(self, item_id: str) -> bool:
        """Remove a specific item by ID. O(n) but safe."""
        with self._not_empty:
            before = len(self._heap)
            self._heap = [ti for ti in self._heap if ti.item.item_id != item_id]
            heapq.heapify(self._heap)
            self._ids.discard(item_id)
            removed = len(self._heap) < before
            if removed:
                logger.debug("Removed item_id=%s.", item_id)
            return removed

    def stats(self) -> dict:
        with self._lock:
            by_pri: dict[str, int] = {}
            for ti in self._heap:
                name = Priority(ti.priority).name
                by_pri[name] = by_pri.get(name, 0) + 1
            return {"total": len(self._heap), "by_priority": by_pri}

    # ── Internal ──────────────────────────────

    def _evict_lowest(self) -> bool:
        """Remove the lowest-priority, oldest item. Returns True if evicted."""
        if not self._heap:
            return False
        # Find lowest priority (smallest value), latest sequence
        worst_idx = max(
            range(len(self._heap)),
            key=lambda i: (-self._heap[i].priority, self._heap[i].sequence),
        )
        evicted = self._heap.pop(worst_idx)
        heapq.heapify(self._heap)
        if evicted.item.item_id:
            self._ids.discard(evicted.item.item_id)
        logger.debug("Evicted: %.40s (pri=%d)", evicted.item.text, evicted.priority)
        return True
