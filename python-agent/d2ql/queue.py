"""Deterministic dispatch order for the cloudlets of one episode.

The agent picks the **host** for a cloudlet; this queue picks **which** pending
cloudlet goes next. Until recently the order was computed but never applied (the
gateway dispatched them in trace order), so the class was decorative.

With the SLA axis retired the trace deadline carries no usable signal: it is an
absolute timestamp (hundreds of thousands of seconds) compared against a
simulation clock that runs for ~170 s, so every deadline looks infinitely far
away. Ordering therefore targets makespan/energy, which is what the reward
measures: **Longest Processing Time first (LPT)** by estimated work per PE
(``mi / num_pes``). Putting the longest jobs in the first hosts avoids the long
tail that keeps the datacenter draining (and consuming idle power) after the
scheduler has nothing left to place.

The order is fixed at ``reset`` because every cloudlet is dispatched exactly once
per step: there is no queue state to re-evaluate mid-episode.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=True)
class CloudletEntry:
    """A pending cloudlet with its dispatch priority (lower key = dispatched first)."""

    # Min-heap key: negative work-per-PE -> longest job first (LPT).
    priority: float
    cloudlet_id: int = field(compare=False)
    mi: float = field(compare=False)
    num_pes: int = field(compare=False)

    @staticmethod
    def compute_priority(mi: float, num_pes: int) -> float:
        """LPT key: ``-mi / num_pes`` (work per processing element, descending)."""
        pes = max(int(num_pes), 1)
        return -float(mi) / float(pes)


class PriorityCloudletQueue:
    """Min-heap of pending cloudlets ordered by LPT.

    Usage:
        q = PriorityCloudletQueue()
        q.push(cloudlet_id=0, mi=10_000.0, num_pes=2)
        ordered_ids = q.drain()   # longest work-per-PE first
    """

    def __init__(self) -> None:
        self._heap: list[CloudletEntry] = []

    def push(self, cloudlet_id: int, mi: float, num_pes: int) -> None:
        """Add a cloudlet to the queue."""
        heapq.heappush(
            self._heap,
            CloudletEntry(
                priority=CloudletEntry.compute_priority(mi, num_pes),
                cloudlet_id=int(cloudlet_id),
                mi=float(mi),
                num_pes=max(int(num_pes), 1),
            ),
        )

    def pop(self) -> Optional[CloudletEntry]:
        """Remove and return the next cloudlet to dispatch."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def drain(self) -> list[int]:
        """Empty the queue and return cloudlet IDs in dispatch order."""
        ordered: list[int] = []
        while self._heap:
            ordered.append(heapq.heappop(self._heap).cloudlet_id)
        return ordered

    def peek(self) -> Optional[CloudletEntry]:
        """Return the next entry without removing it."""
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return f"PriorityCloudletQueue(size={len(self._heap)})"
