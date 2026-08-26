from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=True)
class CloudletEntry:
    """A single cloudlet with a computed dispatch priority."""

    priority: float
    cloudlet_id: int = field(compare=False)
    deadline: float = field(compare=False)
    mi: float = field(compare=False)          # million instructions (workload size)
    num_pes: int = field(compare=False)       # required processing elements
    submitted_at: float = field(compare=False)

    @staticmethod
    def compute_priority(
        deadline: float,
        mi: float,
        num_pes: int,
        submitted_at: float,
        current_time: float,
        urgency_weight: float = 0.6,
        demand_weight: float = 0.4,
    ) -> float:
        """
        Lower priority score = higher urgency (min-heap ordering).

        Combines:
          - Time urgency: how close the deadline is relative to elapsed time.
          - Resource demand: normalized MI * PEs, favouring lighter cloudlets
            when urgency is equal.
        """
        time_remaining = max(deadline - current_time, 1e-6)
        elapsed = max(current_time - submitted_at, 1e-6)
        urgency = elapsed / time_remaining          # higher = more urgent

        demand = mi * num_pes                       # raw compute demand

        # Invert urgency so that the most urgent cloudlet gets the lowest score.
        # Demand is added so that among equally urgent cloudlets, lighter ones
        # are preferred (less risk of blocking the host).
        priority_score = -(urgency_weight * urgency) + (demand_weight * (demand / 1e6))
        return priority_score


class PriorityCloudletQueue:
    """
    Min-heap priority queue for cloudlet dispatch ordering.

    Usage:
        q = PriorityCloudletQueue()
        q.push(cloudlet_id=0, deadline=500.0, mi=10000.0,
               num_pes=2, submitted_at=0.0, current_time=0.0)
        ordered_ids = q.drain()   # returns cloudlet IDs sorted by priority
    """

    def __init__(
        self,
        urgency_weight: float = 0.6,
        demand_weight: float = 0.4,
    ) -> None:
        self._heap: list[CloudletEntry] = []
        self.urgency_weight = urgency_weight
        self.demand_weight = demand_weight

    def push(
        self,
        cloudlet_id: int,
        deadline: float,
        mi: float,
        num_pes: int,
        submitted_at: float,
        current_time: float,
    ) -> None:
        """Add a cloudlet to the queue."""
        priority = CloudletEntry.compute_priority(
            deadline=deadline,
            mi=mi,
            num_pes=num_pes,
            submitted_at=submitted_at,
            current_time=current_time,
            urgency_weight=self.urgency_weight,
            demand_weight=self.demand_weight,
        )
        entry = CloudletEntry(
            priority=priority,
            cloudlet_id=cloudlet_id,
            deadline=deadline,
            mi=mi,
            num_pes=num_pes,
            submitted_at=submitted_at,
        )
        heapq.heappush(self._heap, entry)

    def pop(self) -> Optional[CloudletEntry]:
        """Remove and return the highest-priority cloudlet."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def drain(self) -> list[int]:
        """
        Empty the queue and return cloudlet IDs in priority order
        (most urgent first).
        """
        ordered: list[int] = []
        while self._heap:
            entry = heapq.heappop(self._heap)
            ordered.append(entry.cloudlet_id)
        return ordered

    def peek(self) -> Optional[CloudletEntry]:
        """Return the highest-priority entry without removing it."""
        if not self._heap:
            return None
        return self._heap[0]

    def reprioritize(self, current_time: float) -> None:
        """
        Recompute priorities for all queued cloudlets given a new
        current_time, then rebuild the heap. Call this at the start
        of each environment step to keep urgency scores fresh.
        """
        entries = self._heap[:]
        self._heap = []
        for e in entries:
            updated_priority = CloudletEntry.compute_priority(
                deadline=e.deadline,
                mi=e.mi,
                num_pes=e.num_pes,
                submitted_at=e.submitted_at,
                current_time=current_time,
                urgency_weight=self.urgency_weight,
                demand_weight=self.demand_weight,
            )
            e.priority = updated_priority
            heapq.heappush(self._heap, e)

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return f"PriorityCloudletQueue(size={len(self._heap)})"
