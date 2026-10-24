"""Dispatch order: LPT by work-per-PE (docs/backlog.md, SIM-04)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql.queue import CloudletEntry, PriorityCloudletQueue


def test_compute_priority_is_negative_work_per_pe():
    assert CloudletEntry.compute_priority(mi=100.0, num_pes=1) == -100.0
    assert CloudletEntry.compute_priority(mi=100.0, num_pes=4) == -25.0
    # num_pes is clipped to >= 1 (never divides by zero).
    assert CloudletEntry.compute_priority(mi=100.0, num_pes=0) == -100.0


def test_drain_returns_longest_work_per_pe_first():
    q = PriorityCloudletQueue()
    q.push(cloudlet_id=0, mi=1000.0, num_pes=1)   # 1000 per PE
    q.push(cloudlet_id=1, mi=4000.0, num_pes=8)   # 500 per PE
    q.push(cloudlet_id=2, mi=2000.0, num_pes=1)   # 2000 per PE
    assert q.drain() == [2, 0, 1]
    assert len(q) == 0


def test_pop_matches_drain_order_and_is_exhaustible():
    q = PriorityCloudletQueue()
    for cid, mi, pes in [(0, 10.0, 1), (1, 30.0, 1), (2, 20.0, 1)]:
        q.push(cloudlet_id=cid, mi=mi, num_pes=pes)
    nxt = q.peek()
    assert nxt is not None and nxt.cloudlet_id == 1
    popped = []
    while True:
        entry = q.pop()
        if entry is None:
            break
        popped.append(entry.cloudlet_id)
    assert popped == [1, 2, 0]
    assert q.pop() is None


def test_ties_keep_a_deterministic_order():
    q = PriorityCloudletQueue()
    for cid in (3, 1, 2):
        q.push(cloudlet_id=cid, mi=50.0, num_pes=1)
    # Equal priority -> the heap falls back to insertion/handling order, but it
    # must be stable across two identical builds.
    first = q.drain()
    q2 = PriorityCloudletQueue()
    for cid in (3, 1, 2):
        q2.push(cloudlet_id=cid, mi=50.0, num_pes=1)
    assert q2.drain() == first
