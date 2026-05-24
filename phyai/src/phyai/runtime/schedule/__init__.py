"""phyai.runtime.schedule — scheduler primitives.

Currently exposes the abstract :class:`Scheduler` base. Future
multi-batch / continuous-batching schedulers will live alongside it.
"""

from __future__ import annotations

from phyai.runtime.schedule.base import Scheduler


__all__ = ["Scheduler"]
