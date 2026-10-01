"""Period router for the QSITE 2026 Computational Track.

    from period_router import solve
    placement, routed_program = solve(program, hardware_graph)
"""

from .solve import solve

__all__ = ["solve"]
