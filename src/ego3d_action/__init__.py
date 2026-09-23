"""Ego3D Hand Action Pipeline.

An orchestration layer that turns a monocular egocentric RGB video into a
world-frame, metric, 21-joint 3D hand-action trajectory.

The package is organised by pipeline stage so that every stage can be executed,
inspected and tested on its own:

``io``            video / frames / on-disk serialisation
``detection``     WiLoR detection + conservative tracking (Phase 1)
``hand``          HaWoR temporal reconstruction (Phase 2)
``camera``        VGGT-Omega windows + depth-derived Sim(3) stitching (Phase 3-4)
``geometry``      Sim(3) / Umeyama / transform primitives
``fusion``        hand + camera -> world frame (Phase 5)
``refinement``    camera filter / bone scale / wrist depth (Phase 6)
``evaluation``    HOT3D Action-MPJPE / coverage / FPS (Phase 7)
``visualization`` per-stage debug videos (Phase 7)
``runtime``       device (cpu/cuda) + backend availability helpers
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

