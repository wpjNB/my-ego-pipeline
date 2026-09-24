# Third-party backends

This directory holds **checkouts only** - never vendored or patched code. Each
backend keeps its own environment so that the orchestrator stays importable on a
CPU-only machine.

| Backend | Clone into | Environment | Used by |
| --- | --- | --- | --- |
| WiLoR | `third_party/WiLoR` | `ego3d_wilor` | Phase 1 (detection) |
| HaWoR | `third_party/HaWoR` | `ego3d_hawor` | Phase 2 (hand reconstruction) |
| VGGT-Omega | `third_party/VGGT-Omega` | `ego3d_vggt` | Phase 3 (camera reconstruction) |

```bash
git clone https://github.com/rolpotamias/WiLoR        third_party/WiLoR
git clone https://github.com/ThunderVVV/HaWoR         third_party/HaWoR
git clone https://github.com/facebookresearch/vggt    third_party/VGGT-Omega
```

Weights go into `weights/<backend>/` (`weights/wilor`, `weights/hawor/checkpoints`,
`weights/vggt-omega`) and are never committed. `./scripts/download_weights.sh`
fetches what is publicly available; VGGT-Omega comes from ModelScope
(`facebook/VGGT-Omega`, `vggt_omega_1b_416_reproduce.pt`, FAIR Noncommercial
Research License) and MANO stays a manual, licence-gated download.

## Why not fork HaWoR?

The official HaWoR demo pipeline is
`WiLoR -> HaWoR -> masked DROID-SLAM -> Metric3D -> world space`.
The reference system replaces `DROID-SLAM + Metric3D` with `VGGT-Omega`, so
forking HaWoR and editing it would entangle two mutually exclusive
camera-reconstruction designs in one code base. Instead, HaWoR is invoked as a
backend through `src/ego3d_action/hand/hawor.py`.

## Checkpoint note (2026-09 onwards)

VGGT-Omega publishes `VGGT-Omega-1B-512`, `VGGT-Omega-1B-416-Reproduction` and
`VGGT-Omega-1B-256-Text-Alignment`. This project uses
`VGGT-Omega-1B-416-Reproduction` and does **not** resize the 512 checkpoint.
It is an official reproduction checkpoint, but the blog never states it is the
exact checkpoint behind its reported numbers, so 52.0435 mm is not a promise.
