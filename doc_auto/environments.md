# Environments

Last modified: 2026-09-23 17:12 (+08:00)

## Strategy

| Env | File | Python | Purpose | Created where |
| --- | --- | --- | --- | --- |
| `ego3d_base` | `environment-base.yml` | 3.11 | orchestration, geometry, evaluation, visualisation | CPU laptop **and** GPU server |
| `ego3d_wilor` | `environment-wilor.yml` | 3.10 | WiLoR detection backend | GPU server |
| `ego3d_hawor` | `environment-hawor.yml` | 3.10 | HaWoR (torch 1.13 / CUDA 11.7) | GPU server |
| `ego3d_vggt` | `environment-vggt.yml` | 3.11 | VGGT-Omega | GPU server |

One environment cannot hold all of it: HaWoR pins torch 1.13/CUDA 11.7 while the
orchestrator must stay importable on a CPU-only laptop. The boundary between
environments is the filesystem - each stage reads and writes artefacts under
`data/<clip>/`.

## Device handling

`runtime/device.py` resolves the device from `runtime.device` in the config:

| Request | Behaviour |
| --- | --- |
| `auto` | `cuda:0` when a GPU is visible, otherwise `cpu` |
| `cpu` | always CPU |
| `cuda`, `cuda:N` | raises `BackendNotAvailableError` when CUDA is unavailable |
| `EGO3D_FORCE_CPU=1` | forces CPU even if a GPU exists (useful for CI) |

The same config therefore works unchanged on the laptop and on the server, and a
GPU run can never silently degrade to CPU.

## Base environment (verified locally)

```
python 3.11.15 | numpy 2.4.6 | scipy 1.17.1 | opencv 4.14.0
pyyaml 6.0.3 | pytest 9.0.3 | matplotlib 3.11.0 | ffmpeg 6.1.1 (system)
```

Set `MPLCONFIGDIR` to a writable directory when `$HOME` is read-only, e.g.
`MPLCONFIGDIR=/tmp/mpl-cache`.

