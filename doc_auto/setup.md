# Setup: environments, checkouts and weights

Last modified: 2026-09-23 18:20 (+08:00)

## TL;DR - how to find out what is missing on any machine

```bash
conda run -n ego3d_base python scripts/doctor.py --config configs/macrodata_final.yaml
conda run -n ego3d_base python scripts/doctor.py --config configs/hot3d.yaml --clip hot3d_ep000 --runners
```

`doctor.py` prints every requirement with a status, the fix, and a verdict
(`CPU path: ready` / `GPU path: not complete`). It never raises, so it is safe
to run first on a fresh machine. `--json` gives the same information
machine-readable, `--strict` exits non-zero on warnings too.

## 0. Prerequisites

| Need | Why | Check |
| --- | --- | --- |
| conda (miniconda is fine) | all four environments | `conda --version` |
| ffmpeg + ffprobe | frame extraction, video metadata (`io/video.py`) | `ffmpeg -version` |
| ~10 GB disk | sample + frames + windows + weights | `df -h` |
| NVIDIA driver + CUDA 12.x (GPU server only) | the three model backends | `nvidia-smi` |

Phases 0, 1-tracking, 4, 5, 6 and 7 are pure numpy/scipy and need **no GPU**.

## 1. Orchestrator environment (`ego3d_base`) - required, CPU-only

```bash
conda env create -f environment-base.yml      # or: make env
conda run -n ego3d_base python -m pip install --no-build-isolation -e .   # or: make install
conda run -n ego3d_base python -m pytest -q   # or: make test   -> 249 tests
```

Contents: python 3.11, numpy, scipy, opencv, pyyaml, pyarrow + pandas (LeRobot
v3 parquet), matplotlib, pytest. **No torch** - the orchestrator never imports a
model; it runs them as subprocesses (see `runtime/subprocess_backend.py`).

`make install` is optional: `pyproject.toml` sets `pythonpath = ["src"]` for
pytest and every script inserts `src/` itself, so the tests and the CLIs work
from a checkout without an editable install.

## 2. Model backends - only needed for Phases 1-3

Each backend gets its own environment because their stacks are mutually
incompatible (HaWoR pins torch 1.13/CUDA 11.7; VGGT-Omega wants a modern CUDA
stack). The orchestrator talks to them over files, never imports.

```bash
conda env create -f environment-wilor.yml     # ego3d_wilor  (python 3.10)
conda env create -f environment-hawor.yml     # ego3d_hawor  (python 3.10, torch 1.13, CUDA 11.7)
conda env create -f environment-vggt.yml      # ego3d_vggt   (python 3.11, torch 2.4+, CUDA 12.1)
```

The `environment-*.yml` files list the Python-level requirements; each backend's
own README is authoritative for the CUDA build that matches your driver. Adjust
the torch pin inside those files if the driver is older/newer, then re-verify
with `--check`.

Which interpreter runs which backend is configurable, so an unusual setup (docker,
different env names, a system python) needs no code change:

```yaml
# configs/macrodata_final.yaml
backends:
  mode: real
  python:
    wilor: ["conda", "run", "-n", "ego3d_wilor", "python"]
    hawor: ["conda", "run", "-n", "ego3d_hawor", "python"]
    vggt:  ["conda", "run", "-n", "ego3d_vggt", "python"]
```

## 3. Checkouts

Backends are cloned, never forked or vendored:

```bash
git clone https://github.com/rolpotamias/WiLoR        third_party/WiLoR
git clone https://github.com/ThunderVVV/HaWoR         third_party/HaWoR
git clone https://github.com/facebookresearch/vggt    third_party/VGGT-Omega
```

`third_party/*` is git-ignored. `doctor.py` checks for `third_party/<name>` and
each runner's `--check` imports the package from there.

## 4. Weights - **nothing is bundled**

This is the honest answer to "are the checkpoints included?": **no**. The
repository ships code, tests, the HOT3D sample dataset and a synthetic stand-in
for plumbing runs - no model weights, and the MANO mesh model is not present on
this machine either (its directories contain only a `.gitkeep` plus
`mano_mean_params.npz`, which is just the mean pose/shape, not the model).

| Asset | Where it goes | Source | Needed for |
| --- | --- | --- | --- |
| WiLoR detector checkpoint (`wilor_final.ckpt`, plus `model_config.yaml`) | `weights/wilor/` | WiLoR repo / release page | Phase 1 |
| HaWoR checkpoints (`hawor.ckpt`, `infiller.pt`) | `weights/hawor/checkpoints/` | HaWoR release page | Phase 2 |
| VGGT-Omega `VGGT-Omega-1B-416-Reproduction` | `weights/vggt-omega/` | VGGT-Omega release page | Phase 3 |
| MANO model (`MANO_RIGHT.pkl`) | `weights/mano/` (converted to `.npz`) | mano.is.tue.mpg.de (licence + registration) | 21-joint references |

Copy `third_party/README.md`'s commands and the table above; then verify:

```bash
conda run -n ego3d_base python scripts/doctor.py --config configs/macrodata_final.yaml --runners
# every backend should print ok; --check reports the exact missing file otherwise
```

### MANO specifically

MANO is licence-gated, so it cannot ship here. Once you have it:

```bash
conda run -n ego3d_hawor python scripts/convert_mano.py \
    --input /path/to/MANO_RIGHT.pkl --output-dir weights/mano
# then, in configs/hot3d.yaml:  paths.mano_model: weights/mano
```

The `.pkl` needs `chumpy` (present in `ego3d_hawor`); the `.npz` needs nothing.
Only `MANO_RIGHT` is required - the left hand is mirrored automatically and
flagged as `mano_mirrored` in the reference metadata. Without MANO the pipeline
still runs: references are wrist-only and the evaluation says so (see
`doc_auto/runbook.md`).

## 5. Sample data

`data/samples/lerobot_v3` (8 HOT3D clips, 3600 frames, 21 MB) ships with the
workspace and is git-ignored; `data/*` never enters the repository. Everything
derived from it lands under `data/hot3d/`.

## 6. Environment variables

| Variable | Effect |
| --- | --- |
| `EGO3D_FORCE_CPU=1` | resolve `runtime.device: auto` to `cpu` even when a GPU exists |
| `MPLCONFIGDIR=<writable dir>` | required when `$HOME` is read-only (matplotlib) |

## 7. Known constraints

* `runtime.device: cuda` on a machine without CUDA raises at config validation -
  a GPU run can never silently fall back to the CPU. Use `auto` or `cpu`.
* The pipeline assumes the current directory is the repository root (relative
  `paths.*`). Run the scripts from there or pass `--data-root`.

Last verified on this machine (2026-09-23 18:20): `ego3d_base` complete, the
three backend checkouts and every checkpoint **absent**, MANO **absent** - the
CPU path is fully functional, the GPU path needs the downloads above.
