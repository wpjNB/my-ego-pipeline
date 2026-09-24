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

## 4. Weights - **nothing is bundled**, and one script fetches them

This is the honest answer to "are the checkpoints included?": **no**. The
repository ships code, tests, the HOT3D sample dataset and a synthetic stand-in
for plumbing runs - no model weights, and the MANO mesh model is not present on
this machine either (its directories contain only a `.gitkeep` plus
`mano_mean_params.npz`, which is just the mean pose/shape, not the model).

Two bash scripts do the whole job - no Python environment needed, so they also
work on a bare server:

```bash
./scripts/download_weights.sh --dry-run        # the plan: every URL and destination, no traffic
./scripts/download_weights.sh                  # fetch everything (resumable), then verify
./scripts/download_weights.sh --only wilor,hawor
./scripts/download_weights.sh --with-repos     # also clone third_party/*
DEST=/mnt/weights ./scripts/download_weights.sh

./scripts/verify_weights.sh                    # re-check size + container format
./scripts/verify_weights.sh --quiet --strict   # only problems; optional assets count too
```

* **resumable** - `wget -c` (or `curl -C -`) into `<name>.part`, moved into
  place only after the size check passes;
* **verified** - `verify_weights.sh` checks a size floor *and* the container
  magic bytes (`PK\x03\x04` zip for modern `torch.save`, `\x80` pickle for a
  legacy one, the 8-byte little-endian JSON header for `safetensors`), so the
  classic "wget succeeded on an HTML error page, `torch.load` explodes later"
  failure is caught immediately. A `sha256` is supported per entry;
* **explicit about MANO** - it is licence-gated, so the script never fetches it;
  it prints the registration page, the exact filename, the destination and the
  conversion command;
* **paths match the code** - `weights/wilor/`, `weights/hawor/checkpoints/`,
  `weights/vggt-omega/`, `weights/mano/`. The runners also accept the flat
  variants (`weights/hawor/hawor.ckpt`, `weights/vggt/...`) so a manual download
  is never a dead end.

**The URLs in the script were transcribed from each project's documentation and
could not be verified from the development machine (no network).** A moved URL
shows up as `[FAIL] ... only N bytes`; fix it in the script (the `WILOR_BASE` /
`HAWOR_BASE` / `VGGT_URL` variables at the top) or drop the file in by hand and
re-run `./scripts/verify_weights.sh`. Those same variables point the script at a
local mirror on an air-gapped host:

```bash
WILOR_BASE=file:///srv/mirror/wilor HAWOR_BASE=file:///srv/mirror/hawor \
VGGT_URL=file:///srv/mirror/vggt_omega_1b_416_reproduce.pt \
    ./scripts/download_weights.sh
```

| Asset | Where it goes | Source | Needed for |
| --- | --- | --- | --- |
| WiLoR detector checkpoint (`wilor_final.ckpt`, plus `model_config.yaml`) | `weights/wilor/` | WiLoR repo / release page | Phase 1 |
| HaWoR checkpoints (`hawor.ckpt`, `infiller.pt`) | `weights/hawor/checkpoints/` | HaWoR release page | Phase 2 |
| VGGT-Omega `vggt_omega_1b_416_reproduce.pt` (4.58 GB) | `weights/vggt-omega/` | ModelScope `facebook/VGGT-Omega`, revision `master` | Phase 3 |
| MANO model (`MANO_RIGHT.pkl`) | `weights/mano/` (converted to `.npz`) | mano.is.tue.mpg.de (licence + registration) | 21-joint references |

The VGGT-Omega repository also publishes `vggt_omega_1b_512.pt` (4.58 GB) and
`vggt_omega_1b_256_text.pt` (5.40 GB); the script uses them as fallbacks and the
verifier accepts any of the three. Its licence is the **FAIR Noncommercial
Research License** - respect it. The 416 reproduction file is the one the
reference configuration names, so downloading it means no checkpoint
substitution is reported.

Then verify end to end:

```bash
./scripts/verify_weights.sh --strict
conda run -n ego3d_base python scripts/doctor.py --config configs/macrodata_final.yaml --runners
# every backend should print ok; --check reports the exact missing file otherwise
```

### MANO specifically

MANO is licence-gated (register at <https://mano.is.tue.mpg.de/>), so no script
can download it. Reading the backend sources settled where it is actually
needed - and the answer is *not* "everywhere":

| Consumer | Expected path | Needed for |
| --- | --- | --- |
| HaWoR `run_mano` | `third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl` | **Phase 2 (required)** |
| HaWoR `run_mano_left` | `third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl` | Phase 2 left hand |
| this project's forward kinematics | `weights/mano/MANO_RIGHT.pkl` (+ `.npz`) | 21-joint HOT3D references |
| WiLoR (`MANO.MODEL_PATH='./mano_data/'`) | `third_party/WiLoR/mano_data/MANO_RIGHT.pkl` | only for WiLoR's *own* 3D model - Phase 1 uses just `detector.pt` |

**Phase 1 needs no MANO and no `wilor_final.ckpt`**: WiLoR's demo separates the
YOLO detector (`detector.pt`, boxes + handedness) from the 3D model
(`load_wilor(wilor_final.ckpt, ...)`, whose MANO layer needs the licence-gated
model). This project takes hand *tracking* from WiLoR and hand *reconstruction*
from HaWoR, so only the detector is used - 51 MiB instead of 2.4 GiB.

The HaWoR and WiLoR directories are empty after cloning (they hold only
`.gitkeep` - the model is not redistributable), so the checkouts alone are not
enough. One command installs your copy everywhere:

```bash
./scripts/install_mano.sh --from ~/Downloads/mano          # copy
./scripts/install_mano.sh --from ~/Downloads/mano --link   # or symlink
./scripts/install_mano.sh --from ~/Downloads/mano --dry-run

# it also prints (and runs when ego3d_hawor exists) the conversion this project
# needs for its own forward kinematics:
conda run -n ego3d_hawor python scripts/convert_mano.py \
    --input weights/mano/MANO_RIGHT.pkl --output-dir weights/mano
# then, in configs/hot3d.yaml:  paths.mano_model: weights/mano
```

**No chumpy is needed.** The official archive wraps only ``shapedirs`` in a
chumpy ``Select``; ``hand/mano_model.py::read_mano_pickle`` unpickles it with a
tiny stand-in class and materialises the array, so this project reads the
official pickle directly (and ``convert_mano.py`` can still write a portable
``.npz`` if you prefer). chumpy 0.70 would otherwise require both ``numpy<1.24``
and Python <= 3.10, which is incompatible with the orchestrator environment.

`MANO_RIGHT` alone is enough for this project - our FK mirrors it for the left
hand (`mano_mirrored` in the reference metadata) and HaWoR's `run_mano_left` has
a `fix_shapedirs` workaround - but passing `MANO_LEFT.pkl` too removes that
approximation. `./scripts/verify_weights.sh` lists all four locations, and both
backend runners report the missing file in their `--check`. Without MANO the CPU
path still works: references stay wrist-only and the evaluation says so.

Once `weights/mano` holds the models, `configs/hot3d.yaml` already points at it
(`paths.mano_model: weights/mano`), so `scripts/import_lerobot.py` writes 21-joint
references; `--no-mano` falls back to wrist-only for a quick check.

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
