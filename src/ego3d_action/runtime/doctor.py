"""Environment and asset audit.

Answers one question - "can this project run here, and if not, what exactly is
missing?" - without raising, so it is safe to run at any point. Every failing
check carries the command that fixes it.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

logger = logging.getLogger(__name__)

Status = Literal["ok", "warn", "missing"]
Required = Literal["cpu", "gpu"]

#: installed by environment-base.yml
BASE_DEPENDENCIES = (
    ("numpy", "numerics"),
    ("scipy", "numerics, SLERP, sparse wrist solve"),
    ("cv2", "frame IO, debug videos"),
    ("yaml", "configuration"),
    ("pyarrow", "LeRobot v3 parquet"),
    ("pandas", "LeRobot v3 tables"),
    ("matplotlib", "trajectory plots"),
    ("pytest", "test suite"),
)

#: only needed to *execute* the model backends
BACKEND_DEPENDENCIES = (("torch", "model backends (their own envs)"),)


@dataclass(frozen=True)
class Check:
    """One audited requirement."""

    name: str
    status: Status
    detail: str
    fix: str = ""
    required_for: Required = "cpu"
    section: str = field(default="general")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def check_host_tools() -> list[Check]:
    """External binaries the pipeline shells out to."""
    checks: list[Check] = []
    for name, why in (("ffmpeg", "frame extraction"), ("ffprobe", "video metadata")):
        found = shutil.which(name)
        checks.append(
            Check(
                name=name,
                status="ok" if found else "missing",
                detail=found or f"not on PATH ({why})",
                fix="" if found else f"install ffmpeg (apt install ffmpeg / brew install ffmpeg) - needed for {why}",
                section="host",
            )
        )
    return checks


def check_python_dependencies(
    *,
    dependencies: Sequence[tuple[str, str]] = BASE_DEPENDENCIES,
    required_for: Required = "cpu",
    fix: str = "conda env create -f environment-base.yml",
) -> list[Check]:
    """Import checks for the modules this interpreter must provide."""
    checks: list[Check] = []
    for module, why in dependencies:
        spec = importlib.util.find_spec(module)
        if spec is None:
            checks.append(
                Check(
                    name=f"python:{module}",
                    status="missing",
                    detail=f"not importable ({why})",
                    fix=fix,
                    section="python",
                    required_for=required_for,
                )
            )
            continue
        try:
            version = importlib.import_module(module).__version__
        except Exception:  # noqa: BLE001 - a broken install is a finding, not a crash
            version = "imported"
        checks.append(
            Check(name=f"python:{module}", status="ok", detail=str(version), section="python", required_for=required_for)
        )
    return checks


def check_configuration(config_path: str | Path) -> list[Check]:
    """Load and validate a config file."""
    from ..config import load_config
    from ..errors import ConfigError
    from ..runtime.subprocess_backend import BackendInvocation

    path = Path(config_path)
    if not path.is_file():
        return [
            Check(
                name="config",
                status="missing",
                detail=f"{path} not found",
                fix="use one of configs/{default,macrodata_final,hot3d,mock}.yaml",
                section="config",
            )
        ]
    try:
        config = load_config(path)
    except ConfigError as exc:
        return [
            Check(name="config", status="missing", detail=str(exc), fix="fix the config file", section="config")
        ]
    invocation = BackendInvocation.from_config(config)
    return [
        Check(
            name="config",
            status="ok",
            detail=f"{path} valid (backend mode: {invocation.mode})",
            section="config",
        )
    ]


def check_backends(third_party: Path, weights_root: Path) -> list[Check]:
    """Checkout + weights availability for the three model backends."""
    from ..camera.vggt_omega import probe as probe_vggt
    from ..detection.wilor import probe as probe_wilor
    from ..hand.hawor import probe as probe_hawor

    checks: list[Check] = []
    for label, selector, probe, clone, checkout, weights in (
        (
            "WiLoR",
            "wilor",
            probe_wilor,
            "https://github.com/rolpotamias/WiLoR",
            "WiLoR",
            "weights/wilor",
        ),
        (
            "HaWoR",
            "hawor",
            probe_hawor,
            "https://github.com/ThunderVVV/HaWoR",
            "HaWoR",
            "weights/hawor",
        ),
        (
            "VGGT-Omega",
            "vggt",
            probe_vggt,
            "https://github.com/facebookresearch/vggt",
            "VGGT-Omega",
            "weights/vggt-omega",
        ),
    ):
        status = probe(third_party, weights_root)
        checks.append(
            Check(
                name=f"backend:{label}",
                status="ok" if status.available else "missing",
                detail="checkout + weights present" if status.available else "; ".join(status.missing),
                fix=(
                    f"git clone {clone} {third_party / checkout} && "
                    f"./scripts/download_weights.sh --only {selector} "
                    "(see doc_auto/setup.md)"
                ),
                section="backends",
                required_for="gpu",
            )
        )
    return checks


def check_backend_runners(
    config_path: str | Path, *, timeout: float = 60.0
) -> list[Check]:
    """Run each real runner's ``--check`` through its configured interpreter."""
    from ..config import load_config
    from ..runtime.backend import probe_backend
    from ..runtime.subprocess_backend import BackendInvocation

    from ..camera.vggt_omega import VGGT_SPEC
    from ..detection.wilor import WILOR_SPEC
    from ..hand.hawor import HAWOR_SPEC

    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - already reported by check_configuration
        logger.debug("skipping runner checks: %s", exc)
        return []
    invocation = BackendInvocation.from_config(config)
    specs = {"wilor": WILOR_SPEC, "hawor": HAWOR_SPEC, "vggt": VGGT_SPEC}
    third_party = Path(str(config.get("paths.third_party", "third_party")))
    weights_root = Path(str(config.get("paths.weights", "weights")))

    checks: list[Check] = []
    environments = _conda_environments()
    for name, spec in specs.items():
        backend = probe_backend(spec, third_party=third_party, weights_root=weights_root)
        runner = invocation.backends_dir / f"{name}_runner.py"
        if not runner.is_file():
            checks.append(
                Check(
                    name=f"runner:{name}",
                    status="missing",
                    detail=f"{runner} not found",
                    fix="the runner ships with this repository",
                    section="runners",
                    required_for="gpu",
                )
            )
            continue
        if not backend.available:
            checks.append(
                Check(
                    name=f"runner:{name}",
                    status="missing",
                    detail="checkout or weights missing, cannot import the backend",
                    fix="see the backends section above",
                    section="runners",
                    required_for="gpu",
                )
            )
            continue
        # ``conda run -n <env> python`` needs that env to exist; say so plainly
        # instead of surfacing conda's own error.
        command_template = invocation.python_commands.get(name, ("python",))
        env_name = _conda_env_of(command_template)
        if env_name is not None and environments is not None and env_name not in environments:
            checks.append(
                Check(
                    name=f"runner:{name}",
                    status="missing",
                    detail=f"backend environment '{env_name}' does not exist yet",
                    fix=(
                        f"conda env create -f environment-"
                        f"{'vggt' if name == 'vggt' else name}.yml"
                    ),
                    section="runners",
                    required_for="gpu",
                )
            )
            continue
        command = [*invocation.python_commands.get(name, ("python",)), str(runner), "--check",
                   "--third-party", str(third_party), "--weights", str(weights_root)]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(
                Check(
                    name=f"runner:{name}",
                    status="missing",
                    detail=f"cannot execute {' '.join(command[:4])}...: {type(exc).__name__}",
                    fix="create the backend environment (see doc_auto/setup.md)",
                    section="runners",
                    required_for="gpu",
                )
            )
            continue
        payload: dict[str, object] = {}
        for line in reversed([line for line in proc.stdout.splitlines() if line.strip().startswith("{")]):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        checks.append(
            Check(
                name=f"runner:{name}",
                status="ok" if proc.returncode == 0 else "warn",
                detail=str(payload.get("detail", (proc.stderr or proc.stdout).strip()[-160:])),
                fix="" if proc.returncode == 0 else "check the backend environment and weights",
                section="runners",
                required_for="gpu",
            )
        )
    return checks


def _conda_env_of(command: Sequence[str]) -> str | None:
    """Extract ``<env>`` from ``conda run -n <env> python`` style commands."""
    parts = list(command)
    if "run" in parts:
        index = parts.index("run")
        for flag in ("-n", "--name"):
            if flag in parts[index:]:
                position = parts.index(flag, index)
                if position + 1 < len(parts):
                    return parts[position + 1]
    return None


def _conda_environments() -> set[str] | None:
    """Names of the conda environments visible here; ``None`` if conda is absent."""
    if shutil.which("conda") is None:
        return None
    try:
        proc = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    names: set[str] = set()
    for path in payload.get("envs", []):
        names.add(Path(str(path)).name)
    return names


def check_data(config_path: str | Path, *, clip: str | None = None) -> list[Check]:
    """Sample dataset, decoded frames, MANO model and ground truth."""
    from ..config import load_config
    from ..io.artefacts import ClipLayout

    checks: list[Check] = []
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("skipping data checks: %s", exc)
        return []

    sample = Path(str(config.get("paths.lerobot_root", "data/samples/lerobot_v3")))
    checks.append(
        Check(
            name="sample dataset",
            status="ok" if sample.is_dir() else "warn",
            detail=str(sample) if sample.is_dir() else f"{sample} not found",
            fix="the sample ships with the workspace; re-run the demo once it is back",
            section="data",
        )
    )
    mano = config.get("paths.mano_model", None)
    if mano:
        checks.append(
            Check(
                name="MANO model",
                status="ok" if Path(str(mano)).exists() else "missing",
                detail=str(mano),
                fix="python scripts/convert_mano.py --input MANO_RIGHT.pkl --output-dir weights/mano",
                section="data",
            )
        )
    else:
        checks.append(
            Check(
                name="MANO model",
                status="warn",
                detail="not configured -> references are wrist-only",
                fix="set paths.mano_model (needs the licence-gated MANO asset)",
                section="data",
            )
        )
    if clip:
        layout = ClipLayout(data_root=Path(str(config.require("paths.data_root"))), clip=clip)
        for name, path, fix in (
            ("frames", layout.frames_dir, f"scripts/run_preprocess.py --clip {clip} <video>"),
            ("clip metadata", layout.metadata_path, "produced by Phase 0 / the LeRobot importer"),
            ("detection", layout.detection_path, "scripts/run_detection.py"),
            ("hand", layout.hand_path, "scripts/run_hand.py"),
            ("stitched camera", layout.stitched_camera_path, "scripts/run_stitch.py"),
            ("trajectory", layout.trajectory_path, "scripts/run_fusion.py + scripts/run_refine.py"),
        ):
            exists = path.exists()
            checks.append(
                Check(
                    name=f"clip:{name}",
                    status="ok" if exists else "warn",
                    detail=str(path) if exists else f"missing ({path})",
                    fix="" if exists else f"run {fix}",
                    section="data",
                    required_for="gpu",
                )
            )
    return checks


def run_all(
    config_path: str | Path = "configs/default.yaml",
    *,
    clip: str | None = None,
    with_runners: bool = False,
) -> list[Check]:
    """Collect every check; never raises."""
    checks = check_host_tools()
    checks += check_python_dependencies()
    checks += check_python_dependencies(
        dependencies=BACKEND_DEPENDENCIES,
        required_for="gpu",
        fix="the model backends bring their own env (see doc_auto/setup.md)",
    )
    checks += check_configuration(config_path)
    from ..config import load_config

    try:
        config = load_config(config_path)
        third_party = Path(str(config.get("paths.third_party", "third_party")))
        weights_root = Path(str(config.get("paths.weights", "weights")))
        checks += check_backends(third_party, weights_root)
    except Exception as exc:  # noqa: BLE001
        logger.debug("skipping backend checks: %s", exc)
    checks += check_data(config_path, clip=clip)
    if with_runners:
        checks += check_backend_runners(config_path)
    return checks


def summarise(checks: Sequence[Check], *, strict: bool = False) -> tuple[str, int]:
    """Render a table plus a verdict; returns ``(text, exit_code)``."""
    lines: list[str] = []
    section = None
    for check in checks:
        if check.section != section:
            section = check.section
            lines.append(f"\n[{section}]")
        marker = {"ok": "ok   ", "warn": "warn ", "missing": "MISS "}[check.status]
        lines.append(f"  {marker} {check.name:24s} {check.detail}")
        if check.status != "ok" and check.fix:
            lines.append(f"        fix: {check.fix}")

    missing_cpu = [c for c in checks if c.status == "missing" and c.required_for == "cpu"]
    missing_gpu = [c for c in checks if c.status == "missing" and c.required_for == "gpu"]
    warnings = [c for c in checks if c.status == "warn"]
    lines.append("")
    if missing_cpu:
        lines.append(f"CPU path: NOT ready ({len(missing_cpu)} missing)")
    else:
        lines.append("CPU path: ready (tests, mock pipeline, reference import, evaluation)")
    if missing_gpu or warnings:
        lines.append(
            f"GPU path: not complete ({len(missing_gpu)} missing, {len(warnings)} warnings) - "
            "see doc_auto/setup.md for the download list"
        )
    else:
        lines.append("GPU path: ready")
    exit_code = 1 if missing_cpu or (strict and (missing_gpu or warnings)) else 0
    return "\n".join(lines), exit_code
