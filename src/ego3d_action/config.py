"""Configuration loading and validation.

The reference configuration of the blog post lives in
``configs/macrodata_final.yaml``. Everything the pipeline reads at runtime
comes through this module, so an inconsistency (``overlap >= window``, an
unknown device, a threshold outside ``[0, 1]``) fails at start-up with a
:class:`~ego3d_action.errors.ConfigError` instead of deep inside a stage.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/macrodata_final.yaml")


@dataclass
class Config:
    """A validated, dotted-access view over the YAML configuration tree."""

    data: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Fetch ``"camera.window"`` style keys."""
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None:
            raise ConfigError(f"missing required config key '{key}'")
        return value

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot set '{key}': '{part}' is not a mapping")
        node[parts[-1]] = value

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        clone = Config(data=copy.deepcopy(self.data), source=self.source)
        for key, value in overrides.items():
            clone.set(key, value)
        return clone

    def section(self, name: str) -> dict[str, Any]:
        node = self.get(name, {})
        if not isinstance(node, Mapping):
            raise ConfigError(f"config section '{name}' must be a mapping")
        return dict(node)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML configuration file.

    Raises:
        ConfigError: when the file is missing, not valid YAML, not a mapping,
            or fails :func:`validate_config`.
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not target.is_file():
        raise ConfigError(f"config file not found: {target}")
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{target} must contain a YAML mapping at the top level")
    config = Config(data=raw, source=target)
    validate_config(config)
    logger.debug("loaded config from %s", target)
    return config


def _check_unit_interval(config: Config, key: str, *, allow_none: bool = False) -> None:
    value = config.get(key, None)
    if value is None and allow_none:
        return
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ConfigError(f"'{key}' must be a number in [0, 1], got {value!r}")


def _check_window_overlap(config: Config, prefix: str) -> None:
    window = config.get(f"{prefix}.window")
    overlap = config.get(f"{prefix}.overlap")
    if not isinstance(window, int) or window <= 0:
        raise ConfigError(f"'{prefix}.window' must be a positive integer, got {window!r}")
    if not isinstance(overlap, int) or overlap < 0:
        raise ConfigError(f"'{prefix}.overlap' must be a non-negative integer, got {overlap!r}")
    if overlap >= window:
        raise ConfigError(
            f"'{prefix}.overlap' ({overlap}) must be smaller than '{prefix}.window' ({window})"
        )


def validate_config(config: Config) -> None:
    """Check cross-field invariants; raises :class:`ConfigError`."""
    from .runtime.device import resolve_device

    device = config.get("runtime.device", "auto")
    if device not in {"auto", "cpu", "gpu"} and not str(device).startswith("cuda"):
        raise ConfigError(f"runtime.device must be auto|cpu|cuda[:N], got {device!r}")
    if str(device) == "cuda":
        # Fails loudly on a CPU-only machine instead of silently using the CPU.
        try:
            resolve_device("cuda")
        except Exception as exc:  # noqa: BLE001 - re-raised as a config error
            raise ConfigError(
                f"runtime.device is 'cuda' but CUDA is unavailable: {exc}. "
                "Use 'auto' or 'cpu' on this machine."
            ) from exc

    if not config.get("paths.data_root"):
        raise ConfigError("'paths.data_root' must be set")

    _check_unit_interval(config, "detection.min_confidence")
    _check_unit_interval(config, "detection.iou_threshold")
    gap = config.get("detection.max_gap")
    if not isinstance(gap, int) or gap < 0:
        raise ConfigError(f"'detection.max_gap' must be a non-negative integer, got {gap!r}")

    _check_window_overlap(config, "hand")
    _check_window_overlap(config, "camera")

    resolution = config.get("camera.resolution")
    if not isinstance(resolution, int) or resolution <= 0:
        raise ConfigError(f"'camera.resolution' must be a positive integer, got {resolution!r}")

    stride = config.get("stitch.pixel_stride")
    if not isinstance(stride, int) or stride <= 0:
        raise ConfigError(f"'stitch.pixel_stride' must be a positive integer, got {stride!r}")
    threshold = config.get("stitch.inlier_threshold", None)
    if threshold is not None and (not isinstance(threshold, (int, float)) or threshold <= 0):
        raise ConfigError(f"'stitch.inlier_threshold' must be positive or null, got {threshold!r}")

    bound = config.get("refinement.bone_scale_max_correction")
    if not isinstance(bound, (int, float)) or not 0.0 <= float(bound) <= 1.0:
        raise ConfigError(f"'refinement.bone_scale_max_correction' must be in [0, 1], got {bound!r}")
    lam = config.get("refinement.wrist_depth_lambda")
    if not isinstance(lam, (int, float)) or float(lam) < 0.0:
        raise ConfigError(f"'refinement.wrist_depth_lambda' must be >= 0, got {lam!r}")

    chunk = config.get("evaluation.chunk_seconds")
    if not isinstance(chunk, (int, float)) or float(chunk) <= 0.0:
        raise ConfigError(f"'evaluation.chunk_seconds' must be positive, got {chunk!r}")

    from .runtime.subprocess_backend import VALID_MODES, BackendInvocation

    mode = config.get("backends.mode", "real")
    if mode not in VALID_MODES:
        raise ConfigError(f"'backends.mode' must be one of {VALID_MODES}, got {mode!r}")
    if mode == "real":
        invocation = BackendInvocation.from_config(config)
        for name in ("wilor", "hawor", "vggt"):
            if not invocation.python_commands.get(name):
                raise ConfigError(
                    f"'backends.python.{name}' must be set for backends.mode=real "
                    f'(e.g. ["conda", "run", "-n", "ego3d_{name}", "python"])'
                )
    timeout = config.get("backends.timeout_seconds")
    if not isinstance(timeout, (int, float)) or float(timeout) <= 0:
        raise ConfigError(f"'backends.timeout_seconds' must be positive, got {timeout!r}")
