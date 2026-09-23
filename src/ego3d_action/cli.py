"""Shared command-line plumbing for ``scripts/*.py``."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Config, load_config
from .errors import ConfigError
from .io.artefacts import ClipLayout
from .runtime.device import EnvironmentReport, describe_environment

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Idempotent logging setup shared by every entry point."""
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        raise ConfigError(f"unknown log level '{level}'")
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(numeric)
        return
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def base_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with the flags every stage script understands."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(Path("configs") / "macrodata_final.yaml"),
        help="YAML configuration file (default: configs/macrodata_final.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config entry, e.g. --set camera.window=200",
    )
    parser.add_argument("--data-root", default=None, help="override paths.data_root")
    parser.add_argument("--clip", default=None, help="clip identifier (default: derived from the input)")
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:N | auto")
    parser.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING | ERROR")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan only")
    return parser


def parse_overrides(pairs: list[str]) -> dict[str, object]:
    """Parse ``--set key=value`` pairs into a nested-override mapping."""
    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"--set expects KEY=VALUE, got '{pair}'")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"--set expects KEY=VALUE, got '{pair}'")
        overrides[key] = _parse_scalar(raw.strip())
    return overrides


def _parse_scalar(raw: str) -> object:
    lowered = raw.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


@dataclass(frozen=True)
class RunContext:
    """Everything a stage script needs after parsing its arguments."""

    config: Config
    environment: EnvironmentReport
    data_root: Path
    layout: ClipLayout | None

    @property
    def device(self) -> str:
        return self.environment.resolved_device

    def path(self, key: str) -> Path:
        value = self.config.require(key)
        return Path(str(value))


def build_context(args: argparse.Namespace, *, clip: str | None = None) -> RunContext:
    """Load the config, apply CLI overrides and resolve the environment."""
    config = load_config(args.config)
    overrides = parse_overrides(getattr(args, "overrides", []) or [])
    if getattr(args, "data_root", None):
        overrides["paths.data_root"] = args.data_root
    if getattr(args, "device", None):
        overrides["runtime.device"] = args.device
    if overrides:
        config = config.with_overrides(overrides)

    from .config import validate_config

    validate_config(config)

    level = getattr(args, "log_level", None) or str(config.get("runtime.log_level", "INFO"))
    configure_logging(level)

    environment = describe_environment(str(config.get("runtime.device", "auto")))
    data_root = Path(str(config.require("paths.data_root")))
    clip_name = clip or getattr(args, "clip", None)
    layout = ClipLayout(data_root=data_root, clip=clip_name) if clip_name else None
    return RunContext(config=config, environment=environment, data_root=data_root, layout=layout)


def fail(message: str, *, code: int = 2) -> int:
    """Log an error and return a process exit code."""
    logger.error("%s", message)
    print(f"error: {message}", file=sys.stderr)
    return code
