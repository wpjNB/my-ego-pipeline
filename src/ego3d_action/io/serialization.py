"""Artefact serialisation.

Every pipeline stage writes its own artefact; nothing is passed between stages
as a live Python object. These helpers keep that contract strict: writes are
atomic, and reads validate required keys, so a truncated or schema-drifted file
fails loudly instead of producing a silently wrong trajectory.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray


def _atomic_write(path: Path, write: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_npz(path: str | Path, **arrays: Array) -> Path:
    """Atomically write an uncompressed ``.npz`` archive."""
    target = Path(path)
    if not arrays:
        raise StageIOError(f"refusing to write an empty archive to {target}")

    def _write(handle: Any) -> None:
        np.savez(handle, **arrays)

    _atomic_write(target, _write)
    logger.debug("wrote %s (%s)", target, ", ".join(sorted(arrays)))
    return target


def load_npz(
    path: str | Path,
    *,
    required: tuple[str, ...] | list[str] = (),
) -> dict[str, Array]:
    """Load an ``.npz`` archive and check that ``required`` keys are present."""
    source = Path(path)
    if not source.is_file():
        raise StageIOError(f"artefact not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as handle:
            data = {key: handle[key] for key in handle.files}
    except (OSError, ValueError) as exc:
        raise StageIOError(f"cannot read {source}: {exc}") from exc

    missing = [key for key in required if key not in data]
    if missing:
        raise StageIOError(
            f"{source} is missing required fields {missing}; found {sorted(data)}"
        )
    return data


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a JSON document."""
    target = Path(path)

    def _write(handle: Any) -> None:
        handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")

    _atomic_write(target, _write)
    logger.debug("wrote %s", target)
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON document, raising :class:`StageIOError` on failure."""
    source = Path(path)
    if not source.is_file():
        raise StageIOError(f"metadata not found: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageIOError(f"{source} is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------
# Final trajectory contract (agent spec section 7)
# --------------------------------------------------------------------------

TRAJECTORY_FIELDS: dict[str, str] = {
    "frames": "[T]",
    "timestamps": "[T]",
    "hand_xyz_world": "[T, 2, 21, 3]",
    "hand_xyz_camera": "[T, 2, 21, 3]",
    "hand_valid": "[T, 2]",
    "hand_confidence": "[T, 2]",
    "camera_R_c2w": "[T, 3, 3]",
    "camera_t_c2w": "[T, 3]",
    "camera_K": "[T, 3, 3]",
    "bbox": "[T, 2, 4]",
    "track_id": "[T, 2]",
    "mano_root_rot": "[T, 2, 3, 3]",
    "mano_hand_pose": "[T, 2, 15, 3, 3]",
    "mano_betas": "[T, 2, 10]",
    "postprocess_valid": "[T, 2]",
}


def _shape_of(spec: str) -> tuple[int, ...] | None:
    """Parse ``'[T, 2, 21, 3]'`` into ``(None, 2, 21, 3)`` (``T`` -> ``None``)."""
    body = spec.strip().strip("[]")
    if not body:
        return None
    dims: list[int | None] = []
    for token in body.split(","):
        token = token.strip()
        dims.append(None if token == "T" else int(token))
    return tuple(dims)


def validate_trajectory(data: Mapping[str, Array], *, strict: bool = True) -> list[str]:
    """Validate a final ``trajectory.npz`` payload.

    Args:
        data: mapping of field name to array.
        strict: when ``True`` every field of :data:`TRAJECTORY_FIELDS` must be
            present with the documented shape.

    Returns:
        A list of human readable problems; empty means the artefact is valid.
    """
    problems: list[str] = []
    missing = [name for name in TRAJECTORY_FIELDS if name not in data]
    if missing and strict:
        problems.append(f"missing fields: {sorted(missing)}")

    num_frames: int | None = None
    for name, spec in TRAJECTORY_FIELDS.items():
        if name not in data:
            continue
        arr = np.asarray(data[name])
        expected = _shape_of(spec)
        if expected is not None and arr.ndim != len(expected):
            problems.append(f"{name}: expected {len(expected)} dims ({spec}), got {arr.shape}")
            continue
        if expected is not None:
            for axis, want in enumerate(expected):
                if want is not None and arr.shape[axis] != want:
                    problems.append(f"{name}: axis {axis} should be {want} ({spec}), got {arr.shape}")
        if arr.shape[0] and spec.startswith("[T"):
            if num_frames is None:
                num_frames = int(arr.shape[0])
            elif int(arr.shape[0]) != num_frames:
                problems.append(
                    f"{name}: leading dim {arr.shape[0]} != T={num_frames} used by other fields"
                )
        if arr.dtype == np.dtype("O"):
            problems.append(f"{name}: object dtype is not allowed in a trajectory artefact")
    return problems


def save_trajectory(
    path: str | Path,
    metadata_path: str | Path,
    arrays: Mapping[str, Array],
    metadata: Mapping[str, Any],
    *,
    strict: bool = True,
) -> tuple[Path, Path]:
    """Validate, then atomically write ``trajectory.npz`` + ``metadata.json``."""
    problems = validate_trajectory(arrays, strict=strict)
    if problems:
        raise StageIOError("trajectory artefact is invalid: " + "; ".join(problems))
    npz_path = save_npz(path, **dict(arrays))
    json_path = save_json(metadata_path, metadata)
    return npz_path, json_path
