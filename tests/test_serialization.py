"""Stage artefacts: atomic writes, strict reads, trajectory contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego3d_action.errors import StageIOError
from ego3d_action.io.serialization import (
    load_json,
    load_npz,
    save_json,
    save_npz,
    save_trajectory,
    validate_trajectory,
)


def test_npz_roundtrip_preserves_arrays(tmp_path: Path) -> None:
    boxes = np.arange(8, dtype=np.float64).reshape(2, 4)
    flags = np.array([True, False])
    save_npz(tmp_path / "boxes.npz", boxes=boxes, flags=flags)

    data = load_npz(tmp_path / "boxes.npz", required=("boxes", "flags"))
    assert np.allclose(data["boxes"], boxes)
    assert data["flags"].tolist() == [True, False]


def test_load_npz_reports_missing_keys(tmp_path: Path) -> None:
    save_npz(tmp_path / "a.npz", boxes=np.zeros((2, 4)))
    with pytest.raises(StageIOError, match="missing required fields"):
        load_npz(tmp_path / "a.npz", required=("boxes", "valid"))


def test_load_npz_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(StageIOError):
        load_npz(tmp_path / "nope.npz")


def test_save_npz_rejects_empty_payload(tmp_path: Path) -> None:
    with pytest.raises(StageIOError):
        save_npz(tmp_path / "empty.npz")


def test_json_roundtrip(tmp_path: Path) -> None:
    payload = {"fps": 30.0, "clip": "demo01", "nested": {"a": [1, 2, 3]}}
    path = tmp_path / "metadata.json"
    save_json(path, payload)
    assert load_json(path) == payload


def test_load_json_missing_and_malformed(tmp_path: Path) -> None:
    with pytest.raises(StageIOError):
        load_json(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(StageIOError):
        load_json(broken)


def test_writes_are_atomic_and_leave_no_temporary_files(tmp_path: Path) -> None:
    save_json(tmp_path / "meta.json", {"a": 1})
    save_npz(tmp_path / "array.npz", x=np.zeros(3))
    leftovers = [path.name for path in tmp_path.iterdir() if path.name.startswith(".")]
    assert leftovers == []


def make_valid_arrays(total: int = 6) -> dict[str, np.ndarray]:
    return {
        "frames": np.arange(total, dtype=np.int64),
        "timestamps": np.arange(total, dtype=np.float64) / 30.0,
        "hand_xyz_world": np.zeros((total, 2, 21, 3)),
        "hand_xyz_camera": np.zeros((total, 2, 21, 3)),
        "hand_valid": np.ones((total, 2), dtype=bool),
        "hand_confidence": np.ones((total, 2)),
        "camera_R_c2w": np.broadcast_to(np.eye(3), (total, 3, 3)).copy(),
        "camera_t_c2w": np.zeros((total, 3)),
        "camera_K": np.broadcast_to(np.eye(3), (total, 3, 3)).copy(),
        "bbox": np.zeros((total, 2, 4)),
        "track_id": np.zeros((total, 2), dtype=np.int64),
        "mano_root_rot": np.broadcast_to(np.eye(3), (total, 2, 3, 3)).copy(),
        "mano_hand_pose": np.broadcast_to(np.eye(3), (total, 2, 15, 3, 3)).copy(),
        "mano_betas": np.zeros((total, 2, 10)),
        "postprocess_valid": np.ones((total, 2), dtype=bool),
    }


def test_save_trajectory_validates_and_writes_both_files(tmp_path: Path) -> None:
    arrays = make_valid_arrays()
    metadata = {"fps": 30.0, "world_frame": 0}
    npz_path, json_path = save_trajectory(
        tmp_path / "trajectory.npz", tmp_path / "metadata.json", arrays, metadata
    )
    assert npz_path.is_file() and json_path.is_file()
    loaded = load_npz(npz_path, required=tuple(arrays))
    assert validate_trajectory(loaded) == []
    assert load_json(json_path) == metadata


def test_save_trajectory_rejects_invalid_payloads(tmp_path: Path) -> None:
    arrays = make_valid_arrays()
    del arrays["camera_K"]
    with pytest.raises(StageIOError, match="invalid"):
        save_trajectory(tmp_path / "t.npz", tmp_path / "m.json", arrays, {"fps": 30})


def test_validate_trajectory_flags_object_dtype() -> None:
    arrays = make_valid_arrays()
    arrays["bbox"] = np.empty((6, 2, 4), dtype=object)
    problems = validate_trajectory(arrays)
    assert any("object dtype" in problem for problem in problems)
