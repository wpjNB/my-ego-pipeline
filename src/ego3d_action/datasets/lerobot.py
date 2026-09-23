"""Minimal reader for LeRobot v3 datasets.

The bundled HOT3D sample (``data/samples/lerobot_v3``) is a LeRobot v3 dataset:
one parquet file holds every episode's per-frame labels, one mp4 per episode
holds the egocentric RGB, and ``meta/`` describes the schema.

Only what this project needs is implemented - no LeRobot dependency, no video
decoding (that stays with :mod:`ego3d_action.io.video`), and vector columns are
stacked into ``[T, n]`` arrays instead of being kept as Python lists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..errors import StageIOError

logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass(frozen=True)
class LeRobotInfo:
    """Parsed ``meta/info.json``."""

    root: Path
    codebase_version: str
    robot_type: str
    fps: float
    total_episodes: int
    total_frames: int
    features: Mapping[str, Mapping[str, Any]]
    data_path: str
    video_path: str
    raw: Mapping[str, Any]

    def feature_shape(self, name: str) -> tuple[int, ...]:
        feature = self.features.get(name)
        if feature is None:
            raise StageIOError(f"feature '{name}' is not present in this dataset")
        return tuple(int(dim) for dim in feature.get("shape", []))

    def vector_features(self) -> dict[str, tuple[int, ...]]:
        """Numeric features whose shape is not a scalar."""
        out: dict[str, tuple[int, ...]] = {}
        for name, feature in self.features.items():
            if feature.get("dtype") in {"video", "image"}:
                continue
            shape = tuple(int(dim) for dim in feature.get("shape", []))
            if shape and int(np.prod(shape)) > 1:
                out[name] = shape
        return out


@dataclass(frozen=True)
class LeRobotEpisode:
    """One episode: its labels as arrays plus the path to its video."""

    index: int
    length: int
    task_index: int
    task: str | None
    video_path: Path
    columns: Mapping[str, Array]

    def column(self, name: str) -> Array:
        if name not in self.columns:
            raise StageIOError(
                f"episode {self.index} has no column '{name}'; available: {sorted(self.columns)}"
            )
        return self.columns[name]

    def has(self, name: str) -> bool:
        return name in self.columns


class LeRobotDataset:
    """Read-only view over a LeRobot v3 dataset directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise StageIOError(f"{self.root} is not a LeRobot v3 dataset (no meta/info.json)")
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StageIOError(f"{info_path} is not valid JSON: {exc}") from exc
        self._info = LeRobotInfo(
            root=self.root,
            codebase_version=str(raw.get("codebase_version", "unknown")),
            robot_type=str(raw.get("robot_type", "unknown")),
            fps=float(raw.get("fps", 0.0)),
            total_episodes=int(raw.get("total_episodes", 0)),
            total_frames=int(raw.get("total_frames", 0)),
            features=raw.get("features", {}),
            data_path=str(raw.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")),
            video_path=str(raw.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")),
            raw=raw,
        )
        if self._info.fps <= 0.0:
            raise StageIOError(f"{info_path} has no positive fps")
        self._episodes_meta: dict[int, dict[str, Any]] | None = None
        self._tasks: dict[int, str] | None = None

    @property
    def info(self) -> LeRobotInfo:
        return self._info

    @property
    def video_key(self) -> str:
        for name, feature in self._info.features.items():
            if feature.get("dtype") == "video":
                return name
        raise StageIOError(f"{self.root} declares no video feature")

    # ------------------------------------------------------------------ meta
    def _read_parquet(self, path: Path, *, filters: Any = None) -> Any:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - dependency check
            raise StageIOError(
                "reading LeRobot parquet files needs pyarrow; install it in the base env "
                "(conda install -n ego3d_base pyarrow pandas)"
            ) from exc
        if not path.is_file():
            raise StageIOError(f"dataset file is missing: {path}")
        return pq.read_table(path, filters=filters)

    def episodes_meta(self) -> dict[int, dict[str, Any]]:
        """Per-episode metadata keyed by ``episode_index``."""
        if self._episodes_meta is not None:
            return self._episodes_meta
        directory = self.root / "meta" / "episodes"
        files = sorted(directory.rglob("*.parquet"))
        if not files:
            raise StageIOError(f"no episode metadata parquet files under {directory}")
        meta: dict[int, dict[str, Any]] = {}
        for path in files:
            table = self._read_parquet(path)
            for row in table.to_pylist():
                meta[int(row["episode_index"])] = row
        self._episodes_meta = meta
        return meta

    def tasks(self) -> dict[int, str]:
        """Task index -> instruction text."""
        if self._tasks is not None:
            return self._tasks
        path = self.root / "meta" / "tasks.parquet"
        table = self._read_parquet(path)
        tasks = {int(row["task_index"]): str(row["task"]) for row in table.to_pylist()}
        self._tasks = tasks
        return tasks

    def episode_indices(self) -> list[int]:
        return sorted(self.episodes_meta())

    # --------------------------------------------------------------- episode
    def _data_file(self, meta: Mapping[str, Any]) -> Path:
        relative = self._info.data_path.format(
            chunk_index=int(meta.get("data/chunk_index", 0)),
            file_index=int(meta.get("data/file_index", 0)),
        )
        return self.root / relative

    def _video_file(self, meta: Mapping[str, Any]) -> Path:
        key = self.video_key
        relative = self._info.video_path.format(
            video_key=key,
            chunk_index=int(meta.get(f"videos/{key}/chunk_index", 0)),
            file_index=int(meta.get(f"videos/{key}/file_index", 0)),
        )
        return self.root / relative

    def load_episode(self, index: int) -> LeRobotEpisode:
        """Load one episode's labels and locate its video.

        Raises:
            StageIOError: unknown episode, missing video, or a shape that
                disagrees with ``meta/info.json``.
        """
        meta = self.episodes_meta().get(int(index))
        if meta is None:
            raise StageIOError(
                f"episode {index} is not in this dataset (available: {self.episode_indices()})"
            )
        length = int(meta.get("length", 0))
        video_path = self._video_file(meta)
        if not video_path.is_file():
            raise StageIOError(f"video for episode {index} is missing: {video_path}")

        table = self._read_parquet(self._data_file(meta), filters=[("episode_index", "=", int(index))])
        if table.num_rows != length:
            logger.warning(
                "episode %d: metadata says %d frames but the data file has %d",
                index,
                length,
                table.num_rows,
            )
        length = table.num_rows
        columns: dict[str, Array] = {}
        for name, shape in self._info.vector_features().items():
            if name not in table.column_names:
                continue
            values = np.stack(
                [np.asarray(row, dtype=np.float64) for row in table.column(name).to_pylist()]
            )
            if values.shape[1:] != shape:
                raise StageIOError(
                    f"column '{name}' has shape {values.shape[1:]} but info.json says {shape}"
                )
            columns[name] = values
        for name in table.column_names:
            if name in columns or name in self._info.vector_features():
                continue
            feature = self._info.features.get(name, {})
            dtype = bool if feature.get("dtype") == "bool" else np.float64
            values = np.asarray(table.column(name).to_pylist())
            columns[name] = values.astype(dtype) if values.ndim == 1 else values
        # Columns declared with shape [1] are per-frame series, not vectors.
        for name in list(columns):
            if columns[name].ndim == 2 and columns[name].shape[1] == 1:
                columns[name] = columns[name][:, 0]

        task_index = int(meta.get("task_index", table.column("task_index")[0].as_py()))
        tasks = self.tasks()
        return LeRobotEpisode(
            index=int(index),
            length=length,
            task_index=task_index,
            task=tasks.get(task_index),
            video_path=video_path,
            columns=columns,
        )
