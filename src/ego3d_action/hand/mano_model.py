"""MANO forward kinematics (numpy) and the 21-landmark convention.

The bundled HOT3D sample stores a wrist pose plus 15 joint rotations, so turning
it into the 21 metric joint positions this project evaluates needs the MANO mesh
model (``v_template``, ``shapedirs``, ``J_regressor``, ``weights``,
``posedirs``). That asset is licence-gated and is not bundled here - this module
is what activates the moment it is dropped in, and
``scripts/convert_mano.py`` converts the official pickle to an ``.npz`` that
needs no ``chumpy``.

Joint conventions
-----------------

*MANO* (16 joints, ``manopth`` / ``smplx`` numbering)::

    0 wrist | 1,2,3 index | 4,5,6 middle | 7,8,9 pinky | 10,11,12 ring | 13,14,15 thumb

*This project* (21 landmarks, matching ``hand/mano.py::JOINT_PARENTS``)::

    0 wrist | 1..4 thumb | 5..8 index | 9..12 middle | 13..16 ring | 17..20 pinky

The five fingertips are not MANO joints - they are vertices of the mesh, taken
from :data:`MANO_FINGERTIP_VERTICES` (the standard MANO tip indices).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import StageIOError
from .mano import JOINT_PARENTS, NUM_JOINTS

logger = logging.getLogger(__name__)

Array = np.ndarray

MANO_NUM_JOINTS = 16

#: MANO kinematic tree (index 0 is the root; ``-1`` marks no parent).
MANO_JOINT_PARENTS: tuple[int, ...] = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)

#: Standard MANO fingertip vertex indices.
MANO_FINGERTIP_VERTICES: dict[str, int] = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 555,
    "pinky": 672,
}

#: ``(source, index)`` per landmark of this project's 21-joint convention,
#: where ``source`` is ``"joint"`` (MANO joint) or ``"vertex"`` (mesh vertex).
MANO_TO_LANDMARK: tuple[tuple[str, int], ...] = (
    ("joint", 0),  # 0  wrist
    ("joint", 13),  # 1  thumb mcp
    ("joint", 14),  # 2  thumb pip
    ("joint", 15),  # 3  thumb dip
    ("vertex", MANO_FINGERTIP_VERTICES["thumb"]),  # 4  thumb tip
    ("joint", 1),  # 5  index mcp
    ("joint", 2),  # 6  index pip
    ("joint", 3),  # 7  index dip
    ("vertex", MANO_FINGERTIP_VERTICES["index"]),  # 8  index tip
    ("joint", 4),  # 9  middle mcp
    ("joint", 5),  # 10 middle pip
    ("joint", 6),  # 11 middle dip
    ("vertex", MANO_FINGERTIP_VERTICES["middle"]),  # 12 middle tip
    ("joint", 10),  # 13 ring mcp
    ("joint", 11),  # 14 ring pip
    ("joint", 12),  # 15 ring dip
    ("vertex", MANO_FINGERTIP_VERTICES["ring"]),  # 16 ring tip
    ("joint", 7),  # 17 pinky mcp
    ("joint", 8),  # 18 pinky pip
    ("joint", 9),  # 19 pinky dip
    ("vertex", MANO_FINGERTIP_VERTICES["pinky"]),  # 20 pinky tip
)

REQUIRED_KEYS = ("v_template", "shapedirs", "j_regressor", "weights")
OPTIONAL_KEYS = ("posedirs", "f")


@dataclass(frozen=True)
class ManoModel:
    """The MANO mesh model in the exact form the kinematics below needs."""

    v_template: Array  # [778, 3]
    shapedirs: Array  # [778, 3, n_betas]
    j_regressor: Array  # [16, 778]
    weights: Array  # [778, 16]
    posedirs: Array | None = None  # [778, 3, 9 * 15]
    faces: Array | None = None  # [n_faces, 3]
    hands: str = "right"
    source: str = "unknown"
    mirrored: bool = False

    def __post_init__(self) -> None:
        vertices = int(np.asarray(self.v_template).shape[0])
        if np.asarray(self.v_template).shape != (vertices, 3):
            raise StageIOError(f"v_template must be [V, 3], got {np.shape(self.v_template)}")
        if np.asarray(self.shapedirs).shape[:2] != (vertices, 3):
            raise StageIOError(
                f"shapedirs must be [V, 3, n_betas], got {np.shape(self.shapedirs)}"
            )
        if np.asarray(self.j_regressor).shape != (MANO_NUM_JOINTS, vertices):
            raise StageIOError(
                f"j_regressor must be [{MANO_NUM_JOINTS}, V], got {np.shape(self.j_regressor)}"
            )
        if np.asarray(self.weights).shape != (vertices, MANO_NUM_JOINTS):
            raise StageIOError(
                f"weights must be [V, {MANO_NUM_JOINTS}], got {np.shape(self.weights)}"
            )
        if self.posedirs is not None:
            expected = (vertices, 3, 9 * (MANO_NUM_JOINTS - 1))
            if np.asarray(self.posedirs).shape != expected:
                raise StageIOError(f"posedirs must be {expected}, got {np.shape(self.posedirs)}")
        if vertices <= max(MANO_FINGERTIP_VERTICES.values()):
            raise StageIOError(
                f"the model has {vertices} vertices but the fingertip indices go up to "
                f"{max(MANO_FINGERTIP_VERTICES.values())}"
            )
        if not np.isclose(np.sum(np.asarray(self.weights), axis=1), 1.0, atol=1e-3).all():
            raise StageIOError("weights must sum to 1 per vertex")

    @property
    def num_betas(self) -> int:
        return int(np.asarray(self.shapedirs).shape[2])

    @property
    def num_vertices(self) -> int:
        return int(np.asarray(self.v_template).shape[0])


def load_mano_model(path: str | Path, *, hands: str | None = None) -> ManoModel:
    """Load a MANO model from ``.npz`` (preferred) or the official ``.pkl``.

    Args:
        path: ``.npz`` produced by ``scripts/convert_mano.py``, or the official
            ``MANO_RIGHT.pkl`` / ``MANO_LEFT.pkl`` (needs ``chumpy``).
        hands: override the handedness label; inferred from the file name
            otherwise.

    Raises:
        StageIOError: missing file, missing keys or inconsistent shapes.
    """
    source = Path(path)
    if not source.is_file():
        raise StageIOError(f"MANO model not found: {source}")
    inferred = hands
    if inferred is None:
        lowered = source.name.lower()
        inferred = "left" if "left" in lowered else "right" if "right" in lowered else "unknown"

    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as handle:
            data = {key: handle[key] for key in handle.files}
    elif source.suffix == ".pkl":
        data = _read_mano_pickle(source)
    else:
        raise StageIOError(f"unsupported MANO model format: {source.suffix}")

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise StageIOError(f"{source} is missing {missing}; expected at least {REQUIRED_KEYS}")
    return ManoModel(
        v_template=np.asarray(data["v_template"], dtype=np.float64),
        shapedirs=np.asarray(data["shapedirs"], dtype=np.float64),
        j_regressor=np.asarray(data["j_regressor"], dtype=np.float64),
        weights=np.asarray(data["weights"], dtype=np.float64),
        posedirs=(
            np.asarray(data["posedirs"], dtype=np.float64) if "posedirs" in data else None
        ),
        faces=np.asarray(data["f"], dtype=np.int64) if "f" in data else None,
        hands=str(inferred),
        source=str(source),
    )


def load_mano_models(path: str | Path) -> dict[str, ManoModel]:
    """Load ``{"right": model, "left": model}`` from a file or a directory.

    A directory is scanned for ``MANO_RIGHT.*`` / ``MANO_LEFT.*`` (``.npz``
    preferred, ``.pkl`` accepted). When only a right-hand model exists, the left
    one is derived with :func:`mirror_to_left` and marked ``mirrored=True``.

    Raises:
        StageIOError: nothing loadable at ``path``.
    """
    target = Path(path)
    if target.is_dir():
        found: dict[str, Path] = {}
        for candidate in sorted(target.iterdir()):
            if not candidate.is_file() or candidate.suffix not in {".npz", ".pkl"}:
                continue
            lowered = candidate.name.lower()
            if "left" in lowered:
                found.setdefault("left", candidate)
            elif "right" in lowered:
                found.setdefault("right", candidate)
        if not found:
            raise StageIOError(
                f"no MANO_RIGHT.* / MANO_LEFT.* model found in {target}"
            )
        models = {hands: load_mano_model(file, hands=hands) for hands, file in found.items()}
    else:
        model = load_mano_model(target)
        models = {model.hands if model.hands in {"left", "right"} else "right": model}

    if "right" in models and "left" not in models:
        models["left"] = mirror_to_left(models["right"])
        logger.info("only a right-hand MANO model was found; the left one is mirrored")
    if "left" in models and "right" not in models:
        models["right"] = mirror_to_left(models["left"])
        logger.info("only a left-hand MANO model was found; the right one is mirrored")
    return models


def _read_mano_pickle(path: Path) -> dict[str, Array]:
    try:
        import pickle

        import chumpy  # noqa: F401 - needed to unpickle the official model
    except ImportError as exc:
        raise StageIOError(
            f"{path} is the official pickle and needs chumpy to unpickle. Either run "
            "scripts/convert_mano.py in the HaWoR environment (which has chumpy), or install "
            "chumpy in this one."
        ) from exc
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    out: dict[str, Array] = {}
    for key, value in raw.items():
        out[key] = np.asarray(getattr(value, "r", value))
    logger.info("read MANO pickle %s with %d entries", path.name, len(out))
    return out


def mirror_to_left(model: ManoModel) -> ManoModel:
    """Mirror a right-hand model into a left-hand one.

    Many installations only ship ``MANO_RIGHT.pkl``. Mirroring along ``x`` and
    conjugating the pose rotations is the standard trick; the result is labelled
    ``mirrored=True`` so any downstream number can be audited.
    """
    if model.hands == "left" and not model.mirrored:
        return model
    flip = np.array([-1.0, 1.0, 1.0])
    return ManoModel(
        v_template=np.asarray(model.v_template) * flip,
        shapedirs=np.asarray(model.shapedirs) * flip[None, :, None],
        j_regressor=np.asarray(model.j_regressor),
        weights=np.asarray(model.weights),
        posedirs=(
            None
            if model.posedirs is None
            else np.asarray(model.posedirs) * flip[None, :, None]
        ),
        faces=(
            None if model.faces is None else np.asarray(model.faces)[:, ::-1]  # keep winding
        ),
        hands="left",
        source=model.source,
        mirrored=True,
    )


def mirror_pose(hand_pose: Array, root_rotation: Array | None = None) -> tuple[Array, Array | None]:
    """Conjugate rotations for a mirrored model: ``R -> M R M`` with ``M = diag(-1,1,1)``.

    Mirroring the mesh alone is not enough - a left hand's joint rotations must
    be expressed in the mirrored frame, otherwise the fingers bend the wrong way.
    """
    mirror = np.diag(np.array([-1.0, 1.0, 1.0]))
    pose = np.asarray(hand_pose, dtype=np.float64)
    mirrored_pose = np.einsum("ij,...jk,kl->...il", mirror, pose, mirror)
    if root_rotation is None:
        return mirrored_pose, None
    root = np.asarray(root_rotation, dtype=np.float64)
    return mirrored_pose, np.einsum("ij,...jk,kl->...il", mirror, root, mirror)


def _homogeneous(rotation: Array, translation: Array) -> Array:
    mat = np.zeros((*rotation.shape[:-2], 4, 4), dtype=np.float64)
    mat[..., :3, :3] = rotation
    mat[..., :3, 3] = translation
    mat[..., 3, 3] = 1.0
    return mat


def forward_kinematics(
    model: ManoModel,
    betas: Array,
    hand_pose: Array,
    *,
    root_rotation: Array | None = None,
    root_translation: Array | None = None,
) -> Array:
    """Pose MANO and return the 21 landmarks.

    Args:
        model: the MANO model.
        betas: ``[B, n_betas]`` shape parameters (padded/truncated to the model).
        hand_pose: ``[B, 15, 3, 3]`` **local** rotations of the 15 non-root
            joints, in MANO joint order (matches ``mano_hand_pose`` in the
            trajectory contract).
        root_rotation: optional ``[B, 3, 3]`` wrist orientation.
        root_translation: optional ``[B, 3]`` wrist position. When given, joint 0
            is placed exactly there (the alignment the HOT3D sample uses).

    Returns:
        ``[B, 21, 3]`` landmarks in the project's convention.

    Raises:
        StageIOError: on shape mismatches.
    """
    shaped = np.asarray(betas, dtype=np.float64)
    if shaped.ndim != 2:
        raise StageIOError(f"betas must be [B, n_betas], got {shaped.shape}")
    batch = shaped.shape[0]
    pose = np.asarray(hand_pose, dtype=np.float64)
    if pose.shape != (batch, MANO_NUM_JOINTS - 1, 3, 3):
        raise StageIOError(
            f"hand_pose must be [{batch}, 15, 3, 3], got {pose.shape}"
        )

    coefficients = np.zeros((batch, model.num_betas), dtype=np.float64)
    width = min(model.num_betas, shaped.shape[1])
    coefficients[:, :width] = shaped[:, :width]

    v_shaped = np.asarray(model.v_template)[None] + np.einsum(
        "vck,bk->bvc", np.asarray(model.shapedirs), coefficients
    )
    joints = np.einsum("jv,bvc->bjc", np.asarray(model.j_regressor), v_shaped)

    local_rotation = np.zeros((batch, MANO_NUM_JOINTS, 3, 3), dtype=np.float64)
    local_rotation[:, 0] = (
        np.broadcast_to(np.eye(3), (batch, 3, 3))
        if root_rotation is None
        else np.asarray(root_rotation, dtype=np.float64)
    )
    local_rotation[:, 1:] = pose
    if root_rotation is not None and np.asarray(root_rotation).shape != (batch, 3, 3):
        raise StageIOError(
            f"root_rotation must be [{batch}, 3, 3], got {np.shape(root_rotation)}"
        )

    if model.posedirs is not None:
        pose_feature = (local_rotation[:, 1:] - np.eye(3)).reshape(batch, -1)
        v_posed = v_shaped + np.einsum("vcp,bp->bvc", np.asarray(model.posedirs), pose_feature)
    else:
        v_posed = v_shaped

    transforms: list[Array] = []
    for index, parent in enumerate(MANO_JOINT_PARENTS):
        if parent == -1:
            transform = _homogeneous(local_rotation[:, index], joints[:, index])
        else:
            relative = joints[:, index] - joints[:, parent]
            transform = transforms[parent] @ _homogeneous(local_rotation[:, index], relative)
        transforms.append(transform)
    stacked = np.stack(transforms, axis=1)  # [B, 16, 4, 4]
    rest_inverse = np.zeros((batch, MANO_NUM_JOINTS, 4, 4), dtype=np.float64)
    rest_inverse[:, :, :3, :3] = np.eye(3)
    rest_inverse[:, :, :3, 3] = -joints
    rest_inverse[:, :, 3, 3] = 1.0
    world_transform = stacked @ rest_inverse

    posed_joints = (
        np.einsum("bjik,bjk->bji", world_transform[..., :3, :3], joints)
        + world_transform[..., :3, 3]
    )
    homogeneous_vertices = np.concatenate(
        [v_posed, np.ones((batch, model.num_vertices, 1))], axis=-1
    )
    posed_vertices = np.einsum("bjik,bvk->bvji", world_transform, homogeneous_vertices)
    blended = np.einsum("vj,bvji->bvi", np.asarray(model.weights), posed_vertices)[:, :, :3]

    landmarks = np.zeros((batch, NUM_JOINTS, 3), dtype=np.float64)
    for slot, (source, index) in enumerate(MANO_TO_LANDMARK):
        landmarks[:, slot] = posed_joints[:, index] if source == "joint" else blended[:, index]

    if root_translation is not None:
        translation = np.asarray(root_translation, dtype=np.float64)
        if translation.shape != (batch, 3):
            raise StageIOError(
                f"root_translation must be [{batch}, 3], got {translation.shape}"
            )
        # Place the wrist exactly where the caller says it is.
        landmarks = landmarks - landmarks[:, :1, :] + translation[:, None, :]
    return landmarks


def landmarks_match_topology(landmarks: Array, *, tolerance: float = 0.0) -> bool:
    """Sanity-check that 21 landmarks follow :data:`JOINT_PARENTS`.

    Each finger chain must get *farther* from the wrist along its links. This is
    a cheap guard against the classic mapping bug - emitting a finger tip where a
    proximal joint belongs - not a proof that the mapping is right; the unit
    tests pin the mapping itself against a synthetic model. ``tolerance`` is an
    absolute slack in metres, and callers should treat a ``False`` result as a
    warning rather than a hard failure, since a heavily curled finger can
    legitimately break strict monotonicity.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape[-2:] != (NUM_JOINTS, 3):
        return False
    wrist = points[..., 0, :]
    for child, parent in enumerate(JOINT_PARENTS):
        if parent < 0:
            continue
        child_distance = np.linalg.norm(points[..., child, :] - wrist, axis=-1)
        parent_distance = np.linalg.norm(points[..., parent, :] - wrist, axis=-1)
        if np.any(child_distance + tolerance < parent_distance):
            return False
    return True
