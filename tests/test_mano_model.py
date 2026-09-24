"""MANO forward kinematics, landmark convention and model loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego3d_action.errors import StageIOError
from ego3d_action.hand.mano import JOINT_PARENTS
from ego3d_action.hand.mano_model import (
    MANO_FINGERTIP_VERTICES,
    MANO_JOINT_PARENTS,
    MANO_TO_LANDMARK,
    forward_kinematics,
    landmarks_match_topology,
    load_mano_model,
    mirror_to_left,
    rest_landmarks,
    validate_landmark_mapping,
)
from ego3d_action.testing.synthetic import (
    make_synthetic_mano_model,
    mano_fingertip_positions,
    synthetic_mano_joints,
    write_synthetic_mano_npz,
)


def identity_pose(batch: int = 1) -> np.ndarray:
    return np.broadcast_to(np.eye(3), (batch, 15, 3, 3)).copy()


def test_topology_constants_are_internally_consistent() -> None:
    assert len(MANO_JOINT_PARENTS) == 16
    assert MANO_JOINT_PARENTS[0] == -1
    assert len(MANO_TO_LANDMARK) == 21
    assert MANO_TO_LANDMARK[0] == ("joint", 0)
    # Each project landmark maps to a distinct source.
    assert len({source for source, _ in MANO_TO_LANDMARK if source == "joint"}) == 1
    assert len([1 for source, _ in MANO_TO_LANDMARK if source == "vertex"]) == 5
    # The project topology must be the landmark ordering the mapping produces.
    assert JOINT_PARENTS[0] == -1
    assert len(JOINT_PARENTS) == 21


def test_identity_pose_reproduces_the_template() -> None:
    model = make_synthetic_mano_model()
    landmarks = forward_kinematics(model, np.zeros((1, 10)), identity_pose())
    expected_joints = synthetic_mano_joints()
    expected_tips = mano_fingertip_positions(expected_joints)

    tips = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}
    for slot, (source, index) in enumerate(MANO_TO_LANDMARK):
        if source == "joint":
            assert np.allclose(landmarks[0, slot], expected_joints[index], atol=1e-12), slot
        else:
            finger = next(name for name, vertex in MANO_FINGERTIP_VERTICES.items() if vertex == index)
            assert np.allclose(landmarks[0, slot], expected_tips[tips[finger]], atol=1e-12), slot


def test_landmarks_follow_the_project_topology() -> None:
    model = make_synthetic_mano_model()
    landmarks = forward_kinematics(model, np.zeros((1, 10)), identity_pose())
    assert landmarks_match_topology(landmarks)

    # Reverse the index finger chain (tip before dip): the distance from the
    # wrist must stop being monotone along the chain.
    shuffled = landmarks.copy()
    shuffled[:, 6] = landmarks[:, 8]
    shuffled[:, 8] = landmarks[:, 6]
    assert not landmarks_match_topology(shuffled)


def test_root_translation_places_the_wrist_exactly() -> None:
    model = make_synthetic_mano_model()
    translation = np.array([[0.3, -0.2, 1.4]])
    landmarks = forward_kinematics(
        model,
        np.zeros((1, 10)),
        identity_pose(),
        root_translation=translation,
    )
    assert np.allclose(landmarks[0, 0], translation[0], atol=1e-12)
    # The hand shape is preserved: offsets between landmarks are unchanged.
    reference = forward_kinematics(model, np.zeros((1, 10)), identity_pose())
    assert np.allclose(
        landmarks[0] - landmarks[0, 0], reference[0] - reference[0, 0], atol=1e-12
    )


def test_root_rotation_rotates_the_whole_hand() -> None:
    model = make_synthetic_mano_model()
    reference = forward_kinematics(model, np.zeros((1, 10)), identity_pose())
    rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    rotated = forward_kinematics(
        model, np.zeros((1, 10)), identity_pose(), root_rotation=rotation[None]
    )
    assert np.allclose(rotated[0], reference[0] @ rotation.T, atol=1e-9)


def test_local_rotation_moves_only_its_own_finger() -> None:
    model = make_synthetic_mano_model()
    reference = forward_kinematics(model, np.zeros((1, 10)), identity_pose())

    pose = identity_pose()
    pose[0, 0] = Rotation.from_euler("z", 40.0, degrees=True).as_matrix()  # MANO joint 1 = index mcp
    posed = forward_kinematics(model, np.zeros((1, 10)), pose)

    # Index landmarks are 5..8 in the project convention.
    assert np.linalg.norm(posed[0, 6] - reference[0, 6]) > 1e-3
    assert np.linalg.norm(posed[0, 8] - reference[0, 8]) > 1e-3
    # The rotated joint itself stays put (it rotates about its own origin).
    assert np.allclose(posed[0, 5], reference[0, 5], atol=1e-9)
    for slot in (1, 9, 13, 17):  # thumb, middle, ring, pinky mcps
        assert np.allclose(posed[0, slot], reference[0, slot], atol=1e-9)


def test_shape_parameters_scale_the_hand() -> None:
    model = make_synthetic_mano_model(shape_scale=0.1)
    reference = forward_kinematics(model, np.zeros((1, 10)), identity_pose())
    bigger = forward_kinematics(model, np.array([[1.0] + [0.0] * 9]), identity_pose())
    assert np.allclose(bigger[0], reference[0] * 1.1, atol=1e-9)


def test_pose_blend_shapes_are_applied() -> None:
    model = make_synthetic_mano_model(pose_dirs_scale=0.02)
    reference = forward_kinematics(model, np.zeros((1, 10)), identity_pose())
    pose = identity_pose()
    pose[0, 0] = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    posed = forward_kinematics(model, np.zeros((1, 10)), pose)
    # posedirs act on the mesh, so the fingertip (vertex) must move as well.
    assert not np.allclose(posed[0, 8], reference[0, 8], atol=1e-6)


def test_batch_and_shape_validation() -> None:
    model = make_synthetic_mano_model()
    batch = forward_kinematics(model, np.zeros((4, 10)), identity_pose(4))
    assert batch.shape == (4, 21, 3)
    with pytest.raises(StageIOError):
        forward_kinematics(model, np.zeros((4, 10)), identity_pose(3))
    with pytest.raises(StageIOError):
        forward_kinematics(model, np.zeros(10), identity_pose(1))
    with pytest.raises(StageIOError):
        forward_kinematics(model, np.zeros((1, 10)), identity_pose(1), root_translation=np.zeros((2, 3)))
    with pytest.raises(StageIOError):
        forward_kinematics(model, np.zeros((1, 10)), identity_pose(1), root_rotation=np.eye(3))


def test_mirror_to_left_flips_x() -> None:
    model = make_synthetic_mano_model()
    mirrored = mirror_to_left(model)
    assert mirrored.hands == "left" and mirrored.mirrored
    reference = forward_kinematics(model, np.zeros((1, 10)), identity_pose())
    flipped = forward_kinematics(mirrored, np.zeros((1, 10)), identity_pose())
    expected = reference.copy()
    expected[..., 0] *= -1
    assert np.allclose(flipped, expected, atol=1e-9)
    assert not np.array_equal(mirrored.faces, model.faces)  # winding is flipped


def test_mirroring_an_already_left_model_is_a_noop() -> None:
    model = make_synthetic_mano_model()
    left = mirror_to_left(mirror_to_left(model))
    assert left.hands == "left"
    assert np.allclose(left.v_template, model.v_template)


def test_loader_roundtrip_and_validation(tmp_path: Path) -> None:
    path = write_synthetic_mano_npz(tmp_path / "MANO_RIGHT.npz")
    model = load_mano_model(path)
    assert model.num_vertices == 800
    assert model.num_betas == 10
    assert model.hands == "right"
    assert model.faces is not None

    with pytest.raises(StageIOError, match="not found"):
        load_mano_model(tmp_path / "missing.npz")

    broken = tmp_path / "broken.npz"
    np.savez(broken, v_template=np.zeros((800, 3)))
    with pytest.raises(StageIOError, match="missing"):
        load_mano_model(broken)

    wrong = tmp_path / "MANO_LEFT.npz"
    np.savez(
        wrong,
        v_template=np.zeros((800, 3)),
        shapedirs=np.zeros((800, 3, 10)),
        j_regressor=np.zeros((15, 800)),
        weights=np.zeros((800, 16)),
    )
    with pytest.raises(StageIOError):
        load_mano_model(wrong)


def test_model_rejects_inconsistent_shapes() -> None:
    from ego3d_action.hand.mano_model import ManoModel

    with pytest.raises(StageIOError, match="fingertip indices"):
        ManoModel(
            v_template=np.zeros((100, 3)),
            shapedirs=np.zeros((100, 3, 10)),
            j_regressor=np.zeros((16, 100)),
            weights=np.repeat(np.eye(16)[0][None], 100, axis=0),
        )
    with pytest.raises(StageIOError, match="sum to 1"):
        ManoModel(
            v_template=np.zeros((800, 3)),
            shapedirs=np.zeros((800, 3, 10)),
            j_regressor=np.zeros((16, 800)),
            weights=np.zeros((800, 16)),
        )


# ------------------------------------------------- real MANO (licence-gated)

MANO_DIR = Path("weights/mano")


def test_reads_the_official_pickle_without_chumpy() -> None:
    """The official archive wraps only `shapedirs` in chumpy; we materialise it."""
    source = MANO_DIR / "MANO_RIGHT.pkl"
    if not source.is_file():
        pytest.skip("MANO is licence-gated and not present in this checkout")
    model = load_mano_model(source)
    assert model.num_vertices == 778
    assert model.num_betas == 10
    assert model.j_regressor.shape == (16, 778)
    assert model.weights.shape == (778, 16)
    assert model.posedirs is not None and model.posedirs.shape == (778, 3, 135)
    assert model.faces is not None and model.faces.shape == (1538, 3)
    assert model.j_regressor.sum() > 0  # the archive spells it `J_regressor`


def test_real_model_rest_pose_looks_like_a_hand() -> None:
    """A pose-independent check of the landmark mapping against the real model."""
    source = MANO_DIR / "MANO_RIGHT.pkl"
    if not source.is_file():
        pytest.skip("MANO is licence-gated and not present in this checkout")
    model = load_mano_model(source)
    ok, detail = validate_landmark_mapping(model)
    assert ok, detail
    assert "monotone" in detail

    landmarks = rest_landmarks(model)
    wrist = landmarks[0, 0]
    reach = {
        name: float(np.linalg.norm(landmarks[0, index] - wrist))
        for index, name in ((12, "middle"), (8, "index"), (16, "ring"), (20, "pinky"), (4, "thumb"))
    }
    # A real adult hand: ~15-20 cm reach, middle finger longest, thumb shortest.
    assert 0.13 < reach["middle"] < 0.22
    assert reach["middle"] == max(reach.values())
    assert reach["thumb"] == min(reach.values())
