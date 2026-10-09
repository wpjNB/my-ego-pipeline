"""WiLoR hand backend: crop->camera conversion, joint mapping, shims."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str) -> object:
    """Import a ``backends/*.py`` runner as a module."""
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cam_crop_to_full_matches_the_reference_formula() -> None:
    """tz = 2f/(b*s); the tangent terms shift (tx, ty) by the box offset."""
    runner = load_script("wilor_hand_runner_t1", "backends/wilor_hand_runner.py")
    f = 227.0
    s, tx_c, ty_c, b = 5.0, 0.05, -0.02, 180.0
    cam = np.array([[s, tx_c, ty_c]])
    center = np.array([[256.0, 256.0]])
    size = np.array([b])
    img = np.array([[512.0, 512.0]])
    t = runner.cam_crop_to_full(cam, center, size, img, f)
    assert t.shape == (1, 3)
    assert np.isclose(t[0, 2], 2 * f / (b * s))
    # centred box: only the weak-perspective offset remains
    assert np.isclose(t[0, 0], tx_c)
    assert np.isclose(t[0, 1], ty_c)
    # offset box: tangent term adds 2*(cx - w/2)/(b*s)
    center2 = np.array([[256.0 + 40.0, 256.0 - 30.0]])
    t2 = runner.cam_crop_to_full(cam, center2, size, img, f)
    assert np.isclose(t2[0, 0], tx_c + 2 * 40.0 / (b * s))
    assert np.isclose(t2[0, 1], ty_c + 2 * (-30.0) / (b * s))
    assert np.isclose(t2[0, 2], t[0, 2])


def test_undo_openpose_remap_recovers_the_full_joint_set() -> None:
    """openpose[j] = full[joint_map[j]]; inverting must restore every slot."""
    runner = load_script("wilor_hand_runner_t2", "backends/wilor_hand_runner.py")
    joint_map = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
    full = np.arange(21, dtype=np.float64).reshape(1, 21, 1) * np.ones((1, 21, 3))
    remapped = full[:, joint_map, :]
    recovered = runner.undo_openpose_remap(remapped, joint_map)
    assert np.array_equal(recovered, full)
    # permutation sanity: every source index used exactly once
    assert sorted(joint_map.tolist()) == list(range(21))


def test_our_landmarks_follow_the_project_convention() -> None:
    """wrist first, then thumb..pinky chains; tips come from mesh vertices."""
    runner = load_script("wilor_hand_runner_t3", "backends/wilor_hand_runner.py")
    from ego3d_action.hand.mano_model import MANO_TO_LANDMARK

    assert len(MANO_TO_LANDMARK) == 21
    assert MANO_TO_LANDMARK[0] == ("joint", 0)
    vertices = np.full((1, 778, 3), 7.0)
    joints = np.zeros((1, 21, 3))
    for i in range(21):
        joints[:, i, :] = i
    land = runner.our_landmarks(vertices, joints)
    assert land.shape == (1, 21, 3)
    assert np.all(land[0, 0] == 0)  # wrist
    # sources respected: every landmark matches its (source, index) entry
    for slot, (source, index) in enumerate(MANO_TO_LANDMARK):
        expected = joints[0, index] if source == "joint" else vertices[0, index]
        assert np.array_equal(land[0, slot], expected)


def test_pyrender_stub_install_is_idempotent() -> None:
    runner = load_script("wilor_hand_runner_t4", "backends/wilor_hand_runner.py")
    runner.install_pyrender_stub()
    import pyrender  # noqa: PLC0415 - resolved through the stub when absent

    assert hasattr(pyrender, "OffscreenRenderer")
    runner.install_pyrender_stub()  # second call must not raise or replace


def test_wilor_preview_box_nudge_defaults_off_and_allows_override() -> None:
    runner = load_script("wilor_hand_entry_nudge_test", "scripts/run_hand_wilor.py")

    class Config:
        def __init__(self, values):
            self.values = values

        def get(self, key, default=None):
            return self.values.get(key, default)

    assert runner.resolve_wilor_box_nudge(Config({})) == 0.0
    assert runner.resolve_wilor_box_nudge(
        Config({"visualization.wilor_box_nudge": 0.5})
    ) == 0.5


def test_helpers_locate_checkout_files(tmp_path: Path) -> None:
    runner = load_script("wilor_hand_runner_t5", "backends/wilor_hand_runner.py")
    assert runner.find_checkpoint(tmp_path) is None
    assert runner.find_config(tmp_path) is None
    (tmp_path / "wilor").mkdir()
    (tmp_path / "wilor" / "wilor_final.ckpt").write_bytes(b"stub")
    (tmp_path / "wilor" / "model_config.yaml").write_text("x: 1\n")
    assert runner.find_checkpoint(tmp_path).name == "wilor_final.ckpt"
    assert runner.find_config(tmp_path).name == "model_config.yaml"
