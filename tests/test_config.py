"""Config loading, overrides and cross-field validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ego3d_action.cli import parse_overrides
from ego3d_action.config import load_config, validate_config
from ego3d_action.errors import ConfigError

# Shipped configs with the camera schedule each one pins: the reference
# profiles all carry 200/40; hot3d_p100.yaml is the documented P100 host
# variant (sm_60 has no flash-attention kernel, so 8 is the OOM ceiling).
CONFIG_FILES = {
    "default.yaml": (200, 40),
    "macrodata_final.yaml": (200, 40),
    "hot3d.yaml": (200, 40),
    "hot3d_p100.yaml": (8, 4),
}


@pytest.mark.parametrize("name,expected_camera", sorted(CONFIG_FILES.items()))
def test_shipped_configs_are_valid(name: str, expected_camera: tuple[int, int]) -> None:
    config = load_config(Path("configs") / name)
    assert config.get("hand.window") == 16
    assert config.get("hand.overlap") == 8
    assert config.get("camera.window") == expected_camera[0]
    assert config.get("camera.overlap") == expected_camera[1]


def test_macrodata_final_matches_the_reference_table() -> None:
    config = load_config("configs/macrodata_final.yaml")
    assert config.get("detection.min_confidence") == 0.75
    assert config.get("detection.max_gap") == 4
    assert config.get("detection.iou_threshold") == 0.20
    assert config.get("camera.resolution") == 416
    assert config.get("camera.checkpoint") == "VGGT-Omega-1B-416-Reproduction"
    assert config.get("stitch.blend") is True
    assert config.get("refinement.bone_scale_max_correction") == 0.035
    assert config.get("refinement.wrist_depth_lambda") == 0.2
    assert config.get("evaluation.chunk_seconds") == 1.0


@pytest.mark.parametrize("name", ("macrodata_final.yaml", "hot3d.yaml"))
def test_the_hand_reference_configs_point_at_the_installed_mano(name: str) -> None:
    """Both configs that build 21-joint references must find weights/mano."""
    config = load_config(Path("configs") / name)
    assert config.get("paths.mano_model") == "weights/mano"


def test_missing_config_raises() -> None:
    with pytest.raises(ConfigError):
        load_config("configs/does-not-exist.yaml")


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("hand: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(broken)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    flat = tmp_path / "flat.yaml"
    flat.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(flat)


def test_overlap_must_be_smaller_than_window(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"device": "cpu"},
                "paths": {"data_root": "data"},
                "detection": {"min_confidence": 0.75, "max_gap": 4, "iou_threshold": 0.2},
                "hand": {"window": 16, "overlap": 16},
                "camera": {"window": 200, "overlap": 40, "resolution": 416},
                "stitch": {"pixel_stride": 8},
                "refinement": {"bone_scale_max_correction": 0.035, "wrist_depth_lambda": 0.2},
                "evaluation": {"chunk_seconds": 1.0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="overlap"):
        load_config(path)


def test_out_of_range_threshold_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"device": "cpu"},
                "paths": {"data_root": "data"},
                "detection": {"min_confidence": 1.5, "max_gap": 4, "iou_threshold": 0.2},
                "hand": {"window": 16, "overlap": 8},
                "camera": {"window": 200, "overlap": 40, "resolution": 416},
                "stitch": {"pixel_stride": 8},
                "refinement": {"bone_scale_max_correction": 0.035, "wrist_depth_lambda": 0.2},
                "evaluation": {"chunk_seconds": 1.0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="min_confidence"):
        load_config(path)


def test_explicit_cuda_on_a_cpu_machine_is_an_error(tmp_path: Path) -> None:
    """torch is absent in the base env, so 'cuda' must not be accepted silently."""
    config = load_config("configs/macrodata_final.yaml").with_overrides(
        {"runtime.device": "cuda"}
    )
    if __import__("ego3d_action.runtime.device", fromlist=["cuda_available"]).cuda_available():
        pytest.skip("a GPU is visible on this machine")
    with pytest.raises(ConfigError, match="cuda"):
        validate_config(config)


def test_with_overrides_does_not_mutate_the_source() -> None:
    base = load_config("configs/default.yaml")
    changed = base.with_overrides({"camera.window": 100, "new.section": {"a": 1}})
    assert base.get("camera.window") == 200
    assert changed.get("camera.window") == 100
    assert changed.get("new.section.a") == 1


def test_require_and_set() -> None:
    config = load_config("configs/default.yaml")
    assert config.require("paths.data_root") == "data"
    with pytest.raises(ConfigError):
        config.require("nope.nothing")
    config.set("camera.window", 123)
    assert config.get("camera.window") == 123
    config.set("fresh.key", "value")
    assert config.get("fresh.key") == "value"


def test_parse_overrides_types() -> None:
    parsed = parse_overrides(
        ["camera.window=100", "stitch.blend=false", "stitch.max_depth=null", "camera.checkpoint=abc"]
    )
    assert parsed["camera.window"] == 100
    assert parsed["stitch.blend"] is False
    assert parsed["stitch.max_depth"] is None
    assert parsed["camera.checkpoint"] == "abc"
    with pytest.raises(ConfigError):
        parse_overrides(["broken"])
