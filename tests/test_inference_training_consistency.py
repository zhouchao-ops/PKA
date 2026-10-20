from pathlib import Path

import yaml

from ominicontrol.cli import inference_geometry


ROOT = Path(__file__).parents[1]


def test_default_inference_geometry_matches_training_recipes():
    compact_paa, position_scale = inference_geometry(256, 512, 512)
    assert compact_paa is True
    assert position_scale == 2.0

    for name in ("subject_canny", "subject_depth", "canny_depth"):
        recipe = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
        assert recipe["model"]["compact_paa"] is compact_paa
        assert recipe["model"]["compact_paa_mode"] == "aligned1"
        assert recipe["model"]["condition_layout"] == "aligned1"
        assert recipe["train"]["dataset"]["position_scale"] == position_scale


def test_noncompact_geometry_has_unit_position_scale():
    compact_paa, position_scale = inference_geometry(512, 512, 512)
    assert compact_paa is False
    assert position_scale == 1.0
