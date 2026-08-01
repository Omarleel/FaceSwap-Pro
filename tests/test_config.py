from pathlib import Path

from faceswap_pro.config import load_config


def test_max_speed_profile_loads():
    config = load_config(Path("config/max_speed.yaml"))

    assert config.engine.backend == "insightface_inswapper"
    assert config.engine.plugins == ()
    assert config.engine.allowed_modules == ("detection", "recognition")
    assert config.tracking.detection_interval == 3
    assert config.blend.roi_enabled is True
    assert config.performance.hardware_decode is True


def test_mesh_assisted_profile_uses_honest_backend_name():
    config = load_config(Path("config/quality_mesh_assisted.yaml"))

    assert config.engine.backend == "insightface_inswapper_mediapipe_mesh"
    assert config.engine.options["base_backend"] == "insightface_inswapper"
    assert config.tracking.detection_interval == 1
    assert config.tracking.optical_flow is False


def test_quality_3d_profile_is_true_3dmm_backend():
    config = load_config(Path("config/quality_3d.yaml"))

    assert config.engine.backend == "hififace_3dmm"
    assert config.engine.options["repository_path"] == "third_party/HiFiFace-pytorch"
    assert config.engine.options["checkpoint_iteration"] == 80000
    assert config.tracking.detection_interval == 1
