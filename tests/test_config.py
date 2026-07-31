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
