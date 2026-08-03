from pathlib import Path

from faceswap_pro.config import load_config


def test_max_speed_profile_loads():
    config = load_config(Path("config/max_speed.yaml"))

    assert config.engine.backend == "insightface_inswapper"
    assert config.engine.plugins == ()
    assert config.engine.allowed_modules == ("detection", "recognition")
    assert config.tracking.detection_interval == 3
    assert config.tracking.max_target_faces == 2
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
    assert config.engine.options["checkpoint_iteration"] == 320000
    assert config.tracking.detection_interval == 1


def test_dreamidv_profile_disables_visible_disclosure():
    from pathlib import Path

    config = load_config(Path("config/quality_dreamidv.yaml"))

    assert config.engine.backend == "dreamid_v"
    assert config.provenance.visible_disclosure is False
    assert config.provenance.hash_models is True
    assert config.provenance.c2pa_enabled is True
    assert config.provenance.c2pa_required is False
    assert config.engine.options["chunk_overlap_frames"] == 17
    assert config.engine.options["persistent_worker"] is True
    assert config.engine.options["reference_bank_size"] == 6
    assert config.engine.options["frame_num"] % 4 == 1


def test_hififace_profiles_use_windows_safe_auxiliary_paths():
    for profile in ("config/quality_3d.yaml", "config/quality_3dmm.yaml"):
        config = load_config(Path(profile))
        options = config.engine.options

        assert options["f_3d_checkpoint_path"].startswith("models/hififace/auxiliary/")
        assert options["f_id_checkpoint_path"].startswith("models/hififace/auxiliary/")
        assert options["bfm_folder"] == "models/hififace/auxiliary/BFM"
