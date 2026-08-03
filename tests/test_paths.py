from datetime import datetime
from pathlib import Path

from faceswap_pro.paths import build_manifest_path, build_output_path, create_project_directories


def test_create_project_directories(tmp_path):
    directories = create_project_directories(tmp_path)
    assert directories
    assert all(path.is_dir() for path in directories)
    assert (tmp_path / "inputs/source_faces").is_dir()
    assert (tmp_path / "outputs/manifests").is_dir()


def test_default_output_is_timestamped_and_not_in_project_root(tmp_path):
    input_video = tmp_path / "inputs/videos/scene.mp4"
    output_dir = tmp_path / "outputs/videos"
    result = build_output_path(
        input_video,
        None,
        output_dir,
        now=datetime(2026, 7, 31, 17, 35, 0),
    )
    assert result == output_dir / "scene_faceswap_20260731_173500.mp4"


def test_explicit_output_gets_mp4_extension(tmp_path):
    result = build_output_path(
        Path("input.mp4"),
        tmp_path / "custom/result",
        tmp_path / "unused",
    )
    assert result == tmp_path / "custom/result.mp4"
    assert result.parent.is_dir()


def test_manifest_path_is_separate_from_video(tmp_path):
    output = tmp_path / "outputs/videos/render.mp4"
    manifest = build_manifest_path(output, tmp_path / "outputs/manifests")
    assert manifest == tmp_path / "outputs/manifests/render.manifest.json"


def test_hififace_uses_windows_safe_auxiliary_directory(tmp_path):
    create_project_directories(tmp_path)

    assert (tmp_path / "models/hififace/auxiliary").is_dir()
    assert not (tmp_path / "models/hififace/aux").exists()
