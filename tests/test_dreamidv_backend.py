from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from faceswap_pro.dreamidv_backend import (
    DreamIDVOptions,
    DreamIDVSubprocessBackend,
    build_dreamidv_command,
    dreamidv_readiness,
)
from faceswap_pro.modeling import (
    FaceData,
    ModelCapabilities,
    VideoModelBundle,
    VideoSwapResult,
)
from faceswap_pro.pipeline import run_pipeline
from faceswap_pro.videoio import VideoMetadata


def _config(tmp_path: Path, **overrides):
    options = {
        "repository_path": str(tmp_path / "DreamID-V"),
        "wan_checkpoint_dir": str(tmp_path / "Wan2.1-T2V-1.3B"),
        "python_executable": sys.executable,
        "variant": "faster",
        "size": "832*480",
        "frame_num": 49,
        "sample_fps": 16,
        "sample_steps": 16,
        "offload_model": True,
        "t5_cpu": True,
        "chunking": True,
    }
    options.update(overrides)
    return SimpleNamespace(engine=SimpleNamespace(options=options))


def test_dreamidv_options_require_4n_plus_1_frames(tmp_path):
    with pytest.raises(ValueError, match="4n\\+1"):
        DreamIDVOptions.from_config(_config(tmp_path, frame_num=48))


def test_dreamidv_resolves_relative_runtime_paths_before_changing_cwd(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config = _config(
        Path("."),
        repository_path="third_party/DreamID-V",
        wan_checkpoint_dir="models/Wan2.1-T2V-1.3B",
        python_executable=".venv-dreamidv/Scripts/python.exe",
    )
    options = DreamIDVOptions.from_config(config)
    command = build_dreamidv_command(
        options,
        checkpoint=Path("models/dreamidv_faster.pth"),
        input_video=Path("input.mp4"),
        source_reference=Path("face.png"),
        output_video=Path("output.mp4"),
    )

    assert Path(command[0]).is_absolute()
    assert Path(command[1]).is_absolute()
    for flag in (
        "--ckpt_dir",
        "--dreamidv_ckpt",
        "--save_file",
        "--ref_image",
        "--ref_video",
    ):
        assert Path(command[command.index(flag) + 1]).is_absolute()


def test_dreamidv_command_matches_official_faster_cli(tmp_path):
    options = DreamIDVOptions.from_config(_config(tmp_path))
    command = build_dreamidv_command(
        options,
        checkpoint=tmp_path / "dreamidv_faster.pth",
        input_video=tmp_path / "input.mp4",
        source_reference=tmp_path / "face.png",
        output_video=tmp_path / "output.mp4",
    )

    assert command[0] == sys.executable
    assert command[1].endswith("generate_dreamidv_faster.py")
    assert command[command.index("--size") + 1] == "832*480"
    assert command[command.index("--frame_num") + 1] == "49"
    assert command[command.index("--sample_steps") + 1] == "16"
    assert command[command.index("--offload_model") + 1] == "true"
    assert "--t5_cpu" in command


def test_dreamidv_readiness_checks_official_layout(tmp_path, monkeypatch):
    repository = tmp_path / "DreamID-V"
    package = repository / "dreamidv_wan_faster"
    pose = repository / "pose" / "models"
    package.mkdir(parents=True)
    (package / "context.pth").write_bytes(b"context")
    pose.mkdir(parents=True)
    (repository / "generate_dreamidv_faster.py").write_text("", encoding="utf-8")
    (pose / "dw-ll_ucoco_384.onnx").write_bytes(b"pose")
    (pose / "yolox_l.onnx").write_bytes(b"detector")

    wan = tmp_path / "Wan2.1-T2V-1.3B"
    wan.mkdir()
    (wan / "model.bin").write_bytes(b"wan")
    checkpoint = tmp_path / "dreamidv_faster.pth"
    checkpoint.write_bytes(b"dreamidv")

    monkeypatch.setattr(
        "faceswap_pro.dreamidv_backend.select_ffmpeg",
        lambda: Path("ffmpeg"),
    )
    result = dreamidv_readiness(_config(tmp_path), checkpoint, probe_environment=False)

    assert result["ready"] is True
    assert all(result["checks"].values())


class _Analyzer:
    def find_faces(self, image):
        h, w = image.shape[:2]
        return [
            FaceData(
                bbox=np.array([w * 0.2, h * 0.15, w * 0.8, h * 0.85], dtype=np.float32),
                kps=np.array(
                    [
                        [w * 0.38, h * 0.42],
                        [w * 0.62, h * 0.42],
                        [w * 0.50, h * 0.56],
                        [w * 0.41, h * 0.70],
                        [w * 0.59, h * 0.70],
                    ],
                    dtype=np.float32,
                ),
                det_score=0.99,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        ]

    def analyze(self, frame, previous_bbox, full_scan):
        return [], SimpleNamespace(detected=0, recognized=0, full_scan=full_scan)


class _Processor:
    def __init__(self):
        self.reference_shape = None

    def process(self, request):
        image = cv2.imread(str(request.source_reference))
        self.reference_shape = image.shape
        request.output_video.write_bytes(b"silent-video")
        return VideoSwapResult(request.output_video, {"chunks": 1})


def test_pipeline_dispatches_video_backend_and_builds_512_reference(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "face.png"
    target = tmp_path / "target.png"
    input_video = tmp_path / "input.mp4"
    model = tmp_path / "dreamidv_faster.pth"
    output = tmp_path / "output.mp4"
    manifests = tmp_path / "manifests"

    image = np.full((240, 200, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    assert cv2.imwrite(str(target), image)
    input_video.write_bytes(b"input")
    model.write_bytes(b"model")

    processor = _Processor()
    bundle = VideoModelBundle(
        backend="dreamid_v",
        analyzer=_Analyzer(),
        processor=processor,
        capabilities=ModelCapabilities(
            generator="dreamid_v_wan_1.3b",
            temporal_generation="video_diffusion_transformer",
        ),
        model_artifacts=(model,),
    )
    config = SimpleNamespace(
        identity=SimpleNamespace(source_min_score=0.5, max_source_images=4),
        encoding=SimpleNamespace(audio_bitrate="192k"),
        provenance=SimpleNamespace(visible_disclosure=False),
    )

    def fake_mux(silent, source_video, destination, bitrate):
        shutil.copyfile(silent, destination)

    def fake_manifest(**kwargs):
        path = manifests / "output.manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr("faceswap_pro.pipeline.mux_original_audio", fake_mux)
    monkeypatch.setattr("faceswap_pro.pipeline.write_manifest", fake_manifest)
    monkeypatch.setattr(
        "faceswap_pro.pipeline.probe_video",
        lambda path: VideoMetadata(fps=16.0, width=832, height=480, frame_count=49),
    )

    result, manifest = run_pipeline(
        input_video=input_video,
        source_dir=source_dir,
        target_reference=target,
        model_path=model,
        output_video=output,
        manifest_dir=manifests,
        config=config,
        model_bundle=bundle,
    )

    assert result == output
    assert manifest.is_file()
    assert processor.reference_shape == (512, 512, 3)
