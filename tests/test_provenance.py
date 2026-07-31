from pathlib import Path

import numpy as np

from faceswap_pro.provenance import add_disclosure, write_manifest


def test_disclosure_only_changes_label_region():
    frame = np.full((240, 320, 3), 127, dtype=np.uint8)
    result = add_disclosure(frame, "IA")

    assert np.array_equal(result[-20:, -20:], np.full((20, 20, 3), 127, dtype=np.uint8))
    assert not np.array_equal(result[:80, :120], np.full((80, 120, 3), 127, dtype=np.uint8))


def test_manifest_is_written_to_dedicated_directory(tmp_path):
    input_video = tmp_path / "input.mp4"
    target = tmp_path / "target.jpg"
    source = tmp_path / "source.jpg"
    model = tmp_path / "model.onnx"
    output = tmp_path / "videos" / "result.mp4"
    manifests = tmp_path / "manifests"

    for path in (input_video, target, source, model):
        path.write_bytes(path.name.encode("utf-8"))

    manifest = write_manifest(
        output_video=output,
        manifest_dir=manifests,
        input_video=input_video,
        source_images=[source],
        target_reference=target,
        model_path=model,
        runtime={"effective_fps": 30.0},
    )

    assert manifest == manifests / "result.manifest.json"
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")