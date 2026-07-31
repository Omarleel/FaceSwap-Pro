from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faceswap_pro.engine import load_configured_model_plugins

from faceswap_pro.modeling import (
    DetectionStats,
    FaceData,
    ModelBundle,
    SwapResult,
    available_model_backends,
    create_model_bundle,
    register_model_backend,
)


class DummyAnalyzer:
    def find_faces(self, image):
        return []

    def analyze(self, frame, previous_bbox, full_scan):
        return [], DetectionStats(0, 0, full_scan)


class DummySwapper:
    def swap(self, frame, target_face, source_face):
        return SwapResult(np.zeros((1, 1, 3), dtype=np.uint8), np.eye(2, 3))


def test_face_data_clone_detaches_mutable_arrays():
    face = FaceData(
        bbox=np.array([1, 2, 3, 4], dtype=np.float32),
        kps=np.ones((5, 2), dtype=np.float32),
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    clone = face.clone()
    clone.bbox[0] = 99
    clone.kps[0, 0] = 88
    clone.embedding[0] = 0

    assert face.bbox[0] == 1
    assert face.kps[0, 0] == 1
    assert face.embedding[0] == 1


def test_registered_backend_can_be_created_without_changing_pipeline():
    backend_name = "unit_test_backend"

    def factory(config, model_path):
        return ModelBundle(
            backend=backend_name,
            analyzer=DummyAnalyzer(),
            swapper=DummySwapper(),
            runtime={"model": str(model_path), "option": config.engine.options["value"]},
        )

    register_model_backend(backend_name, factory, replace=True)
    config = SimpleNamespace(engine=SimpleNamespace(options={"value": 7}))

    bundle = create_model_bundle(backend_name, config, Path("dummy.model"))

    assert backend_name in available_model_backends()
    assert bundle.runtime["option"] == 7


def test_unknown_backend_reports_available_implementations():
    with pytest.raises(ValueError, match="Backend de modelo desconocido"):
        create_model_bundle("missing_backend", SimpleNamespace(), Path("dummy.model"))


def test_backend_plugin_is_loaded_from_configured_module(tmp_path, monkeypatch):
    plugin = tmp_path / "unit_backend_plugin.py"
    plugin.write_text(
        "from faceswap_pro.modeling import ModelBundle, register_model_backend\n"
        "def build(config, model_path):\n"
        "    return ModelBundle(backend='filesystem_backend', "
        "analyzer=object(), swapper=object())\n"
        "register_model_backend('filesystem_backend', build, replace=True)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = SimpleNamespace(
        engine=SimpleNamespace(plugins=("unit_backend_plugin",), options={})
    )

    load_configured_model_plugins(config)
    bundle = create_model_bundle("filesystem_backend", config, Path("model.bin"))

    assert bundle.backend == "filesystem_backend"
