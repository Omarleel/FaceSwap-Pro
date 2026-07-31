from copy import copy

import numpy as np
import pytest

from faceswap_pro.face_utils import clone_face


class DummyFace(dict):
    """Replica el comportamiento relevante de insightface.app.common.Face."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __getstate__ = None
    __setstate__ = None


def test_standard_copy_reproduces_insightface_failure():
    face = DummyFace({"bbox": np.array([0, 0, 10, 10], dtype=np.float32)})

    with pytest.raises(TypeError, match="NoneType"):
        copy(face)


def test_clone_face_avoids_copy_protocol_and_detaches_arrays():
    face = DummyFace(
        {
            "bbox": np.array([0, 0, 10, 10], dtype=np.float32),
            "kps": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "det_score": 0.99,
        }
    )

    cloned = clone_face(face)

    assert isinstance(cloned, DummyFace)
    assert cloned is not face
    assert np.array_equal(cloned.bbox, face.bbox)
    assert cloned.bbox is not face.bbox
    assert cloned.kps is not face.kps
    assert cloned.det_score == face.det_score
