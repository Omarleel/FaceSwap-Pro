import numpy as np

from faceswap_pro.alignment import arcface_template
from faceswap_pro.hififace_backend import HifiFace3DMMSwapper
from faceswap_pro.modeling import FaceData


class DummyHifiFaceRuntime:
    output_size = 64

    def __init__(self):
        self.calls = 0

    def infer(self, source_bgr, target_bgr, *, shape_rate, identity_rate):
        self.calls += 1
        assert source_bgr.shape == (64, 64, 3)
        assert target_bgr.shape == (64, 64, 3)
        generated = np.full((64, 64, 3), 180, dtype=np.uint8)
        mask = np.zeros((64, 64, 1), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        return generated, mask


def _face(image):
    return FaceData(
        bbox=np.array([0, 0, 63, 63], dtype=np.float32),
        kps=arcface_template(64),
        det_score=1.0,
        embedding=np.ones(4, dtype=np.float32),
        reference_image=image,
    )


def test_hififace_swapper_uses_internal_3dmm_runtime_and_learned_mask():
    runtime = DummyHifiFaceRuntime()
    swapper = HifiFace3DMMSwapper(
        runtime,
        iterations=2,
        mask_dilate_ratio=0.02,
        mask_blur_ratio=0.01,
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    result = swapper.swap(frame, _face(frame), _face(frame.copy()))

    assert runtime.calls == 2
    assert result.crop.shape == (64, 64, 3)
    assert result.mask is not None
    assert result.mask_mode == "replace"
    assert result.metadata["geometry_conditioning"] == "3dmm_internal"
    assert result.metadata["semantic_fusion_internal"] is True
