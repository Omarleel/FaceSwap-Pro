import numpy as np

from faceswap_pro.blend import ProfessionalBlender


class IdentityRestorer:
    def __call__(self, image):
        return image


def test_roi_blend_does_not_modify_distant_pixels():
    blender = ProfessionalBlender(
        aligned_size=16,
        mask_shrink=0.8,
        mask_blur_ratio=0.1,
        color_match_strength=0.0,
        detail_strength=0.0,
        roi_enabled=True,
        roi_margin=0.0,
        interpolation="linear",
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    fake = np.full((16, 16, 3), 255, dtype=np.uint8)
    affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    result = blender.composite(frame, fake, affine, IdentityRestorer())

    assert result[:16, :16].max() > 0
    assert np.count_nonzero(result[24:, 24:]) == 0


def test_roi_and_full_frame_blend_are_equivalent():
    class Restorer:
        def __call__(self, image):
            return image

    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    fake = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    affine = np.array([[0.5, 0, -20], [0, 0.5, -18]], dtype=np.float32)
    common = dict(
        aligned_size=16,
        mask_shrink=0.8,
        mask_blur_ratio=0.1,
        color_match_strength=0.0,
        detail_strength=0.0,
        roi_margin=0.2,
        interpolation="linear",
    )

    roi = ProfessionalBlender(roi_enabled=True, **common).composite(
        frame.copy(), fake, affine, Restorer()
    )
    full = ProfessionalBlender(roi_enabled=False, **common).composite(
        frame.copy(), fake, affine, Restorer()
    )

    assert np.array_equal(roi, full)
