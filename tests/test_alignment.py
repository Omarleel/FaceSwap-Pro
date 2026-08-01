import numpy as np

from faceswap_pro.alignment import arcface_template, estimate_face_affine


def test_arcface_template_maps_to_itself():
    template = arcface_template(256)
    affine = estimate_face_affine(template, 256)
    homogeneous = np.concatenate(
        [template, np.ones((template.shape[0], 1), dtype=np.float32)], axis=1
    )
    mapped = homogeneous @ affine.T

    assert np.allclose(mapped, template, atol=1e-3)
