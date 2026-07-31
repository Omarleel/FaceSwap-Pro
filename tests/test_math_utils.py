import numpy as np

from faceswap_pro.math_utils import bbox_iou, cosine_similarity, l2_normalize


def test_l2_normalize():
    value = l2_normalize(np.array([3.0, 4.0]))
    assert np.allclose(value, [0.6, 0.8])


def test_cosine_similarity():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0


def test_bbox_iou():
    assert np.isclose(bbox_iou(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15])), 25 / 175)
