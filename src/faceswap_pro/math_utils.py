from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("No se puede normalizar un vector casi nulo.")
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float32)
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float32)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else float(inter / union)
