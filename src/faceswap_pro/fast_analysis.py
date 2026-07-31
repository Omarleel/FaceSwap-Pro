"""Compatibilidad para el analizador selectivo de InsightFace.

Las integraciones nuevas deben usar ``FaceAnalyzer`` desde ``modeling.py``. Esta
reexportación mantiene estable la API utilizada por versiones anteriores y pruebas.
"""

from .insightface_backend import SelectiveFaceAnalyzer
from .modeling import DetectionStats

__all__ = ["DetectionStats", "SelectiveFaceAnalyzer"]
