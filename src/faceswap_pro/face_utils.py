from __future__ import annotations

from typing import Any

import numpy as np


def clone_face(face: Any):
    """Clona un registro ``Face`` sin usar ``copy.copy``.

    ``insightface.app.common.Face`` desactiva ``__setstate__`` y, por ello,
    ``copy.copy(face)`` falla en algunas versiones de Python con
    ``TypeError: 'NoneType' object is not callable``. La clase acepta un
    diccionario en su constructor, así que se reconstruye a partir de sus
    campos y se copian de forma independiente los arreglos NumPy mutables.
    """

    if not hasattr(face, "items"):
        raise TypeError("El objeto de rostro debe exponer items() como un diccionario.")

    payload = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in face.items()
    }

    face_type = type(face)
    try:
        return face_type(payload)
    except Exception as exc:
        raise TypeError(
            f"No se pudo reconstruir el rostro usando {face_type.__name__}(dict)."
        ) from exc
