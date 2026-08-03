from __future__ import annotations

"""Worker persistente de DWPose separado del modelo Wan.

Precalcula pose y máscara para todos los clips. El proceso padre lo cierra antes
de iniciar ``dreamidv_worker.py``, de modo que las sesiones ONNX y sus buffers
CUDA se liberan por completo antes de reservar VRAM para DreamID-V.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

PREFIX = "FACESWAP_RESULT "


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


class PoseRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        repository = args.repository.expanduser().resolve()
        sys.path.insert(0, str(repository))
        sys.path.insert(0, str(repository / "pose"))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device_id))

        from pose.extract import process_dwpose

        self.process_dwpose = process_dwpose

    def generate(self, request: dict[str, Any]) -> None:
        input_video = Path(request["input_video"]).expanduser().resolve()
        pose_video = Path(request["pose_video"]).expanduser().resolve()
        mask_video = Path(request["mask_video"]).expanduser().resolve()
        pose_video.parent.mkdir(parents=True, exist_ok=True)
        mask_video.parent.mkdir(parents=True, exist_ok=True)

        if pose_video.is_file() and mask_video.is_file():
            return

        self.process_dwpose(str(input_video), str(pose_video), str(mask_video))
        if not pose_video.is_file() or not mask_video.is_file():
            raise RuntimeError(
                "DWPose terminó sin crear pose.mp4 y mask.mp4 para " f"{input_video.name}."
            )


def _reply(payload: dict[str, Any]) -> None:
    print(PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    args = _arguments()
    os.chdir(args.repository)
    try:
        runtime = PoseRuntime(args)
    except Exception as exc:  # noqa: BLE001 - informa al proceso padre
        _reply({"ok": False, "error": f"Inicialización: {type(exc).__name__}: {exc}"})
        traceback.print_exc()
        return 1

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
            if request.get("command") == "shutdown":
                _reply({"ok": True, "shutdown": True})
                return 0
            runtime.generate(request)
            _reply(
                {
                    "ok": True,
                    "pose_video": request["pose_video"],
                    "mask_video": request["mask_video"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - protocolo de proceso externo
            traceback.print_exc()
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
