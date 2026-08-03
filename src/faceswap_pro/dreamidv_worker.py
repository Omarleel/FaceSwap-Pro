from __future__ import annotations

"""Worker persistente ejecutado dentro del entorno aislado de DreamID-V.

El protocolo es JSON Lines por stdin. Cada respuesta comienza con
``FACESWAP_RESULT `` para poder convivir con los logs del runtime oficial.
"""

import argparse
import importlib
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
    parser.add_argument("--variant", choices=["faster", "dwpose"], required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--sample-fps", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dreamidv-checkpoint", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--t5-cpu", action="store_true")
    return parser.parse_args()


class WorkerRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        repository = args.repository.expanduser().resolve()
        sys.path.insert(0, str(repository))
        sys.path.insert(0, str(repository / "pose"))
        self.args = args
        self.repository = repository
        package_name = "dreamidv_wan_faster" if args.variant == "faster" else "dreamidv_wan"
        package = importlib.import_module(package_name)
        configs = importlib.import_module(f"{package_name}.configs")
        utils = importlib.import_module(f"{package_name}.utils.utils")
        self.size_configs = configs.SIZE_CONFIGS
        self.cache_video = utils.cache_video
        self.cfg = configs.WAN_CONFIGS["swapface"]
        self.cfg.sample_fps = args.sample_fps
        self.pipeline = package.DreamIDV(
            config=self.cfg,
            checkpoint_dir=str(args.checkpoint_dir.resolve()),
            dreamidv_ckpt=str(args.dreamidv_checkpoint.resolve()),
            device_id=args.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=args.t5_cpu,
        )
        from pose.extract import process_dwpose

        self.process_dwpose = process_dwpose

    def generate(self, request: dict[str, Any]) -> None:
        input_video = Path(request["input_video"]).resolve()
        source_reference = Path(request["source_reference"]).resolve()
        output_video = Path(request["output_video"]).resolve()
        output_video.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = input_video.parent / f"{input_video.stem}_dreamidv_pose"
        temp_dir.mkdir(parents=True, exist_ok=True)
        pose_path = temp_dir / "pose.mp4"
        mask_path = temp_dir / "mask.mp4"
        if not pose_path.is_file() or not mask_path.is_file():
            self.process_dwpose(str(input_video), str(pose_path), str(mask_path))

        if self.args.variant == "faster":
            ref_paths = [str(input_video), str(mask_path), str(source_reference)]
        else:
            ref_paths = [
                str(input_video),
                str(mask_path),
                str(source_reference),
                str(pose_path),
            ]
        video = self.pipeline.generate(
            "chang face",
            ref_paths,
            size=self.size_configs[self.args.size],
            frame_num=int(request["frame_num"]),
            shift=float(request["sample_shift"]),
            sample_solver=str(request["sample_solver"]),
            sampling_steps=int(request["sample_steps"]),
            guide_scale_img=float(request["guide_scale"]),
            seed=int(request["seed"]),
            offload_model=bool(request["offload_model"]),
        )
        self.cache_video(
            tensor=video[None],
            save_file=str(output_video),
            fps=self.cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )


def _reply(payload: dict[str, Any]) -> None:
    print(PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    args = _arguments()
    os.chdir(args.repository)
    try:
        runtime = WorkerRuntime(args)
    except Exception as exc:  # noqa: BLE001 - reporta al proceso padre
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
            _reply({"ok": True, "output_video": request["output_video"]})
        except Exception as exc:  # noqa: BLE001 - protocolo de proceso externo
            traceback.print_exc()
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
