from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .runtime import ffmpeg_has_encoder, ffmpeg_has_hwaccel, select_ffmpeg


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoReader(Protocol):
    metadata: VideoMetadata
    backend: str

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def close(self) -> None: ...


def probe_video(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"No se pudo abrir el video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("El video no reporta FPS o resolución válidos.")
    return VideoMetadata(fps=fps, width=width, height=height, frame_count=count)


class OpenCVVideoReader:
    backend = "opencv"

    def __init__(self, path: Path, metadata: VideoMetadata | None = None) -> None:
        self.metadata = metadata or probe_video(path)
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"No se pudo abrir el video: {path}")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.capture.read()
        return bool(ok), frame if ok else None

    def close(self) -> None:
        self.capture.release()


class FFmpegRawReader:
    """Decodifica con FFmpeg y entrega BGR24 por pipe.

    En modo CUDA se solicita NVDEC. FFmpeg descarga los cuadros a memoria del
    sistema porque InsightFace/OpenCV reciben arreglos NumPy; aun así, la
    decodificación del códec se solapa con el resto del pipeline.
    """

    def __init__(self, path: Path, metadata: VideoMetadata, use_cuda: bool) -> None:
        self.metadata = metadata
        self.ffmpeg = select_ffmpeg()
        if self.ffmpeg is None:
            raise RuntimeError("FFmpeg no está disponible para decodificar el video.")
        self.backend = "ffmpeg_cuda" if use_cuda else "ffmpeg"
        command = [str(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if use_cuda:
            command += ["-hwaccel", "cuda", "-hwaccel_device", "0"]
        command += [
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=max(1024 * 1024, metadata.width * metadata.height * 3 * 2),
        )
        self.frame_bytes = metadata.width * metadata.height * 3
        self._first_frame: np.ndarray | None = self._read_frame()
        if self._first_frame is None:
            self.close()
            raise RuntimeError(f"FFmpeg no pudo iniciar decodificación con backend {self.backend}.")

    def _read_exact(self, size: int) -> bytes:
        if self.process.stdout is None:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            block = self.process.stdout.read(remaining)
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def _read_frame(self) -> np.ndarray | None:
        raw = self._read_exact(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.metadata.height, self.metadata.width, 3
        )
        return frame.copy()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
            return True, frame
        frame = self._read_frame()
        return frame is not None, frame

    def close(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def open_video_reader(path: Path, performance) -> VideoReader:
    metadata = probe_video(path)
    decoder = performance.decoder
    ffmpeg = select_ffmpeg()
    cuda_available = bool(ffmpeg and ffmpeg_has_hwaccel(ffmpeg, "cuda"))

    if decoder == "opencv":
        return OpenCVVideoReader(path, metadata)

    candidates: list[bool] = []
    if decoder == "ffmpeg_cuda":
        candidates = [True]
    elif decoder == "ffmpeg":
        candidates = [False]
    elif decoder == "auto":
        if performance.hardware_decode and cuda_available:
            candidates.append(True)
        candidates.append(False)

    for use_cuda in candidates:
        try:
            return FFmpegRawReader(path, metadata, use_cuda=use_cuda)
        except (OSError, RuntimeError):
            if decoder != "auto":
                raise

    return OpenCVVideoReader(path, metadata)


class RawFFmpegWriter:
    def __init__(self, output: Path, width: int, height: int, fps: float, encoding) -> None:
        self.ffmpeg = select_ffmpeg(encoding.codec)
        if self.ffmpeg is None:
            raise RuntimeError(
                "FFmpeg no está disponible. Ejecuta scripts/setup_windows.ps1 "
                "o define FACESWAP_PRO_FFMPEG."
            )
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.encoding = encoding
        use_fallback = not ffmpeg_has_encoder(self.ffmpeg, encoding.codec)
        self.used_codec = encoding.fallback_codec if use_fallback else encoding.codec
        self.process = self._start(use_fallback=use_fallback)

    def _command(self, use_fallback: bool) -> list[str]:
        common = [
            str(self.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
        ]
        if use_fallback:
            video = [
                "-c:v",
                self.encoding.fallback_codec,
                "-preset",
                self.encoding.fallback_preset,
                "-crf",
                str(self.encoding.fallback_crf),
                "-pix_fmt",
                "yuv420p",
            ]
        else:
            video = [
                "-c:v",
                self.encoding.codec,
                "-preset",
                self.encoding.preset,
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(self.encoding.cq),
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
            ]
        return common + video + [str(self.output)]

    def _start(self, use_fallback: bool):
        return subprocess.Popen(self._command(use_fallback), stdin=subprocess.PIPE)

    def write(self, frame) -> None:
        if self.process.stdin is None:
            raise RuntimeError("El proceso FFmpeg no tiene stdin disponible.")
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(
                "FFmpeg dejó de aceptar cuadros; revisa soporte NVENC y el controlador."
            ) from exc

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"FFmpeg terminó con código {code} usando {self.used_codec}.")


def mux_original_audio(silent_video: Path, source_video: Path, output: Path, audio_bitrate: str) -> None:
    ffmpeg = select_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg no está disponible para multiplexar el audio.")
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        "-metadata",
        "comment=Contenido sintetico mediante reemplazo facial autorizado",
        str(output),
    ]
    subprocess.run(command, check=True)
