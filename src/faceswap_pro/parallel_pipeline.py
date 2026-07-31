from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .fast_analysis import SelectiveFaceAnalyzer
from .provenance import add_disclosure

_SENTINEL = object()


@dataclass(frozen=True)
class FramePacket:
    index: int
    frame: np.ndarray


@dataclass(frozen=True)
class AnalyzedPacket:
    index: int
    frame: np.ndarray
    target_face: Any | None


@dataclass(frozen=True)
class ProcessedPacket:
    index: int
    frame: np.ndarray
    swapped: bool
    postprocess_seconds: float


@dataclass
class PipelineStats:
    decoded_frames: int = 0
    analyzed_frames: int = 0
    detection_frames: int = 0
    full_scans: int = 0
    optical_flow_frames: int = 0
    detected_faces: int = 0
    recognition_inferences: int = 0
    swapped_frames: int = 0
    written_frames: int = 0
    decode_seconds: float = 0.0
    analysis_seconds: float = 0.0
    swap_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    encode_feed_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **values: float | int) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, getattr(self, key) + value)

    def as_dict(self) -> dict[str, float | int]:
        with self._lock:
            return {
                key: value
                for key, value in self.__dict__.items()
                if key != "_lock"
            }


def resolve_parallelism(performance, restorer_enabled: bool) -> tuple[int, int]:
    cpu_count = max(2, os.cpu_count() or 4)
    if performance.postprocess_workers > 0:
        workers = performance.postprocess_workers
    else:
        workers = min(4, max(1, cpu_count // 4))
    if restorer_enabled:
        # El restaurador puede usar otra sesión GPU; mantener una sola llamada
        # evita saturar VRAM y conserva el solapamiento con el swapper.
        workers = 1

    if performance.opencv_threads > 0:
        cv_threads = performance.opencv_threads
    else:
        cv_threads = max(1, cpu_count // max(2, workers + 2))
    return workers, cv_threads


def _queue_put(target: queue.Queue, item, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            target.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _raise_pipeline_error(errors: queue.Queue) -> None:
    try:
        exc = errors.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError("Falló una etapa del pipeline paralelo.") from exc


def _reader_worker(reader, output: queue.Queue, stop, errors, stats: PipelineStats) -> None:
    index = 0
    try:
        while not stop.is_set():
            started = time.perf_counter()
            ok, frame = reader.read()
            elapsed = time.perf_counter() - started
            if not ok or frame is None:
                break
            stats.add(decoded_frames=1, decode_seconds=elapsed)
            if not _queue_put(output, FramePacket(index, frame), stop):
                break
            index += 1
    except BaseException as exc:  # noqa: BLE001 - se propaga al hilo principal
        errors.put(exc)
        stop.set()
    finally:
        reader.close()
        if not stop.is_set():
            _queue_put(output, _SENTINEL, stop)


def _analysis_worker(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    stop: threading.Event,
    errors: queue.Queue,
    stats: PipelineStats,
    analyzer: SelectiveFaceAnalyzer,
    tracker,
    tracking_config,
) -> None:
    try:
        while not stop.is_set():
            try:
                packet = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if packet is _SENTINEL:
                _queue_put(output_queue, _SENTINEL, stop)
                return

            started = time.perf_counter()
            gray, scene_cut = tracker.observe(packet.frame)
            detection_due = (
                scene_cut
                or tracker.needs_redetect
                or packet.index % tracking_config.detection_interval == 0
            )
            target_face = None
            used_flow = False

            if detection_due:
                full_scan = (
                    scene_cut
                    or tracker.needs_redetect
                    or packet.index % tracking_config.full_scan_interval == 0
                )
                faces, detection_stats = analyzer.analyze(
                    packet.frame,
                    previous_bbox=tracker.current_bbox,
                    full_scan=full_scan,
                )
                target_face = tracker.select_detected(packet.frame, gray, faces)
                stats.add(
                    detection_frames=1,
                    full_scans=int(detection_stats.full_scan),
                    detected_faces=detection_stats.detected,
                    recognition_inferences=detection_stats.recognized,
                )
            else:
                target_face = tracker.propagate(packet.frame, gray)
                used_flow = target_face is not None
                if target_face is None:
                    # Redetección inmediata: evita perder un frame cuando LK falla.
                    faces, detection_stats = analyzer.analyze(
                        packet.frame,
                        previous_bbox=tracker.current_bbox,
                        full_scan=True,
                    )
                    target_face = tracker.select_detected(packet.frame, gray, faces)
                    stats.add(
                        detection_frames=1,
                        full_scans=1,
                        detected_faces=detection_stats.detected,
                        recognition_inferences=detection_stats.recognized,
                    )

            stats.add(
                analyzed_frames=1,
                optical_flow_frames=int(used_flow),
                analysis_seconds=time.perf_counter() - started,
            )
            if not _queue_put(
                output_queue,
                AnalyzedPacket(packet.index, packet.frame, target_face),
                stop,
            ):
                return
    except BaseException as exc:  # noqa: BLE001
        errors.put(exc)
        stop.set()


def _writer_worker(
    writer,
    input_queue: queue.Queue,
    stop: threading.Event,
    errors: queue.Queue,
    stats: PipelineStats,
) -> None:
    try:
        while not stop.is_set():
            try:
                packet = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if packet is _SENTINEL:
                return
            started = time.perf_counter()
            writer.write(packet.frame)
            stats.add(
                written_frames=1,
                swapped_frames=int(packet.swapped),
                postprocess_seconds=packet.postprocess_seconds,
                encode_feed_seconds=time.perf_counter() - started,
            )
    except BaseException as exc:  # noqa: BLE001
        errors.put(exc)
        stop.set()


def _postprocess(
    packet: AnalyzedPacket,
    fake_crop: np.ndarray | None,
    affine: np.ndarray | None,
    blender,
    restorer,
    watermark_text: str,
) -> ProcessedPacket:
    started = time.perf_counter()
    frame = packet.frame
    swapped = fake_crop is not None and affine is not None
    if swapped:
        frame = blender.composite(frame, fake_crop, affine, restorer)
    frame = add_disclosure(frame, watermark_text)
    return ProcessedPacket(
        index=packet.index,
        frame=np.ascontiguousarray(frame),
        swapped=swapped,
        postprocess_seconds=time.perf_counter() - started,
    )


def run_parallel_frames(
    *,
    reader,
    writer,
    face_app,
    swapper,
    source_face,
    tracker,
    blender,
    restorer,
    config,
) -> tuple[PipelineStats, dict[str, Any]]:
    workers, cv_threads = resolve_parallelism(config.performance, config.restorer.enabled)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(cv_threads)

    analyzer = SelectiveFaceAnalyzer(
        face_app,
        max_faces=config.engine.max_faces,
        max_recognition_candidates=config.tracking.max_recognition_candidates,
    )
    read_queue: queue.Queue = queue.Queue(maxsize=config.performance.reader_queue)
    analysis_queue: queue.Queue = queue.Queue(maxsize=config.performance.analysis_queue)
    write_queue: queue.Queue = queue.Queue(maxsize=config.performance.writer_queue)
    errors: queue.Queue = queue.Queue()
    stop = threading.Event()
    stats = PipelineStats()

    reader_thread = threading.Thread(
        target=_reader_worker,
        name="faceswap-reader",
        args=(reader, read_queue, stop, errors, stats),
        daemon=True,
    )
    analysis_thread = threading.Thread(
        target=_analysis_worker,
        name="faceswap-analysis",
        args=(
            read_queue,
            analysis_queue,
            stop,
            errors,
            stats,
            analyzer,
            tracker,
            config.tracking,
        ),
        daemon=True,
    )
    writer_thread = threading.Thread(
        target=_writer_worker,
        name="faceswap-writer",
        args=(writer, write_queue, stop, errors, stats),
        daemon=True,
    )

    reader_thread.start()
    analysis_thread.start()
    writer_thread.start()

    pending: dict[int, Future] = {}
    next_emit = 0
    progress = tqdm(
        total=reader.metadata.frame_count if reader.metadata.frame_count > 0 else None,
        unit="frame",
        dynamic_ncols=True,
    )

    def emit_next(*, block: bool) -> bool:
        nonlocal next_emit
        future = pending.get(next_emit)
        if future is None or (not block and not future.done()):
            return False
        packet = future.result()
        del pending[next_emit]
        if not _queue_put(write_queue, packet, stop):
            _raise_pipeline_error(errors)
            raise RuntimeError("El escritor se detuvo antes de recibir todos los frames.")
        next_emit += 1
        return True

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="faceswap-post")
    try:
        while not stop.is_set():
            _raise_pipeline_error(errors)
            try:
                packet = analysis_queue.get(timeout=0.1)
            except queue.Empty:
                if not analysis_thread.is_alive():
                    _raise_pipeline_error(errors)
                    break
                continue
            if packet is _SENTINEL:
                break

            fake_crop = None
            affine = None
            if packet.target_face is not None:
                swap_started = time.perf_counter()
                fake_crop, affine = swapper.get(
                    packet.frame,
                    packet.target_face,
                    source_face,
                    paste_back=False,
                )
                stats.add(swap_seconds=time.perf_counter() - swap_started)

            pending[packet.index] = executor.submit(
                _postprocess,
                packet,
                fake_crop,
                affine,
                blender,
                restorer,
                config.watermark.text,
            )
            progress.update(1)

            while emit_next(block=False):
                pass
            while len(pending) >= config.performance.max_inflight:
                emit_next(block=True)
                _raise_pipeline_error(errors)

        while pending:
            emit_next(block=True)
            _raise_pipeline_error(errors)

        if not _queue_put(write_queue, _SENTINEL, stop):
            _raise_pipeline_error(errors)
        writer_thread.join()
        _raise_pipeline_error(errors)
    finally:
        progress.close()
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
        for thread in (reader_thread, analysis_thread, writer_thread):
            thread.join(timeout=5)

    settings = {
        "postprocess_workers": workers,
        "opencv_threads": cv_threads,
        "reader_queue": config.performance.reader_queue,
        "analysis_queue": config.performance.analysis_queue,
        "writer_queue": config.performance.writer_queue,
        "max_inflight": config.performance.max_inflight,
        "detection_interval": config.tracking.detection_interval,
        "full_scan_interval": config.tracking.full_scan_interval,
        "optical_flow": config.tracking.optical_flow,
    }
    return stats, settings
