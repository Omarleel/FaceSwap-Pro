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

from .modeling import FaceAnalyzer, FaceData, FaceRestorer, FaceSwapper, SwapResult
from .observability import (
    function_profile,
    log_exception,
    log_problem,
    profile_event,
    profile_span,
)
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
    target_faces: tuple[FaceData, ...] = ()

    @property
    def target_face(self) -> FaceData | None:
        """Compatibilidad con extensiones que consumían una sola cara."""
        return self.target_faces[-1] if self.target_faces else None


@dataclass(frozen=True)
class ProcessedPacket:
    index: int
    frame: np.ndarray
    swapped: bool
    swapped_faces: int
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
    swapped_faces: int = 0
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


def _queue_put(
    target: queue.Queue,
    item,
    stop: threading.Event,
    *,
    queue_name: str,
    frame_index: int | None = None,
) -> bool:
    started_ns = time.perf_counter_ns()
    retries = 0
    while not stop.is_set():
        try:
            target.put(item, timeout=0.1)
            wait_ns = time.perf_counter_ns() - started_ns
            profile_event(
                "queue_put",
                queue=queue_name,
                frame_index=frame_index,
                wait_ns=wait_ns,
                retries=retries,
                queue_size=target.qsize(),
            )
            if wait_ns >= 1_000_000_000:
                log_problem(
                    "Bloqueo prolongado al insertar en una cola del pipeline",
                    queue=queue_name,
                    frame_index=frame_index,
                    wait_ns=wait_ns,
                    retries=retries,
                )
            return True
        except queue.Full:
            retries += 1
            continue
    profile_event(
        "queue_put_cancelled",
        queue=queue_name,
        frame_index=frame_index,
        wait_ns=time.perf_counter_ns() - started_ns,
        retries=retries,
    )
    return False


def _raise_pipeline_error(errors: queue.Queue) -> None:
    try:
        exc = errors.get_nowait()
    except queue.Empty:
        return
    log_exception("Falló una etapa del pipeline paralelo", exc)
    raise RuntimeError("Falló una etapa del pipeline paralelo.") from exc


def _reader_worker(reader, output: queue.Queue, stop, errors, stats: PipelineStats) -> None:
    index = 0
    with function_profile("thread.reader"):
        try:
            while not stop.is_set():
                started = time.perf_counter()
                with profile_span("frame.decode", frame_index=index):
                    ok, frame = reader.read()
                elapsed = time.perf_counter() - started
                if not ok or frame is None:
                    profile_event("reader_end_of_stream", frame_index=index)
                    break
                stats.add(decoded_frames=1, decode_seconds=elapsed)
                if not _queue_put(
                    output,
                    FramePacket(index, frame),
                    stop,
                    queue_name="reader_to_analysis",
                    frame_index=index,
                ):
                    break
                index += 1
        except BaseException as exc:  # noqa: BLE001 - se propaga al hilo principal
            log_exception("Falló el hilo de decodificación", exc, stage="reader")
            errors.put(exc)
            stop.set()
        finally:
            with profile_span("reader.close"):
                reader.close()
            if not stop.is_set():
                _queue_put(
                    output,
                    _SENTINEL,
                    stop,
                    queue_name="reader_to_analysis",
                )


def _analysis_worker(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    stop: threading.Event,
    errors: queue.Queue,
    stats: PipelineStats,
    analyzer: FaceAnalyzer,
    tracker,
    tracking_config,
) -> None:
    def previous_regions():
        bboxes = getattr(tracker, "current_bboxes", None)
        if bboxes and bool(
            getattr(analyzer, "supports_multiple_previous_bboxes", False)
        ):
            return bboxes
        return getattr(tracker, "current_bbox", None)

    def select_faces(frame, gray, faces) -> tuple[FaceData, ...]:
        method = getattr(tracker, "select_all_detected", None)
        if callable(method):
            return tuple(method(frame, gray, faces))
        selected = tracker.select_detected(frame, gray, faces)
        return () if selected is None else (selected,)

    def propagate_faces(frame, gray) -> tuple[FaceData, ...]:
        method = getattr(tracker, "propagate_all", None)
        if callable(method):
            return tuple(method(frame, gray))
        selected = tracker.propagate(frame, gray)
        return () if selected is None else (selected,)

    with function_profile("thread.analysis"):
        try:
            while not stop.is_set():
                wait_started_ns = time.perf_counter_ns()
                try:
                    packet = input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                profile_event(
                    "queue_get",
                    queue="reader_to_analysis",
                    frame_index=getattr(packet, "index", None),
                    wait_ns=time.perf_counter_ns() - wait_started_ns,
                    queue_size=input_queue.qsize(),
                )
                if packet is _SENTINEL:
                    _queue_put(
                        output_queue,
                        _SENTINEL,
                        stop,
                        queue_name="analysis_to_main",
                    )
                    return

                started = time.perf_counter()
                with profile_span("frame.analysis", frame_index=packet.index):
                    with profile_span("frame.tracker.observe", frame_index=packet.index):
                        gray, scene_cut = tracker.observe(packet.frame)
                    detection_due = (
                        scene_cut
                        or tracker.needs_redetect
                        or packet.index % tracking_config.detection_interval == 0
                    )
                    target_faces: tuple[FaceData, ...] = ()
                    used_flow = False
                    full_scan = False
                    redetection = False

                    if detection_due:
                        full_scan = (
                            scene_cut
                            or tracker.needs_redetect
                            or packet.index % tracking_config.full_scan_interval == 0
                        )
                        with profile_span(
                            "frame.face_analysis",
                            frame_index=packet.index,
                            full_scan=full_scan,
                            reason="scheduled_or_required",
                        ):
                            faces, detection_stats = analyzer.analyze(
                                packet.frame,
                                previous_bbox=previous_regions(),
                                full_scan=full_scan,
                            )
                        with profile_span(
                            "frame.target_selection",
                            frame_index=packet.index,
                            candidates=len(faces),
                        ):
                            target_faces = select_faces(packet.frame, gray, faces)
                        stats.add(
                            detection_frames=1,
                            full_scans=int(detection_stats.full_scan),
                            detected_faces=detection_stats.detected,
                            recognition_inferences=detection_stats.recognized,
                        )
                    else:
                        with profile_span(
                            "frame.optical_flow",
                            frame_index=packet.index,
                        ):
                            target_faces = propagate_faces(packet.frame, gray)
                        used_flow = bool(target_faces)
                        if not target_faces or tracker.needs_redetect:
                            redetection = True
                            # Redetección inmediata: evita perder un frame cuando LK falla
                            # en una o en todas las apariciones del sujeto.
                            with profile_span(
                                "frame.face_analysis",
                                frame_index=packet.index,
                                full_scan=True,
                                reason="optical_flow_fallback",
                            ):
                                faces, detection_stats = analyzer.analyze(
                                    packet.frame,
                                    previous_bbox=previous_regions(),
                                    full_scan=True,
                                )
                            with profile_span(
                                "frame.target_selection",
                                frame_index=packet.index,
                                candidates=len(faces),
                            ):
                                target_faces = select_faces(packet.frame, gray, faces)
                            stats.add(
                                detection_frames=1,
                                full_scans=1,
                                detected_faces=detection_stats.detected,
                                recognition_inferences=detection_stats.recognized,
                            )

                    profile_event(
                        "frame_analysis_result",
                        frame_index=packet.index,
                        scene_cut=bool(scene_cut),
                        detection_due=bool(detection_due),
                        full_scan=bool(full_scan),
                        optical_flow_used=bool(used_flow),
                        redetection=redetection,
                        target_faces=len(target_faces),
                    )

                stats.add(
                    analyzed_frames=1,
                    optical_flow_frames=int(used_flow),
                    analysis_seconds=time.perf_counter() - started,
                )
                if not _queue_put(
                    output_queue,
                    AnalyzedPacket(packet.index, packet.frame, target_faces),
                    stop,
                    queue_name="analysis_to_main",
                    frame_index=packet.index,
                ):
                    return
        except BaseException as exc:  # noqa: BLE001
            log_exception("Falló el hilo de análisis", exc, stage="analysis")
            errors.put(exc)
            stop.set()


def _writer_worker(
    writer,
    input_queue: queue.Queue,
    stop: threading.Event,
    errors: queue.Queue,
    stats: PipelineStats,
) -> None:
    with function_profile("thread.writer"):
        try:
            while not stop.is_set():
                wait_started_ns = time.perf_counter_ns()
                try:
                    packet = input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                profile_event(
                    "queue_get",
                    queue="main_to_writer",
                    frame_index=getattr(packet, "index", None),
                    wait_ns=time.perf_counter_ns() - wait_started_ns,
                    queue_size=input_queue.qsize(),
                )
                if packet is _SENTINEL:
                    return
                started = time.perf_counter()
                with profile_span("frame.encode_feed", frame_index=packet.index):
                    writer.write(packet.frame)
                stats.add(
                    written_frames=1,
                    swapped_frames=int(packet.swapped),
                    swapped_faces=packet.swapped_faces,
                    postprocess_seconds=packet.postprocess_seconds,
                    encode_feed_seconds=time.perf_counter() - started,
                )
        except BaseException as exc:  # noqa: BLE001
            log_exception("Falló el hilo de escritura", exc, stage="writer")
            errors.put(exc)
            stop.set()


def _postprocess(
    packet: AnalyzedPacket,
    swap_results: tuple[SwapResult, ...],
    blender,
    restorer,
    visible_disclosure: bool,
) -> ProcessedPacket:
    started = time.perf_counter()
    with profile_span(
        "frame.postprocess",
        frame_index=packet.index,
        swap_result_count=len(swap_results),
    ):
        frame = packet.frame
        swapped_faces = 0
        for face_index, swap_result in enumerate(swap_results):
            if swap_result.opacity <= 0.0:
                profile_event(
                    "face_composite_skipped",
                    frame_index=packet.index,
                    face_index=face_index,
                    opacity=swap_result.opacity,
                )
                continue
            with profile_span(
                "face.composite",
                frame_index=packet.index,
                face_index=face_index,
                mask_mode=swap_result.mask_mode,
                opacity=swap_result.opacity,
                restorer_enabled=bool(restorer.enabled),
            ):
                frame = blender.composite(
                    frame,
                    swap_result.crop,
                    swap_result.affine,
                    restorer,
                    mask=swap_result.mask,
                    opacity=swap_result.opacity,
                    mask_mode=swap_result.mask_mode,
                )
            swapped_faces += 1
        if visible_disclosure:
            with profile_span("frame.visible_disclosure", frame_index=packet.index):
                frame = add_disclosure(frame)
        with profile_span("frame.ascontiguousarray", frame_index=packet.index):
            contiguous_frame = np.ascontiguousarray(frame)
    return ProcessedPacket(
        index=packet.index,
        frame=contiguous_frame,
        swapped=swapped_faces > 0,
        swapped_faces=swapped_faces,
        postprocess_seconds=time.perf_counter() - started,
    )


def run_parallel_frames(
    *,
    reader,
    writer,
    analyzer: FaceAnalyzer,
    swapper: FaceSwapper,
    source_face,
    tracker,
    blender,
    restorer: FaceRestorer,
    config,
) -> tuple[PipelineStats, dict[str, Any]]:
    workers, cv_threads = resolve_parallelism(config.performance, restorer.enabled)
    provenance = getattr(config, "provenance", None)
    visible_disclosure = bool(getattr(provenance, "visible_disclosure", True))
    cv2.setUseOptimized(True)
    cv2.setNumThreads(cv_threads)
    profile_event(
        "parallel_pipeline_start",
        postprocess_workers=workers,
        opencv_threads=cv_threads,
        reader_queue=config.performance.reader_queue,
        analysis_queue=config.performance.analysis_queue,
        writer_queue=config.performance.writer_queue,
        max_inflight=config.performance.max_inflight,
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

    with profile_span("parallel.start_threads"):
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
        wait_started_ns = time.perf_counter_ns()
        with profile_span(
            "frame.await_postprocess",
            frame_index=next_emit,
            blocking=block,
        ):
            packet = future.result()
        profile_event(
            "future_result",
            frame_index=next_emit,
            blocking=block,
            wait_ns=time.perf_counter_ns() - wait_started_ns,
            pending_count=len(pending),
        )
        del pending[next_emit]
        if not _queue_put(
            write_queue,
            packet,
            stop,
            queue_name="main_to_writer",
            frame_index=packet.index,
        ):
            _raise_pipeline_error(errors)
            raise RuntimeError("El escritor se detuvo antes de recibir todos los frames.")
        next_emit += 1
        return True

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="faceswap-post")
    try:
        while not stop.is_set():
            _raise_pipeline_error(errors)
            wait_started_ns = time.perf_counter_ns()
            try:
                packet = analysis_queue.get(timeout=0.1)
            except queue.Empty:
                if not analysis_thread.is_alive():
                    _raise_pipeline_error(errors)
                    break
                continue
            profile_event(
                "queue_get",
                queue="analysis_to_main",
                frame_index=getattr(packet, "index", None),
                wait_ns=time.perf_counter_ns() - wait_started_ns,
                queue_size=analysis_queue.qsize(),
            )
            if packet is _SENTINEL:
                break

            swap_results: list[SwapResult] = []
            if packet.target_faces:
                swap_started = time.perf_counter()
                for face_index, target_face in enumerate(packet.target_faces):
                    with profile_span(
                        "face.swap",
                        frame_index=packet.index,
                        face_index=face_index,
                    ):
                        swap_results.append(
                            swapper.swap(
                                packet.frame,
                                target_face,
                                source_face,
                            )
                        )
                stats.add(swap_seconds=time.perf_counter() - swap_started)

            with profile_span(
                "frame.submit_postprocess",
                frame_index=packet.index,
                target_faces=len(packet.target_faces),
            ):
                pending[packet.index] = executor.submit(
                    _postprocess,
                    packet,
                    tuple(swap_results),
                    blender,
                    restorer,
                    visible_disclosure,
                )
            profile_event(
                "postprocess_submitted",
                frame_index=packet.index,
                pending_count=len(pending),
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

        if not _queue_put(
            write_queue,
            _SENTINEL,
            stop,
            queue_name="main_to_writer",
        ):
            _raise_pipeline_error(errors)
        with profile_span("parallel.join_writer"):
            writer_thread.join()
        _raise_pipeline_error(errors)
    finally:
        progress.close()
        stop.set()
        with profile_span("parallel.shutdown_executor"):
            executor.shutdown(wait=True, cancel_futures=True)
        for thread in (reader_thread, analysis_thread, writer_thread):
            with profile_span("parallel.join_thread", joined_thread=thread.name):
                thread.join(timeout=5)
            if thread.is_alive():
                log_problem(
                    "Un hilo del pipeline no terminó dentro del límite de cierre",
                    thread=thread.name,
                )

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
        "max_target_faces": int(
            getattr(config.tracking, "max_target_faces", 1)
        ),
    }
    profile_event(
        "parallel_pipeline_end",
        settings=settings,
        stats=stats.as_dict(),
    )
    return stats, settings
