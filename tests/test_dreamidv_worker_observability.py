from __future__ import annotations

import json

import numpy as np

from faceswap_pro.dreamidv_backend import (
    _ingest_worker_log,
    _ingest_worker_profile,
)
from faceswap_pro.dreamidv_worker import WorkerProfiler, _is_single_frame_input
from faceswap_pro.observability import configure_observability


def _json_lines(text: str) -> list[dict]:
    records: list[dict] = []
    for line in text.splitlines():
        _, payload = line.split(" ", 1)
        records.append(json.loads(payload))
    return records


def test_worker_profiler_emits_atomic_span_summary_and_log(capsys):
    profiler = WorkerProfiler(enabled=True, device_id=0)
    profiler.set_request(
        {
            "request_id": "request-1",
            "clip_index": 2,
            "frame_num": 49,
            "sample_steps": 8,
        }
    )
    with profiler.span("dreamidv.dit.forward", cuda=False, diffusion_step_index=0):
        sum(range(20))
    profiler.log("WARNING", "problema controlado", code="unit")
    profiler.summary(elapsed_seconds=1.0)

    records = _json_lines(capsys.readouterr().out)
    assert any(
        record.get("event") == "span"
        and record.get("name") == "dreamidv.dit.forward"
        and record.get("worker_request_id") == "request-1"
        for record in records
    )
    assert any(record.get("event") == "dreamidv.clip_summary" for record in records)
    assert any(record.get("message") == "problema controlado" for record in records)


def test_reference_cache_classifier_only_accepts_single_frame_temporal_inputs():
    image = np.zeros((3, 1, 32, 32), dtype=np.float32)
    video = np.zeros((3, 49, 32, 32), dtype=np.float32)

    assert _is_single_frame_input(([image], {})) is True
    assert _is_single_frame_input(([video], {})) is False


def test_parent_ingests_worker_profile_and_log_into_two_jsonl_files(tmp_path):
    session = configure_observability(tmp_path / "logs", command="worker-ingest-test")
    try:
        _ingest_worker_profile(
            {
                "event": "span",
                "name": "dreamidv.vae.encode",
                "duration_ns": 123,
                "thread_cpu_ns": 45,
                "process_id": 999,
                "source": "dreamidv_worker",
            }
        )
        _ingest_worker_log(
            {
                "level": "WARNING",
                "message": "VRAM alta",
                "source": "dreamidv_worker",
                "cuda_reserved_bytes": 42,
            }
        )
    finally:
        session.close()

    profile = [
        json.loads(line)
        for line in session.profile_path.read_text(encoding="utf-8").splitlines()
    ]
    logs = [
        json.loads(line)
        for line in session.logs_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row.get("event") == "span"
        and row.get("name") == "dreamidv.vae.encode"
        and row.get("process_id") == 999
        for row in profile
    )
    assert any(
        row.get("event") == "external_log"
        and row.get("message") == "VRAM alta"
        and row.get("cuda_reserved_bytes") == 42
        for row in logs
    )
