import json
import threading

from faceswap_pro.observability import (
    configure_observability,
    function_profile,
    log_exception,
    profile_event,
    profile_span,
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_observability_creates_two_structured_files_and_records_atomic_metrics(tmp_path):
    session = configure_observability(tmp_path / "logs", command="test")
    try:
        with profile_span("test.outer", frame_index=7):
            with profile_span("test.inner", operation="unit"):
                profile_event("test_metric", value=3)

        def worker():
            with function_profile("test.worker"):
                sum(range(100))

        thread = threading.Thread(target=worker, name="test-profile-worker")
        thread.start()
        thread.join()

        try:
            raise ValueError("fallo controlado")
        except ValueError as exc:
            log_exception("Problema de prueba", exc, stage="unit")
    finally:
        session.close()

    assert session.logs_path.is_file()
    assert session.profile_path.is_file()
    assert sorted(path.name for path in session.log_dir.iterdir()) == [
        "logs.jsonl",
        "profile.jsonl",
    ]

    log_records = _read_jsonl(session.logs_path)
    profile_records = _read_jsonl(session.profile_path)

    assert any(record.get("event") == "exception" for record in log_records)
    assert any(
        record.get("event") == "span" and record.get("name") == "test.inner"
        for record in profile_records
    )
    assert any(record.get("event") == "test_metric" for record in profile_records)
    assert any(record.get("event") == "cprofile_function" for record in profile_records)
    assert any(
        record.get("scope") == "test.worker"
        and record.get("event") in {"cprofile_function", "cprofile_unavailable"}
        for record in profile_records
    )
    assert any(record.get("event") == "span_summary" for record in profile_records)
    assert any(record.get("event") == "session_end" for record in profile_records)
