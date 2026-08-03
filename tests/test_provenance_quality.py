from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from faceswap_pro import provenance


def test_model_hash_cache_reuses_digest_until_artifact_changes(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    model.write_bytes(b"version-one")
    cache = tmp_path / "cache" / "hashes.json"
    calls = 0
    original = provenance.sha256_path

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(provenance, "sha256_path", counted)

    first = provenance.sha256_path_cached(model, cache)
    second = provenance.sha256_path_cached(model, cache)
    assert first == second
    assert calls == 1

    model.write_bytes(b"version-two-is-longer")
    os.utime(model, None)
    third = provenance.sha256_path_cached(model, cache)
    assert third != first
    assert calls == 2


def test_c2pa_is_non_blocking_when_optional_tool_is_missing(tmp_path, monkeypatch):
    output = tmp_path / "output.mp4"
    source = tmp_path / "input.mp4"
    output.write_bytes(b"output")
    source.write_bytes(b"input")
    monkeypatch.setattr(provenance, "_resolve_tool", lambda value: None)
    config = SimpleNamespace(c2pa_enabled=True, c2pa_required=False, c2pa_tool="missing")

    result = provenance.embed_c2pa_manifest(
        output_video=output,
        input_video=source,
        runtime={"model_backend": "dreamid_v"},
        manifest_dir=tmp_path / "manifests",
        config=config,
    )

    assert result["status"] == "skipped"
    assert "c2patool" in result["error"]


def test_c2pa_required_fails_closed_when_tool_is_missing(tmp_path, monkeypatch):
    output = tmp_path / "output.mp4"
    source = tmp_path / "input.mp4"
    output.write_bytes(b"output")
    source.write_bytes(b"input")
    monkeypatch.setattr(provenance, "_resolve_tool", lambda value: None)
    config = SimpleNamespace(c2pa_enabled=True, c2pa_required=True, c2pa_tool="missing")

    with pytest.raises(RuntimeError, match="c2patool"):
        provenance.embed_c2pa_manifest(
            output_video=output,
            input_video=source,
            runtime={},
            manifest_dir=tmp_path / "manifests",
            config=config,
        )


def test_c2pa_definition_includes_standard_generative_action(tmp_path, monkeypatch):
    import json
    import shutil
    import subprocess

    output = tmp_path / "output.mp4"
    source = tmp_path / "input.mp4"
    output.write_bytes(b"output")
    source.write_bytes(b"input")
    monkeypatch.setattr(provenance, "_resolve_tool", lambda value: "c2patool")

    def fake_run(command, **kwargs):
        signed = Path(command[command.index("-o") + 1])
        shutil.copyfile(output, signed)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    config = SimpleNamespace(
        c2pa_enabled=True,
        c2pa_required=True,
        c2pa_tool="c2patool",
        c2pa_algorithm="es256",
        c2pa_sign_cert=None,
        c2pa_private_key=None,
        c2pa_timestamp_url=None,
    )

    result = provenance.embed_c2pa_manifest(
        output_video=output,
        input_video=source,
        runtime={"model_backend": "dreamid_v"},
        manifest_dir=tmp_path / "manifests",
        config=config,
    )

    definition = json.loads(Path(result["definition"]).read_text(encoding="utf-8"))
    actions = next(item for item in definition["assertions"] if item["label"] == "c2pa.actions.v2")
    action = actions["data"]["actions"][0]
    assert action["action"] == "c2pa.created"
    assert action["digitalSourceType"].endswith("compositeWithTrainedAlgorithmicMedia")
    assert action["softwareAgent"]["name"] == "FaceSwap-Pro"
