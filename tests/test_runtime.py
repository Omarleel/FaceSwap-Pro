from pathlib import Path

from faceswap_pro import runtime


def test_ffmpeg_override_has_priority(tmp_path, monkeypatch):
    override = tmp_path / ("ffmpeg.exe" if runtime.os.name == "nt" else "ffmpeg")
    override.write_bytes(b"")
    monkeypatch.setenv(runtime.FFMPEG_ENV_VAR, str(override))
    monkeypatch.setenv("PATH", "")

    assert runtime.ffmpeg_candidates()[0] == override


def test_select_ffmpeg_prefers_requested_encoder(tmp_path, monkeypatch):
    first = tmp_path / "first_ffmpeg"
    second = tmp_path / "second_ffmpeg"
    first.write_bytes(b"")
    second.write_bytes(b"")

    monkeypatch.setattr(runtime, "ffmpeg_candidates", lambda: [first, second])
    monkeypatch.setattr(
        runtime,
        "ffmpeg_has_encoder",
        lambda path, encoder: Path(path) == second and encoder == "h264_nvenc",
    )

    assert runtime.select_ffmpeg("h264_nvenc") == second
