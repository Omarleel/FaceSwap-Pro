from types import SimpleNamespace

from faceswap_pro.cli import _hififace_readiness


def test_hififace_readiness_does_not_treat_missing_paths_as_current_directory(tmp_path):
    config = SimpleNamespace(
        engine=SimpleNamespace(
            options={
                "checkpoint_iteration": 80000,
                "device": "cpu",
            }
        )
    )

    result = _hififace_readiness(config, tmp_path / "missing-checkpoint")

    assert result["ready"] is False
    assert result["repository_path"] is None
    assert result["checks"]["repository"] is False
    assert result["checks"]["checkpoint_directory"] is False
    assert result["checks"]["generator_checkpoint"] is False


def test_hififace_readiness_rejects_incomplete_repository_checkout(tmp_path):
    repository = tmp_path / "HiFiFace-pytorch"
    repository.mkdir()
    config = SimpleNamespace(
        engine=SimpleNamespace(
            options={
                "repository_path": str(repository),
                "checkpoint_iteration": 80000,
                "device": "cpu",
            }
        )
    )

    result = _hififace_readiness(config, tmp_path / "standard_model")

    assert result["checks"]["repository"] is False

    models = repository / "models"
    models.mkdir()
    (models / "model.py").write_text("", encoding="utf-8")
    (models / "generator.py").write_text("", encoding="utf-8")

    result = _hififace_readiness(config, tmp_path / "standard_model")

    assert result["checks"]["repository"] is True
