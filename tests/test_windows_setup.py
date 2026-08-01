from pathlib import Path


def test_windows_setup_runs_multiline_python_probes_from_temporary_files():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert "function Invoke-EnvironmentPythonCode" in script
    assert 'Invoke-EnvironmentPythonCode -Code $cudaProbe -Name "cuda-probe"' in script
    assert 'Invoke-EnvironmentPythonCode -Code $torchProbe -Name "torch-probe"' in script
    assert 'Invoke-EnvironmentPython -Arguments @("-c", $cudaProbe)' not in script
    assert 'Invoke-EnvironmentPython -Arguments @("-c", $torchProbe)' not in script


def test_windows_setup_uses_sparse_checkout_for_reserved_aux_filename():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert "function Install-HifiFaceRepository" in script
    assert "--no-checkout" in script
    assert "config core.protectNTFS false" in script
    assert '"!/AdaptiveWingLoss/aux.py"' in script
    assert "Install-HifiFaceRepository" in script


def test_windows_setup_keeps_onnxruntime_and_pytorch_on_cuda_12():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert '[string]$OnnxRuntimeVersion = "1.26.0"' in script
    assert '[string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"' in script


def test_windows_setup_does_not_redownload_compatible_pytorch():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert "function Test-EnvironmentPythonCode" in script
    assert "$torchAlreadyReady" in script
    assert "no se reinstalará" in script


def test_windows_setup_keeps_setuptools_compatible_with_pytorch():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert '"setuptools<82"' in script
    assert '"pip", "install", "--upgrade", "pip", "setuptools", "wheel"' not in script


def test_windows_setup_does_not_collect_third_party_tests():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'Invoke-EnvironmentPython -Arguments @("-m", "pytest", "-q", "tests")' in script
    assert 'testpaths = ["tests"]' in pyproject
    assert 'norecursedirs = ["third_party"' in pyproject


def test_windows_setup_reuses_onnxruntime_gpu_and_pip_cache():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert '"onnxruntime", "onnxruntime-directml"' in script
    assert '"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"' not in script
    assert '$onnxRuntimeReady = Test-EnvironmentPythonCode' in script
    assert '"--force-reinstall", "--no-deps"' in script

    ort_section = script.split(
        'Write-Step "Configurando ONNX Runtime exclusivamente para CUDA"', 1
    )[1].split('if (-not $SkipMeshAssist)', 1)[0]
    assert '"--no-cache-dir"' not in ort_section


def test_windows_setup_supports_onnxruntime_without_preload_dlls():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert 'getattr(ort, "preload_dlls", None)' in script
    assert "if callable(preload_dlls):" in script
    assert 'ort.preload_dlls(directory="")' not in script


def test_windows_setup_warning_block_accepts_blank_separator_lines():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert "[AllowEmptyString()][string[]]$Lines" in script


def test_windows_setup_installs_insightface_without_cpu_onnxruntime_dependency():
    script = Path("scripts/setup_windows.ps1").read_text(encoding="utf-8-sig")

    assert '"--no-deps", "insightface==1.0.1"' in script
    assert '"--no-deps", "-e", "."' in script
    assert '"onnxruntime-gpu[cuda,cudnn]==$OnnxRuntimeVersion"' in script
