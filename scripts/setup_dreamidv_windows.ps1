param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11",
    [string]$RepositoryPath = ".\third_party\DreamID-V",
    [string]$VenvPath = ".\.venv-dreamidv",
    [switch]$InstallRequirements
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "El comando falló ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

New-Item -ItemType Directory -Force -Path ".\third_party" | Out-Null
New-Item -ItemType Directory -Force -Path ".\models\dreamidv\Wan2.1-T2V-1.3B" | Out-Null

if (-not (Test-Path $RepositoryPath)) {
    Invoke-Checked "git" @(
        "clone",
        "--depth", "1",
        "https://github.com/bytedance/DreamID-V.git",
        $RepositoryPath
    )
}

if (-not (Test-Path $VenvPath)) {
    Invoke-Checked $PythonLauncher @($PythonVersion, "-m", "venv", $VenvPath)
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if ($InstallRequirements) {
    Write-Host "Instala primero una build CUDA de PyTorch compatible con tu RTX 5070 Ti." -ForegroundColor Yellow
    Invoke-Checked $VenvPython @(
        "-m", "pip", "install", "-r", (Join-Path $RepositoryPath "requirements.txt")
    )
}

Write-Host "Entorno DreamID-V preparado: $VenvPython" -ForegroundColor Green
Write-Host "Faltan los checkpoints; consulta docs\dreamidv_5070ti.md." -ForegroundColor Cyan
