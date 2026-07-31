[CmdletBinding()]
param(
    [string]$EnvironmentName = "FaceSwap-Pro",
    [switch]$Recreate,
    [switch]$SkipFfmpeg,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Find-CondaExecutable {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "No se encontró Conda. Instala Anaconda/Miniconda o abre Anaconda Prompt."
}

function Invoke-CondaCommand {
    param([string[]]$Arguments)
    & $script:CondaExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Conda terminó con código ${LASTEXITCODE}: conda $($Arguments -join ' ')"
    }
}

function Get-EnvironmentPrefix {
    $jsonText = & $script:CondaExecutable env list --json
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo consultar la lista de entornos Conda."
    }
    $environmentList = $jsonText | ConvertFrom-Json
    foreach ($prefix in $environmentList.envs) {
        if ((Split-Path $prefix -Leaf) -ieq $EnvironmentName) {
            return $prefix
        }
    }
    return $null
}

function Install-GyanFfmpeg {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "WinGet no está disponible. Instala App Installer o ejecuta con -SkipFfmpeg."
    }

    $listOutput = & $winget.Source list -e --id Gyan.FFmpeg --accept-source-agreements 2>&1 | Out-String
    $installed = ($LASTEXITCODE -eq 0) -and ($listOutput -match "Gyan\.FFmpeg")
    if (-not $installed) {
        Write-Step "Instalando FFmpeg de Gyan con NVENC mediante WinGet"
        & $winget.Source install -e --id Gyan.FFmpeg --scope user `
            --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "WinGet no pudo instalar Gyan.FFmpeg."
        }
    }

    $ffmpegLink = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\ffmpeg.exe"
    if (-not (Test-Path $ffmpegLink)) {
        throw "Gyan.FFmpeg figura instalado, pero no existe el alias: $ffmpegLink"
    }

    $encoders = & $ffmpegLink -hide_banner -encoders 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or $encoders -notmatch "h264_nvenc") {
        throw "La build instalada de FFmpeg no expone h264_nvenc."
    }
    return $ffmpegLink
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$script:CondaExecutable = Find-CondaExecutable

Write-Step "Comprobando controlador y GPU NVIDIA"
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    throw "No se encontró nvidia-smi. Instala o actualiza el controlador NVIDIA y reinicia Windows."
}
& $nvidiaSmi.Source --query-gpu=name,driver_version --format=csv,noheader
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi no pudo consultar la GPU NVIDIA."
}

Write-Step "Preparando entorno Conda '$EnvironmentName' con Python 3.11"
$environmentPrefix = Get-EnvironmentPrefix
if ($Recreate -and $environmentPrefix) {
    if ($env:CONDA_DEFAULT_ENV -ieq $EnvironmentName) {
        throw "Desactiva el entorno '$EnvironmentName' antes de usar -Recreate."
    }
    Invoke-CondaCommand -Arguments @("env", "remove", "-n", $EnvironmentName, "-y")
    $environmentPrefix = $null
}

if (-not $environmentPrefix) {
    Invoke-CondaCommand -Arguments @("create", "-n", $EnvironmentName, "python=3.11", "pip", "-y")
    $environmentPrefix = Get-EnvironmentPrefix
} else {
    Invoke-CondaCommand -Arguments @("install", "-n", $EnvironmentName, "python=3.11", "pip", "-y")
}

if (-not $environmentPrefix) {
    throw "No se pudo determinar la ruta del entorno '$EnvironmentName'."
}

Write-Step "Preparando estructura de entradas y salidas"
$projectDirectories = @(
    "inputs\videos",
    "inputs\source_faces",
    "inputs\target_faces",
    "models",
    "outputs\videos",
    "outputs\manifests"
)
foreach ($relativePath in $projectDirectories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $relativePath) | Out-Null
}

Write-Step "Instalando FaceSwap-Pro y sus dependencias"
Invoke-CondaCommand -Arguments @(
    "run", "--no-capture-output", "-n", $EnvironmentName,
    "python", "-m", "pip", "install", "--upgrade", "setuptools", "wheel"
)

& $script:CondaExecutable run --no-capture-output -n $EnvironmentName `
    python -m pip uninstall -y faceswap-pro
Invoke-CondaCommand -Arguments @(
    "run", "--no-capture-output", "-n", $EnvironmentName,
    "python", "-m", "pip", "install", "-e", ".[dev]"
)

Write-Step "Reparando ONNX Runtime para usar exclusivamente la build GPU"
# InsightFace declara 'onnxruntime' CPU en sus metadatos. Se eliminan ambas
# variantes y se reinstala al final la build GPU con los runtimes CUDA/cuDNN.
& $script:CondaExecutable run --no-capture-output -n $EnvironmentName `
    python -m pip uninstall -y onnxruntime onnxruntime-gpu
if ($LASTEXITCODE -ne 0) {
    throw "No se pudieron limpiar las variantes de ONNX Runtime."
}
Invoke-CondaCommand -Arguments @(
    "run", "--no-capture-output", "-n", $EnvironmentName,
    "python", "-m", "pip", "install", "--no-cache-dir", "--upgrade",
    "onnxruntime-gpu[cuda,cudnn]==1.28.0"
)
Invoke-CondaCommand -Arguments @(
    "run", "--no-capture-output", "-n", $EnvironmentName,
    "python", "-m", "pip", "install", "-e", ".", "--no-deps"
)

Write-Step "Validando CUDAExecutionProvider"
$cudaProbe = @'
import sys
import onnxruntime as ort
try:
    ort.preload_dlls(directory="")
except Exception as exc:
    print(f"Aviso preload_dlls: {exc}")
providers = ort.get_available_providers()
print("ONNX Runtime:", ort.__version__)
print("Providers:", providers)
sys.exit(0 if "CUDAExecutionProvider" in providers else 2)
'@
& $script:CondaExecutable run --no-capture-output -n $EnvironmentName python -c $cudaProbe
if ($LASTEXITCODE -ne 0) {
    throw "CUDAExecutionProvider no está disponible. Verifica el controlador NVIDIA y reinicia Windows."
}

$ffmpegLink = $null
if (-not $SkipFfmpeg) {
    $ffmpegLink = Install-GyanFfmpeg
    Write-Step "Configurando el entorno para preferir FFmpeg con NVENC"
    Invoke-CondaCommand -Arguments @(
        "env", "config", "vars", "set", "-n", $EnvironmentName,
        "FACESWAP_PRO_FFMPEG=$ffmpegLink"
    )
}

if (-not $SkipTests) {
    Write-Step "Ejecutando pruebas unitarias"
    Invoke-CondaCommand -Arguments @(
        "run", "--no-capture-output", "-n", $EnvironmentName,
        "python", "-m", "pytest", "-q"
    )
}

Write-Step "Ejecutando diagnóstico final"
$cli = Join-Path $environmentPrefix "Scripts\faceswap-pro.exe"
if (-not (Test-Path $cli)) {
    throw "No se encontró el comando instalado: $cli"
}
if ($ffmpegLink) {
    $env:FACESWAP_PRO_FFMPEG = $ffmpegLink
}
& $cli doctor
if ($LASTEXITCODE -ne 0) {
    throw "El diagnóstico final no fue satisfactorio."
}

Write-Host "`nInstalación terminada correctamente." -ForegroundColor Green
Write-Host "Activa el entorno con: conda activate $EnvironmentName"
Write-Host "Ejecuta el ejemplo con: powershell -ExecutionPolicy Bypass -File scripts\run_example.ps1"
