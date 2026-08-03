[CmdletBinding()]
param(
    [string]$EnvironmentName = "FaceSwap-Pro",
    [switch]$Recreate,
    [switch]$SkipFfmpeg,
    [switch]$SkipTests,

    [Alias("SkipGeometry3D")]
    [switch]$SkipMeshAssist,

    [switch]$SkipHifiFace3DMM,

    # ORT 1.26.x usa CUDA 12.8 en PyPI y es compatible con la build cu128 de
    # PyTorch instalada más abajo. ORT 1.27+ usa CUDA 13 por defecto y mezclar
    # ambas familias puede hacer que torch.cuda.is_available() falle en Windows.
    [string]$OnnxRuntimeVersion = "1.26.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$HifiFaceRepositoryUrl = "https://github.com/xuehy/HiFiFace-pytorch.git",

    # Carpeta o ZIP opcional con la estructura models\hififace.
    # Se usa para copiar checkpoints obtenidos legítimamente por el usuario.
    [string]$HifiFaceAssetsPath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-WarningBlock {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$Lines)
    Write-Host ""
    foreach ($line in $Lines) {
        Write-Host $line -ForegroundColor Yellow
    }
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
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & $script:CondaExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Conda terminó con código ${LASTEXITCODE}: conda $($Arguments -join ' ')"
    }
}

function Invoke-EnvironmentPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $condaArguments = @(
        "run", "--no-capture-output", "-n", $EnvironmentName, "python"
    ) + $Arguments

    Invoke-CondaCommand -Arguments $condaArguments
}

function Invoke-EnvironmentPythonCode {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [Parameter(Mandatory = $true)][string]$Name
    )

    # Conda para Windows no admite saltos de línea dentro de un argumento de
    # ``python -c``. Escribimos las comprobaciones en un archivo temporal para
    # conservar la activación de ``conda run`` sin depender de escapes frágiles.
    $temporaryScript = Join-Path ([System.IO.Path]::GetTempPath()) (
        "faceswap-pro-$Name-" + [Guid]::NewGuid().ToString("N") + ".py"
    )

    try {
        $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
        [System.IO.File]::WriteAllText($temporaryScript, $Code, $utf8WithoutBom)
        Invoke-EnvironmentPython -Arguments @($temporaryScript)
    } finally {
        Remove-Item -Path $temporaryScript -Force -ErrorAction SilentlyContinue
    }
}

function Test-EnvironmentPythonCode {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $temporaryScript = Join-Path ([System.IO.Path]::GetTempPath()) (
        "faceswap-pro-$Name-" + [Guid]::NewGuid().ToString("N") + ".py"
    )

    try {
        $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
        [System.IO.File]::WriteAllText($temporaryScript, $Code, $utf8WithoutBom)
        & $script:CondaExecutable run --no-capture-output -n $EnvironmentName `
            python $temporaryScript
        return ($LASTEXITCODE -eq 0)
    } finally {
        Remove-Item -Path $temporaryScript -Force -ErrorAction SilentlyContinue
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

    $listOutput = & $winget.Source list -e --id Gyan.FFmpeg --accept-source-agreements 2>&1 |
        Out-String
    $installed = ($LASTEXITCODE -eq 0) -and ($listOutput -match "Gyan\.FFmpeg")

    if (-not $installed) {
        Write-Step "Instalando FFmpeg con NVENC"
        & $winget.Source install -e --id Gyan.FFmpeg --scope user `
            --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "WinGet no pudo instalar Gyan.FFmpeg."
        }
    }

    $ffmpegLink = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\ffmpeg.exe"
    if (-not (Test-Path $ffmpegLink)) {
        throw "FFmpeg figura instalado, pero no existe el alias: $ffmpegLink"
    }

    $encoders = & $ffmpegLink -hide_banner -encoders 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or $encoders -notmatch "h264_nvenc") {
        throw "La build instalada de FFmpeg no expone h264_nvenc."
    }

    return $ffmpegLink
}

function Copy-HifiFaceAssets {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    $resolvedSource = (Resolve-Path $SourcePath).Path
    New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null

    if (Test-Path $resolvedSource -PathType Container) {
        $sourceModels = Join-Path $resolvedSource "models\hififace"
        if (Test-Path $sourceModels -PathType Container) {
            Copy-Item -Path (Join-Path $sourceModels "*") `
                -Destination $DestinationPath -Recurse -Force
            return
        }

        Copy-Item -Path (Join-Path $resolvedSource "*") `
            -Destination $DestinationPath -Recurse -Force
        return
    }

    if ([System.IO.Path]::GetExtension($resolvedSource) -ieq ".zip") {
        $temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) (
            "faceswap-pro-hififace-" + [Guid]::NewGuid().ToString("N")
        )

        try {
            New-Item -ItemType Directory -Force -Path $temporaryDirectory | Out-Null
            Expand-Archive -Path $resolvedSource -DestinationPath $temporaryDirectory -Force

            $sourceModels = Join-Path $temporaryDirectory "models\hififace"
            if (Test-Path $sourceModels -PathType Container) {
                Copy-Item -Path (Join-Path $sourceModels "*") `
                    -Destination $DestinationPath -Recurse -Force
            } else {
                Copy-Item -Path (Join-Path $temporaryDirectory "*") `
                    -Destination $DestinationPath -Recurse -Force
            }
        } finally {
            Remove-Item -Path $temporaryDirectory -Recurse -Force -ErrorAction SilentlyContinue
        }

        return
    }

    throw "HifiFaceAssetsPath debe ser una carpeta o un archivo ZIP."
}

function Install-HifiFaceRepository {
    param(
        [Parameter(Mandatory = $true)][string]$GitExecutable,
        [Parameter(Mandatory = $true)][string]$RepositoryUrl,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )

    $gitDirectory = Join-Path $DestinationPath ".git"
    $requiredFile = Join-Path $DestinationPath "models\model.py"

    # El repositorio contiene AdaptiveWingLoss/aux.py. AUX es un nombre de
    # dispositivo reservado por Windows, incluso con extensión, y Git no puede
    # materializarlo en NTFS. No se usa durante la inferencia de HiFiFace, así que
    # hacemos un sparse checkout de todo el árbol salvo ese único archivo.
    if (Test-Path $DestinationPath) {
        if (-not (Test-Path $gitDirectory -PathType Container)) {
            throw "Existe '$DestinationPath', pero no parece un clon válido de HifiFace."
        }

        if (Test-Path $requiredFile -PathType Leaf) {
            Write-Host "HifiFace ya existe: $DestinationPath"
            return
        }

        Write-Host "Eliminando un checkout incompleto de HifiFace: $DestinationPath"
        Remove-Item -Path $DestinationPath -Recurse -Force
    }

    & $GitExecutable clone --depth 1 --no-checkout $RepositoryUrl $DestinationPath
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -Path $DestinationPath -Recurse -Force -ErrorAction SilentlyContinue
        throw "No se pudo descargar HifiFace desde $RepositoryUrl"
    }

    # Git for Windows protege por defecto rutas incompatibles con NTFS. Se
    # desactiva solo para este clon y, antes del checkout, se excluye AUX.py.
    & $GitExecutable -C $DestinationPath config core.protectNTFS false
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo configurar Git para el checkout compatible con Windows."
    }

    & $GitExecutable -C $DestinationPath sparse-checkout set --no-cone `
        "/*" "!/AdaptiveWingLoss/aux.py"
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo configurar el checkout compatible con Windows de HifiFace."
    }

    & $GitExecutable -C $DestinationPath checkout --force
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo completar el checkout compatible con Windows de HifiFace."
    }

    if (-not (Test-Path $requiredFile -PathType Leaf)) {
        throw "El checkout de HifiFace terminó, pero falta el archivo requerido: $requiredFile"
    }
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$script:CondaExecutable = Find-CondaExecutable

Write-Step "Comprobando GPU y controlador NVIDIA"
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
    Invoke-CondaCommand -Arguments @(
        "create", "-n", $EnvironmentName, "python=3.11", "pip", "-y"
    )
} else {
    Invoke-CondaCommand -Arguments @(
        "install", "-n", $EnvironmentName, "python=3.11", "pip", "-y"
    )
}

$environmentPrefix = Get-EnvironmentPrefix
if (-not $environmentPrefix) {
    throw "No se pudo determinar la ruta del entorno '$EnvironmentName'."
}

Write-Step "Preparando directorios del proyecto"
@(
    "inputs\videos",
    "inputs\source_faces",
    "inputs\target_faces",
    "models",
    "models\hififace\standard_model",
    "models\hififace\auxiliary\Deep3DFaceRecon",
    "models\hififace\auxiliary\arcface",
    "models\hififace\auxiliary\BFM",
    "third_party",
    "outputs\videos",
    "outputs\manifests"
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $_) | Out-Null
}

Write-Step "Actualizando herramientas de instalación"
Invoke-EnvironmentPython -Arguments @(
    "-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel"
)

Write-Step "Instalando dependencias de FaceSwap-Pro"

# InsightFace declara la distribución CPU ``onnxruntime`` como dependencia.
# Para un entorno NVIDIA instalamos sus dependencias explícitamente, luego
# InsightFace y el proyecto con --no-deps. Así una segunda ejecución no vuelve
# a introducir ONNX Runtime CPU ni obliga a reparar la rueda GPU.
$projectRequirements = @(
    "onnxruntime-gpu[cuda,cudnn]==$OnnxRuntimeVersion",
    "onnx>=1.18,<2",
    "opencv-python-headless>=4.10,<5",
    "numpy>=1.26,<3",
    "scipy>=1.13,<2",
    "PyYAML>=6,<7",
    "typer>=0.16,<1",
    "rich>=13.9,<15",
    "tqdm>=4.66,<5",
    "requests>=2,<3",
    "scikit-image>=0.24,<1",
    "pytest>=8.3,<10",
    "ruff>=0.9,<1"
)

if (-not $SkipMeshAssist) {
    $projectRequirements += "mediapipe>=0.10.35,<0.11"
}

$pipInstallArguments = @("-m", "pip", "install") + $projectRequirements
Invoke-EnvironmentPython -Arguments $pipInstallArguments

Invoke-EnvironmentPython -Arguments @(
    "-m", "pip", "install", "--no-deps", "insightface==1.0.1"
)

Write-Step "Instalando FaceSwap-Pro en modo editable"
Invoke-EnvironmentPython -Arguments @(
    "-m", "pip", "install", "--no-deps", "-e", "."
)

$cudaProbe = @'
import importlib.metadata as metadata
import sys

import onnxruntime as ort

distributions = {
    (dist.metadata.get("Name") or "").strip().lower()
    for dist in metadata.distributions()
}
ort_distributions = sorted(
    name for name in distributions if name.startswith("onnxruntime")
)

preload_dlls = getattr(ort, "preload_dlls", None)
if callable(preload_dlls):
    preload_dlls(directory="")
providers = ort.get_available_providers()

print("ONNX Runtime:", ort.__version__)
print("Distribuciones:", ort_distributions)
print("Providers:", providers)

if "onnxruntime" in distributions:
    raise SystemExit("ERROR: continúa instalada la distribución CPU 'onnxruntime'.")
if "onnxruntime-gpu" not in distributions:
    raise SystemExit("ERROR: no está instalada la distribución 'onnxruntime-gpu'.")
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("ERROR: CUDAExecutionProvider no está disponible.")

sys.exit(0)
'@

Write-Step "Configurando ONNX Runtime exclusivamente para CUDA"

# Limpia distribuciones CPU/DirectML que pudieran quedar de ejecuciones
# anteriores. La instalación actual de InsightFace usa --no-deps, por lo que
# no volverán a introducirse en ejecuciones posteriores.
Invoke-EnvironmentPython -Arguments @(
    "-m", "pip", "uninstall", "-y",
    "onnxruntime", "onnxruntime-directml"
)

# Una desinstalación heredada de la rueda CPU puede retirar archivos que
# comparte con la rueda GPU. Solo en ese caso restauramos ORT GPU desde la
# caché de pip; en instalaciones limpias esta reparación no se ejecuta.
$onnxRuntimeReady = Test-EnvironmentPythonCode `
    -Code $cudaProbe `
    -Name "onnxruntime-compatibility"

if (-not $onnxRuntimeReady) {
    Write-Host "Restaurando los archivos de ONNX Runtime GPU desde la caché de pip."
    Invoke-EnvironmentPython -Arguments @(
        "-m", "pip", "install",
        "--force-reinstall", "--no-deps",
        "onnxruntime-gpu==$OnnxRuntimeVersion"
    )
}

Invoke-EnvironmentPythonCode -Code $cudaProbe -Name "cuda-probe"

if (-not $SkipMeshAssist) {
    Write-Step "Preparando MediaPipe Face Landmarker para el backend asistido por malla"
    $geometryModel = Join-Path $ProjectRoot "models\face_landmarker.task"

    if (-not (Test-Path $geometryModel) -or (Get-Item $geometryModel).Length -lt 1000000) {
        $geometryUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        Invoke-WebRequest -Uri $geometryUrl -OutFile $geometryModel
    }

    if (-not (Test-Path $geometryModel) -or (Get-Item $geometryModel).Length -lt 1000000) {
        throw "El bundle Face Landmarker no parece válido."
    }
}

if (-not $SkipHifiFace3DMM) {
    Write-Step "Instalando PyTorch CUDA para HifiFace 3DMM"

    $expectedTorchCudaMajor = ""
    if ($TorchIndexUrl -match "/cu(?<major>\d{2})\d") {
        $expectedTorchCudaMajor = $Matches.major
    }

    $previousExpectedCudaMajor = $env:FACESWAP_PRO_EXPECTED_TORCH_CUDA_MAJOR
    $env:FACESWAP_PRO_EXPECTED_TORCH_CUDA_MAJOR = $expectedTorchCudaMajor
    $torchCompatibilityProbe = @'
import os
import sys

try:
    import torch
except Exception as exc:
    print(f"PyTorch actual no se puede importar: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

cuda_runtime = torch.version.cuda or ""
expected_major = os.environ.get("FACESWAP_PRO_EXPECTED_TORCH_CUDA_MAJOR", "")
compatible = bool(torch.cuda.is_available())
if expected_major:
    compatible = compatible and cuda_runtime.startswith(expected_major + ".")

print("PyTorch existente:", torch.__version__)
print("CUDA runtime existente:", cuda_runtime)
print("CUDA disponible:", torch.cuda.is_available())
raise SystemExit(0 if compatible else 1)
'@

    try {
        $torchAlreadyReady = Test-EnvironmentPythonCode `
            -Code $torchCompatibilityProbe `
            -Name "torch-compatibility"
    } finally {
        if ($null -eq $previousExpectedCudaMajor) {
            Remove-Item Env:FACESWAP_PRO_EXPECTED_TORCH_CUDA_MAJOR `
                -ErrorAction SilentlyContinue
        } else {
            $env:FACESWAP_PRO_EXPECTED_TORCH_CUDA_MAJOR = $previousExpectedCudaMajor
        }
    }

    if ($torchAlreadyReady) {
        Write-Host "La instalación existente de PyTorch CUDA es compatible; no se reinstalará."
    } else {
        # Evita conservar accidentalmente una build CPU o una familia CUDA distinta.
        Invoke-EnvironmentPython -Arguments @(
            "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"
        )

        Invoke-EnvironmentPython -Arguments @(
            "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", $TorchIndexUrl
        )
    }

    $torchProbe = @'
import sys
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch no puede usar CUDA.")

sys.exit(0)
'@

    Invoke-EnvironmentPythonCode -Code $torchProbe -Name "torch-probe"

    Write-Step "Instalando dependencias Python de HifiFace"
    Invoke-EnvironmentPython -Arguments @(
        "-m", "pip", "install", "kornia>=0.7,<1", "loguru>=0.7,<1"
    )

    Write-Step "Preparando el runtime externo HifiFace"
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if (-not $git) {
        throw "Git no está instalado. Instala Git for Windows para preparar HifiFace."
    }

    $hifiFaceRepository = Join-Path $ProjectRoot "third_party\HiFiFace-pytorch"
    Install-HifiFaceRepository `
        -GitExecutable $git.Source `
        -RepositoryUrl $HifiFaceRepositoryUrl `
        -DestinationPath $hifiFaceRepository

    if ($HifiFaceAssetsPath) {
        Write-Step "Copiando checkpoints HifiFace proporcionados por el usuario"
        Copy-HifiFaceAssets `
            -SourcePath $HifiFaceAssetsPath `
            -DestinationPath (Join-Path $ProjectRoot "models\hififace")
    }
}

$ffmpegLink = $null
if (-not $SkipFfmpeg) {
    $ffmpegLink = Install-GyanFfmpeg

    Write-Step "Configurando FFmpeg con NVENC para el entorno"
    Invoke-CondaCommand -Arguments @(
        "env", "config", "vars", "set", "-n", $EnvironmentName,
        "FACESWAP_PRO_FFMPEG=$ffmpegLink"
    )
}

if (-not $SkipTests) {
    Write-Step "Ejecutando pruebas unitarias"
    Invoke-EnvironmentPython -Arguments @("-m", "pytest", "-q", "tests")
}

Write-Step "Ejecutando diagnóstico base"
$cli = Join-Path $environmentPrefix "Scripts\faceswap-pro.exe"
if (-not (Test-Path $cli)) {
    throw "No se encontró el comando instalado: $cli"
}

if ($ffmpegLink) {
    $env:FACESWAP_PRO_FFMPEG = $ffmpegLink
}

& $cli doctor
if ($LASTEXITCODE -ne 0) {
    throw "El diagnóstico base no fue satisfactorio."
}

if (-not $SkipHifiFace3DMM) {
    Write-Step "Validando el perfil HifiFace 3DMM"
    & $cli doctor --config ".\config\quality_3dmm.yaml"
    $hifiFaceReady = ($LASTEXITCODE -eq 0)

    if (-not $hifiFaceReady) {
        Write-WarningBlock -Lines @(
            "El runtime HifiFace y PyTorch CUDA quedaron instalados, pero faltan uno o más checkpoints.",
            "No se incluye ni se descarga automáticamente Basel Face Model (BFM): su licencia prohíbe redistribuirlo",
            "y exige que cada usuario acepte sus condiciones y solicite acceso directamente.",
            "",
            "Cuando tengas los archivos autorizados, colócalos bajo models\hififace o vuelve a ejecutar:",
            "  .\scripts\setup_windows.ps1 -HifiFaceAssetsPath C:\ruta\hififace-assets.zip",
            "",
            "El perfil INSwapper asistido por malla continúa disponible:",
            "  faceswap-pro run --config .\config\quality_mesh_assisted.yaml"
        )
    }
}

Write-Host "`nInstalación terminada." -ForegroundColor Green
Write-Host "Activa el entorno con: conda activate $EnvironmentName"
Write-Host "Ejecuta con: powershell -ExecutionPolicy Bypass -File .\scripts\run_example.ps1"
