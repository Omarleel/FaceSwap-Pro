[CmdletBinding()]
param(
    [string]$RepositoryPath = ".\third_party\DreamID-V",
    [string]$ConfigPath = ".\config\quality_dreamidv.yaml",
    [string]$ModelsPath = ".\models\dreamidv",

    [string]$NumpyVersion = "1.26.4",
    [string]$OpenCvVersion = "4.10.0.84",
    [string]$OnnxRuntimeVersion = "1.26.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",

    [switch]$SkipDependencies,
    [switch]$SkipModelDownloads,
    [switch]$SkipDoctor,
    [switch]$ForceModelDownloads,

    # Opcionales. No son necesarios para DreamID-V Faster en una sola GPU.
    [switch]$InstallFlashAttention,
    [switch]$InstallXfuser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Info {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host $Message -ForegroundColor DarkCyan
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "El comando falló ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    Invoke-Checked -FilePath $script:PythonExecutable -Arguments $Arguments
}

function Invoke-PythonCode {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $temporaryScript = Join-Path ([System.IO.Path]::GetTempPath()) (
        "faceswap-pro-dreamidv-$Name-" + [Guid]::NewGuid().ToString("N") + ".py"
    )

    try {
        $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
        [System.IO.File]::WriteAllText($temporaryScript, $Code, $utf8WithoutBom)
        Invoke-Python -Arguments @($temporaryScript)
    } finally {
        Remove-Item -Path $temporaryScript -Force -ErrorAction SilentlyContinue
    }
}

function Test-PythonCode {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $temporaryScript = Join-Path ([System.IO.Path]::GetTempPath()) (
        "faceswap-pro-dreamidv-$Name-" + [Guid]::NewGuid().ToString("N") + ".py"
    )

    try {
        $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
        [System.IO.File]::WriteAllText($temporaryScript, $Code, $utf8WithoutBom)
        & $script:PythonExecutable $temporaryScript
        return ($LASTEXITCODE -eq 0)
    } finally {
        Remove-Item -Path $temporaryScript -Force -ErrorAction SilentlyContinue
    }
}

function Set-Utf8FileContent {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8WithoutBom)
}

function Get-ProjectRoot {
    $scriptDirectory = $PSScriptRoot
    $parentDirectory = Split-Path -Parent $scriptDirectory

    if ($parentDirectory -and (Test-Path (Join-Path $parentDirectory "pyproject.toml"))) {
        return (Resolve-Path $parentDirectory).Path
    }

    if (Test-Path (Join-Path (Get-Location) "pyproject.toml")) {
        return (Resolve-Path (Get-Location)).Path
    }

    throw "No se encontró pyproject.toml. Ejecuta el script desde FaceSwap-Pro o colócalo dentro de scripts\."
}

function Resolve-HfExecutable {
    $pythonDirectory = Split-Path -Parent $script:PythonExecutable
    $candidate = Join-Path $pythonDirectory "Scripts\hf.exe"
    if (Test-Path $candidate -PathType Leaf) {
        return $candidate
    }

    $command = Get-Command hf.exe -ErrorAction SilentlyContinue
    if (-not $command) {
        $command = Get-Command hf -ErrorAction SilentlyContinue
    }
    if ($command) {
        return $command.Source
    }

    throw "No se encontró hf.exe después de instalar huggingface_hub. Reabre la terminal Conda y vuelve a ejecutar el script."
}

function Invoke-HfDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [string[]]$Files = @(),
        [Parameter(Mandatory = $true)][string]$LocalDirectory
    )

    New-Item -ItemType Directory -Force -Path $LocalDirectory | Out-Null

    $arguments = @("download", $Repository)
    if ($Files.Count -gt 0) {
        $arguments += $Files
    }
    $arguments += @("--local-dir", $LocalDirectory)

    Invoke-Checked -FilePath $script:HfExecutable -Arguments $arguments
}

function Update-ProjectDependencyPins {
    $pyprojectPath = Join-Path $script:ProjectRoot "pyproject.toml"
    $content = [System.IO.File]::ReadAllText($pyprojectPath)
    $updated = $content

    $updated = [regex]::Replace(
        $updated,
        '"numpy[^"\r\n]*"',
        ('"numpy==' + $NumpyVersion + '"'),
        1
    )
    $updated = [regex]::Replace(
        $updated,
        '"opencv-python-headless[^"\r\n]*"',
        ('"opencv-python-headless==' + $OpenCvVersion + '"'),
        1
    )

    if ($updated -ne $content) {
        Set-Utf8FileContent -Path $pyprojectPath -Content $updated
        Write-Info "pyproject.toml fijado a NumPy $NumpyVersion y OpenCV headless $OpenCvVersion."
    }
}

function Configure-DreamIdVToUseCurrentPython {
    $resolvedConfig = Join-Path $script:ProjectRoot $ConfigPath
    if (-not (Test-Path $resolvedConfig -PathType Leaf)) {
        throw "No se encontró la configuración DreamID-V: $resolvedConfig"
    }

    $content = [System.IO.File]::ReadAllText($resolvedConfig)
    $updated = [regex]::Replace(
        $content,
        '(?m)^(\s*)python_executable\s*:\s*.*$',
        '${1}python_executable: python'
    )

    if ($updated -eq $content -and $content -notmatch '(?m)^\s*python_executable\s*:') {
        throw "No se encontró python_executable en $resolvedConfig"
    }

    if ($updated -ne $content) {
        Set-Utf8FileContent -Path $resolvedConfig -Content $updated
        Write-Info "quality_dreamidv.yaml configurado para usar el entorno Conda activo."
    }
}

function Fix-DreamIdVDoctorImportOrder {
    $backendPath = Join-Path $script:ProjectRoot "src\faceswap_pro\dreamidv_backend.py"
    if (-not (Test-Path $backendPath -PathType Leaf)) {
        throw "No se encontró $backendPath"
    }

    $content = [System.IO.File]::ReadAllText($backendPath)
    $oldImport = "    import cv2, decord, diffusers, numpy, onnxruntime, torch, torchvision, transformers"
    $newImport = "    import torch`r`n    import torchvision`r`n    import cv2, decord, diffusers, numpy, onnxruntime, transformers"

    if ($content.Contains($oldImport)) {
        $content = $content.Replace($oldImport, $newImport)
        Set-Utf8FileContent -Path $backendPath -Content $content
        Write-Info "Corregido el orden de carga de DLL: torch se importa antes de ONNX Runtime."
    }
}

function Enable-DreamIdVTorchAttentionFallback {
    $moduleDirectory = Join-Path $script:ProjectRoot "third_party\DreamID-V\dreamidv_wan_faster\modules"
    $modelPath = Join-Path $moduleDirectory "model.py"
    $attentionPath = Join-Path $moduleDirectory "attention.py"
    if (-not (Test-Path $modelPath -PathType Leaf)) {
        throw "No se encontró el módulo Faster de DreamID-V: $modelPath"
    }
    if (-not (Test-Path $attentionPath -PathType Leaf)) {
        throw "No se encontró attention.py de DreamID-V: $attentionPath"
    }

    $content = [System.IO.File]::ReadAllText($modelPath)
    $updated = $content.Replace(
        "from .attention import flash_attention",
        "from .attention import attention"
    )
    $updated = $updated.Replace("flash_attention(", "attention(")

    if ($updated -ne $content) {
        Set-Utf8FileContent -Path $modelPath -Content $updated
    }
    if ($updated.Contains("flash_attention(")) {
        throw "No se pudieron reemplazar todas las llamadas directas a flash_attention."
    }

    $backupPath = "$attentionPath.upstream.bak"
    if (-not (Test-Path $backupPath -PathType Leaf)) {
        Copy-Item -Path $attentionPath -Destination $backupPath -Force
    }

    $facade = @'
"""Adaptador instalado por FaceSwap-Pro.

La implementación real vive en faceswap_pro.dreamidv_sdpa para poder corregir
máscaras, seleccionar kernels fusionados y registrar el backend efectivo sin
mantener un fork completo de DreamID-V.
"""
from faceswap_pro.dreamidv_sdpa import (
    attention,
    flash_attention,
    install_attention_override,
    sdpa_runtime_summary,
)

__all__ = [
    "attention",
    "flash_attention",
    "install_attention_override",
    "sdpa_runtime_summary",
]
'@
    Set-Utf8FileContent -Path $attentionPath -Content $facade
    Write-Info "DreamID-V Faster configurado con SDPA nativo, máscara correcta y selección explícita de kernels."
}

function Test-WanCheckpoint {
    param([Parameter(Mandatory = $true)][string]$Directory)

    $required = @(
        "config.json",
        "Wan2.1_VAE.pth",
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google\umt5-xxl\spiece.model"
    )

    foreach ($relativePath in $required) {
        if (-not (Test-Path (Join-Path $Directory $relativePath) -PathType Leaf)) {
            return $false
        }
    }
    return $true
}

$ProjectRoot = Get-ProjectRoot
$script:ProjectRoot = $ProjectRoot
Push-Location $ProjectRoot

try {
    Write-Step "Validando el entorno Conda activo"

    if (-not $env:CONDA_PREFIX) {
        throw "No hay un entorno Conda activo. Ejecuta: conda activate FaceSwap-Pro"
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    }
    if (-not $pythonCommand) {
        throw "No se encontró Python en el entorno activo."
    }

    $script:PythonExecutable = $pythonCommand.Source
    $condaPrefix = (Resolve-Path $env:CONDA_PREFIX).Path.TrimEnd('\')
    $pythonResolved = (Resolve-Path $script:PythonExecutable).Path
    if (-not $pythonResolved.StartsWith($condaPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "python apunta fuera del Conda activo: $pythonResolved"
    }

    Write-Info "Entorno: $env:CONDA_DEFAULT_ENV"
    Write-Info "Python:  $pythonResolved"
    Write-Info "Raíz:    $ProjectRoot"

    $pythonVersionProbe = @'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Se requiere Python 3.11; encontrado: {sys.version}")
print("Python:", sys.version.split()[0])
print("Ejecutable:", sys.executable)
'@
    Invoke-PythonCode -Code $pythonVersionProbe -Name "python-version"

    Write-Step "Clonando o validando DreamID-V"
    New-Item -ItemType Directory -Force -Path ".\third_party" | Out-Null

    $resolvedRepositoryPath = Join-Path $ProjectRoot $RepositoryPath
    if (-not (Test-Path $resolvedRepositoryPath)) {
        $gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
        if (-not $gitCommand) {
            $gitCommand = Get-Command git -ErrorAction SilentlyContinue
        }
        if (-not $gitCommand) {
            throw "Git no está instalado o no está en PATH."
        }

        Invoke-Checked -FilePath $gitCommand.Source -Arguments @(
            "clone",
            "--depth", "1",
            "https://github.com/bytedance/DreamID-V.git",
            $resolvedRepositoryPath
        )
    }

    $entryScript = Join-Path $resolvedRepositoryPath "generate_dreamidv_faster.py"
    $contextCheckpoint = Join-Path $resolvedRepositoryPath "dreamidv_wan_faster\context.pth"
    if (-not (Test-Path $entryScript -PathType Leaf)) {
        throw "El checkout de DreamID-V está incompleto: falta $entryScript"
    }
    if (-not (Test-Path $contextCheckpoint -PathType Leaf)) {
        throw "El checkout de DreamID-V está incompleto: falta $contextCheckpoint"
    }

    Update-ProjectDependencyPins
    Configure-DreamIdVToUseCurrentPython
    Fix-DreamIdVDoctorImportOrder
    Enable-DreamIdVTorchAttentionFallback

    if (-not $SkipDependencies) {
        Write-Step "Actualizando herramientas de instalación"
        Invoke-Python -Arguments @(
            "-m", "pip", "install", "--upgrade",
            "pip", "setuptools<82", "wheel"
        )

        Write-Step "Validando PyTorch CUDA"
        $torchProbe = @'
from packaging.version import Version
import torch
import torchvision
version = Version(torch.__version__.split("+")[0])
if version < Version("2.4.0"):
    raise SystemExit(f"PyTorch demasiado antiguo: {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch no detecta CUDA")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
'@
        $torchReady = Test-PythonCode -Code $torchProbe -Name "torch-cuda"
        if (-not $torchReady) {
            Write-Host "PyTorch CUDA no está listo; instalando la build cu128." -ForegroundColor Yellow
            Invoke-Python -Arguments @(
                "-m", "pip", "install", "--upgrade",
                "torch", "torchvision", "torchaudio",
                "--index-url", $TorchIndexUrl
            )
            Invoke-PythonCode -Code $torchProbe -Name "torch-cuda-after-install"
        } else {
            Write-Info "PyTorch CUDA ya es compatible; no se reinstalará."
        }

        Write-Step "Normalizando NumPy, OpenCV y ONNX Runtime GPU"
        $runtimeProbe = @"
import importlib.metadata as metadata
import cv2
import numpy
import onnxruntime as ort

names = {dist.metadata.get("Name", "").lower() for dist in metadata.distributions()}
if numpy.__version__ != "$NumpyVersion":
    raise SystemExit(f"NumPy inesperado: {numpy.__version__}")
if cv2.__version__ != "4.10.0":
    raise SystemExit(f"OpenCV inesperado: {cv2.__version__}")
if "opencv-python-headless" not in names:
    raise SystemExit("Falta opencv-python-headless")
for conflicting in (
    "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless",
    "onnxruntime", "onnxruntime-directml"
):
    if conflicting in names:
        raise SystemExit(f"Distribución incompatible instalada: {conflicting}")
if "onnxruntime-gpu" not in names:
    raise SystemExit("Falta onnxruntime-gpu")
if ort.__version__ != "$OnnxRuntimeVersion":
    raise SystemExit(f"ONNX Runtime inesperado: {ort.__version__}")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("ONNX Runtime no expone CUDAExecutionProvider")
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("ONNX Runtime:", ort.__version__)
print("Providers:", ort.get_available_providers())
"@
        $runtimeReady = Test-PythonCode -Code $runtimeProbe -Name "runtime-stack"
        if (-not $runtimeReady) {
            Invoke-Python -Arguments @(
                "-m", "pip", "uninstall", "-y",
                "opencv-python",
                "opencv-python-headless",
                "opencv-contrib-python",
                "opencv-contrib-python-headless",
                "onnxruntime",
                "onnxruntime-directml"
            )

            $sitePackages = (& $script:PythonExecutable -c "import site; print(site.getsitepackages()[0])").Trim()
            if ($LASTEXITCODE -ne 0 -or -not $sitePackages) {
                throw "No se pudo resolver site-packages para limpiar residuos de cv2."
            }
            $cv2Residual = Join-Path $sitePackages "cv2"
            if (Test-Path $cv2Residual) {
                Remove-Item -Path $cv2Residual -Recurse -Force
            }

            Invoke-Python -Arguments @(
                "-m", "pip", "install", "--force-reinstall",
                "numpy==$NumpyVersion"
            )
            Invoke-Python -Arguments @(
                "-m", "pip", "install", "--force-reinstall", "--no-deps",
                "opencv-python-headless==$OpenCvVersion"
            )
            Invoke-Python -Arguments @(
                "-m", "pip", "install", "--force-reinstall",
                "onnxruntime-gpu[cuda,cudnn]==$OnnxRuntimeVersion"
            )
            Invoke-PythonCode -Code $runtimeProbe -Name "runtime-stack-after-repair"
        } else {
            Write-Info "NumPy, OpenCV y ONNX Runtime GPU ya están normalizados."
        }

        Write-Step "Instalando dependencias compatibles de DreamID-V"
        $constraintsPath = Join-Path $ProjectRoot "dreamidv-constraints.txt"
        $constraints = @"
numpy==$NumpyVersion
opencv-python-headless==$OpenCvVersion
onnxruntime-gpu==$OnnxRuntimeVersion
"@
        Set-Utf8FileContent -Path $constraintsPath -Content $constraints

        # No se instala requirements.txt directamente: introduciría ONNX Runtime
        # CPU, otra variante de OpenCV, MediaPipe 0.10.5, flash-attn y xfuser.
        $dreamDependencies = @(
            "diffusers==0.35.2",
            "transformers==4.49.0",
            "tokenizers==0.21.4",
            "accelerate==1.11.0",
            "imageio",
            "imageio-ffmpeg",
            "easydict",
            "ftfy",
            "dashscope",
            "decord==0.6.0",
            "huggingface_hub[cli]",
            "hf_xet",
            "safetensors",
            "einops",
            "omegaconf"
        )
        Invoke-Python -Arguments (
            @("-m", "pip", "install", "-c", $constraintsPath) + $dreamDependencies
        )

        if ($InstallFlashAttention) {
            Write-Step "Instalando flash-attn bajo solicitud explícita"
            Invoke-Python -Arguments @(
                "-m", "pip", "install", "flash-attn", "--no-build-isolation"
            )
        }

        if ($InstallXfuser) {
            Write-Step "Instalando xFuser para ejecución distribuida"
            Invoke-Python -Arguments @(
                "-m", "pip", "install", "xfuser==0.4.4"
            )
        }

        Write-Step "Actualizando la instalación editable de FaceSwap-Pro"
        Invoke-Python -Arguments @("-m", "pip", "install", "--no-deps", "-e", ".")

        Write-Step "Comprobando importaciones de DreamID-V"
        $dependencyProbe = @'
import torch
import torchvision
import cv2
import decord
import diffusers
import numpy
import onnxruntime
import transformers
import accelerate
import imageio
import easydict
import ftfy
import safetensors
import einops
import omegaconf

if not torch.cuda.is_available():
    raise SystemExit("CUDA no está disponible")
if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
    raise SystemExit("ONNX Runtime no tiene CUDAExecutionProvider")

print("Dependencias DreamID-V: OK")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("Diffusers:", diffusers.__version__)
print("Transformers:", transformers.__version__)
'@
        Invoke-PythonCode -Code $dependencyProbe -Name "dreamidv-dependencies"
    } else {
        Write-Info "Instalación de dependencias omitida por -SkipDependencies."
        Invoke-Python -Arguments @("-m", "pip", "install", "--no-deps", "-e", ".")
    }

    if (-not $SkipModelDownloads) {
        Write-Step "Preparando Hugging Face CLI"
        $script:HfExecutable = Resolve-HfExecutable
        Write-Info "Hugging Face CLI: $script:HfExecutable"

        $resolvedModelsPath = Join-Path $ProjectRoot $ModelsPath
        $dreamCheckpoint = Join-Path $resolvedModelsPath "dreamidv_faster.pth"
        $wanDirectory = Join-Path $resolvedModelsPath "Wan2.1-T2V-1.3B"
        $poseDirectory = Join-Path $resolvedRepositoryPath "pose\models"
        $dwPoseModel = Join-Path $poseDirectory "dw-ll_ucoco_384.onnx"
        $yoloModel = Join-Path $poseDirectory "yolox_l.onnx"

        Write-Host "Los checkpoints ocupan más de 20 GB. Las descargas incompletas se pueden reanudar." -ForegroundColor Yellow

        Write-Step "Descargando DreamID-V Faster"
        if ($ForceModelDownloads -or -not (Test-Path $dreamCheckpoint -PathType Leaf)) {
            Invoke-HfDownload `
                -Repository "XuGuo699/DreamID-V" `
                -Files @("dreamidv_faster.pth") `
                -LocalDirectory $resolvedModelsPath
        } else {
            Write-Info "Checkpoint DreamID-V ya presente; se omite la descarga."
        }

        Write-Step "Descargando Wan 2.1 T2V 1.3B"
        if ($ForceModelDownloads -or -not (Test-WanCheckpoint -Directory $wanDirectory)) {
            Invoke-HfDownload `
                -Repository "Wan-AI/Wan2.1-T2V-1.3B" `
                -LocalDirectory $wanDirectory
        } else {
            Write-Info "Checkpoint Wan 2.1 ya completo; se omite la descarga."
        }

        Write-Step "Descargando modelos DWPose"
        if (
            $ForceModelDownloads -or
            -not (Test-Path $dwPoseModel -PathType Leaf) -or
            -not (Test-Path $yoloModel -PathType Leaf)
        ) {
            Invoke-HfDownload `
                -Repository "yzd-v/DWPose" `
                -Files @("dw-ll_ucoco_384.onnx", "yolox_l.onnx") `
                -LocalDirectory $poseDirectory
        } else {
            Write-Info "Modelos DWPose ya presentes; se omite la descarga."
        }
    } else {
        Write-Info "Descarga de modelos omitida por -SkipModelDownloads."
    }

    if (-not $SkipDoctor) {
        Write-Step "Ejecutando diagnóstico final"

        $faceswapCommand = Get-Command faceswap-pro.exe -ErrorAction SilentlyContinue
        if (-not $faceswapCommand) {
            $faceswapCommand = Get-Command faceswap-pro -ErrorAction SilentlyContinue
        }

        if (-not $faceswapCommand) {
            $pythonDirectory = Split-Path -Parent $script:PythonExecutable
            $candidate = Join-Path $pythonDirectory "Scripts\faceswap-pro.exe"
            if (Test-Path $candidate -PathType Leaf) {
                $faceswapExecutable = $candidate
            } else {
                throw "No se encontró faceswap-pro.exe después de instalar el proyecto."
            }
        } else {
            $faceswapExecutable = $faceswapCommand.Source
        }

        & $faceswapExecutable doctor --config $ConfigPath
        $doctorExitCode = $LASTEXITCODE
        if ($doctorExitCode -ne 0) {
            if ($SkipModelDownloads) {
                Write-Host "El diagnóstico no está listo porque se omitieron los modelos." -ForegroundColor Yellow
            } else {
                throw "faceswap-pro doctor terminó con código $doctorExitCode."
            }
        }
    }

    Write-Host "`nDreamID-V quedó preparado en el entorno Conda actual." -ForegroundColor Green
    Write-Host "Ejemplo de ejecución:" -ForegroundColor Cyan
    Write-Host "faceswap-pro run ``" -ForegroundColor White
    Write-Host "  --input .\inputs\videos\input.mp4 ``" -ForegroundColor White
    Write-Host "  --source-dir .\inputs\source_faces ``" -ForegroundColor White
    Write-Host "  --target-ref .\inputs\target_faces\target.jpg ``" -ForegroundColor White
    Write-Host "  --config .\config\quality_dreamidv.yaml ``" -ForegroundColor White
    Write-Host "  --output .\outputs\videos\resultado_dreamidv.mp4" -ForegroundColor White
} finally {
    Pop-Location
}