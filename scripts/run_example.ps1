[CmdletBinding()]
param(
    [string]$EnvironmentName = "FaceSwap-Pro",
    [string]$Config = "config\quality_3dmm.yaml",
    [string]$InputVideo = "inputs\videos\input.mp4",
    [string]$SourceDir = "inputs\source_faces",
    [string]$TargetReference = "inputs\target_faces\target.jpg",

    # Déjalo vacío para usar engine.options.model_path del YAML.
    [string]$ModelPath = "",

    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$runArguments = @(
    "run",
    "--input", $InputVideo,
    "--source-dir", $SourceDir,
    "--target-ref", $TargetReference,
    "--config", $Config,
    "--output-dir", "outputs\videos",
    "--manifest-dir", "outputs\manifests"
)

if ($ModelPath) {
    $runArguments += @("--model-path", $ModelPath)
}

if ($Output) {
    $runArguments += @("--output", $Output)
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "El comando terminó con código $LASTEXITCODE`: $Command $($Arguments -join ' ')"
    }
}

$faceswapCommand = Get-Command faceswap-pro.exe -ErrorAction SilentlyContinue
if ($faceswapCommand -and $env:CONDA_DEFAULT_ENV -ieq $EnvironmentName) {
    Invoke-CheckedCommand `
        -Command $faceswapCommand.Source `
        -Arguments @("doctor", "--config", $Config)

    Invoke-CheckedCommand `
        -Command $faceswapCommand.Source `
        -Arguments $runArguments

    exit 0
}

$conda = Get-Command conda.exe -ErrorAction SilentlyContinue
if (-not $conda) {
    throw "Activa el entorno con 'conda activate $EnvironmentName' o instala Conda."
}

$doctorArguments = @(
    "run", "--no-capture-output", "-n", $EnvironmentName,
    "faceswap-pro", "doctor", "--config", $Config
)

& $conda.Source @doctorArguments
if ($LASTEXITCODE -ne 0) {
    throw "El backend configurado no está listo. Revisa el diagnóstico anterior."
}

$condaRunArguments = @(
    "run", "--no-capture-output", "-n", $EnvironmentName,
    "faceswap-pro"
) + $runArguments

& $conda.Source @condaRunArguments
if ($LASTEXITCODE -ne 0) {
    throw "FaceSwap-Pro terminó con código $LASTEXITCODE."
}

exit 0
