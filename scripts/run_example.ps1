[CmdletBinding()]
param(
    [string]$EnvironmentName = "FaceSwap-Pro",
    [string]$Config = "config\max_speed.yaml",
    [string]$InputVideo = "inputs\videos\input.mp4",
    [string]$SourceDir = "inputs\source_faces",
    [string]$TargetReference = "inputs\target_faces\target.jpg",
    [string]$SwapperModel = "models\inswapper_128.onnx",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$arguments = @(
    "run",
    "--input", $InputVideo,
    "--source-dir", $SourceDir,
    "--target-ref", $TargetReference,
    "--swapper-model", $SwapperModel,
    "--config", $Config,
    "--output-dir", "outputs\videos",
    "--manifest-dir", "outputs\manifests"
)

if ($Output) {
    $arguments += @("--output", $Output)
}

$command = Get-Command faceswap-pro.exe -ErrorAction SilentlyContinue
if ($command -and $env:CONDA_DEFAULT_ENV -ieq $EnvironmentName) {
    & $command.Source @arguments
    exit $LASTEXITCODE
}

$conda = Get-Command conda.exe -ErrorAction SilentlyContinue
if (-not $conda) {
    throw "Activa el entorno con 'conda activate $EnvironmentName' o instala Conda."
}

& $conda.Source run --no-capture-output -n $EnvironmentName faceswap-pro @arguments
exit $LASTEXITCODE
