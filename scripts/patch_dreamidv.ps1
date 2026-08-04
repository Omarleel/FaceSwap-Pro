$moduleDirectory = ".\third_party\DreamID-V\dreamidv_wan_faster\modules"
$modelPath = Join-Path $moduleDirectory "model.py"
$attentionPath = Join-Path $moduleDirectory "attention.py"

if (-not (Test-Path $modelPath -PathType Leaf)) {
    throw "No se encontró $modelPath"
}
if (-not (Test-Path $attentionPath -PathType Leaf)) {
    throw "No se encontró $attentionPath"
}

Copy-Item $modelPath "$modelPath.bak" -Force
if (-not (Test-Path "$attentionPath.upstream.bak" -PathType Leaf)) {
    Copy-Item $attentionPath "$attentionPath.upstream.bak" -Force
}

$content = Get-Content $modelPath -Raw
$content = $content.Replace(
    "from .attention import flash_attention",
    "from .attention import attention"
)
$content = $content.Replace("flash_attention(", "attention(")
Set-Content $modelPath $content -Encoding UTF8

$facade = @'
"""Adaptador instalado por FaceSwap-Pro."""
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
Set-Content $attentionPath $facade -Encoding UTF8

Write-Host "DreamID-V actualizado con SDPA nativo de FaceSwap-Pro." -ForegroundColor Green
