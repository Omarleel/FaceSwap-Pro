$path = ".\third_party\DreamID-V\dreamidv_wan_faster\modules\model.py"

Copy-Item $path "$path.bak" -Force

$content = Get-Content $path -Raw

$content = $content.Replace(
    "from .attention import flash_attention",
    "from .attention import attention"
)

$content = $content.Replace(
    "flash_attention(",
    "attention("
)

Set-Content $path $content -Encoding UTF8