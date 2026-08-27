# run_camera_demo.ps1 - Windows 摄像头仪表读数 Demo
# 用法:
#   .\scripts\run_camera_demo.ps1                      (默认摄像头0, 640x480)
#   .\scripts\run_camera_demo.ps1 -Camera 1 -Width 1280 -Height 720
#   .\scripts\run_camera_demo.ps1 -ImageDir assets\demo_images   (图片回放)
param(
    [int]$Camera = 0,
    [int]$Width = 640,
    [int]$Height = 480,
    [string]$ImageDir = "",
    [int]$MaxFps = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found: $Python"
}

Push-Location $ProjectRoot
try {
    $ArgsList = @("-m", "style_reader.demo_camera", "--camera", $Camera, "--width", $Width, "--height", $Height)
    if ($MaxFps -gt 0) { $ArgsList += @("--max-fps", $MaxFps) }
    if ($ImageDir) { $ArgsList += @("--image-dir", $ImageDir) }
    & $Python @ArgsList
}
finally {
    Pop-Location
}
