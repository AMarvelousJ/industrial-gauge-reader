param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment not found: $Python"
}

$Arguments = @(
    "-m", "data_premark.annotation_app",
    "--review-csv", "outputs/data_premark_v1/review.csv",
    "--manifest", "outputs/data_premark_v1/review_manifest.json",
    "--source-root", "all_set",
    "--port", $Port
)
if ($NoBrowser) {
    $Arguments += "--no-browser"
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
}
finally {
    Pop-Location
}
