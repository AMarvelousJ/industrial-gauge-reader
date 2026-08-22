param(
    [string]$ReviewDir = "outputs/pointer_keypoint_review_v1",
    [int]$Port = 8766
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found: $Python"
}

Push-Location $ProjectRoot
try {
    & $Python -m data_premark.annotation_app `
        --review-csv (Join-Path $ReviewDir "review.csv") `
        --manifest (Join-Path $ReviewDir "review_manifest.json") `
        --source-root "all_set" `
        --port $Port
}
finally {
    Pop-Location
}
