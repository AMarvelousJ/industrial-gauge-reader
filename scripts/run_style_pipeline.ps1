param(
    [int]$Epochs = 12,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found: $python. Create .venv and install requirements.txt first."
}

Push-Location $projectRoot
try {
    if (-not $SkipTrain) {
        & $python -m style_classifier.train --epochs $Epochs --workers 2
        if ($LASTEXITCODE -ne 0) { throw "Style classifier training failed." }
    }

    & $python -m style_classifier.predict_manifest
    if ($LASTEXITCODE -ne 0) { throw "Frozen-YOLO manifest prediction failed." }

    & $python -m style_classifier.evaluate --target 0.80
    if ($LASTEXITCODE -ne 0) { throw "The 80% acceptance target was not met." }

    & $python -m pytest tests\style_classifier -q
    if ($LASTEXITCODE -ne 0) { throw "Regression tests failed." }

    Write-Host "Pipeline passed. Report: outputs\style_classifier\final_report.md"
}
finally {
    Pop-Location
}
