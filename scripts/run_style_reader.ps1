param(
    [string]$OutputDir = "outputs/style_reader/latest"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found: $Python"
}

Push-Location $ProjectRoot
try {
    & $Python -m style_reader.run_manifest --output-dir $OutputDir
    & $Python -m style_reader.evaluate_readings `
        --truth "docs/reading_ground_truth_audit.json" `
        --predictions (Join-Path $OutputDir "predictions.json") `
        --output (Join-Path $OutputDir "evaluation.json")
    & $Python -m style_reader.build_report --output-dir $OutputDir
}
finally {
    Pop-Location
}
