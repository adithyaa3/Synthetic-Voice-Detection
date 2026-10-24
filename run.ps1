# Windows helper.  Usage:  .\run.ps1 <setup|pipeline|refit|test|app|e2e|all>
param([Parameter(Position = 0)][string]$Task = "help")
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Need-Venv { if (-not (Test-Path $py)) { throw "No .venv found. Run: .\run.ps1 setup" } }

switch ($Task) {
  "setup" {
    if (-not (Test-Path $py)) { python -m venv (Join-Path $PSScriptRoot ".venv") }
    & $py -m pip install --upgrade pip
    & $py -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    & $py -m pip install -e $PSScriptRoot --no-deps
  }
  "fetch"    { Need-Venv; & $py scripts/fetch_supplementary_asvspoof.py }
  "pipeline" { Need-Venv; & $py -m voicedetect.pipeline all }
  "refit"    { Need-Venv; & $py scripts/train.py --refit }
  "test"     { Need-Venv; & $py -m pytest }
  "app"      { Need-Venv; & $py -m streamlit run app/streamlit_app.py --server.port 8501 }
  "e2e"      { Need-Venv; & $py scripts/e2e_check.py }
  "all"      { & $PSCommandPath setup; & $PSCommandPath fetch; & $PSCommandPath pipeline; & $PSCommandPath test }
  default    { Write-Host "Usage: .\run.ps1 <setup|fetch|pipeline|refit|test|app|e2e|all>" }
}
