# Lanzador local del Verificador de Identidad (backend únicamente)
# Uso:
#   .\run.ps1            -> arranca el backend en el puerto 8000
#   .\run.ps1 -Port 9000 -> backend en otro puerto
param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# Preparar venv si no existe
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "[setup] Creando venv..." -ForegroundColor Yellow
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el venv. Verifica que 'py' apunte a Python 3." }
    & $python -m pip install --upgrade pip
    if (Test-Path -LiteralPath "requirements.txt") {
        & $python -m pip install -r requirements.txt
    }
}

Write-Host "[server] Arrancando backend en http://localhost:$Port ..." -ForegroundColor Cyan
Write-Host "[server] Ctrl+C para detener." -ForegroundColor DarkGray

& $python -m uvicorn main:app --host 127.0.0.1 --port $Port --reload
