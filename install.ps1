# ============================================================================
# Instalador del Verificador de Identidad - Opción 1
# Uso: Ejecutar como Administrador
#   Set-ExecutionPolicy Bypass -Scope Process; .\install.ps1
# ============================================================================
param(
    [switch]$SkipChrome
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  INSTALADOR - Verificador de Identidad" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Verificar Python 3.11
# ---------------------------------------------------------------------------
Write-Host "[1/5] Verificando Python 3.11..." -ForegroundColor Yellow

$pythonCmd = $null
try {
    $ver = & py -3.11 --version 2>&1
    if ($ver -match "Python 3\.11") {
        $pythonCmd = "py -3.11"
        Write-Host "  OK: $ver" -ForegroundColor Green
    }
} catch {}

if (-not $pythonCmd) {
    Write-Host "  Python 3.11 no encontrado. Intentando instalar..." -ForegroundColor Yellow
    
    # Verificar si winget está disponible
    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    if ($hasWinget) {
        Write-Host "  Instalando Python 3.11 via winget..." -ForegroundColor Yellow
        winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
        
        # Recargar PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        $ver = & py -3.11 --version 2>&1
        if ($ver -match "Python 3\.11") {
            $pythonCmd = "py -3.11"
            Write-Host "  OK: $ver instalado" -ForegroundColor Green
        }
    }
    
    if (-not $pythonCmd) {
        Write-Host "  ERROR: No se pudo instalar Python 3.11 automaticamente." -ForegroundColor Red
        Write-Host "  Descargalo manualmente desde: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  Asegurate de marcar 'Add Python to PATH' durante la instalacion." -ForegroundColor Yellow
        exit 1
    }
}

# ---------------------------------------------------------------------------
# 2. Crear virtual environment
# ---------------------------------------------------------------------------
Write-Host "[2/5] Creando entorno virtual..." -ForegroundColor Yellow

$venv = Join-Path $PSScriptRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    & $pythonCmd.Split()[0] $pythonCmd.Split()[1] -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el venv" }
    Write-Host "  OK: .venv creado" -ForegroundColor Green
} else {
    Write-Host "  OK: .venv ya existe" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 3. Instalar dependencias
# ---------------------------------------------------------------------------
Write-Host "[3/5] Instalando dependencias (puede tardar 5-10 min)..." -ForegroundColor Yellow

& $python -m pip install --upgrade pip --quiet
& $python -m pip install -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) { throw "Error instalando dependencias" }
Write-Host "  OK: Dependencias instaladas" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Verificar GPU/CUDA
# ---------------------------------------------------------------------------
Write-Host "[4/5] Verificando GPU..." -ForegroundColor Yellow

$gpuCheck = & $python -c "import torch; print('CUDA:' + ('YES' if torch.cuda.is_available() else 'NO'))" 2>&1

if ($gpuCheck -match "CUDA:YES") {
    $gpuName = & $python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "  OK: GPU detectada - $gpuName" -ForegroundColor Green
} else {
    Write-Host "  WARN: GPU no detectada, usando CPU (mas lento)" -ForegroundColor Yellow
    Write-Host "  Verifica que tengas NVIDIA drivers instalados" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 5. Verificar Chrome extension
# ---------------------------------------------------------------------------
Write-Host "[5/5] Verificando extension de Chrome..." -ForegroundColor Yellow

$extPath = Join-Path $PSScriptRoot "extension-id-verifier"
$manifest = Join-Path $extPath "manifest.json"

if (Test-Path -LiteralPath $manifest) {
    Write-Host "  OK: Extension encontrada en $extPath" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Para instalar la extension:" -ForegroundColor Cyan
    Write-Host "  1. Abrir chrome://extensions" -ForegroundColor White
    Write-Host "  2. Activar 'Modo desarrollador'" -ForegroundColor White
    Write-Host "  3. Click 'Cargar extension descomprimida'" -ForegroundColor White
    Write-Host "  4. Seleccionar carpeta: $extPath" -ForegroundColor White
} else {
    Write-Host "  WARN: No se encontro la extension" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Completado
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  INSTALACION COMPLETADA" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Para iniciar:" -ForegroundColor Cyan
Write-Host "  .\run.ps1" -ForegroundColor White
Write-Host ""
