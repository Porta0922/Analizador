# AI System Launcher
# Runs the AI document analyzer on port 8001

$ErrorActionPreference = "Stop"

# Check Python
try {
    $pythonVersion = py -3.11 --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.11 not found. Please install Python 3.11." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Python 3.11 not found. Please install Python 3.11." -ForegroundColor Red
    exit 1
}

# Create venv if needed
if (-not (Test-Path "ai_system\.venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    py -3.11 -m venv ai_system\.venv
}

# Activate venv
& "ai_system\.venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "Installing AI system dependencies..." -ForegroundColor Yellow
pip install -r ai_system\requirements.txt --quiet

# Check if Ollama is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2
    Write-Host "Ollama is running." -ForegroundColor Green
} catch {
    Write-Host "Warning: Ollama is not running on localhost:11434" -ForegroundColor Yellow
    Write-Host "Please start Ollama before using AI features." -ForegroundColor Yellow
}

# Kill any existing process on port 8001
$existing = Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Stopping existing process on port 8001..." -ForegroundColor Yellow
    $existing | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

# Start AI backend on port 8001
Write-Host "Starting AI Document Analyzer on port 8001..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray

python -m uvicorn ai_system.main_ai:app --host 0.0.0.0 --port 8001 --reload
