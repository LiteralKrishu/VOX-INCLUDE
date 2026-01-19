# VOX-INCLUDE Development Startup Script

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   VOX-INCLUDE DEV ENVIRONMENT STARTUP    " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Start Backend Server in a new window
Write-Host "1. Starting FastAPI Backend (Port 8000)..." -ForegroundColor Yellow
$backendCommand = "& '.\venv\Scripts\Activate.ps1'; uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$backendCommand"

# Wait for server to initialize
Write-Host "   Waiting 5 seconds for backend to warm up..." -ForegroundColor DarkGray
Start-Sleep -Seconds 5

# 2. Start Flutter App (Default: Chrome)
Write-Host "2. Starting Flutter App..." -ForegroundColor Yellow
Write-Host "   Note: To run on Android, close this script and run: flutter run -d emulator-id" -ForegroundColor DarkGray
Set-Location "mobile_app"
flutter run -d chrome

# Note: The backend window remains open after Flutter closes.
Write-Host "Frontend closed. Backend is still running in the other window." -ForegroundColor Green
